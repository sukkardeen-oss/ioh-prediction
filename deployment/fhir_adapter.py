"""HL7 FHIR R4 <-> model feature vector, via the two-layer concept map.

Inbound:  Bundle (Patient, Encounter, Observations) -> coded primitives
          -> app.derivations.compute(...)           -> model feature vector
Outbound: probability                               -> RiskAssessment

The structural change from a static-feature service: most model columns are not
present in the payload at all. The payload carries *primitives* (a MAP series,
an HR series, a rhythm code, an episode period), and the derived features are
computed here by transforms that shipped inside the model artifact. See
app/derivations.py for why derived features are versioned rather than coded.

Two rejection policies, both chosen because the alternative fails quietly:

  * Unit mismatch is rejected, never converted. A MAP series sent in kPa and
    read as mmHg is off by ~7.5x and still returns a plausible probability.
  * Samples after the prediction anchor are rejected, not truncated. At
    training time they are outcome leakage; at serving time their presence
    means the caller assembled the window wrongly, which deserves a 422 rather
    than a silently different feature vector.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from .concept_map import ConceptMap
from .derivations import DerivationError, Series, compute
from .schemas import (
    FeatureResolution,
    FHIRBundle,
    FHIREncounter,
    FHIRObservation,
    FHIRPatient,
)

LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"

_UNIT_SYNONYMS: dict[str, set[str]] = {
    "mm[Hg]": {"mm[hg]", "mm[Hg]", "mmhg", "mmHg", "mm hg"},
    "/min": {"/min", "min-1", "bpm", "{beats}/min"},
    "a": {"a", "year", "years", "yr"},
}


class PayloadError(ValueError):
    """Payload is structurally valid but cannot be mapped to a feature vector."""


def _units_compatible(expected: Optional[str], received: Optional[str]) -> bool:
    if expected is None or received is None:
        return True
    if expected == received:
        return True
    return received.strip() in _UNIT_SYNONYMS.get(expected, set())


def _age_years(birth_date: str, reference: datetime) -> float:
    born = date.fromisoformat(birth_date[:10])
    return (reference.date() - born).days / 365.25


def resolve_coded_value(binding, sent_codes, feature_name):
    """Resolve a set of sent codes against a pre-expanded value set.

    Returns (member, unmapped_code_or_None). Applies the binding's
    unmapped_policy; records the outcome so the unmapped RATE is observable
    whichever policy is in force.
    """
    from . import unmapped_stats

    index = binding.value_code_index or {}
    hit = next((index[c] for c in sent_codes if c in index), None)
    if hit is not None:
        unmapped_stats.record(feature_name, unmapped=False)
        return hit, None

    offending = sorted(sent_codes)[0] if sent_codes else None
    unmapped_stats.record(feature_name, unmapped=True, code=offending)

    if binding.unmapped_policy == "map_to_other" and "other" in (binding.value_set or {}):
        return "other", offending

    raise PayloadError(
        f"Code(s) {sorted(sent_codes)} are not in the bound value set for "
        f"'{feature_name}'. Rejected rather than folded into a catch-all: an "
        "unmapped code is a terminology gap, not a clinical finding, and quietly "
        "bucketing it would let a broken binding score every episode on a value "
        "the model never saw. If the code is a legitimate descendant of a bound "
        "concept, expand the closure in the concept map; if scoring should "
        "continue regardless, set unmapped_policy='map_to_other'."
    )


def _split_bundle(bundle: FHIRBundle):
    patient: Optional[FHIRPatient] = None
    encounter: Optional[FHIREncounter] = None
    observations: list[FHIRObservation] = []
    for entry in bundle.entry:
        r = entry.resource
        if isinstance(r, FHIRPatient):
            if patient is not None:
                raise PayloadError(
                    "Bundle contains more than one Patient resource. This endpoint scores "
                    "exactly one episode for one subject; batching risks cross-attribution."
                )
            patient = r
        elif isinstance(r, FHIREncounter):
            encounter = r
        elif isinstance(r, FHIRObservation):
            observations.append(r)
    if patient is None:
        raise PayloadError("Bundle must contain a Patient resource.")
    return patient, encounter, observations


def _codings(obs: FHIRObservation) -> set[tuple[str, str]]:
    return {(c.system, c.code) for c in obs.code.coding}


def _episode_window(observations) -> tuple[datetime, datetime]:
    for obs in observations:
        p = obs.effectivePeriod
        if p is not None and p.start is not None and p.end is not None:
            if p.end <= p.start:
                raise PayloadError("Episode effectivePeriod ends at or before it starts.")
            return p.start, p.end
    raise PayloadError(
        "No Observation carries an effectivePeriod defining the episode window. The "
        "prediction anchor cannot be located, so no feature can be computed over the "
        "correct interval."
    )


def _collect_series(observations, system, code, onset, anchor, expected_unit,
                    name) -> tuple[Series, Optional[str]]:
    """Gather every sample for one coded primitive into a time series.

    Times are seconds since episode onset. A single Observation may carry one
    value, or a whole series via component[] — a common shape for
    waveform-derived data.
    """
    samples: list[tuple[float, float]] = []
    unit_seen: Optional[str] = None

    for obs in observations:
        if (system, code) not in _codings(obs):
            continue

        points: list[tuple[Optional[datetime], float, Optional[str]]] = []
        if obs.valueQuantity is not None:
            q = obs.valueQuantity
            points.append((obs.effectiveDateTime, float(q.value), q.code or q.unit))
        for comp in (obs.component or []):
            if comp.valueQuantity is None:
                continue
            q = comp.valueQuantity
            points.append((comp.effectiveDateTime or obs.effectiveDateTime,
                           float(q.value), q.code or q.unit))

        for t, v, unit in points:
            if t is None:
                raise PayloadError(
                    f"A sample for '{name}' ({system}|{code}) has no timestamp. Series "
                    "primitives must be timestamped to be windowed correctly.")
            if unit_seen is None:
                unit_seen = unit
            if not _units_compatible(expected_unit, unit):
                raise PayloadError(
                    f"Unit mismatch for '{name}' ({system}|{code}): concept map expects "
                    f"'{expected_unit}', payload sent '{unit}'. Rejected rather than "
                    "converted, so the conversion stays explicit and reviewable.")
            if t > anchor:
                raise PayloadError(
                    f"'{name}' contains a sample timestamped after the prediction anchor "
                    f"({anchor.isoformat()}). Features must be computed only over the "
                    "observation window; a post-anchor sample means the payload was "
                    "assembled with the wrong window.")
            if t < onset:
                continue  # pre-episode context lies outside the window
            samples.append(((t - onset).total_seconds(), v))

    if not samples:
        raise PayloadError(
            f"No samples found for required primitive '{name}' ({system}|{code}) "
            "within the observation window.")

    samples.sort(key=lambda s: s[0])
    return (Series(t=np.array([s[0] for s in samples], dtype=float),
                   v=np.array([s[1] for s in samples], dtype=float)),
            unit_seen)


def bundle_to_features(bundle: FHIRBundle, cm: ConceptMap, reject_implausible: bool = True):
    """Map a FHIR Bundle onto the concept map's feature space.

    Returns (feature_dict, resolution trail, warnings). feature_dict holds model
    columns: direct primitives plus computed derivations.
    """
    patient, _encounter, observations = _split_bundle(bundle)
    resolutions: list[FeatureResolution] = []
    warnings: list[str] = []
    primitives: dict[str, Any] = {}
    features: dict[str, Any] = {}

    # --- episode window and prediction anchor -----------------------------
    onset, end = _episode_window(observations)
    window_s = float((cm.target or {}).get("observation_window_s") or 300)
    anchor = min(end, onset + timedelta(seconds=window_s))
    primitives["episode_period"] = (0.0, (anchor - onset).total_seconds())
    resolutions.append(FeatureResolution(
        feature="episode_period", resolved=True, source_system="FHIR",
        source_code="Observation.effectivePeriod",
        value=(anchor - onset).total_seconds(), unit="s",
        note=f"anchor at onset + {window_s:.0f}s"))

    # --- primitives ---------------------------------------------------------
    for b in cm.features:
        name = b.feature
        if name == "episode_period":
            continue

        if name == "age":
            if patient.birthDate is None:
                raise PayloadError("Patient.birthDate absent; required primitive 'age'.")
            features["age"] = primitives["age"] = _age_years(patient.birthDate, onset)
            resolutions.append(FeatureResolution(
                feature="age", resolved=True, source_system="FHIR",
                source_code="Patient.birthDate", value=features["age"], unit="a"))
            continue

        if name == "sex":
            vmap = (b.fhir or {}).get("value_map", {})
            if patient.gender is None or patient.gender not in vmap:
                raise PayloadError(
                    f"Patient.gender={patient.gender!r} is not mappable to the concept "
                    "map's sex value map.")
            features["sex"] = primitives["sex"] = vmap[patient.gender]
            resolutions.append(FeatureResolution(
                feature="sex", resolved=True, source_system="FHIR",
                source_code="Patient.gender", note=f"{patient.gender} -> {features['sex']}"))
            continue

        if b.dtype == "coded":
            code = b.snomed or b.loinc
            system = SNOMED_SYSTEM if b.snomed else LOINC_SYSTEM
            match = None
            if code is not None:
                match = next((o for o in observations if (system, code) in _codings(o)), None)
            if match is None or match.valueCodeableConcept is None:
                raise PayloadError(
                    f"Required coded primitive '{name}' absent from Bundle. It must be "
                    "sent as an Observation with a valueCodeableConcept.")
            sent = {c.code for c in match.valueCodeableConcept.coding}
            hit, unmapped = resolve_coded_value(b, sent, name)
            features[name] = primitives[name] = hit
            if unmapped:
                warnings.append(
                    f"'{name}': code {unmapped} is not in the bound value set; mapped to "
                    "'other' under unmapped_policy. Check whether the concept map's "
                    "descendant closure needs re-expanding against the current "
                    "vocabulary release.")
            resolutions.append(FeatureResolution(
                feature=name, resolved=True, source_system=system, source_code=code,
                note=f"-> {hit}" + (f" (unmapped {unmapped})" if unmapped else "")))
            continue

        if b.dtype_series:
            code = b.loinc or b.snomed
            system = LOINC_SYSTEM if b.loinc else SNOMED_SYSTEM
            if code is None:
                raise PayloadError(f"Series primitive '{name}' has no code bound.")
            series, unit = _collect_series(observations, system, code, onset, anchor,
                                           b.unit, name)
            if len(series) < b.min_samples:
                raise PayloadError(
                    f"Primitive '{name}' has {len(series)} sample(s) in the observation "
                    f"window; at least {b.min_samples} required. A derived feature "
                    "computed from too few samples is not comparable to training.")
            if b.plausible_range is not None:
                lo, hi = b.plausible_range
                n_out = int(((series.v < lo) | (series.v > hi)).sum())
                if n_out:
                    msg = (f"'{name}' has {n_out} sample(s) outside plausible range "
                           f"[{lo}, {hi}]")
                    if reject_implausible:
                        raise PayloadError(
                            msg + ". Rejected: more likely artefact or a unit error than "
                            "real measurement, and it would distort every derived feature "
                            "computed from this series.")
                    warnings.append(msg)
            primitives[name] = series
            resolutions.append(FeatureResolution(
                feature=name, resolved=True, source_system=system, source_code=code,
                value=float(len(series)), unit=unit,
                note=f"{len(series)} samples over {series.duration_s:.0f}s"))

    # --- derived features ----------------------------------------------------
    try:
        derived_values, derived_warnings = compute(cm.derived_names, primitives)
    except DerivationError as exc:
        raise PayloadError(str(exc)) from exc
    features.update(derived_values)
    warnings.extend(derived_warnings)

    for d in cm.derived:
        resolutions.append(FeatureResolution(
            feature=d.feature, resolved=True, source_system="derived",
            source_code=d.identifier, value=derived_values.get(d.feature),
            unit=d.unit, note=f"computed from {', '.join(d.inputs)}"))

    missing = [f for f in cm.model_feature_names if f not in features]
    if missing:
        raise PayloadError(f"Feature(s) unresolved after derivation: {missing}")

    return features, resolutions, warnings


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------
def build_risk_assessment(
    probability: float,
    subject_reference: str,
    cm: ConceptMap,
    model_id: str,
    model_version: str,
    request_id: str,
    is_clinical_artifact: bool,
    occurrence: Optional[datetime] = None,
) -> dict[str, Any]:
    """Render the prediction as a FHIR R4 RiskAssessment.

    `subject_reference` is the pseudonymised reference: the service never emits
    an identifier it was unwilling to write to its own audit log.
    """
    target = cm.target or {}
    when = (occurrence or datetime.now(timezone.utc)).isoformat()
    horizon_min = float(target.get("prediction_horizon_s") or 600) / 60.0

    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
        "id": request_id,
        "status": "final",
        "subject": {"reference": subject_reference},
        "occurrenceDateTime": when,
        "method": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "survey",
                "display": "Statistical model estimate",
            }],
            "text": (f"{model_id} v{model_version} "
                     f"(concept map {cm.concept_map_id} v{cm.version})"),
        },
        "prediction": [{
            "outcome": {"text": target.get("description", "Predicted outcome")},
            "probabilityDecimal": round(float(probability), 6),
            "whenRange": {
                "low": {"value": 0, "unit": "min", "system": "http://unitsofmeasure.org"},
                "high": {"value": horizon_min, "unit": "min",
                         "system": "http://unitsofmeasure.org"},
            },
        }],
        "note": [],
    }

    if not is_clinical_artifact:
        # Carried inside the resource so the warning survives being forwarded,
        # stored, or rendered somewhere the documentation is not.
        resource["status"] = "preliminary"
        resource["note"].append({"text": (
            "NON-CLINICAL PROOF OF CONCEPT. Produced by a model artifact not validated "
            "for clinical use; must not inform patient care.")})
    return resource
