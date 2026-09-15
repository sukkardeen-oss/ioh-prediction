"""OMOP CDM 5.4 -> model feature vector, via the two-layer concept map.

Sites running an OHDSI stack usually hold their EHR extract already normalised
into the CDM, so making them round-trip through FHIR would add a lossy hop for
no benefit. Both adapters converge on the same feature dict, the same
resolution trail, and the same derivation layer, so audit output and downstream
logic stay payload-format independent.

Resolution is by standard concept_id only. `measurement_source_value` is
site-local free text, and matching on it would silently reintroduce exactly the
string-matching fragility the CDM exists to remove.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from .concept_map import ConceptMap
from .derivations import DerivationError, Series, compute
from .fhir_adapter import PayloadError, resolve_coded_value
from .schemas import FeatureResolution, OMOPPayload

OMOP_GENDER = {8507: "M", 8532: "F"}


def _as_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def omop_to_features(payload: OMOPPayload, cm: ConceptMap, reject_implausible: bool = True):
    resolutions: list[FeatureResolution] = []
    warnings: list[str] = []
    primitives: dict[str, Any] = {}
    features: dict[str, Any] = {}

    person = payload.person
    ids = {person.person_id} | {r.person_id for r in
                                list(payload.measurement) + list(payload.observation)}
    if len(ids) > 1:
        raise PayloadError(
            f"Payload mixes multiple person_ids {sorted(ids)}. One episode, one subject.")

    # --- episode window -------------------------------------------------
    vo = payload.visit_occurrence
    if vo is None or not (vo.visit_start_datetime or vo.visit_start_date):
        raise PayloadError(
            "visit_occurrence.visit_start_datetime is required: it anchors the "
            "observation window over which every derived feature is computed.")
    if vo.visit_start_datetime is None:
        raise PayloadError(
            "visit_occurrence carries only visit_start_date (date precision). An "
            "intraoperative observation window cannot be anchored to midnight; send "
            "visit_start_datetime.")
    onset = _as_dt(vo.visit_start_datetime)
    window_s = float((cm.target or {}).get("observation_window_s") or 300)
    anchor = onset + timedelta(seconds=window_s)
    primitives["episode_period"] = (0.0, window_s)
    resolutions.append(FeatureResolution(
        feature="episode_period", resolved=True, source_system="OMOP",
        source_code="visit_occurrence.visit_start_date", value=window_s, unit="s"))

    # --- demographics ------------------------------------------------------
    if cm.by_feature("age") is not None:
        if not person.year_of_birth:
            raise PayloadError("person.year_of_birth absent; required primitive 'age'.")
        born = date(person.year_of_birth, person.month_of_birth or 7,
                    person.day_of_birth or 1)
        features["age"] = primitives["age"] = (onset.date() - born).days / 365.25
        if not (person.month_of_birth and person.day_of_birth):
            warnings.append("person birth month/day absent; age accurate to roughly +/-6 months")
        resolutions.append(FeatureResolution(
            feature="age", resolved=True, source_system="OMOP",
            source_code="person.year_of_birth", value=features["age"], unit="a"))

    sex_b = cm.by_feature("sex")
    if sex_b is not None:
        vc = {int(v): k for k, v in (sex_b.value_concept_ids or {}).items()} or OMOP_GENDER
        mapped = vc.get(person.gender_concept_id)
        if mapped is None:
            raise PayloadError(
                f"gender_concept_id={person.gender_concept_id} is not in the concept map.")
        features["sex"] = primitives["sex"] = mapped
        resolutions.append(FeatureResolution(
            feature="sex", resolved=True, source_system="OMOP",
            source_code=str(person.gender_concept_id), note=f"-> {mapped}"))

    # --- series and coded primitives ----------------------------------------
    for b in cm.features:
        name = b.feature
        if name in features or name == "episode_period":
            continue
        if b.omop_concept_id is None:
            raise PayloadError(
                f"Primitive '{name}' has no OMOP concept_id bound in the concept map; "
                "OMOP payloads cannot be resolved until SNUH populates it.")
        cid = b.omop_concept_id

        if b.dtype_series:
            samples: list[tuple[float, float]] = []
            for m in payload.measurement:
                if m.measurement_concept_id != cid or m.value_as_number is None:
                    continue
                t = _as_dt(m.measurement_datetime or m.measurement_date)
                if t is None:
                    raise PayloadError(
                        f"A sample for '{name}' has no measurement_datetime. Series "
                        "primitives must be timestamped to be windowed correctly.")
                if t > anchor:
                    raise PayloadError(
                        f"'{name}' contains a sample after the prediction anchor "
                        f"({anchor.isoformat()}). Rejected rather than truncated: this "
                        "means the payload was assembled with the wrong window.")
                if t < onset:
                    continue
                samples.append(((t - onset).total_seconds(), float(m.value_as_number)))

            if len(samples) < b.min_samples:
                raise PayloadError(
                    f"Primitive '{name}' has {len(samples)} sample(s) in the window; at "
                    f"least {b.min_samples} required.")
            samples.sort(key=lambda s: s[0])
            series = Series(t=np.array([s[0] for s in samples], dtype=float),
                            v=np.array([s[1] for s in samples], dtype=float))
            if b.plausible_range is not None:
                lo, hi = b.plausible_range
                n_out = int(((series.v < lo) | (series.v > hi)).sum())
                if n_out:
                    msg = f"'{name}' has {n_out} sample(s) outside [{lo}, {hi}]"
                    if reject_implausible:
                        raise PayloadError(
                            msg + ". Rejected: likely artefact or a unit_concept_id "
                            "mismatch, and it would distort every derived feature.")
                    warnings.append(msg)
            primitives[name] = series
            resolutions.append(FeatureResolution(
                feature=name, resolved=True, source_system="OMOP", source_code=str(cid),
                value=float(len(series)), unit=b.unit,
                note=f"{len(series)} samples over {series.duration_s:.0f}s"))
            continue

        if b.dtype == "coded":
            row = next((o for o in payload.observation
                        if o.observation_concept_id == cid), None)
            if row is None or row.value_as_concept_id is None:
                raise PayloadError(
                    f"Required coded primitive '{name}' absent, or has no "
                    "value_as_concept_id.")
            hit, unmapped = resolve_coded_value(b, {str(row.value_as_concept_id)}, name)
            features[name] = primitives[name] = hit
            if unmapped:
                warnings.append(
                    f"'{name}': concept_id {unmapped} is not in the bound value set; "
                    "mapped to 'other' under unmapped_policy.")
            resolutions.append(FeatureResolution(
                feature=name, resolved=True, source_system="OMOP",
                source_code=str(cid), note=f"-> {hit}"))

    # --- derived --------------------------------------------------------------
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
