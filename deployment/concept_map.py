"""Concept map loading and validation.

The concept map is the interoperability contract: it is the single place that
says "LOINC 2160-0, reported in mg/dL, is the feature the model calls
preop_cr". Every adapter (FHIR, OMOP) resolves through it, and no adapter is
allowed to hardcode a code anywhere else.

Validation is fail-closed. An unbound or internally inconsistent map makes the
service unready rather than making it guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class ConceptMapError(RuntimeError):
    """Concept map is missing, malformed, or not safe to serve with."""


@dataclass(frozen=True)
class FeatureBinding:
    feature: str
    dtype: str
    required: bool
    unit: Optional[str]
    loinc: Optional[str]
    snomed: Optional[str]
    dtype_series: bool
    min_samples: int
    value_set: dict
    # code -> member name, covering the primary concept AND every pre-expanded
    # descendant. Built once at load; request-time resolution is a dict lookup,
    # because an air-gapped service cannot call a terminology server to test
    # subsumption per request.
    value_code_index: dict
    unmapped_policy: str
    value_set_expansions: dict
    local_system: Optional[str]
    local_code: Optional[str]
    local_is_placeholder: bool
    omop_concept_id: Optional[int]
    omop_domain: Optional[str]
    plausible_range: Optional[tuple[float, float]]
    categories: Optional[list[str]]
    levels: Optional[list[int]]
    value_concept_ids: dict[str, int]
    fhir: dict[str, Any]
    display: Optional[str]
    verify: bool
    verify_reason: Optional[str]
    derived_from: Optional[list[str]]


@dataclass(frozen=True)
class DerivedBinding:
    """A feature computed from primitives.

    Carries no terminology code by design; see app/derivations.py for why. The
    binding is to `identifier`, whose version segment changes whenever the
    transform changes.
    """

    feature: str
    identifier: str
    version: str
    inputs: tuple[str, ...]
    unit: Optional[str]


@dataclass(frozen=True)
class ConceptMap:
    concept_map_id: str
    version: str
    status: str
    omop_cdm_version: Optional[str]
    omop_vocabulary_version: Optional[str]
    fhir_version: Optional[str]
    target: dict[str, Any]
    unit_policy: dict[str, Any]
    features: tuple[FeatureBinding, ...]
    derived: tuple[DerivedBinding, ...] = ()
    derived_writeback: dict[str, Any] = field(default_factory=dict)

    # --- lookups -------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        """Primitive names only. These are what an adapter resolves from a payload."""
        return [f.feature for f in self.features]

    @property
    def derived_names(self) -> list[str]:
        return [d.feature for d in self.derived]

    @property
    def model_feature_names(self) -> list[str]:
        """Everything the model can be fed: primitives that are directly usable
        plus every derived feature. Series and period primitives are inputs to
        derivations, not model columns, so they are excluded."""
        direct = [f.feature for f in self.features
                  if f.dtype not in {"series", "period"}]
        return direct + [d.feature for d in self.derived]

    @property
    def required_features(self) -> list[str]:
        return [f.feature for f in self.features if f.required]

    def by_feature(self, name: str) -> Optional[FeatureBinding]:
        return next((f for f in self.features if f.feature == name), None)

    def by_loinc(self, code: str) -> Optional[FeatureBinding]:
        return next((f for f in self.features if f.loinc == code), None)

    def by_omop_concept_id(self, concept_id: int) -> Optional[FeatureBinding]:
        return next((f for f in self.features if f.omop_concept_id == concept_id), None)

    def derived_by_feature(self, name: str) -> Optional[DerivedBinding]:
        return next((d for d in self.derived if d.feature == name), None)

    @property
    def unbound_features(self) -> list[str]:
        """Features whose OMOP concept_id has not been populated by SNUH."""
        return [
            f.feature
            for f in self.features
            if f.omop_domain != "Person" and f.omop_concept_id is None and f.derived_from is None
        ]

    @property
    def placeholder_local_codes(self) -> list[str]:
        """Site-local codings still carrying a placeholder system/code."""
        return [f.feature for f in self.features if f.local_is_placeholder]

    @property
    def unverified_features(self) -> list[str]:
        return [f.feature for f in self.features if f.verify]


def _index_value_set(feature: str, value_set: dict) -> tuple[dict[str, str], dict]:
    """Flatten a value set into a code -> member lookup.

    Three shapes are accepted per member, so a simple map stays simple:
      null                        - unbound; blocks readiness
      "49436004"                  - a single concept, no expansion
      {"primary": ..., "includes": [...], "_expansion": "..."}

    The expanded form is what a real SNOMED binding needs. Descendants are
    materialised here rather than tested at request time because the SDDC has
    no terminology server to ask.
    """
    index: dict[str, str] = {}
    expansions: dict[str, Any] = {}
    for member, spec in (value_set or {}).items():
        if spec is None:
            continue
        if isinstance(spec, str):
            codes, meta = [spec], None
        elif isinstance(spec, dict):
            primary = spec.get("primary")
            if primary is None:
                continue
            codes = [primary] + list(spec.get("includes") or [])
            meta = spec.get("_expansion")
        else:
            raise ConceptMapError(
                f"{feature}.{member}: value-set member must be null, a code string, or "
                "an object with 'primary'.")
        for c in codes:
            c = str(c)
            if c in index and index[c] != member:
                # One code resolving to two members would make rhythm
                # assignment non-deterministic and depend on dict order.
                raise ConceptMapError(
                    f"{feature}: code {c} is claimed by both '{index[c]}' and "
                    f"'{member}'. Overlapping descendant closures must be "
                    "disambiguated before binding.")
            index[c] = member
        expansions[member] = {"n_codes": len(codes), "expansion": meta}
    return index, expansions


def _parse_feature(raw: dict[str, Any]) -> FeatureBinding:
    omop = raw.get("omop") or {}
    local = raw.get("local") or {}
    pr = raw.get("plausible_range")
    value_set = raw.get("value_set") or {}
    code_index, expansions = _index_value_set(raw.get("feature", "?"), value_set)
    return FeatureBinding(
        value_code_index=code_index,
        unmapped_policy=raw.get("unmapped_policy", "reject"),
        value_set_expansions=expansions,
        dtype_series=raw.get("dtype") == "series",
        min_samples=int(raw.get("min_samples", 1)),
        value_set=raw.get("value_set") or {},
        feature=raw["feature"],
        dtype=raw["dtype"],
        required=bool(raw.get("required", False)),
        unit=raw.get("unit"),
        loinc=raw.get("loinc"),
        snomed=raw.get("snomed"),
        local_system=local.get("system"),
        local_code=local.get("code"),
        local_is_placeholder=bool(local.get("_placeholder", False)),
        omop_concept_id=omop.get("concept_id"),
        omop_domain=omop.get("domain"),
        plausible_range=(float(pr[0]), float(pr[1])) if pr else None,
        categories=raw.get("categories"),
        levels=raw.get("levels"),
        value_concept_ids=omop.get("value_concept_ids") or {},
        fhir=raw.get("fhir") or {},
        display=raw.get("display"),
        verify=bool(raw.get("verify", False)),
        verify_reason=raw.get("verify_reason"),
        derived_from=raw.get("derived_from"),
    )


def load_concept_map(path: Path) -> ConceptMap:
    if not path.exists():
        raise ConceptMapError(f"Concept map not found at {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConceptMapError(f"Concept map at {path} is not valid JSON: {exc}") from exc

    features = tuple(_parse_feature(f) for f in raw.get("features", []))
    if not features:
        raise ConceptMapError("Concept map declares no features.")

    derived = tuple(
        DerivedBinding(
            feature=d["feature"], identifier=d["identifier"], version=d["version"],
            inputs=tuple(d.get("inputs", [])), unit=d.get("unit"),
        )
        for d in raw.get("derived", [])
    )

    # A derived feature sharing a name with a primitive would make the feature
    # space ambiguous and silently shadow one of them.
    primitive_names = {f.feature for f in features}
    clash = sorted(primitive_names & {d.feature for d in derived})
    if clash:
        raise ConceptMapError(
            f"Name(s) declared as both a primitive and a derived feature: {clash}")

    # Every derivation input must be a declared primitive: the whole point of
    # the two-layer split is that derivations compose coded things.
    for d in derived:
        unknown = [i for i in d.inputs if i not in primitive_names]
        if unknown:
            raise ConceptMapError(
                f"Derived feature '{d.feature}' names input(s) {unknown} that are not "
                "declared primitives in this concept map.")

    # The version inside the identifier must match the declared version, or the
    # identifier stops being a reliable key for the transform.
    for d in derived:
        if not d.identifier.endswith(f":v{d.version}"):
            raise ConceptMapError(
                f"Derived feature '{d.feature}' has identifier '{d.identifier}' whose "
                f"version segment does not match declared version '{d.version}'.")

    names = [f.feature for f in features]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ConceptMapError(f"Duplicate feature entries in concept map: {dupes}")

    # A LOINC code resolving to two different features would make FHIR
    # ingestion non-deterministic.
    loincs = [f.loinc for f in features if f.loinc]
    if len(loincs) != len(set(loincs)):
        dupes = sorted({c for c in loincs if loincs.count(c) > 1})
        raise ConceptMapError(f"LOINC code bound to more than one feature: {dupes}")

    return ConceptMap(
        concept_map_id=raw.get("concept_map_id", "unknown"),
        version=raw.get("version", "0"),
        status=raw.get("status", "UNBOUND"),
        omop_cdm_version=raw.get("omop_cdm_version"),
        omop_vocabulary_version=raw.get("omop_vocabulary_version"),
        fhir_version=raw.get("fhir_version"),
        target=raw.get("target") or {},
        unit_policy=raw.get("unit_policy") or {},
        features=features,
        derived=derived,
        derived_writeback=raw.get("derived_writeback") or {},
    )


def check_serving_readiness(cm: ConceptMap, require_bound: bool) -> list[str]:
    """Return a list of blocking problems. Empty list means safe to serve."""
    problems: list[str] = []
    if not require_bound:
        return problems

    if cm.status != "BOUND":
        problems.append(
            f"concept_map status is '{cm.status}', expected 'BOUND'. "
            "SNUH terminology sign-off is required before serving."
        )
    if cm.omop_vocabulary_version is None:
        problems.append(
            "concept_map.omop_vocabulary_version is null; concept_ids cannot be "
            "trusted without the vocabulary release they were drawn from."
        )
    unbound = cm.unbound_features
    if unbound:
        problems.append(f"{len(unbound)} feature(s) have no OMOP concept_id: {unbound}")
    # Value sets must be fully bound too: an unbound rhythm member would fall
    # through to a reject at request time, which is safe but only discoverable
    # in production.
    unbound_vs = [
        f"{f.feature}.{k}" for f in cm.features
        for k, v in (f.value_set or {}).items()
        if v is None or (isinstance(v, dict) and v.get("primary") is None)
    ]
    if unbound_vs:
        problems.append(
            f"{len(unbound_vs)} value-set member(s) unbound: {unbound_vs[:6]}"
            + (" ..." if len(unbound_vs) > 6 else ""))

    # A closure with no recorded vocabulary release cannot be trusted: it is
    # only valid for the release it was expanded from, and a later release
    # adding concepts silently shrinks coverage.
    no_prov = [
        f"{f.feature}.{m}" for f in cm.features
        for m, meta in (f.value_set_expansions or {}).items()
        if meta.get("n_codes", 0) > 1 and not meta.get("expansion")
    ]
    if no_prov:
        problems.append(
            f"{len(no_prov)} value-set member(s) have an expanded code list but no "
            f"'_expansion' provenance: {no_prov[:6]}")

    bad_policy = [f.feature for f in cm.features
                  if f.value_set and f.unmapped_policy not in {"reject", "map_to_other"}]
    if bad_policy:
        problems.append(f"invalid unmapped_policy on: {bad_policy}")

    placeholders = cm.placeholder_local_codes
    if placeholders:
        problems.append(
            f"{len(placeholders)} feature(s) still use placeholder site-local codings: "
            f"{placeholders}. Replace with SNUH's real CodeSystem URIs."
        )
    return problems
