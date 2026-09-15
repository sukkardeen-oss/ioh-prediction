"""Contract tests for the SNUH interoperability module.

Focused on failures that are dangerous because they are quiet: derivation
drift, post-anchor leakage, unit mismatch, cross-patient bundles, PHI in logs,
and concept-map/model disagreement.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

DEPLOY_DIR = Path(__file__).resolve().parent.parent
SHIPPED_CM_PATH = DEPLOY_DIR / "concept_map" / "snuh_intraop_hypotension_concept_map.json"


def _write_bound_fixture() -> Path:
    """The shipped map is UNBOUND by design, so the service refuses to serve it.
    Tests need a bound copy to exercise the happy path; the fail-closed
    behaviour of the shipped map is asserted separately."""
    raw = json.loads(SHIPPED_CM_PATH.read_text())
    for i, f in enumerate(raw["features"]):
        if f["omop"].get("domain") in {"Measurement", "Observation"}:
            f["omop"]["concept_id"] = 4000000 + i
        if f["feature"] == "rhythm_code":
            f["snomed"] = "364074009"
            for j, k in enumerate(f["value_set"]):
                # Primary concept plus a pre-expanded descendant, mimicking a
                # real SNOMED closure.
                f["value_set"][k] = {
                    "primary": str(4100000 + j),
                    "includes": [str(4200000 + j)],
                    "_expansion": "is-a descendants, ATHENA v5.0-TEST, expanded 2026-08-30",
                }
    raw["status"] = "BOUND"
    raw["omop_vocabulary_version"] = "v5.0-TEST"
    out = Path("/tmp/snuh_bound_concept_map.json")
    out.write_text(json.dumps(raw))
    return out


BOUND_CM_PATH = _write_bound_fixture()
os.environ.setdefault("CONCEPT_MAP_PATH", str(BOUND_CM_PATH))
os.environ.setdefault("ALLOW_EPHEMERAL_PEPPER", "true")
os.environ.setdefault("REQUIRE_CLINICAL_ARTIFACT", "false")
os.environ.setdefault("REQUIRE_BOUND_CONCEPT_MAP", "false")
os.environ.setdefault("AUDIT_LOG_PATH", "/tmp/snuh_test_audit.jsonl")
os.environ.setdefault("ENFORCE_AIRGAP", "false")

from fastapi.testclient import TestClient  # noqa: E402

from app.concept_map import load_concept_map  # noqa: E402
from app.main import app  # noqa: E402

CM_PATH = SHIPPED_CM_PATH          # the map as delivered: UNBOUND
CM = load_concept_map(CM_PATH)
CM_BOUND = load_concept_map(BOUND_CM_PATH)   # what the running service uses
RHYTHM_VS = {m: spec["primary"]
             for m, spec in CM_BOUND.by_feature("rhythm_code").value_set.items()}
RHYTHM_DESCENDANTS = {m: spec["includes"][0]
                      for m, spec in CM_BOUND.by_feature("rhythm_code").value_set.items()}

ONSET = datetime(2026, 8, 30, 10, 0, 0, tzinfo=timezone.utc)
LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
RHYTHM_SNOMED = None  # set below from the bound fixture


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _series_obs(code, values, unit, start_offset=0, step=30):
    """One Observation carrying a sample series via component[]."""
    return {"resource": {
        "resourceType": "Observation", "status": "final",
        "code": {"coding": [{"system": LOINC, "code": code}]},
        "subject": {"reference": "Patient/p1"},
        "effectivePeriod": {"start": ONSET.isoformat(),
                            "end": (ONSET + timedelta(seconds=300)).isoformat()},
        "component": [
            {"effectiveDateTime": (ONSET + timedelta(seconds=start_offset + i * step)).isoformat(),
             "valueQuantity": {"value": v, "unit": unit, "code": unit,
                               "system": "http://unitsofmeasure.org"}}
            for i, v in enumerate(values)],
    }}


def build_bundle(map_values=None, hr_values=None, map_unit="mm[Hg]", extra=None,
                 rhythm_code=None):
    rhythm_code = rhythm_code or RHYTHM_VS["atrial_fibrillation"]
    map_values = map_values if map_values is not None else [82, 79, 76, 74, 71, 70, 69, 68, 67, 66]
    hr_values = hr_values if hr_values is not None else [88, 90, 92, 95, 97, 99, 101, 103, 104, 106]
    entries = [
        {"resource": {"resourceType": "Patient", "id": "p1",
                      "gender": "male", "birthDate": "1962-04-11"}},
        _series_obs("8478-0", map_values, map_unit),
        _series_obs("8867-4", hr_values, "/min"),
        {"resource": {"resourceType": "Observation", "status": "final",
                      "code": {"coding": [{"system": SNOMED, "code": "364074009"}]},
                      "effectiveDateTime": ONSET.isoformat(),
                      "valueCodeableConcept": {
                          "coding": [{"system": SNOMED, "code": rhythm_code}]}}},
    ]
    if extra:
        entries += extra
    return {"resourceType": "Bundle", "id": "b1", "type": "collection",
            "timestamp": ONSET.isoformat(), "entry": entries}


def bound_cm(tmp_path=None, **overrides):
    """The bound fixture, simulating SNUH terminology sign-off."""
    if not overrides:
        return CM_BOUND
    raw = json.loads(BOUND_CM_PATH.read_text())
    raw.update(overrides)
    p = (tmp_path or Path("/tmp")) / "bound_override.json"
    p.write_text(json.dumps(raw))
    return load_concept_map(p)


def fhir_request(**kw):
    return {"format": "fhir", "subject_identifier": "MRN-12345678",
            "fhir_bundle": build_bundle(**kw)}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def test_liveness_does_not_depend_on_model(client):
    assert client.get("/healthz").json()["status"] == "alive"


def test_readiness_reports_derivation_state(client):
    checks = client.get("/readyz").json()["checks"]
    assert checks["model_loaded"] is True
    assert checks["model_warmup_ok"] is True
    assert len(checks["derivation_transforms_registered"]) == 9
    assert checks["derived_features_declared"] == CM_BOUND.derived_names


def test_readiness_fails_closed_on_unbound_concept_map():
    from app.concept_map import check_serving_readiness
    problems = check_serving_readiness(CM, require_bound=True)
    assert any("BOUND" in p for p in problems)
    assert any("value-set" in p for p in problems)


# ---------------------------------------------------------------------------
# Two-layer concept map
# ---------------------------------------------------------------------------
def test_derived_features_carry_no_terminology_code():
    """The design claim, asserted as a test: derived features bind to a
    version, never to a code."""
    raw = json.loads(CM_PATH.read_text())
    for d in raw["derived"]:
        assert "loinc" not in d and "snomed" not in d
        assert d["identifier"].startswith("urn:snuh:derived:")
        assert d["identifier"].endswith(f":v{d['version']}")


def test_derivation_inputs_must_be_declared_primitives():
    for d in CM.derived:
        for inp in d.inputs:
            assert CM.by_feature(inp) is not None


def test_concept_map_rejects_derived_input_that_is_not_a_primitive(tmp_path):
    raw = json.loads(CM_PATH.read_text())
    raw["derived"][1]["inputs"] = ["not_a_primitive"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw))
    from app.concept_map import ConceptMapError
    with pytest.raises(ConceptMapError, match="not declared primitives"):
        load_concept_map(p)


def test_concept_map_rejects_version_identifier_mismatch(tmp_path):
    raw = json.loads(CM_PATH.read_text())
    raw["derived"][0]["version"] = "9.9.9"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw))
    from app.concept_map import ConceptMapError
    with pytest.raises(ConceptMapError, match="version segment"):
        load_concept_map(p)


def test_series_primitives_are_not_model_columns():
    """map_series feeds derivations; it is not itself a model feature."""
    assert "map_series" in CM.feature_names
    assert "map_series" not in CM.model_feature_names
    assert "map_mean" in CM.model_feature_names


# ---------------------------------------------------------------------------
# Derivation registry integrity (readiness gate 5)
# ---------------------------------------------------------------------------
def test_artifact_records_derivation_registry_hash():
    meta = json.loads((DEPLOY_DIR / "artifacts" / "model_metadata.json").read_text())
    from app.derivations import registry_hash
    assert meta["derivation_registry_sha256"] == registry_hash()
    assert len(meta["derivation_manifest"]) == 9


def test_changed_transform_is_detected(monkeypatch):
    """The failure this whole layer exists to catch: a transform edited without
    a version bump. Same column name, same units, plausible value, different
    model."""
    from app import derivations
    from app.model_runtime import runtime

    assert runtime.validate_derivations(CM_BOUND) == []

    original = derivations._REGISTRY["map_mean"]
    tampered = derivations.Derivation(
        name=original.name, version=original.version, inputs=original.inputs,
        unit=original.unit, spec=original.spec + " (edited)", fn=original.fn,
        plausible_range=original.plausible_range)
    monkeypatch.setitem(derivations._REGISTRY, "map_mean", tampered)

    problems = runtime.validate_derivations(CM_BOUND)
    assert any("registry hash mismatch" in p for p in problems)


def test_missing_transform_blocks_readiness(monkeypatch):
    from app import derivations
    from app.model_runtime import runtime
    reg = dict(derivations._REGISTRY)
    reg.pop("map_auc_below_65")
    monkeypatch.setattr(derivations, "_REGISTRY", reg)
    problems = runtime.validate_derivations(CM_BOUND)
    assert any("no registered transform" in p for p in problems)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_fhir_prediction_returns_risk_assessment(client):
    r = client.post("/v1/predict", json=fhir_request())
    assert r.status_code == 200, r.text
    body = r.json()
    ra = body["fhir_risk_assessment"]
    assert ra["resourceType"] == "RiskAssessment"
    assert 0.0 <= ra["prediction"][0]["probabilityDecimal"] <= 1.0
    assert ra["prediction"][0]["whenRange"]["high"]["value"] == 10.0


def test_response_carries_derivation_provenance(client):
    body = client.post("/v1/predict", json=fhir_request()).json()
    derivs = {d["feature"]: d for d in body["derivations"]}
    assert len(derivs) == 9
    assert derivs["map_auc_below_65"]["identifier"].endswith(":v1.0.0")
    assert derivs["map_mean"]["inputs"] == ["map_series"]
    assert derivs["map_mean"]["value"] is not None
    assert body["prediction"]["derivation_registry_sha256"]


def test_derived_values_are_correct(client):
    """Spot-check one derivation end-to-end against a hand computation."""
    values = [80.0] * 5 + [60.0] * 5  # 5 samples at 30s spacing each
    body = client.post("/v1/predict", json=fhir_request(map_values=values)).json()
    d = {x["feature"]: x["value"] for x in body["derivations"]}
    assert d["map_min"] == 60.0
    assert 60.0 < d["map_mean"] < 80.0
    assert d["map_auc_below_65"] > 0        # time spent below 65
    assert 0.0 < d["map_frac_below_65"] < 1.0
    assert d["map_slope"] < 0               # falling pressure


def test_no_hypotension_gives_zero_auc(client):
    body = client.post("/v1/predict", json=fhir_request(map_values=[90] * 10)).json()
    d = {x["feature"]: x["value"] for x in body["derivations"]}
    assert d["map_auc_below_65"] == 0.0
    assert d["map_frac_below_65"] == 0.0


# ---------------------------------------------------------------------------
# The dangerous-because-quiet failures
# ---------------------------------------------------------------------------
def test_post_anchor_sample_is_rejected(client):
    """Leakage guard: a sample after the prediction anchor is rejected, not
    silently dropped."""
    late = _series_obs("8478-0", [55.0], "mm[Hg]", start_offset=900)
    r = client.post("/v1/predict", json=fhir_request(extra=[late]))
    assert r.status_code == 422
    assert "after the prediction anchor" in r.json()["detail"]


def test_unit_mismatch_is_rejected_not_converted(client):
    """MAP in kPa read as mmHg is a ~7.5x error that still returns a plausible
    probability."""
    r = client.post("/v1/predict", json=fhir_request(map_unit="kPa"))
    assert r.status_code == 422
    assert "unit mismatch" in r.json()["detail"].lower()


def test_implausible_series_sample_is_rejected(client):
    r = client.post("/v1/predict",
                    json=fhir_request(map_values=[82, 79, 76, 74, 900, 70, 69, 68, 67, 66]))
    assert r.status_code == 422
    assert "plausible range" in r.json()["detail"].lower()


def test_too_few_samples_is_rejected(client):
    r = client.post("/v1/predict", json=fhir_request(map_values=[80, 78]))
    assert r.status_code == 422
    assert "at least 3" in r.json()["detail"]


def test_unmapped_rhythm_is_rejected_not_folded_into_other(client):
    r = client.post("/v1/predict", json=fhir_request(rhythm_code="999999999"))
    assert r.status_code == 422
    assert "value set" in r.json()["detail"].lower()


def test_multiple_patients_in_bundle_rejected(client):
    extra = [{"resource": {"resourceType": "Patient", "id": "p2",
                           "gender": "female", "birthDate": "1970-01-01"}}]
    r = client.post("/v1/predict", json=fhir_request(extra=extra))
    assert r.status_code == 422
    assert "more than one patient" in r.json()["detail"].lower()


def test_missing_episode_period_is_rejected(client):
    bundle = build_bundle()
    for e in bundle["entry"]:
        e["resource"].pop("effectivePeriod", None)
    r = client.post("/v1/predict", json={"format": "fhir",
                                         "subject_identifier": "MRN-1",
                                         "fhir_bundle": bundle})
    assert r.status_code == 422
    assert "effectivePeriod" in r.json()["detail"]


def test_patient_name_is_refused_at_the_boundary(client):
    bundle = build_bundle()
    bundle["entry"][0]["resource"]["name"] = [{"family": "Kim", "given": ["Minjun"]}]
    r = client.post("/v1/predict", json={"format": "fhir",
                                         "subject_identifier": "MRN-1",
                                         "fhir_bundle": bundle})
    assert r.status_code == 422


def test_unknown_field_is_rejected(client):
    payload = fhir_request()
    payload["unexpected_field"] = "surprise"
    assert client.post("/v1/predict", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Pseudonymisation and audit
# ---------------------------------------------------------------------------
def test_raw_identifier_never_appears_in_response(client):
    body = client.post("/v1/predict", json=fhir_request()).json()
    assert "MRN-12345678" not in json.dumps(body)
    assert body["fhir_risk_assessment"]["subject"]["reference"].startswith("Patient/sub_")


def test_audit_log_has_no_phi_and_records_derivation_hash(client):
    from app import audit
    log = Path(os.environ["AUDIT_LOG_PATH"])
    if log.exists():
        log.unlink()
    audit.configure_logging(log)
    client.post("/v1/predict", json=fhir_request())
    text = log.read_text()
    assert "MRN-12345678" not in text
    assert "1962-04-11" not in text  # birthDate must not leak
    lines = [json.loads(l) for l in text.strip().splitlines()]
    pred = [l for l in lines if l.get("event") == "prediction"][-1]
    assert pred["subject_pseudonym"].startswith("sub_")
    assert "probability" in pred
    assert "derivation_registry_sha256" in pred


def test_pepper_rotation_breaks_linkage():
    from app import audit
    audit.init_pepper(Path("/nonexistent"), allow_ephemeral=True)
    first = audit.pseudonymise("MRN-777")
    audit._PEPPER = None
    audit.init_pepper(Path("/nonexistent"), allow_ephemeral=True)
    assert audit.pseudonymise("MRN-777") != first


# ---------------------------------------------------------------------------
# Concept map / model agreement
# ---------------------------------------------------------------------------
def test_model_feature_space_matches_concept_map():
    from app.model_runtime import runtime
    assert runtime.validate_against_concept_map(CM_BOUND) == []


def test_half_the_feature_space_is_derived():
    n_derived = len(CM.derived_names)
    n_total = len(CM.model_feature_names)
    assert n_derived / n_total >= 0.5


def test_post_anchor_samples_documented_as_excluded():
    raw = json.loads(CM_PATH.read_text())
    assert "post_anchor_samples" in raw["excluded_features"]


# ---------------------------------------------------------------------------
# OMOP path
# ---------------------------------------------------------------------------
def test_omop_rejected_while_concepts_unbound():
    """Against the SHIPPED (unbound) map, OMOP inputs cannot resolve. Better a
    422 naming the gap than a prediction computed from defaults."""
    from app.fhir_adapter import PayloadError
    from app.omop_adapter import omop_to_features
    from app.schemas import OMOPPayload
    payload = OMOPPayload(
        person={"person_id": 1, "gender_concept_id": 8507, "year_of_birth": 1962},
        visit_occurrence={"visit_occurrence_id": 9, "person_id": 1,
                          "visit_start_datetime": "2026-08-30T10:00:00+00:00"},
        measurement=[{"person_id": 1, "measurement_concept_id": 3004249,
                      "value_as_number": 80.0,
                      "measurement_datetime": "2026-08-30T10:00:00+00:00"}],
    )
    with pytest.raises(PayloadError, match="no OMOP concept_id bound"):
        omop_to_features(payload, CM)


def test_omop_resolves_and_derives_once_bound(tmp_path):
    from app.omop_adapter import omop_to_features
    from app.schemas import OMOPPayload

    cm = bound_cm(tmp_path)
    assert cm.unbound_features == []

    ids = {f.feature: f.omop_concept_id for f in cm.features}
    start = datetime(2026, 8, 30, 10, 0, 0, tzinfo=timezone.utc)
    meas = []
    for i, v in enumerate([82, 78, 74, 70, 66, 63]):
        meas.append({"person_id": 1, "measurement_concept_id": ids["map_series"],
                     "value_as_number": float(v),
                     "measurement_datetime": (start + timedelta(seconds=i * 40)).isoformat()})
    for i, v in enumerate([88, 92, 96, 99, 103, 107]):
        meas.append({"person_id": 1, "measurement_concept_id": ids["hr_series"],
                     "value_as_number": float(v),
                     "measurement_datetime": (start + timedelta(seconds=i * 40)).isoformat()})

    payload = OMOPPayload(
        person={"person_id": 1, "gender_concept_id": 8507, "year_of_birth": 1962,
                "month_of_birth": 4, "day_of_birth": 11},
        visit_occurrence={"visit_occurrence_id": 9, "person_id": 1,
                          "visit_start_date": "2026-08-30",
                          "visit_start_datetime": start.isoformat()},
        measurement=meas,
        observation=[{"person_id": 1,
                      "observation_concept_id": ids["rhythm_code"],
                      "value_as_concept_id": int(RHYTHM_VS["atrial_fibrillation"])}],
    )
    features, resolutions, _ = omop_to_features(payload, cm)
    assert features["rhythm_code"] == "atrial_fibrillation"
    assert features["map_min"] == 63.0
    assert features["map_auc_below_65"] > 0
    assert set(cm.model_feature_names) <= set(features)
    assert any(r.source_system == "derived" for r in resolutions)


def test_both_adapters_produce_the_same_feature_space(tmp_path):
    """FHIR and OMOP must converge on one feature space, or audit output and
    downstream logic become format-dependent."""
    from app.fhir_adapter import bundle_to_features
    from app.schemas import FHIRBundle
    cm = bound_cm(tmp_path)
    f_feats, _, _ = bundle_to_features(FHIRBundle(**build_bundle(
        rhythm_code=RHYTHM_VS["sinus"])), cm)
    assert set(f_feats) == set(cm.model_feature_names)


def test_shipped_concept_map_is_unbound_by_design():
    """The delivered map must NOT be servable: binding is SNUH's act, not ours."""
    from app.concept_map import check_serving_readiness
    assert CM.status == "UNBOUND"
    assert check_serving_readiness(CM, require_bound=True)


def test_date_only_visit_start_is_rejected(tmp_path):
    """An intraoperative window cannot be anchored to midnight."""
    from app.fhir_adapter import PayloadError
    from app.omop_adapter import omop_to_features
    from app.schemas import OMOPPayload
    payload = OMOPPayload(
        person={"person_id": 1, "gender_concept_id": 8507, "year_of_birth": 1962},
        visit_occurrence={"visit_occurrence_id": 9, "person_id": 1,
                          "visit_start_date": "2026-08-30"},
    )
    with pytest.raises(PayloadError, match="date precision"):
        omop_to_features(payload, CM_BOUND)


# ---------------------------------------------------------------------------
# Value-set expansion, unmapped policy, and coverage monitoring
# ---------------------------------------------------------------------------
def test_descendant_code_resolves_without_a_terminology_server(client):
    """The point of pre-expansion: a more specific SNOMED descendant of a bound
    concept must resolve, even though the air-gapped service cannot call a
    terminology server to test subsumption."""
    r = client.post("/v1/predict", json=fhir_request(
        rhythm_code=RHYTHM_DESCENDANTS["atrial_fibrillation"]))
    assert r.status_code == 200, r.text
    assert not r.json()["warnings"]


def test_primary_and_descendant_resolve_to_the_same_member(client):
    a = client.post("/v1/predict", json=fhir_request(
        rhythm_code=RHYTHM_VS["svt"])).json()
    b = client.post("/v1/predict", json=fhir_request(
        rhythm_code=RHYTHM_DESCENDANTS["svt"])).json()
    assert a["prediction"]["probability"] == b["prediction"]["probability"]


def test_unmapped_code_rejected_under_default_policy(client):
    r = client.post("/v1/predict", json=fhir_request(rhythm_code="999999999"))
    assert r.status_code == 422
    assert "not in the bound value set" in r.json()["detail"]


def test_unmapped_code_maps_to_other_when_policy_says_so(tmp_path):
    """The alternative policy: keep scoring, but say so loudly in warnings."""
    from app.fhir_adapter import bundle_to_features
    from app.schemas import FHIRBundle
    import json as _json

    raw = _json.loads(BOUND_CM_PATH.read_text())
    for f in raw["features"]:
        if f["feature"] == "rhythm_code":
            f["unmapped_policy"] = "map_to_other"
    p = tmp_path / "lenient.json"
    p.write_text(_json.dumps(raw))
    cm = load_concept_map(p)

    feats, _, warns = bundle_to_features(
        FHIRBundle(**build_bundle(rhythm_code="999999999")), cm)
    assert feats["rhythm_code"] == "other"
    assert any("not in the bound value set" in w for w in warns)


def test_overlapping_closures_are_rejected_at_load(tmp_path):
    """One code claimed by two members would make rhythm assignment depend on
    dict ordering."""
    import json as _json
    from app.concept_map import ConceptMapError

    raw = _json.loads(BOUND_CM_PATH.read_text())
    for f in raw["features"]:
        if f["feature"] == "rhythm_code":
            vs = f["value_set"]
            vs["atrial_flutter"]["includes"].append(vs["atrial_fibrillation"]["primary"])
    p = tmp_path / "overlap.json"
    p.write_text(_json.dumps(raw))
    with pytest.raises(ConceptMapError, match="claimed by both"):
        load_concept_map(p)


def test_expansion_without_provenance_blocks_readiness(tmp_path):
    """A closure is only valid for the vocabulary release it came from."""
    import json as _json
    from app.concept_map import check_serving_readiness

    raw = _json.loads(BOUND_CM_PATH.read_text())
    for f in raw["features"]:
        if f["feature"] == "rhythm_code":
            f["value_set"]["sinus"]["_expansion"] = None
    p = tmp_path / "noprov.json"
    p.write_text(_json.dumps(raw))
    problems = check_serving_readiness(load_concept_map(p), require_bound=True)
    assert any("_expansion" in x for x in problems)


def test_invalid_unmapped_policy_blocks_readiness(tmp_path):
    import json as _json
    from app.concept_map import check_serving_readiness

    raw = _json.loads(BOUND_CM_PATH.read_text())
    for f in raw["features"]:
        if f["feature"] == "rhythm_code":
            f["unmapped_policy"] = "guess"
    p = tmp_path / "badpolicy.json"
    p.write_text(_json.dumps(raw))
    assert any("unmapped_policy" in x
               for x in check_serving_readiness(load_concept_map(p), require_bound=True))


def test_unmapped_rate_is_observable_on_readiness(client):
    """The monitoring point: whichever policy is set, the RATE must be visible
    without running a log query."""
    from app import unmapped_stats
    unmapped_stats.reset()

    for _ in range(3):
        client.post("/v1/predict", json=fhir_request())
    client.post("/v1/predict", json=fhir_request(rhythm_code="999999999"))

    stats = client.get("/readyz").json()["checks"]["unmapped_value_stats"]
    assert stats["requests_seen"] == 4
    assert stats["unmapped_total"] == 1
    assert 0.0 < stats["unmapped_rate_recent"] < 1.0
    assert "rhythm_code:999999999" in stats["top_unmapped_codes"]


def test_readiness_reports_value_set_coverage(client):
    cov = client.get("/readyz").json()["checks"]["value_set_coverage"]["rhythm_code"]
    assert cov["unmapped_policy"] == "reject"
    assert cov["codes_bound"] == 14  # 7 members x (primary + 1 descendant)
    assert cov["expansions"]["atrial_fibrillation"]["expansion"]


def test_shipped_map_value_set_is_unbound_but_structured():
    """As delivered the closures are empty — SNUH's binding task — but the
    structure is present so it is clear what must be filled in."""
    raw = json.loads(SHIPPED_CM_PATH.read_text())
    vs = next(f for f in raw["features"] if f["feature"] == "rhythm_code")["value_set"]
    assert all(m["primary"] is None for m in vs.values())
    assert all("includes" in m for m in vs.values())
