"""SNUH preoperative risk inference service.

Endpoints
---------
GET  /healthz   liveness  — is the process alive and its event loop responsive
GET  /readyz    readiness — is it safe to send this pod clinical traffic
GET  /model-card            provenance of the loaded artifact
GET  /concept-map           the active interoperability contract
POST /v1/predict            FHIR Bundle or OMOP payload -> RiskAssessment

Liveness and readiness are deliberately different checks. Liveness answers "is
this process wedged, should the kubelet restart it"; it must not depend on the
model or the concept map, because restarting a pod cannot fix an unbound
concept map and a restart loop would only obscure the real problem. Readiness
answers "should this pod receive traffic" and is where every safety gate lives.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from . import audit
from .airgap import assert_offline
from .concept_map import ConceptMapError, check_serving_readiness, load_concept_map
from .config import settings
from .fhir_adapter import PayloadError, bundle_to_features, build_risk_assessment
from .model_runtime import ModelLoadError, runtime
from .omop_adapter import omop_to_features
from . import unmapped_stats
from .derivations import registry, registry_hash
from .schemas import (
    DerivationRecord,
    FeatureResolution,
    HealthResponse,
    PayloadFormat,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    RiskPrediction,
)

SERVICE_VERSION = "0.1.0"

_STATE: dict[str, Any] = {
    "concept_map": None,
    "concept_map_error": None,
    "startup_problems": [],
    "pepper_source": None,
}

NON_CLINICAL_DISCLAIMER = (
    "Proof-of-concept output. Not a validated clinical decision support device and "
    "not authorised under Korean MFDS medical device software rules. Must not be used "
    "to inform patient care."
)
CLINICAL_DISCLAIMER = (
    "Statistical estimate intended to supplement, not replace, clinical judgement."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enforce_airgap:
        assert_offline()

    _STATE["pepper_source"] = audit.init_pepper(
        settings.audit_pepper_file,
        allow_ephemeral=os.getenv("ALLOW_EPHEMERAL_PEPPER", "false").lower() == "true",
    )
    logger = audit.configure_logging(settings.audit_log_path)

    problems: list[str] = []

    # --- concept map ---------------------------------------------------
    try:
        cm = load_concept_map(settings.concept_map_path)
        _STATE["concept_map"] = cm
        problems.extend(check_serving_readiness(cm, settings.require_bound_concept_map))
    except ConceptMapError as exc:
        _STATE["concept_map_error"] = str(exc)
        problems.append(f"concept map failed to load: {exc}")
        cm = None

    # --- model ------------------------------------------------------------
    try:
        runtime.load(settings.model_path, settings.model_metadata_path)
        if cm is not None:
            problems.extend(runtime.validate_against_concept_map(cm))
            # Gate 5: derivation transforms must match those used at training.
            problems.extend(runtime.validate_derivations(cm))
            runtime.warmup(cm)
        if runtime.metadata and not runtime.metadata.is_clinical_artifact:
            msg = ("loaded artifact is flagged is_clinical_artifact=false "
                   "(non-validated proof-of-concept model)")
            if settings.require_clinical_artifact:
                problems.append(msg + " and REQUIRE_CLINICAL_ARTIFACT=true")
            else:
                logger.warning("Serving a NON-CLINICAL artifact: %s", msg)
    except (ModelLoadError, Exception) as exc:  # noqa: BLE001 - startup must not raise
        problems.append(f"model failed to load: {type(exc).__name__}: {exc}")

    _STATE["startup_problems"] = problems

    audit.write_audit_event(
        event="service_startup",
        request_id=audit.new_correlation_id(),
        status="degraded" if problems else "ok",
        model_id=runtime.metadata.model_id if runtime.metadata else None,
        model_version=runtime.metadata.model_version if runtime.metadata else None,
        concept_map_version=cm.version if cm else None,
        is_clinical_artifact=runtime.metadata.is_clinical_artifact if runtime.metadata else None,
        extra={
            "blocking_problems": problems,
            "pepper_source": _STATE["pepper_source"],
            "airgap_enforced": settings.enforce_airgap,
            "artifact_sha256": runtime.metadata.artifact_sha256[:16] if runtime.metadata else None,
            "derivation_registry_sha256": registry_hash()[:16],
        },
    )
    yield


app = FastAPI(
    title="SNUH Preoperative Risk Inference Service",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness. Intentionally checks nothing but the event loop."""
    return HealthResponse(status="alive", service=settings.service_name, version=SERVICE_VERSION)


@app.get("/readyz", response_model=ReadinessResponse, tags=["ops"])
async def readyz(response: Response) -> ReadinessResponse:
    """Readiness. Fails closed on any startup problem."""
    cm = _STATE["concept_map"]
    problems = list(_STATE["startup_problems"])

    checks: dict[str, Any] = {
        "concept_map_loaded": cm is not None,
        "concept_map_status": cm.status if cm else None,
        "concept_map_version": cm.version if cm else None,
        "unbound_omop_concepts": cm.unbound_features if cm else None,
        "features_pending_verification": cm.unverified_features if cm else None,
        "model_loaded": runtime.loaded,
        "model_warmup_ok": runtime.warmup_ok,
        "is_clinical_artifact": runtime.metadata.is_clinical_artifact if runtime.metadata else None,
        "audit_pepper_source": _STATE["pepper_source"],
        "airgap_enforced": settings.enforce_airgap,
        "derivation_registry_sha256": registry_hash()[:16],
        "derived_features_declared": cm.derived_names if cm else None,
        "derivation_transforms_registered": sorted(registry()),
        # Surfaced on the probe so a broken terminology binding is visible
        # without anyone having to run a log query first. Under
        # unmapped_policy='reject' a rising rate looks like a caller bug;
        # under 'map_to_other' it is otherwise invisible, because every
        # episode still scores.
        "value_set_coverage": {
            f.feature: {
                "codes_bound": len(f.value_code_index),
                "unmapped_policy": f.unmapped_policy,
                "expansions": f.value_set_expansions,
            }
            for f in (cm.features if cm else []) if f.value_set
        },
        "unmapped_value_stats": unmapped_stats.snapshot(),
    }

    if not runtime.warmup_ok:
        problems.append("model warmup has not completed successfully")

    ready = not problems
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(ready=ready, checks=checks, blocking_problems=problems)


@app.get("/model-card", tags=["ops"])
async def model_card() -> JSONResponse:
    if runtime.metadata is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    m = runtime.metadata
    return JSONResponse({
        "model_id": m.model_id,
        "model_version": m.model_version,
        "trained_on": m.trained_on,
        "target": m.target,
        "feature_order": m.feature_order,
        "is_clinical_artifact": m.is_clinical_artifact,
        "artifact_sha256": m.artifact_sha256,
        "training_provenance": m.training_provenance,
        "derivation_registry_sha256": m.derivation_registry_sha256,
        "derivation_manifest": m.derivation_manifest,
        "notes": m.notes,
    })


@app.get("/concept-map", tags=["ops"])
async def concept_map_endpoint() -> JSONResponse:
    cm = _STATE["concept_map"]
    if cm is None:
        return JSONResponse({"error": _STATE["concept_map_error"]}, status_code=503)
    return JSONResponse({
        "concept_map_id": cm.concept_map_id,
        "version": cm.version,
        "status": cm.status,
        "omop_cdm_version": cm.omop_cdm_version,
        "omop_vocabulary_version": cm.omop_vocabulary_version,
        "fhir_version": cm.fhir_version,
        "target": cm.target,
        "features": [
            {
                "feature": f.feature, "dtype": f.dtype, "required": f.required,
                "unit": f.unit, "loinc": f.loinc, "snomed": f.snomed,
                "omop_concept_id": f.omop_concept_id, "pending_verification": f.verify,
            }
            for f in cm.features
        ],
        "derived": [
            {
                "feature": d.feature, "identifier": d.identifier, "version": d.version,
                "inputs": list(d.inputs), "unit": d.unit,
                "_note": "Not terminology-bound by design; see app/derivations.py",
            }
            for d in cm.derived
        ],
        "derived_writeback": cm.derived_writeback,
    })


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(req: PredictionRequest, request: Request) -> Any:
    started = time.perf_counter()
    request_id = req.request_id or audit.new_correlation_id()
    cm = _STATE["concept_map"]

    # Readiness gate is re-checked per request: a pod can be routed to before
    # its readiness probe next fires, and serving a prediction from a degraded
    # pod is worse than returning 503.
    if cm is None or not runtime.loaded or _STATE["startup_problems"]:
        audit.write_audit_event(event="prediction_rejected", request_id=request_id,
                                status="unavailable", error_type="ServiceNotReady")
        return JSONResponse(
            {"error": "service not ready", "detail": _STATE["startup_problems"],
             "request_id": request_id},
            status_code=503,
        )

    pseudonym = audit.pseudonymise(req.subject_identifier, settings.pseudonym_length)

    # --- map payload -> features -----------------------------------------
    try:
        if req.format == PayloadFormat.fhir:
            features, resolutions, warnings = bundle_to_features(
                req.fhir_bundle, cm, settings.reject_implausible_values)
        else:
            features, resolutions, warnings = omop_to_features(
                req.omop_payload, cm, settings.reject_implausible_values)
    except PayloadError as exc:
        audit.write_audit_event(
            event="prediction_rejected", request_id=request_id,
            subject_pseudonym=pseudonym, payload_format=req.format.value,
            status="rejected", error_type="PayloadError",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        # The message may name a feature and a range but never a patient value
        # beyond what the caller itself sent, and it goes to the caller only —
        # not to the audit log.
        return JSONResponse({"error": "payload could not be mapped", "detail": str(exc),
                             "request_id": request_id}, status_code=422)

    # --- inference ---------------------------------------------------------
    try:
        probability = runtime.predict_proba(features)
    except Exception as exc:  # noqa: BLE001
        audit.write_audit_event(
            event="prediction_failed", request_id=request_id, subject_pseudonym=pseudonym,
            payload_format=req.format.value, status="error", error_type=type(exc).__name__,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return JSONResponse({"error": "inference failed", "request_id": request_id},
                            status_code=500)

    meta = runtime.metadata
    assert meta is not None
    latency_ms = (time.perf_counter() - started) * 1000

    risk_assessment = build_risk_assessment(
        probability=probability,
        subject_reference=f"Patient/{pseudonym}",
        cm=cm,
        model_id=meta.model_id,
        model_version=meta.model_version,
        request_id=request_id,
        is_clinical_artifact=meta.is_clinical_artifact,
    )

    reg = registry()
    derivation_records = [
        DerivationRecord(
            feature=d.feature, identifier=d.identifier, version=d.version,
            inputs=list(d.inputs), unit=d.unit,
            value=float(features[d.feature]) if d.feature in features else None,
        )
        for d in cm.derived
    ]

    resolved = [r.feature for r in resolutions if r.resolved]
    unresolved = [r.feature for r in resolutions if not r.resolved]

    audit.write_audit_event(
        event="prediction", request_id=request_id, subject_pseudonym=pseudonym,
        model_id=meta.model_id, model_version=meta.model_version,
        concept_map_version=cm.version, payload_format=req.format.value,
        probability=probability, resolved_features=resolved, unresolved_features=unresolved,
        latency_ms=latency_ms, is_clinical_artifact=meta.is_clinical_artifact,
        extra={"derivation_registry_sha256": registry_hash()[:16],
               "unmapped_rate_recent": unmapped_stats.snapshot()["unmapped_rate_recent"]},
    )

    return PredictionResponse(
        request_id=request_id,
        prediction=RiskPrediction(
            probability=probability,
            outcome=cm.target.get("name", "unknown"),
            model_id=meta.model_id,
            model_version=meta.model_version,
            concept_map_version=cm.version,
            derivation_registry_sha256=registry_hash(),
            is_clinical_artifact=meta.is_clinical_artifact,
            disclaimer=CLINICAL_DISCLAIMER if meta.is_clinical_artifact
            else NON_CLINICAL_DISCLAIMER,
        ),
        fhir_risk_assessment=risk_assessment,
        feature_resolution=resolutions,
        derivations=derivation_records,
        unresolved_optional_features=unresolved,
        warnings=warnings,
        latency_ms=round(latency_ms, 2),
    )
