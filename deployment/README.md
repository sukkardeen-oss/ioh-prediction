# SNUH Intraoperative Hypotension Risk Service

Proof-of-concept interoperability and deployment module that exposes the
tabular classification pipeline as an OMOP/FHIR-native inference service,
containerised for SNUH's air-gapped SDDC private cloud.

> ### Regulatory status — read before deploying
>
> Software that informs a clinical decision about an individual patient is a
> **medical device** under Korean MFDS rules. This package has not been through
> device classification, prospective clinical validation, or IRB review.
> It is suitable for **technical integration testing against synthetic or
> de-identified data only**.
>
> The service enforces this rather than merely stating it: it refuses to serve
> until the artifact is explicitly attested as clinical, and it stamps
> non-clinical output as `status: preliminary` *inside* the FHIR resource so
> the warning survives being forwarded or stored somewhere this file is not.

---

## What this is, and what it is not

| | |
|---|---|
| **Is** | A working interoperability boundary: FHIR/OMOP → feature vector → probability → FHIR RiskAssessment, with the mapping externalised into a reviewable concept map |
| **Is** | A deployment shape for an air-gapped SDDC: offline build, probes, anonymised audit, hardened container, VDI delivery |
| **Is not** | A validated risk model. The shipped artifact is trained on synthetic VitalDB-shaped data and encodes no clinical relationship |
| **Is not** | Terminology-complete. OMOP concept IDs are deliberately unbound; SNUH's terminology team must populate them |

---

## Implementation status — what's actually in this folder

This snapshot is a **design document with a working core**, not the runnable
service the rest of this README describes. Concretely:

**Present and real:**
- `concept_map.py` — loading, validation, and the readiness-gate logic
- `fhir_adapter.py` / `omop_adapter.py` — both payload adapters, including the
  value-set resolution and unit/plausibility rejection logic
- `unmapped_stats.py` — the rolling unmapped-rate counter
- `snuh_intraop_hypotension_concept_map.json` — the concept map itself
- `main.py` and `test_contract.py` — written against the *complete* system
  described below, so they name the missing pieces directly

**Referenced throughout this README and the code above, but not present:**
- `schemas.py` (every Pydantic model — `FeatureResolution`, `FHIRBundle`,
  `PredictionRequest`, etc.), `config.py` (`settings`), `audit.py`
  (pseudonymisation + audit logging), `airgap.py` (`assert_offline`),
  `model_runtime.py` (loads/validates the sklearn artifact),
  `derivations.py` (the 9 feature-computation transforms)
- The `app/` package wrapper itself — every file here uses relative imports
  (`from . import audit`, `from app.main import app`) that only resolve once
  these files live inside a folder literally named `app/` with an
  `__init__.py`. They currently sit flat in this folder.
- A `concept_map/` subfolder (`test_contract.py` looks for the JSON there,
  not in the folder root where it actually is) and an `artifacts/` folder
  with a trained model + `model_metadata.json`
- `scripts/export_model.py`, `vendor_wheels.sh`, `build_offline.sh`,
  `verify_offline.sh`, `package_release.sh`, and `k8s/deployment.yaml`

**Practical effect:** `main.py`'s first import (`from . import audit`) fails
immediately, so the service cannot start, and `test_contract.py` cannot
collect because it imports `app.main`. Everything below this point in the
README describes the *intended* complete design — the interoperability
contract, the safety gates, the audit model — which the present files
implement in part; treat it as an architecture spec plus a partial reference
implementation, not a quick start.

---

## Architecture

```
EHR / OHDSI extract
      │
      │  FHIR R4 Bundle          OMOP CDM 5.4 slice
      │  (Patient, Encounter,    (person, measurement,
      │   Observation)            observation)
      ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI  (app/main.py)                                 │
│                                                          │
│  schemas.py       strict pydantic, extra="forbid"        │
│        │          direct identifiers refused at boundary │
│        ▼                                                 │
│  fhir_adapter.py / omop_adapter.py                       │
│        │          resolve via concept_map.py ONLY        │
│        │          reject on unit mismatch / implausible  │
│        ▼                                                 │
│  model_runtime.py  sklearn pipeline, loaded once,        │
│        │           digest-verified, warmed at startup    │
│        ▼                                                 │
│  FHIR RiskAssessment  +  audit.py (HMAC pseudonym)       │
└─────────────────────────────────────────────────────────┘
      │                                  │
      ▼                                  ▼
  EHR integration layer            /var/log/snuh/audit.jsonl
```

Both adapters converge on one feature dict and one resolution trail, so audit
output and downstream logic are payload-format independent.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | **Liveness.** Event loop only |
| `GET` | `/readyz` | **Readiness.** Every safety gate; 503 when blocked |
| `GET` | `/model-card` | Artifact provenance, digest, calibration |
| `GET` | `/concept-map` | Active interoperability contract |
| `POST` | `/v1/predict` | FHIR Bundle or OMOP payload → RiskAssessment |

### Why liveness and readiness check different things

`/healthz` deliberately checks **nothing but the event loop**. It is tempting
to make liveness verify the model too, but restarting a pod cannot fix an
unbound concept map or a missing artifact — it just produces a restart loop
that buries the real error under `CrashLoopBackOff`. Keeping liveness dumb
means a misconfigured pod stays **up and diagnosable**: you can still curl
`/readyz` and `/concept-map` and read exactly what is wrong.

`/readyz` is where the gates live, and it fails closed:

```json
{
  "ready": false,
  "blocking_problems": [
    "concept_map status is 'UNBOUND', expected 'BOUND'...",
    "concept_map.omop_vocabulary_version is null...",
    "16 feature(s) have no OMOP concept_id: [...]",
    "1 feature(s) still use placeholder site-local codings: ['asa']...",
    "loaded artifact is flagged is_clinical_artifact=false ... and REQUIRE_CLINICAL_ARTIFACT=true"
  ]
}
```

A `startupProbe` holds both off during load; without it a slow artifact load
trips liveness and restart-loops forever.

---

## The five deployment gates

The service will not serve traffic until all four are satisfied. This is the
core safety property: **the failure mode being prevented is a well-formatted,
confident, clinically meaningless number reaching a chart.**

1. **Concept map `BOUND`** — `omop_vocabulary_version` populated and every
   `omop.concept_id` filled against SNUH's ATHENA build.
2. **No placeholder local codings** — currently `asa`.
3. **`is_clinical_artifact: true`** — requires a validation report and named
   approver via `export_model.py --attest-clinical`.
4. **Audit pepper mounted** — ≥32 bytes at `/run/secrets/audit_pepper`.
5. **Derivation registry hash matches the artifact** — the transforms in this
   image must be byte-identical to those used at training. See below.

For PoC runs, override explicitly:
`REQUIRE_BOUND_CONCEPT_MAP=false REQUIRE_CLINICAL_ARTIFACT=false ALLOW_EPHEMERAL_PEPPER=true`

---

## The concept map is the contract

`concept_map/snuh_vitaldb_concept_map.json` is the single place a code is bound
to a feature. No adapter hardcodes a code anywhere else, so terminology review
is a review of one JSON file rather than an audit of the codebase.

**LOINC codes are shipped; OMOP concept IDs are deliberately null.** Standard
concept IDs are specific to an ATHENA vocabulary release, and binding them from
outside SNUH's own build would be a guess dressed up as a mapping. Readiness
fails until they are populated.

Six features are flagged `verify: true` and need SNUH review. Two are worth
calling out because the failure is silent:

- **`preop_cr`** — mg/dL vs µmol/L is an ~88× error that still returns a
  plausible-looking probability.
- **`preop_pt`** — VitalDB reports PT as **percent activity**, not INR
  (LOINC 6301-6) and not seconds (LOINC 5902-2). Three different scales,
  routinely confused.

**ASA** has no clean universal LOINC, so it uses an explicit *site-local*
coding slot with a placeholder URI that blocks readiness until replaced. This
reflects reality — perioperative sites typically carry ASA locally pending
SNOMED binding — rather than inventing a code to make the demo pass.

### Intraoperative features are excluded by design

`intraop_*` variables (EBL, urine output, transfusion, vasopressor doses) are
**not** in the contract. They are unavailable at the moment a preoperative risk
estimate is clinically actionable. Including them would inflate retrospective
performance and produce a model that cannot run when it is needed. The
exclusion is recorded in the concept map so it stays a reviewable decision
rather than an oversight.

---

---

## Derived features: bound to a version, not a code

Roughly three quarters of the model's columns are computed rather than
observed. That forced a decision the terminology layer alone cannot express.

**Why derived features are not given codes.** LOINC and SNOMED CT describe
observations made about a patient. `map_auc_below_65` — the area between
65 mmHg and MAP(t) over the episode — is not an observation; it is a
computation over observations. Minting a local code for it produces a concept
that *looks* bound but that no other site, and no future version of this
pipeline, can interpret without also holding our source.

Worse, coding it hides the failure that matters. Two sites can both hold a
legitimately-bound concept called "time-weighted mean MAP" and compute it
differently: trapezoidal versus sample mean, 20 s versus 60 s resampling,
different handling of artefact gaps. The terminology layer reports success, the
model receives a different quantity than it was trained on, and nothing errors.
It is the mg/dL-versus-µmol/L failure again, except no unit check can catch it.

**The two-layer split.** `concept_map/*.json` now has two arrays:

| | `features` (primitives) | `derived` |
|---|---|---|
| What | things an instrument or clinician observed | computations over primitives |
| Bound to | LOINC / SNOMED CT concepts | a version-bearing local identifier |
| Rebindable per site | yes | no — travels with the artifact |
| Example | `map_series` → LOINC 8478-0 | `map_auc_below_65` → `urn:snuh:derived:map-auc-below-65:v1.0.0` |

The version lives **inside** the identifier, so a changed derivation becomes a
different concept rather than a silent redefinition of an existing one.

**The transform ships in the artifact.** `app/derivations.py` holds the
executable transforms; `export_model.py` hashes the registry into
`model_metadata.json`; readiness gate 5 refuses to serve if the image's
registry differs from the one used at training. This is what makes the
portability claim real rather than asserted — rebinding the primitives to
another institution's terminology *cannot* change what a derived feature means,
because the derivation is part of the artifact, not part of the site config.

`test_changed_transform_is_detected` is the regression test: it edits a
transform's spec without bumping its version and asserts readiness fails.

**Leakage guard.** Every derivation reads a series truncated to the observation
window ending at the prediction anchor (`onset + observation_window_s`).
Post-anchor samples are **rejected, not dropped** — at training time they are
outcome leakage, and at serving time their presence means the caller assembled
the window wrongly, which deserves a 422 rather than a silently different
feature vector.

**Writeback.** If derived values are persisted for audit, `derived_writeback`
in the concept map specifies the idiom: FHIR `Observation.derivedFrom` pointing
at source Observations with `Observation.method` naming the transform version;
OMOP `MEASUREMENT` with `measurement_concept_id = 0`, which is the honest
encoding for "no standard concept exists" rather than a forced mis-mapping.

## Coded value sets: pre-expansion and the unmapped rate

SNOMED CT is a hierarchy, and EHRs record at different levels of detail. If
`atrial_fibrillation` is bound only to a generic AF concept, a payload carrying
*chronic* or *permanent* AF is a legitimate descendant that would still be
rejected. In live traffic that could reject a large share of episodes, and it
would look like a caller bug rather than a terminology gap.

Because the SDDC is air-gapped, the service cannot call a terminology server to
test subsumption per request. So the descendant closure is **pre-expanded into
the concept map at bind time**:

```json
"atrial_fibrillation": {
  "primary": "49436004",
  "includes": ["426749004", "440028005"],
  "_expansion": "is-a descendants, ATHENA <release>, expanded <date>"
}
```

The loader flattens this into a code → member index, so request-time resolution
is a dict lookup. Two guards: overlapping closures (one code claimed by two
members) are rejected at load, because assignment would otherwise depend on
dict ordering; and an expanded list with no `_expansion` provenance blocks
readiness, because a closure is only valid for the vocabulary release it came
from — a later release adding concepts silently shrinks coverage.

**`unmapped_policy` is a deliberate choice, not a default.**

| | `reject` (default) | `map_to_other` |
|---|---|---|
| Unmapped code | 422 | folded into `other`, with a warning |
| Argument for | `other` in training meant "residual rhythms we observed", not "codes we failed to map" | in a live theatre, a 422 means no score at all for that episode |
| Failure mode | rising 422 rate looks like a caller bug | invisible: every episode still scores, on a value the model never saw |

Since both failure modes are hard to spot, the **unmapped rate is counted
either way** and exposed on `/readyz` under `unmapped_value_stats`, with the
offending codes named (a code is schema, not patient data). A rising rate is
how a vocabulary update that broke a binding becomes visible.

## Patient-grouped splitting

Episodes are nested within patients (1,284 episodes, 457 patients, ~2.8 each).
A plain `train_test_split` puts a patient's own episodes on both sides and
inflates every metric. `export_model.py` uses `StratifiedGroupKFold` on
`patient_id` and hard-fails if any patient appears in both partitions.

---

## Input handling: reject, do not coerce

| Condition | Behaviour |
|---|---|
| Unit mismatch vs concept map | **422** — convert upstream so it is explicit and reviewable |
| Value outside `plausible_range` | **422** — more likely a unit/mapping error than a real observation |
| Required feature unresolved | **422** — required inputs are never imputed |
| >1 `Patient` in a Bundle | **422** — prevents cross-attributing a result |
| `Patient.name` / `telecom` / `address` | **422** — direct identifiers refused at the boundary |
| Unknown field anywhere | **422** — `extra="forbid"` throughout |

Observations are indexed on `(system, code)`, not code alone: a site-local
code colliding with a LOINC code would otherwise silently overwrite a real lab.

---

## Audit logging

Two requirements pull against each other: governance needs a durable record of
every inference; privacy requires the log not become a second PHI store outside
the EHR's access controls.

- **Identifiers** → `HMAC-SHA256(pepper, mrn)`, truncated. Plain SHA-256 would
  be useless: the MRN space is small and enumerable, so an unkeyed digest is
  brute-forceable in seconds. The pepper is read from a **mounted secret file**,
  not an env var — env vars surface in `kubectl describe`, crash dumps, and
  every child process.
- **Clinical values are never logged.** Feature *names* are (they are schema);
  values are not. Reconstructing a patient's lab panel from the audit log
  should be impossible.
- **The risk score is logged.** That is the point — it lets SNUH answer "what
  did the model say about this case, and when".
- **Error class only**, never the message, which could quote a payload value.

Rotating the pepper intentionally breaks linkage to historical entries. That is
correct behaviour for a privacy control; if longitudinal linkage is needed
across a rotation, keep a sealed mapping in the EHR domain, not here.

```json
{"ts":"2026-08-17T00:52:23Z","event":"prediction","status":"ok",
 "subject_pseudonym":"sub_909986630b9c3d45","model_id":"snuh-preop-mortality-logreg",
 "model_version":"0.1.0","concept_map_version":"0.1.0-POC","payload_format":"fhir",
 "probability":0.607053,"n_features_resolved":14,
 "features_resolved":["age","asa","bmi",...],"features_unresolved":["preop_pt",...],
 "latency_ms":7.58,"is_clinical_artifact":false}
```

---

## Air-gap

The SDDC is air-gapped at the perimeter; the module adds defence in depth
because the failure it prevents is **quiet**. An ML dependency that phones home
for a model card or telemetry does not fail fast on an air-gapped network — it
*hangs* until the socket times out, turning a 40 ms inference into a 30 s one
and tripping liveness probes long before anyone suspects a network call.

- `app/airgap.py` patches `socket.connect` to reject non-loopback destinations,
  converting the hang into an immediate error naming the responsible library.
- Build runs `--network=none`, installing only from vendored wheels.
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `NO_PROXY=*` set in the image.
- NetworkPolicy denies egress except kube-dns; ingress only from the EHR
  integration namespace.

**The serving image excludes torch, XGBoost, SHAP, TabPFN and TabFM.** The
selected artifact is a scikit-learn pipeline, so the training stack is not
needed at inference. This drops the image from several GB to ~250 MB and, more
importantly, removes every dependency that would otherwise try to reach a model
hub at import time.

BLAS threading is pinned (`OMP_NUM_THREADS=2`). Left unset, OpenMP sizes its
pool to the **host** core count rather than the container's CPU limit — on a
large SDDC node that means dozens of threads fighting over a 1-core cgroup, so
latency gets *worse* under load.

---

## Calibration is a safety property here

The export script **blocks** on calibration-in-the-large (mean predicted risk
vs observed event rate must be within 0.5–2.0×).

This gate exists because of a bug caught during development. With
`class_weight="balanced"`, the model is calibrated to a ~50% event rate instead
of the true ~1.9%, and returned a **0.989** mortality probability for a
sick-but-survivable case. ROC AUC was unaffected — which is exactly why it is
easy to miss: the model looks fine on the metric everyone checks, then emits a
number a clinician reads as near-certain death. Since this service publishes a
probability into a `RiskAssessment`, calibration is a patient-safety property,
not a modelling nicety. After the fix the same case returns 0.61.

For imbalanced outcomes, prefer threshold selection at point of use over
reweighting at fit time.

---

## Quick start (PoC, local)

> Does not currently run as-is — see
> [Implementation status](#implementation-status--whats-actually-in-this-folder).
> `scripts/`, the missing `app/` modules, and the `artifacts/` folder this
> section depends on are not present in this snapshot. Shown here as the
> intended workflow once they are.

```bash
python scripts/export_model.py --synthetic      # build a VitalDB-shaped artifact
python -m pytest tests/ -q                      # 25 contract tests

ALLOW_EPHEMERAL_PEPPER=true \
REQUIRE_CLINICAL_ARTIFACT=false \
REQUIRE_BOUND_CONCEPT_MAP=false \
uvicorn app.main:app --port 8000
```

Swap in the real artifact:

```bash
python scripts/export_model.py --from-vitaldb /path/to/vitaldb_clinical.csv
```

Still writes `is_clinical_artifact=false` — training on real data does not make
a model validated. Flipping the flag is a separate governance act requiring
`--attest-clinical --validation-report <ref> --approved-by <name>`.

---

## Build and deliver to SNUH

### On the connected machine

```bash
./scripts/vendor_wheels.sh            # 31 wheels, ~81MB; verifies offline install
python scripts/export_model.py --synthetic
./scripts/build_offline.sh            # docker build --network=none
./scripts/verify_offline.sh           # runs container with --network=none
VERSION=0.1.0 ./scripts/package_release.sh
```

`vendor_wheels.sh` passes **several** manylinux platform tags. Projects publish
against different baselines — scikit-learn ships `manylinux_2_28`,
pydantic-core `manylinux_2_17` — so a single tag makes pip report "no matching
distribution" for a version that plainly exists on PyPI. It then verifies the
vendored set installs with `--no-index`, so a missing wheel fails here rather
than on the air-gapped side where debugging is far more expensive.

### VDI transfer

```bash
# 1. Move tarball + .sha256 through the VDI file broker
# 2. On the SDDC side, verify BEFORE extracting:
sha256sum -c snuh-preop-risk-0.1.0.tar.gz.sha256
# 3. Extract and verify every file:
tar xzf snuh-preop-risk-0.1.0.tar.gz
cd snuh-preop-risk-0.1.0 && sha256sum -c SHA256SUMS
# 4-5. Build and verify offline
./scripts/build_offline.sh && ./scripts/verify_offline.sh
# 6. Submit to the Secure Code Repository with MANIFEST.txt attached
```

A VDI hop is one-way with no retry: whatever lands on the far side is what gets
deployed. The manifest and per-file checksums are the only way the receiving
side can distinguish a truncated transfer from a complete one before building.

---

## Deployment

```bash
kubectl -n snuh-ml create secret generic snuh-audit-pepper \
  --from-literal=audit_pepper="$(openssl rand -hex 32)"
kubectl apply -f k8s/deployment.yaml
```

Hardening: non-root uid 10001, read-only root filesystem, all capabilities
dropped, `no-new-privileges`, seccomp `RuntimeDefault`, no service-account
token, memory-backed `/tmp`, image **pinned by digest** (a mutable tag means
the running model is not determined by the manifest, which defeats the audit
trail).

---

## Test suite

`test_contract.py` holds 45 tests, focused on failures that are dangerous
because they are quiet. They cannot currently execute — see
[Implementation status](#implementation-status--whats-actually-in-this-folder)
— but this is the full contract they check:

```
test_liveness_does_not_depend_on_model
test_readiness_reports_derivation_state
test_readiness_fails_closed_on_unbound_concept_map
test_derived_features_carry_no_terminology_code
test_derivation_inputs_must_be_declared_primitives
test_concept_map_rejects_derived_input_that_is_not_a_primitive
test_concept_map_rejects_version_identifier_mismatch
test_series_primitives_are_not_model_columns
test_artifact_records_derivation_registry_hash
test_changed_transform_is_detected
test_fhir_prediction_returns_risk_assessment
test_response_carries_derivation_provenance
test_post_anchor_sample_is_rejected
test_unit_mismatch_is_rejected_not_converted
test_multiple_patients_in_bundle_rejected
test_patient_name_is_refused_at_the_boundary
test_raw_identifier_never_appears_in_response
test_audit_log_has_no_phi_and_records_derivation_hash
test_pepper_rotation_breaks_linkage
test_model_feature_space_matches_concept_map
test_omop_resolves_and_derives_once_bound
test_both_adapters_produce_the_same_feature_space
test_descendant_code_resolves_without_a_terminology_server
test_unmapped_rate_is_observable_on_readiness
...
```

`test_model_feature_space_matches_concept_map` guards the most dangerous drift
in the design: if the artifact's feature order and the concept map disagree,
predictions are computed on misaligned columns and look entirely normal.
Checked at startup, not per request.

---

## Open items before any clinical use

1. Bind all OMOP concept IDs; set `omop_vocabulary_version`; flip to `BOUND`.
2. Replace the ASA placeholder CodeSystem URI; supply SNOMED concepts.
3. Resolve the six `verify: true` features against SNUH's lab catalogue.
4. Retrain on the real VitalDB extract; assess calibration and subgroup
   performance, not just AUC.
5. Prospective validation, MFDS device classification, IRB review.
6. Decide the clinical workflow: who sees the score, at what point, and what
   action it is meant to support. A risk number with no defined decision
   attached to it changes nothing except liability.
7. Drift monitoring — the audit log carries what is needed for it.
