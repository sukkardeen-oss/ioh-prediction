"""
Quick check (not part of the main pipeline): how many of our 477 cases have
Solar8000/ART_SBP and Solar8000/ART_DBP, as a possible fallback for computing
MAP via the formula MAP = DBP + 1/3*(SBP - DBP) when ART_MBP itself is missing.
"""

from pathlib import Path

import pandas as pd
import vitaldb

CLINICAL_CSV = Path(__file__).parent / "arrhythmia_ep_data" / "imputed_477_cases.csv"

TRACKS = ["Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP"]

clinical = pd.read_csv(CLINICAL_CSV)
case_ids = clinical["caseid"].tolist()
n_total = len(case_ids)
print(f"Cohort size: {n_total}")

track_names_df = vitaldb.get_track_names(caseids=case_ids)

has = {}
for track in TRACKS:
    has[track] = track_names_df["tnames"].apply(lambda names: track in names)
    print(f"{track}: present in {has[track].sum()} / {n_total} ({has[track].mean()*100:.1f}%)")

has_mbp = has["Solar8000/ART_MBP"]
has_sbp = has["Solar8000/ART_SBP"]
has_dbp = has["Solar8000/ART_DBP"]

has_both_sbp_dbp = has_sbp & has_dbp
print(f"\nHave BOTH ART_SBP and ART_DBP: {has_both_sbp_dbp.sum()} / {n_total} ({has_both_sbp_dbp.mean()*100:.1f}%)")

rescuable = (~has_mbp) & has_both_sbp_dbp
print(f"Missing ART_MBP but COULD compute it from SBP+DBP: {rescuable.sum()} / {n_total} ({rescuable.mean()*100:.1f}%)")

still_missing = (~has_mbp) & ~has_both_sbp_dbp
print(f"Missing ART_MBP and CANNOT rescue it (missing SBP or DBP too): {still_missing.sum()} / {n_total} ({still_missing.mean()*100:.1f}%)")

combined_coverage = has_mbp | has_both_sbp_dbp
print(f"\nTotal coverage if using real MBP where available, formula-derived MBP otherwise: "
      f"{combined_coverage.sum()} / {n_total} ({combined_coverage.mean()*100:.1f}%)")

print("\nCase IDs that would be rescued by the SBP/DBP formula:")
print(clinical.loc[rescuable.values, "caseid"].tolist())

print("\nCase IDs still missing ART_MBP even with the SBP/DBP fallback:")
print(clinical.loc[still_missing.values, "caseid"].tolist())
