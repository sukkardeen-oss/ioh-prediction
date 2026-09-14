"""
Quick check (not part of the main pipeline), to inform a pending decision on whether
ART_MBP-missing patients can be rescued with a NIBP-based hybrid approach:

1. Of the 1295 episodes in arrhythmia_episodes_482.csv, how many belong to the 369
   patients who DO have Solar8000/ART_MBP available?
2. For the 108 patients who DON'T have ART_MBP, what % have the non-invasive cuff
   tracks (Solar8000/NIBP_MBP / NIBP_SBP / NIBP_DBP) instead?
"""

from pathlib import Path

import pandas as pd
import vitaldb

DATA_DIR = Path(__file__).parent / "arrhythmia_ep_data"
CLINICAL_CSV = DATA_DIR / "imputed_477_cases.csv"
EPISODES_CSV = DATA_DIR / "arrhythmia_episodes_482.csv"

ART_TRACK = "Solar8000/ART_MBP"
NIBP_TRACKS = ["Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]

clinical = pd.read_csv(CLINICAL_CSV)
episodes = pd.read_csv(EPISODES_CSV)
case_ids = clinical["caseid"].tolist()
print(f"Cohort size: {len(case_ids)}")
print(f"Total episodes in arrhythmia_episodes_482.csv: {len(episodes)}")

track_names_df = vitaldb.get_track_names(caseids=case_ids)
has_art_mbp = track_names_df["tnames"].apply(lambda names: ART_TRACK in names)

caseids_with_art = track_names_df.loc[has_art_mbp, "caseid"].tolist()
caseids_without_art = track_names_df.loc[~has_art_mbp, "caseid"].tolist()
print(f"\nPatients WITH {ART_TRACK}: {len(caseids_with_art)}")
print(f"Patients WITHOUT {ART_TRACK}: {len(caseids_without_art)}")

# ---- Part 1: episode count restricted to the 369 ART_MBP-available patients ----
episodes_with_art = episodes[episodes["caseid"].isin(caseids_with_art)]
print(f"\n--- Part 1 ---")
print(f"Episodes belonging to ART_MBP-available patients: {len(episodes_with_art)} / {len(episodes)}")
print(f"Episodes belonging to ART_MBP-missing patients (would be lost): {len(episodes) - len(episodes_with_art)} / {len(episodes)}")

# ---- Part 2: NIBP coverage among the 108 ART_MBP-missing patients ----
print(f"\n--- Part 2 ---")
n_missing = len(caseids_without_art)
nibp_track_df = track_names_df[track_names_df["caseid"].isin(caseids_without_art)]

has_nibp = {}
for track in NIBP_TRACKS:
    has_nibp[track] = nibp_track_df["tnames"].apply(lambda names: track in names)
    print(f"{track}: present in {has_nibp[track].sum()} / {n_missing} "
          f"({has_nibp[track].mean()*100:.1f}% of the ART_MBP-missing patients)")

has_all_nibp = has_nibp[NIBP_TRACKS[0]] & has_nibp[NIBP_TRACKS[1]] & has_nibp[NIBP_TRACKS[2]]
print(f"\nHave ALL THREE NIBP tracks: {has_all_nibp.sum()} / {n_missing} ({has_all_nibp.mean()*100:.1f}%)")

rescued_caseids = nibp_track_df.loc[has_nibp["Solar8000/NIBP_MBP"].values, "caseid"].tolist()
print(f"\nCaseids rescuable via NIBP_MBP: {rescued_caseids}")

still_unrescuable = nibp_track_df.loc[~has_nibp["Solar8000/NIBP_MBP"].values, "caseid"].tolist()
print(f"\nCaseids with NEITHER ART_MBP NOR NIBP_MBP: {still_unrescuable}")

# How many episodes would the NIBP-rescued patients add back, for context
episodes_rescuable = episodes[episodes["caseid"].isin(rescued_caseids)]
print(f"\nEpisodes belonging to NIBP-rescuable patients: {len(episodes_rescuable)}")
print(f"Combined episode coverage (ART_MBP + NIBP-rescued) would be: "
      f"{len(episodes_with_art) + len(episodes_rescuable)} / {len(episodes)}")
