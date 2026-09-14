"""
Quick check (not part of the main pipeline): for the 108 patients who lack ART_MBP
but have NIBP_MBP, how sparse is NIBP_MBP actually within the 5-minute post-episode
window used for the hypotension outcome label? For each of their 313 episodes, count
how many valid (non-NaN) NIBP_MBP readings fall in [episode_start_sec, +300s).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

DATA_DIR = Path(__file__).parent / "arrhythmia_ep_data"
CLINICAL_CSV = DATA_DIR / "imputed_477_cases.csv"
EPISODES_CSV = DATA_DIR / "arrhythmia_episodes_482.csv"

ART_TRACK = "Solar8000/ART_MBP"
NIBP_TRACK = "Solar8000/NIBP_MBP"
WINDOW_SEC = 300

clinical = pd.read_csv(CLINICAL_CSV)
episodes = pd.read_csv(EPISODES_CSV)
case_ids = clinical["caseid"].tolist()

# Re-derive the 108 ART_MBP-missing caseids (same logic as the earlier coverage check)
track_names_df = vitaldb.get_track_names(caseids=case_ids)
has_art = track_names_df["tnames"].apply(lambda names: ART_TRACK in names)
nibp_only_caseids = track_names_df.loc[~has_art, "caseid"].tolist()
print(f"NIBP-only patients: {len(nibp_only_caseids)}")

nibp_only_episodes = episodes[episodes["caseid"].isin(nibp_only_caseids)].copy()
print(f"Their episodes: {len(nibp_only_episodes)}")

reading_counts = []
for caseid, group in nibp_only_episodes.groupby("caseid"):
    nibp_vals = vitaldb.load_case(caseid, [NIBP_TRACK], 1.0)[:, 0]
    n_total_seconds = len(nibp_vals)

    for _, ep in group.iterrows():
        start_idx = int(ep["episode_start_sec"])
        end_idx = min(start_idx + WINDOW_SEC, n_total_seconds)
        if start_idx >= n_total_seconds:
            n_readings = 0
        else:
            window_vals = nibp_vals[start_idx:end_idx]
            non_nan = window_vals[~np.isnan(window_vals)]
            # IMPORTANT: NIBP_MBP is broadcast every ~1-2s like the other Solar8000 numeric
            # tracks, but the cuff only actually measures every several minutes - the same
            # value gets repeated across many consecutive broadcast slots in between. Simply
            # counting non-NaN samples (like we did for ART_MBP) would massively overcount the
            # number of REAL readings. Instead, count value CHANGES: a run of repeated values
            # is one actual reading, not one-per-second.
            if len(non_nan) == 0:
                n_readings = 0
            else:
                n_readings = int(1 + np.sum(np.diff(non_nan) != 0))
        reading_counts.append({
            "caseid": caseid,
            "episode_number": ep["episode_number"],
            "episode_start_sec": ep["episode_start_sec"],
            "n_nibp_readings_in_5min": n_readings,
        })

results = pd.DataFrame(reading_counts)
print(f"\nProcessed {len(results)} episodes")
print(f"Mean DISTINCT readings per episode in the 5-minute window: {results['n_nibp_readings_in_5min'].mean():.2f}")
print(f"Median: {results['n_nibp_readings_in_5min'].median()}")

bins = pd.cut(
    results["n_nibp_readings_in_5min"],
    bins=[-0.1, 0, 1, 2, np.inf],
    labels=["0", "1", "2", "3+"],
)
print("\nDistribution of NIBP_MBP reading counts within the 5-minute post-episode window:")
print(bins.value_counts().sort_index())

results.to_csv(Path(__file__).parent / "nibp_reading_counts_per_episode.csv", index=False)
print(f"\nSaved per-episode detail to nibp_reading_counts_per_episode.csv")
