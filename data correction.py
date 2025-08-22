#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean draw names (LAST, First -> First Last), match to Sackmann 2024 ATP matches,
compute per-player season features required by your RF model, and export a merged Excel.

Outputs columns:
name, draw_pos, seed, entry_type, surface,
rank, age, ht, ace_rate, df_rate, first_in_pct, first_win_pct, second_win_pct, bp_saved_pct,
matches_count, match_found

Assumptions:
- Draw is US Open (surface = "Hard") but you can change SURFACE below.
- Input draw file has at least: draw_pos, name, (optional) seed, entry_type
- Input matches file is Jeff Sackmann-style atp_matches_2024.csv
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from pathlib import Path

# =======================
# USER PATHS (EDIT THESE)
# =======================
MATCHES_FILE = r"D:\US open\clean data\atp_matches_2024.csv"
DRAW_FILE    = r"D:\US open\clean data\USOpen2025_MS_Draw.xlsx"
OUTPUT_FILE  = r"D:\US open\clean data\USOpen2025_PlayerFeatures_MERGED.xlsx"
SURFACE      = "Hard"  # US Open surface
# =======================


# ---------- Name cleaning helpers ----------

def strip_accents(s: str) -> str:
    # robust accent/diacritic stripping (keeps ASCII only)
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def to_canonical_key(full_name: str) -> str:
    """
    Build a *matching key* that is case/spacing/punctuation/diacritic insensitive.
    Example: "O'CONNELL, Christopher" and "Christopher O Connell" both -> "christopheroconnell"
    """
    if not isinstance(full_name, str):
        return ""
    s = strip_accents(full_name)
    s = s.lower().strip()
    # keep letters only for the key (drop punctuation, spaces, hyphens, apostrophes)
    s = re.sub(r"[^a-z]", "", s)
    return s

def draw_name_to_first_last(draw_name: str) -> str:
    """
    Convert 'SINNER, Jannik' -> 'Jannik Sinner'.
    If the name already looks like 'First Last', just tidy spacing.
    """
    if not isinstance(draw_name, str):
        return ""
    n = draw_name.strip()
    # skip placeholders
    if "QUALIFIER" in n.upper() or "LUCKY LOSER" in n.upper():
        return ""
    if "," in n:
        parts = n.split(",")
        last = parts[0].strip()
        first = parts[1].strip()
        # handle possible extra pieces after comma (e.g., middle names)
        # keep only first token of "first" if you prefer; here we keep as-is
        rebuilt = f"{first} {last}"
    else:
        rebuilt = n
    # collapse whitespace
    rebuilt = re.sub(r"\s+", " ", rebuilt).strip()
    return rebuilt


# ---------- Feature computation ----------

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def compute_features_for_player(matches_df: pd.DataFrame, name_key: str):
    """
    Aggregate season stats for a player given the canonical key.
    Returns dict of features + matches_count.
    """
    w = matches_df[matches_df["winner_key"] == name_key].copy()
    l = matches_df[matches_df["loser_key"]  == name_key].copy()

    if w.empty and l.empty:
        return {
            "rank": np.nan, "age": np.nan, "ht": np.nan,
            "ace_rate": np.nan, "df_rate": np.nan,
            "first_in_pct": np.nan, "first_win_pct": np.nan,
            "second_win_pct": np.nan, "bp_saved_pct": np.nan,
            "matches_count": 0, "match_found": 0
        }

    # Map winner/loser rows into unified columns
    w_assign = pd.DataFrame({
        "svpt":       safe_num(w["w_svpt"]),
        "ace":        safe_num(w["w_ace"]),
        "df":         safe_num(w["w_df"]),
        "first_in":   safe_num(w["w_1stIn"]),
        "first_won":  safe_num(w["w_1stWon"]),
        "second_won": safe_num(w["w_2ndWon"]),
        "bp_saved":   safe_num(w["w_bpSaved"]),
        "bp_faced":   safe_num(w["w_bpFaced"]),
        "rank":       safe_num(w["winner_rank"]),
        "age":        safe_num(w["winner_age"]),
        "ht":         safe_num(w["winner_ht"]),
    })

    l_assign = pd.DataFrame({
        "svpt":       safe_num(l["l_svpt"]),
        "ace":        safe_num(l["l_ace"]),
        "df":         safe_num(l["l_df"]),
        "first_in":   safe_num(l["l_1stIn"]),
        "first_won":  safe_num(l["l_1stWon"]),
        "second_won": safe_num(l["l_2ndWon"]),
        "bp_saved":   safe_num(l["l_bpSaved"]),
        "bp_faced":   safe_num(l["l_bpFaced"]),
        "rank":       safe_num(l["loser_rank"]),
        "age":        safe_num(l["loser_age"]),
        "ht":         safe_num(l["loser_ht"]),
    })

    stats = pd.concat([w_assign, l_assign], ignore_index=True)

    total_svpt = stats["svpt"].sum(skipna=True)
    total_first_in = stats["first_in"].sum(skipna=True)

    ace_rate       = stats["ace"].sum(skipna=True) / total_svpt if total_svpt > 0 else np.nan
    df_rate        = stats["df"].sum(skipna=True) / total_svpt if total_svpt > 0 else np.nan
    first_in_pct   = total_first_in / total_svpt if total_svpt > 0 else np.nan
    first_win_pct  = stats["first_won"].sum(skipna=True) / total_first_in if total_first_in > 0 else np.nan
    second_den     = (total_svpt - total_first_in)
    second_win_pct = stats["second_won"].sum(skipna=True) / second_den if second_den > 0 else np.nan
    bp_faced_sum   = stats["bp_faced"].sum(skipna=True)
    bp_saved_pct   = stats["bp_saved"].sum(skipna=True) / bp_faced_sum if bp_faced_sum > 0 else np.nan

    features = {
        "rank": stats["rank"].median(skipna=True),
        "age":  stats["age"].median(skipna=True),
        "ht":   stats["ht"].median(skipna=True),
        "ace_rate": ace_rate,
        "df_rate": df_rate,
        "first_in_pct": first_in_pct,
        "first_win_pct": first_win_pct,
        "second_win_pct": second_win_pct,
        "bp_saved_pct": bp_saved_pct,
        "matches_count": int(len(stats)),
        "match_found": 1
    }
    return features


def main():
    # Load data
    draw = pd.read_excel(DRAW_FILE)
    matches = pd.read_csv(MATCHES_FILE, low_memory=False)

    # Clean draw names
    draw["name_firstlast"] = draw["name"].apply(draw_name_to_first_last)
    draw["name_key"] = draw["name_firstlast"].apply(to_canonical_key)

    # Build canonical keys for matches file (winner & loser)
    matches["winner_key"] = matches["winner_name"].fillna("").apply(to_canonical_key)
    matches["loser_key"]  = matches["loser_name"].fillna("").apply(to_canonical_key)

    # Compute features for each draw row
    features = []
    for _, r in draw.iterrows():
        nk = r.get("name_key", "")
        # Skip placeholders
        if not nk:
            f = {
                "rank": np.nan, "age": np.nan, "ht": np.nan,
                "ace_rate": np.nan, "df_rate": np.nan,
                "first_in_pct": np.nan, "first_win_pct": np.nan,
                "second_win_pct": np.nan, "bp_saved_pct": np.nan,
                "matches_count": 0, "match_found": 0
            }
        else:
            f = compute_features_for_player(matches, nk)

        f["surface"] = SURFACE  # constant for US Open
        features.append(f)

    feat_df = pd.DataFrame(features)

    # Merge features back to draw
    merged = pd.concat([draw.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # Reorder columns nicely if present
    cols_front = [c for c in ["draw_pos", "name", "name_firstlast", "seed", "entry_type"] if c in merged.columns]
    feature_cols = ["surface","rank","age","ht","ace_rate","df_rate","first_in_pct","first_win_pct","second_win_pct","bp_saved_pct","matches_count","match_found"]
    other_cols = [c for c in merged.columns if c not in cols_front + feature_cols + ["name_key"]]
    merged = merged[cols_front + feature_cols + other_cols + ["name_key"]]

    # Save
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_path, index=False)
    print(f"✅ Saved merged player features to: {out_path}")

    # Quick sanity print
    sample = merged.loc[merged["match_found"] == 1, ["name","rank","ace_rate","first_in_pct","first_win_pct","second_win_pct","bp_saved_pct","matches_count"]].head(12)
    print("\nSample rows with matches found:")
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
