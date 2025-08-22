import pandas as pd
import numpy as np

# Paths
matches_file = r"D:\US open\clean data\atp_matches_2024.csv"
draw_file = r"D:/US open/clean data/USOpen2025_MS_Draw.xlsx"  # your converted draw
output_file = r"D:\US open\clean data\USOpen2025_PlayerFeatures.xlsx"

# Load datasets
matches = pd.read_csv(matches_file, low_memory=False)
draw = pd.read_excel(draw_file)

# Clean names (remove spaces and commas for consistency)
def normalize_name(name):
    if not isinstance(name, str):
        return name
    return name.strip().upper().replace(",", "")

draw["name_clean"] = draw["name"].apply(normalize_name)
matches["winner_name_clean"] = matches["winner_name"].apply(normalize_name)
matches["loser_name_clean"] = matches["loser_name"].apply(normalize_name)

# Function to compute player stats
def compute_player_stats(player, matches):
    # Matches where player was winner
    w = matches[matches["winner_name_clean"] == player].assign(
        svpt=lambda x: pd.to_numeric(x["w_svpt"], errors="coerce"),
        ace=lambda x: pd.to_numeric(x["w_ace"], errors="coerce"),
        df=lambda x: pd.to_numeric(x["w_df"], errors="coerce"),
        first_in=lambda x: pd.to_numeric(x["w_1stIn"], errors="coerce"),
        first_won=lambda x: pd.to_numeric(x["w_1stWon"], errors="coerce"),
        second_won=lambda x: pd.to_numeric(x["w_2ndWon"], errors="coerce"),
        bp_saved=lambda x: pd.to_numeric(x["w_bpSaved"], errors="coerce"),
        bp_faced=lambda x: pd.to_numeric(x["w_bpFaced"], errors="coerce"),
        rank=lambda x: pd.to_numeric(x["winner_rank"], errors="coerce"),
        age=lambda x: pd.to_numeric(x["winner_age"], errors="coerce"),
        ht=lambda x: pd.to_numeric(x["winner_ht"], errors="coerce"),
    )

    # Matches where player was loser
    l = matches[matches["loser_name_clean"] == player].assign(
        svpt=lambda x: pd.to_numeric(x["l_svpt"], errors="coerce"),
        ace=lambda x: pd.to_numeric(x["l_ace"], errors="coerce"),
        df=lambda x: pd.to_numeric(x["l_df"], errors="coerce"),
        first_in=lambda x: pd.to_numeric(x["l_1stIn"], errors="coerce"),
        first_won=lambda x: pd.to_numeric(x["l_1stWon"], errors="coerce"),
        second_won=lambda x: pd.to_numeric(x["l_2ndWon"], errors="coerce"),
        bp_saved=lambda x: pd.to_numeric(x["l_bpSaved"], errors="coerce"),
        bp_faced=lambda x: pd.to_numeric(x["l_bpFaced"], errors="coerce"),
        rank=lambda x: pd.to_numeric(x["loser_rank"], errors="coerce"),
        age=lambda x: pd.to_numeric(x["loser_age"], errors="coerce"),
        ht=lambda x: pd.to_numeric(x["loser_ht"], errors="coerce"),
    )

    stats = pd.concat([w, l])
    if stats.empty:
        return None

    total_svpt = stats["svpt"].sum()
    total_first_in = stats["first_in"].sum()

    ace_rate = stats["ace"].sum() / total_svpt if total_svpt > 0 else np.nan
    df_rate = stats["df"].sum() / total_svpt if total_svpt > 0 else np.nan
    first_in_pct = total_first_in / total_svpt if total_svpt > 0 else np.nan
    first_win_pct = stats["first_won"].sum() / total_first_in if total_first_in > 0 else np.nan
    second_win_pct = stats["second_won"].sum() / (total_svpt - total_first_in) if (total_svpt - total_first_in) > 0 else np.nan
    bp_saved_pct = stats["bp_saved"].sum() / stats["bp_faced"].sum() if stats["bp_faced"].sum() > 0 else np.nan

    return {
        "rank": stats["rank"].median(),
        "age": stats["age"].median(),
        "ht": stats["ht"].median(),
        "ace_rate": ace_rate,
        "df_rate": df_rate,
        "first_in_pct": first_in_pct,
        "first_win_pct": first_win_pct,
        "second_win_pct": second_win_pct,
        "bp_saved_pct": bp_saved_pct,
    }

# Loop over all players
features_list = []
for _, row in draw.iterrows():
    player = row["name_clean"]
    if "QUALIFIER" in player or "LUCKY LOSER" in player:
        # Leave empty for qualifiers
        feat = {"rank": np.nan, "age": np.nan, "ht": np.nan,
                "ace_rate": np.nan, "df_rate": np.nan,
                "first_in_pct": np.nan, "first_win_pct": np.nan,
                "second_win_pct": np.nan, "bp_saved_pct": np.nan}
    else:
        res = compute_player_stats(player, matches)
        feat = res if res is not None else {
            "rank": np.nan, "age": np.nan, "ht": np.nan,
            "ace_rate": np.nan, "df_rate": np.nan,
            "first_in_pct": np.nan, "first_win_pct": np.nan,
            "second_win_pct": np.nan, "bp_saved_pct": np.nan
        }
    feat["name"] = row["name"]
    features_list.append(feat)

# Save output
features_df = pd.DataFrame(features_list)
features_df.to_excel(output_file, index=False)

print(f"✅ Player features saved to {output_file}")
