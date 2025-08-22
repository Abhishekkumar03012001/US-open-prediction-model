import pandas as pd
import joblib
import numpy as np

# Paths
model_path = r"D:/US open/clean data/rf_tennis_model_compressed.pkl"
draw_path  = r"D:/US open/clean data/USOpen2025_PlayerFeatures_MERGED.xlsx"  # your draw excel

# Load trained model
pipe = joblib.load(model_path)

# Helper: build features for a match
def build_match_features(p1, p2):
    rec = {
        "surface": p1.get("surface", np.nan),
        "rank_diff": p2["rank"] - p1["rank"],
        "age_diff": p1["age"] - p2["age"],
        "ht_diff": p1["ht"] - p2["ht"],
        "ace_rate_diff": p1["ace_rate"] - p2["ace_rate"],
        "df_rate_diff": p1["df_rate"] - p2["df_rate"],
        "first_in_diff": p1["first_in_pct"] - p2["first_in_pct"],
        "first_win_diff": p1["first_win_pct"] - p2["first_win_pct"],
        "second_win_diff": p1["second_win_pct"] - p2["second_win_pct"],
        "bp_saved_diff": p1["bp_saved_pct"] - p2["bp_saved_pct"],
    }
    return pd.DataFrame([rec])

# Predict match winner
def predict_winner(p1, p2):
    X = build_match_features(p1, p2)
    prob = pipe.predict_proba(X)[0,1]
    return p1 if prob >= 0.5 else p2

# Run knockout tournament
def simulate_tournament(players):
    round_num = 1
    while len(players) > 8:  # stop at quarterfinals
        winners = []
        for i in range(0, len(players), 2):
            p1, p2 = players[i], players[i+1]
            winner = predict_winner(p1, p2)
            winners.append(winner)
        players = winners
        round_num += 1
    return players

def main():
    # Load 128-player draw
    df = pd.read_excel(draw_path)

    # Keep only the *first instance* of duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Select only necessary columns
    keep_cols = [
        "name", "surface", "rank", "age", "ht",
        "ace_rate", "df_rate", "first_in_pct", "first_win_pct",
        "second_win_pct", "bp_saved_pct"
    ]
    df = df[keep_cols]

    # Fill any missing values with median of the column
    df = df.apply(lambda col: col.fillna(col.median()) if col.dtype != "object" else col)

    # Convert to list of dicts
    players = df.to_dict(orient="records")

    # Simulate tournament
    top8 = simulate_tournament(players)

    # Pair them for quarterfinals
    matchups = [(top8[i]["name"], top8[i+1]["name"]) for i in range(0, 8, 2)]

    print("Quarterfinal Matchups:")
    for m in matchups:
        print(f"{m[0]} vs {m[1]}")

if __name__ == "__main__":
    main()
