import pandas as pd
import random
import joblib

# Load the trained pipeline
pipe = joblib.load("rf_tennis_model_compressed.pkl")

# Load draw list (player names + stats)
draw = pd.read_csv("usopen_2025_draw_stats.csv")  # must include raw stats

def match_win_prob(p1, p2):
    # Build diff features with correct names (must match training pipeline)
    diff = {
        "rank_diff": p1["rank"] - p2["rank"],
        "first_in_diff": p1["first_in"] - p2["first_in"],
        "age_diff": p1["age"] - p2["age"],
        "ace_rate_diff": p1["ace_rate"] - p2["ace_rate"],
        "first_win_diff": p1["first_win"] - p2["first_win"],
        "df_rate_diff": p1["df_rate"] - p2["df_rate"],
        "second_win_diff": p1["second_win"] - p2["second_win"],
        "bp_saved_diff": p1["bp_saved"] - p2["bp_saved"],
        "ht_diff": p1["ht"] - p2["ht"],
        "surface": "Hard"
    }
    X = pd.DataFrame([diff])
    return pipe.predict_proba(X)[0, 1]

def simulate(draw_list, n_sims=5000):
    top8_counts = {p: 0 for p in draw_list['player']}
    for _ in range(n_sims):
        bracket = draw_list.copy()

        # Round of 128 → 64 → 32 → 16 → 8
        for round_size in [128, 64, 32, 16]:
            winners = []
            for i in range(0, round_size, 2):
                p1 = bracket.iloc[i]
                p2 = bracket.iloc[i+1]
                prob = match_win_prob(p1.drop('player'), p2.drop('player'))
                winners.append(p1 if random.random() < prob else p2)
            bracket = pd.DataFrame(winners)

        # Count quarterfinalists
        for p in bracket['player']:
            top8_counts[p] += 1

    return sorted(
        [(p, cnt/n_sims) for p, cnt in top8_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )

if __name__ == "__main__":
    draw_list = pd.read_csv("usopen_2025_draw_stats.csv")
    top8 = simulate(draw_list, n_sims=10)  # you can adjust sims
    print("Predicted Top-8 Probabilities:")
    for player, prob in top8[:8]:
        print(f"{player}: {prob:.1%}")
