import pandas as pd
import random
import joblib

# Load the trained pipeline
pipe = joblib.load("rf_tennis_model_compressed.pkl")

# Load draw list (player names + stats)
draw = pd.read_csv("usopen_2025_draw_stats.csv")  # must include engineered stats

def match_win_prob(p1, p2):
    # Assume `p1` and `p2` are pandas Series of engineered features
    # Compute diff features just like training
    diff = p1 - p2
    diff['surface'] = 'Hard'
    return pipe.predict_proba(pd.DataFrame([diff]))[0, 1]

def simulate(draw_list, n_sims=5000):
    top8_counts = {p: 0 for p in draw_list['player']}
    for _ in range(n_sims):
        bracket = draw_list.copy()
        # Round of 128 → 64 → 32 → 16 → 8 (stop here)
        for round_size in [128, 64, 32, 16]:
            winners = []
            for i in range(0, round_size, 2):
                p1 = bracket.iloc[i]
                p2 = bracket.iloc[i+1]
                prob = match_win_prob(p1.drop('player'), p2.drop('player'))
                winners.append(p1 if random.random() < prob else p2)
            bracket = pd.DataFrame(winners)
        for p in bracket['player']:
            top8_counts[p] += 1
    return sorted([(p, cnt/n_sims) for p, cnt in top8_counts.items()], key=lambda x: x[1], reverse=True)

draw_list = pd.read_csv("usopen_2025_draw_stats.csv")
top8 = simulate(draw_list)
print("Predicted Top-8 Probabilities:")
for player, prob in top8[:8]:
    print(f"{player}: {prob:.1%}")
