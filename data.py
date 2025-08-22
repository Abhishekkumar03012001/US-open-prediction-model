import pandas as pd
import numpy as np

n_players = 128
np.random.seed(42)

data = {
    "player": [f"Player_{i+1}" for i in range(n_players)],
    "rank": np.random.randint(1, 200, size=n_players),
    "first_in": np.random.uniform(40, 70, size=n_players),
    "age": np.random.randint(18, 36, size=n_players),
    "ace_rate": np.random.uniform(0, 20, size=n_players),
    "first_win": np.random.uniform(50, 80, size=n_players),
    "df_rate": np.random.uniform(0, 10, size=n_players),
    "second_win": np.random.uniform(30, 60, size=n_players),
    "bp_saved": np.random.uniform(40, 80, size=n_players),
    "ht": np.random.randint(165, 210, size=n_players)
}

df = pd.DataFrame(data)
df.to_csv("usopen_2025_draw_stats.csv", index=False)
print("Sample dataset saved!")
