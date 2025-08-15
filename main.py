#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Random Forest match-outcome model from Jeff Sackmann-style ATP/WTA match files.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
import joblib

# ======================
# USER CONFIGURATION
# ======================
# Change these to where your files are stored
input_path = r"D:/US open/clean data/atp_matches_2024.csv"
output_path = r"D:/US open/clean data/rf_tennis_model_compressed.pkl"
# ======================

def safe_div(n: pd.Series, d: pd.Series) -> np.ndarray:
    n = pd.to_numeric(n, errors="coerce")
    d = pd.to_numeric(d, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((d == 0) | (~np.isfinite(d)), np.nan, n / d)
    return out

def build_pair_rows(row: pd.Series) -> List[dict]:
    a = {
        "surface": row.get("surface", np.nan),
        "rank_A": row.get("winner_rank", np.nan),
        "rank_B": row.get("loser_rank", np.nan),
        "age_A": row.get("winner_age", np.nan),
        "age_B": row.get("loser_age", np.nan),
        "ht_A": row.get("winner_ht", np.nan),
        "ht_B": row.get("loser_ht", np.nan),
        "ace_rate_A": row.get("w_ace_rate", np.nan),
        "ace_rate_B": row.get("l_ace_rate", np.nan),
        "df_rate_A": row.get("w_df_rate", np.nan),
        "df_rate_B": row.get("l_df_rate", np.nan),
        "first_in_A": row.get("w_first_in_pct", np.nan),
        "first_in_B": row.get("l_first_in_pct", np.nan),
        "first_win_A": row.get("w_first_win_pct", np.nan),
        "first_win_B": row.get("l_first_win_pct", np.nan),
        "second_win_A": row.get("w_second_win_pct", np.nan),
        "second_win_B": row.get("l_second_win_pct", np.nan),
        "bp_saved_A": row.get("w_bp_saved_pct", np.nan),
        "bp_saved_B": row.get("l_bp_saved_pct", np.nan),
        "target": 1
    }
    b = {
        "surface": row.get("surface", np.nan),
        "rank_A": row.get("loser_rank", np.nan),
        "rank_B": row.get("winner_rank", np.nan),
        "age_A": row.get("loser_age", np.nan),
        "age_B": row.get("winner_age", np.nan),
        "ht_A": row.get("loser_ht", np.nan),
        "ht_B": row.get("winner_ht", np.nan),
        "ace_rate_A": row.get("l_ace_rate", np.nan),
        "ace_rate_B": row.get("w_ace_rate", np.nan),
        "df_rate_A": row.get("l_df_rate", np.nan),
        "df_rate_B": row.get("w_df_rate", np.nan),
        "first_in_A": row.get("l_first_in_pct", np.nan),
        "first_in_B": row.get("w_first_in_pct", np.nan),
        "first_win_A": row.get("l_first_win_pct", np.nan),
        "first_win_B": row.get("w_first_win_pct", np.nan),
        "second_win_A": row.get("l_second_win_pct", np.nan),
        "second_win_B": row.get("w_second_win_pct", np.nan),
        "bp_saved_A": row.get("l_bp_saved_pct", np.nan),
        "bp_saved_B": row.get("w_bp_saved_pct", np.nan),
        "target": 0
    }
    return [a, b]

def engineer_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w_svpt = pd.to_numeric(df.get("w_svpt"), errors="coerce")
    l_svpt = pd.to_numeric(df.get("l_svpt"), errors="coerce")
    w_1stIn = pd.to_numeric(df.get("w_1stIn"), errors="coerce")
    l_1stIn = pd.to_numeric(df.get("l_1stIn"), errors="coerce")

    df["w_first_in_pct"]   = safe_div(w_1stIn, w_svpt)
    df["l_first_in_pct"]   = safe_div(l_1stIn, l_svpt)
    df["w_first_win_pct"]  = safe_div(df.get("w_1stWon"), w_1stIn)
    df["l_first_win_pct"]  = safe_div(df.get("l_1stWon"), l_1stIn)
    df["w_second_win_pct"] = safe_div(df.get("w_2ndWon"), (w_svpt - w_1stIn))
    df["l_second_win_pct"] = safe_div(df.get("l_2ndWon"), (l_svpt - l_1stIn))
    df["w_bp_saved_pct"]   = safe_div(df.get("w_bpSaved"), df.get("w_bpFaced"))
    df["l_bp_saved_pct"]   = safe_div(df.get("l_bpSaved"), df.get("l_bpFaced"))
    df["w_ace_rate"]       = safe_div(df.get("w_ace"), w_svpt)
    df["l_ace_rate"]       = safe_div(df.get("l_ace"), l_svpt)
    df["w_df_rate"]        = safe_div(df.get("w_df"), w_svpt)
    df["l_df_rate"]        = safe_div(df.get("l_df"), l_svpt)

    return df

def build_pair_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_rates(df)
    recs = []
    for _, r in df.iterrows():
        recs.extend(build_pair_rows(r))
    pairs = pd.DataFrame(recs)
    pairs["rank_diff"]       = pd.to_numeric(pairs["rank_B"], errors="coerce") - pd.to_numeric(pairs["rank_A"], errors="coerce")
    pairs["age_diff"]        = pd.to_numeric(pairs["age_A"], errors="coerce") - pd.to_numeric(pairs["age_B"], errors="coerce")
    pairs["ht_diff"]         = pd.to_numeric(pairs["ht_A"], errors="coerce") - pd.to_numeric(pairs["ht_B"], errors="coerce")
    pairs["ace_rate_diff"]   = pairs["ace_rate_A"] - pairs["ace_rate_B"]
    pairs["df_rate_diff"]    = pairs["df_rate_A"] - pairs["df_rate_B"]
    pairs["first_in_diff"]   = pairs["first_in_A"] - pairs["first_in_B"]
    pairs["first_win_diff"]  = pairs["first_win_A"] - pairs["first_win_B"]
    pairs["second_win_diff"] = pairs["second_win_A"] - pairs["second_win_B"]
    pairs["bp_saved_diff"]   = pairs["bp_saved_A"] - pairs["bp_saved_B"]
    return pairs

def train_model(pairs: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    feature_cols = [
        "surface",
        "rank_diff","age_diff","ht_diff",
        "ace_rate_diff","df_rate_diff","first_in_diff","first_win_diff","second_win_diff","bp_saved_diff"
    ]
    target_col = "target"
    missing = [c for c in feature_cols + [target_col] if c not in pairs.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    X = pairs[feature_cols].copy()
    y = pairs[target_col].astype(int)

    numeric_cols = [c for c in feature_cols if c != "surface"]
    categorical_cols = ["surface"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("rf", rf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("Metrics:", {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba))
    })
    print("\nClassification report:\n", classification_report(y_test, preds, digits=4))

    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    num_feature_names = numeric_cols
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = num_feature_names + cat_feature_names
    feat_imp = pd.DataFrame({
        "feature": all_feature_names,
        "importance": pipe.named_steps["rf"].feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop feature importances:\n", feat_imp.head(20).to_string(index=False))
    return pipe, feat_imp

def save_model(pipe: Pipeline, out_path: str):
    joblib.dump(pipe, out_path, compress=3)
    print(f"\nSaved compressed model to: {out_path}")

def main():
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    needed_cols = [
        "surface","winner_rank","loser_rank","winner_age","loser_age","winner_ht","loser_ht",
        "w_svpt","l_svpt","w_1stIn","l_1stIn","w_1stWon","l_1stWon","w_2ndWon","l_2ndWon",
        "w_bpSaved","w_bpFaced","l_bpSaved","l_bpFaced","w_ace","l_ace","w_df","l_df"
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    pairs = build_pair_dataframe(df)
    pipe, feat_imp = train_model(pairs)
    save_model(pipe, output_path)
    feat_imp.to_csv(os.path.splitext(output_path)[0] + "_feature_importances.csv", index=False)

if __name__ == "__main__":
    main()
