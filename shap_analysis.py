import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")

# This file runs SHAP analysis on the trained models from predict_second_try.py
# Run AFTER predict_second_try.py has been run successfully.
# It imports the training functions directly so you don't retrain from scratch.

# Import everything from  predict file
from predict_second_try import (
    train_group_models,
    prepare_xy,
    GROUP_FEATURES,
    TRAIN_DATA_PATH,
    assign_position_group,
    normalize_aav,
    compute_inflation_index,
)

# 
# STEP 1: Retrain models (same as predict_second_try.py)


print("Training models for SHAP analysis...")
models, feat_cols, inflation_index, mean_2026 = train_group_models(TRAIN_DATA_PATH)

# Reload training data to get X matrices for SHAP
df = pd.read_csv(TRAIN_DATA_PATH)

# Force numeric columns
for col in ["AAV", "Guarantee", "Age", "Years", "FA_Year",
            "stat_GS", "stat_ERA", "stat_PA", "stat_G", "stat_IP"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[\$,\s]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )

stat_cols = [c for c in df.columns if c.startswith("stat_")]
for col in stat_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(r"[\$,\s]", "", regex=True),
        errors="coerce"
    )

df = df[df["AAV"].notna() & (df["AAV"] > 0)].copy()
df["group"] = df.apply(assign_position_group, axis=1)
df = df[df["group"] != "UNKNOWN"].copy()
df = normalize_aav(df, inflation_index)



# STEP 2: SHAP Summary Plot (feature importance across all players in group)
# Shows which features matter most overall for each position group


def plot_shap_summary(model, X, group, save_path):
    """
    Beeswarm plot — each dot is one player.
    Color = feature value (red = high, blue = low)
    X axis = impact on prediction (right = pushed salary up)
    """
    print(f"  Computing SHAP values for {group}...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Clean up feature names for display
    clean_names = [f.replace("stat_stat_", "").replace("stat_", "")
                   for f in X.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=clean_names,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Feature Impact — {group}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")



# STEP 3: SHAP Bar Plot (mean absolute impact — cleaner for paper figures)

def plot_shap_bar(model, X, group, save_path, top_n=12):
    """
    Bar chart of mean |SHAP value| per feature.
    Cleaner than beeswarm for paper figures.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    indices       = np.argsort(mean_abs_shap)[::-1][:top_n]

    clean_names = [f.replace("stat_stat_", "").replace("stat_", "")
                   for f in X.columns]
    labels = [clean_names[i] for i in indices]
    values = mean_abs_shap[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP Value| (impact on log-normalized AAV)")
    ax.set_title(f"Top Features by SHAP Impact — {group}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


# STEP 4: SHAP Waterfall for individual players
# Shows exactly WHY the model predicted what it did for a specific player
# pick 2-3 interesting cases (overpaid, underpaid, accurate)

def plot_shap_waterfall(model, X, player_name, group, row_index, save_path):
    """
    Waterfall plot for a single player.
    Shows each feature's contribution to the final prediction.
    row_index: integer position of the player in X (use X.index to find it)
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    clean_names = [f.replace("stat_stat_", "").replace("stat_", "")
                   for f in X.columns]

    # Rename features for display
    explanation.feature_names = clean_names

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation[row_index], show=False, max_display=12)
    plt.title(f"SHAP Waterfall — {player_name} ({group})", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# STEP 5: Run everything

print("\n=== Running SHAP Analysis ===\n")

for group, model in models.items():
    print(f"\n--- {group} ---")
    sub          = df[df["group"] == group]
    features     = feat_cols[group]
    X, y, sub_matched = prepare_xy(sub, features, target_col="AAV_norm")

    if len(X) < 5:
        print(f"  Not enough rows for {group}, skipping")
        continue

    # Summary plot (beeswarm)
    plot_shap_summary(model, X, group,
                      save_path=f"shap_summary_{group}.png")

    # Bar plot 
    plot_shap_bar(model, X, group,
                  save_path=f"shap_bar_{group}.png")

print("\n=== Individual Player Waterfall Examples ===")
print("(Edit the PLAYERS_TO_EXPLAIN list below to pick your own examples)\n")

#  Waterfall plots for specific players 
# Edit this list to pick the players we want to explain in your paper.
# Format: ("Player Name as in CSV", "GROUP")
# Good choices: one accurate prediction, one overpaid, one underpaid

PLAYERS_TO_EXPLAIN = [
    ("Merrill Kelly",   "SP"),   # very accurate prediction
    ("Gleyber Torres",  "MI"),   # underpredicted (QO suppressed market)
    ("Max Scherzer",    "SP"),   # injury discount — interesting failure case
    ("J.T. Realmuto",   "C"),    # decent prediction for a catcher
    ("Devin Williams",  "RP"),   # overpredicted closer
]

for player_name, group in PLAYERS_TO_EXPLAIN:
    if group not in models:
        print(f"  {player_name}: group {group} not in models, skipping")
        continue

    sub      = df[df["group"] == group]
    features = feat_cols[group]
    X, y, sub_matched = prepare_xy(sub, features, target_col="AAV_norm")

    # Find player row — try CleanName or Name column
    name_col = "CleanName" if "CleanName" in sub_matched.columns else "Name"
    if name_col not in sub_matched.columns:
        print(f"  {player_name}: name column not found, skipping")
        continue

    matches = sub_matched[
        sub_matched[name_col].str.lower().str.contains(
            player_name.split()[-1].lower(), na=False
        )
    ]

    if matches.empty:
        print(f"  {player_name}: not found in training data, skipping")
        continue

    # Get position in X (integer index)
    player_idx = X.index.get_loc(matches.index[0])

    plot_shap_waterfall(
        models[group], X, player_name, group, player_idx,
        save_path=f"shap_waterfall_{player_name.replace(' ', '_')}.png"
    )

print("\nAll SHAP plots saved. Done.")