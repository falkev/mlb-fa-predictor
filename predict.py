import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATA_PATH = "mlb_fa_data_cleaned - mlb fa_training_data_v1.csv.csv"
FA_2026_PATH    = "mlb_fa_merged.csv"   # your 2026 FA CSV from the pipeline
OUTPUT_PATH     = "2026_predictions.csv"
RANDOM_SEED     = 42

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS  (identical to xgboost_model.py)
# ─────────────────────────────────────────────────────────────────────────────

HITTER_TRADITIONAL = [
    "stat_G", "stat_PA", "stat_HR", "stat_R", "stat_RBI", "stat_SB",
    "stat_BA", "stat_OBP", "stat_SLG", "stat_OPS",
    "stat_BB", "stat_SO", "stat_2B", "stat_3B", "stat_TB",
]
HITTER_ADVANCED = [
    "stat_WAR", "stat_OPS+", "stat_rOBA", "stat_Rbat+",
    "stat_stat_ISO", "stat_stat_BB_K",
]
HITTER_CONTEXT = ["Age", "Years"]
HITTER_ALL = HITTER_TRADITIONAL + HITTER_ADVANCED + HITTER_CONTEXT

PITCHER_TRADITIONAL = [
    "stat_G", "stat_GS", "stat_IP", "stat_W", "stat_L", "stat_SV",
    "stat_ERA", "stat_WHIP", "stat_H9", "stat_HR9", "stat_BB9", "stat_SO9",
    "stat_BB", "stat_SO", "stat_HR",
]
PITCHER_ADVANCED = [
    "stat_WAR", "stat_FIP", "stat_ERA+",
    "stat_SO/BB", "stat_stat_SO_W",
]
PITCHER_CONTEXT = ["Age", "Years"]
PITCHER_ALL = PITCHER_TRADITIONAL + PITCHER_ADVANCED + PITCHER_CONTEXT


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def classify_pos(df):
    df["pos_type"] = "unknown"
    # Find whichever ERA column exists
    era_col = next((c for c in ["stat_ERA", "stat_era", "ERA"] if c in df.columns), None)
    pa_col  = next((c for c in ["stat_PA",  "stat_pa",  "PA"]  if c in df.columns), None)
    if era_col:
        df.loc[df[era_col].notna(), "pos_type"] = "pitcher"
    if pa_col:
        df.loc[df[pa_col].notna(),  "pos_type"] = "hitter"
    # Fallback: use position string if stats didn't classify the player
    if "Position" in df.columns:
        mask = df["pos_type"] == "unknown"
        pitcher_pos = df["Position"].astype(str).str.lower().str.contains(
            "rhp|lhp|sp|rp|cp|-s|-c", na=False)
        df.loc[mask & pitcher_pos,  "pos_type"] = "pitcher"
        df.loc[mask & ~pitcher_pos, "pos_type"] = "hitter"

    print(f"  Classified: {(df['pos_type']=='pitcher').sum()} pitchers, "
          f"{(df['pos_type']=='hitter').sum()} hitters, "
          f"{(df['pos_type']=='unknown').sum()} unknown")
    return df


def get_features(df, feature_list):
    return [f for f in feature_list if f in df.columns]


def prepare_xy(df, feature_list):
    cols = get_features(df, feature_list)
    X = df[cols].copy()
    y = np.log1p(df["AAV"])
    mask = X.notna().any(axis=1) & y.notna()
    return X[mask], y[mask], df[mask]


def prepare_X_only(df, feature_list):
    """For prediction rows that may not have AAV yet."""
    cols = get_features(df, feature_list)
    X = df[cols].copy()
    return X


def make_model():
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Train on ALL historical data (2015-2025)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_models(train_path):
    df = pd.read_csv(train_path)
    df = df[df["AAV"].notna() & (df["AAV"] > 0)].copy()
    df = classify_pos(df)
    df = df[df["pos_type"] != "unknown"].copy()

    pitchers = df[df["pos_type"] == "pitcher"]
    hitters  = df[df["pos_type"] == "hitter"]

    print(f"Training on full dataset: {len(df)} rows "
          f"({len(pitchers)} pitchers, {len(hitters)} hitters)")

    X_p, y_p, _ = prepare_xy(pitchers, PITCHER_ALL)
    X_h, y_h, _ = prepare_xy(hitters,  HITTER_ALL)

    pitcher_model = make_model()
    hitter_model  = make_model()

    pitcher_model.fit(X_p, y_p)
    hitter_model.fit(X_h, y_h)

    print(f"  Pitcher model trained on {len(X_p)} rows, "
          f"{len(X_p.columns)} features")
    print(f"  Hitter model trained on  {len(X_h)} rows, "
          f"{len(X_h.columns)} features")

    return pitcher_model, hitter_model, list(X_p.columns), list(X_h.columns)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load and classify 2026 FA data
# ─────────────────────────────────────────────────────────────────────────────

def load_2026(path):
    df = pd.read_csv(path)

    # Standardize name column
    for col in ["PlayerName", "Name", "Player"]:
        if col in df.columns:
            df.rename(columns={col: "Name"}, inplace=True)
            break

    # Standardize position column
    for col in ["Position", "Pos'n", "Pos"]:
        if col in df.columns:
            df.rename(columns={col: "Position"}, inplace=True)
            break

    # Standardize age column
    if "Age" not in df.columns:
        for col in ["age", "AGE"]:
            if col in df.columns:
                df.rename(columns={col: "Age"}, inplace=True)
                break

    df = classify_pos(df)

    signed   = df[df["AAV"].notna() & (df["AAV"] > 0)]
    unsigned = df[df["AAV"].isna()  | (df["AAV"] == 0)]

    print(f"\n2026 FA class: {len(df)} players loaded")
    print(f"  Signed (have AAV):   {len(signed)}")
    print(f"  Unsigned (no AAV):   {len(unsigned)}")
    print(f"  Pitchers: {(df['pos_type']=='pitcher').sum()} | "
          f"Hitters: {(df['pos_type']=='hitter').sum()} | "
          f"Unknown: {(df['pos_type']=='unknown').sum()}")

    return df, signed, unsigned


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate predictions
# ─────────────────────────────────────────────────────────────────────────────

def predict_players(df, pitcher_model, hitter_model,
                    pitcher_features, hitter_features):
    results = []

    for _, row in df.iterrows():
        pos_type = row.get("pos_type", "unknown")
        if pos_type == "unknown":
            continue

        if pos_type == "pitcher":
            features = pitcher_features
            model    = pitcher_model
        else:
            features = hitter_features
            model    = hitter_model

        # Build feature row — use NaN for missing stats
        x = pd.DataFrame([{f: row.get(f, np.nan) for f in features}])
        pred_log = model.predict(x)[0]
        pred_aav = np.expm1(pred_log)

        result = {
            "Name":          row.get("Name", "Unknown"),
            "Position":      row.get("Position", ""),
            "Age":           row.get("Age", np.nan),
            "pos_type":      pos_type,
            "Actual_AAV":    row.get("AAV", np.nan),
            "Predicted_AAV": round(pred_aav, 0),
        }

        # If player has signed, compute difference
        actual = row.get("AAV", np.nan)
        if pd.notna(actual) and actual > 0:
            result["Difference"]   = round(pred_aav - actual, 0)
            result["Pct_Diff"]     = round((pred_aav - actual) / actual * 100, 1)
            result["Over_Under"]   = "UNDERPAID" if pred_aav > actual else "OVERPAID"

        results.append(result)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_signed_predictions(signed_preds, save_path="2026_pred_vs_actual.png"):
    """Predicted vs actual for signed players."""
    df = signed_preds[signed_preds["Actual_AAV"].notna()].copy()
    actual = df["Actual_AAV"] / 1e6
    pred   = df["Predicted_AAV"] / 1e6

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["salmon" if x == "OVERPAID" else "steelblue"
              for x in df["Over_Under"]]
    ax.scatter(actual, pred, c=colors, alpha=0.7, edgecolors="k", linewidths=0.3, s=60)

    lims = [0, max(actual.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual AAV ($M)", fontsize=12)
    ax.set_ylabel("Predicted AAV ($M)", fontsize=12)
    ax.set_title("2026 FA Class — Predicted vs Actual AAV", fontsize=13)

    # Label the most interesting outliers (biggest absolute difference)
    df["abs_diff"] = (df["Predicted_AAV"] - df["Actual_AAV"]).abs()
    top_outliers = df.nlargest(8, "abs_diff")
    for _, r in top_outliers.iterrows():
        ax.annotate(r["Name"],
                    xy=(r["Actual_AAV"]/1e6, r["Predicted_AAV"]/1e6),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Underpaid (model > actual)"),
        Patch(facecolor="salmon",    label="Overpaid  (model < actual)"),
        plt.Line2D([0], [0], color="k", linestyle="--", label="Perfect prediction"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_top_unsigned(unsigned_preds, n=20, save_path="2026_unsigned_predictions.png"):
    """Bar chart of predicted AAV for unsigned players."""
    df = unsigned_preds.nlargest(n, "Predicted_AAV").copy()
    df["label"] = df["Name"] + " (" + df["pos_type"].str[0].str.upper() + ")"
    df = df.sort_values("Predicted_AAV")

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["steelblue" if p == "pitcher" else "forestgreen"
              for p in df["pos_type"]]
    ax.barh(df["label"], df["Predicted_AAV"] / 1e6, color=colors)
    ax.set_xlabel("Predicted AAV ($M)")
    ax.set_title(f"Top {n} Unsigned 2026 FAs — Predicted AAV")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue",   label="Pitcher"),
        Patch(facecolor="forestgreen", label="Hitter"),
    ])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== 2026 MLB FA Salary Predictor ===\n")

    # Step 1: Train on all historical data
    print("[Step 1] Training final models on 2015-2025 data...")
    pitcher_model, hitter_model, pitcher_features, hitter_features = \
        train_final_models(TRAIN_DATA_PATH)

    # Step 2: Load 2026 FA data
    print("\n[Step 2] Loading 2026 FA class...")
    df_2026, signed, unsigned = load_2026(FA_2026_PATH)

    # Step 3: Predict signed players (compare vs actual)
    print("\n[Step 3] Predicting signed players...")
    signed_preds = predict_players(
        signed, pitcher_model, hitter_model,
        pitcher_features, hitter_features
    )
    signed_preds = signed_preds.sort_values("Difference", key=abs, ascending=False)

    print("\n--- TOP 10 MOST MISPRICED (signed players) ---")
    display_cols = ["Name", "Position", "Age", "Actual_AAV",
                    "Predicted_AAV", "Difference", "Pct_Diff", "Over_Under"]
    display_cols = [c for c in display_cols if c in signed_preds.columns]
    top10 = signed_preds.head(10).copy()
    top10["Actual_AAV"]    = top10["Actual_AAV"].apply(lambda x: f"${x/1e6:.1f}M")
    top10["Predicted_AAV"] = top10["Predicted_AAV"].apply(lambda x: f"${x/1e6:.1f}M")
    top10["Difference"]    = top10["Difference"].apply(lambda x: f"${x/1e6:+.1f}M")
    print(top10[display_cols].to_string(index=False))

    # Test set metrics on signed players
    has_actual = signed_preds["Actual_AAV"].notna()
    if has_actual.sum() > 5:
        y_true = np.log1p(signed_preds.loc[has_actual, "Actual_AAV"])
        y_pred = np.log1p(signed_preds.loc[has_actual, "Predicted_AAV"])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        rmse_dollars = np.mean(
            signed_preds.loc[has_actual, "Actual_AAV"] * np.abs(np.exp(rmse) - 1)
        )
        print(f"\n2026 Signed Players — Final Model Performance:")
        print(f"  RMSE (log):  {rmse:.4f}")
        print(f"  RMSE ($M):   ${rmse_dollars/1e6:.2f}M")
        print(f"  R²:          {r2:.4f}")

    # Step 4: Predict unsigned players
    print("\n[Step 4] Predicting unsigned players...")
    unsigned_preds = predict_players(
        unsigned, pitcher_model, hitter_model,
        pitcher_features, hitter_features
    )
    unsigned_preds = unsigned_preds.sort_values("Predicted_AAV", ascending=False)

    if len(unsigned_preds) > 0:
        print("\n--- TOP 10 UNSIGNED PLAYERS BY PREDICTED AAV ---")
        top_unsigned = unsigned_preds.head(10).copy()
        top_unsigned["Predicted_AAV"] = top_unsigned["Predicted_AAV"].apply(
            lambda x: f"${x/1e6:.1f}M")
        print(top_unsigned[["Name", "Position", "Age",
                             "Predicted_AAV"]].to_string(index=False))

    # Step 5: Save everything to CSV
    all_preds = pd.concat([signed_preds, unsigned_preds], ignore_index=True)
    all_preds.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll predictions saved → {OUTPUT_PATH}")

    # Step 6: Plots
    print("\n[Step 5] Generating plots...")
    if len(signed_preds) > 5:
        plot_signed_predictions(signed_preds)
    if len(unsigned_preds) > 0:
        plot_top_unsigned(unsigned_preds)

    print("\nDone.")


if __name__ == "__main__":
    main()