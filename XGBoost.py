import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files, no display window
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings



DATA_PATH   = "mlb_fa_data_cleaned - mlb fa_training_data_v1.csv.csv"
TEST_PATH   = "mlb_fa_2026.csv"
TARGET      = "AAV"
RANDOM_SEED = 42

# FEATURE SETS  — matched to actual B-Ref column names in the CSV


# --- Hitter feature groups ---
HITTER_TRADITIONAL = [
    "stat_G", "stat_PA", "stat_HR", "stat_R", "stat_RBI", "stat_SB",
    "stat_BA", "stat_OBP", "stat_SLG", "stat_OPS",
    "stat_BB", "stat_SO", "stat_2B", "stat_3B", "stat_TB",
]
HITTER_ADVANCED = [
    "stat_WAR", "stat_OPS+", "stat_rOBA", "stat_Rbat+",
    "stat_stat_ISO",    # SLG - BA  (engineered in bbref.py)
    "stat_stat_BB_K",   # BB / SO   (engineered in bbref.py)
]
HITTER_CONTEXT = ["Age", "Years"]

HITTER_ALL = HITTER_TRADITIONAL + HITTER_ADVANCED + HITTER_CONTEXT

# --- Pitcher feature groups ---
PITCHER_TRADITIONAL = [
    "stat_G", "stat_GS", "stat_IP", "stat_W", "stat_L", "stat_SV",
    "stat_ERA", "stat_WHIP", "stat_H9", "stat_HR9", "stat_BB9", "stat_SO9",
    "stat_BB", "stat_SO", "stat_HR",
]
PITCHER_ADVANCED = [
    "stat_WAR", "stat_FIP", "stat_ERA+",
    "stat_SO/BB",           # native B-Ref column
    "stat_stat_SO_W",       # SO / BB  (engineered in bbref.py — same ratio, kept both)
]
PITCHER_CONTEXT = ["Age", "Years"]

PITCHER_ALL = PITCHER_TRADITIONAL + PITCHER_ADVANCED + PITCHER_CONTEXT


# POSITION MAPPING  (used by load_2026)

POSITION_MAP = {
    "sp": "pitcher", "rp": "pitcher", "p": "pitcher",
    "starting pitcher": "pitcher", "relief pitcher": "pitcher",
    "c": "hitter", "1b": "hitter", "2b": "hitter", "3b": "hitter",
    "ss": "hitter", "lf": "hitter", "cf": "hitter", "rf": "hitter",
    "of": "hitter", "dh": "hitter", "if": "hitter", "util": "hitter",
}

def assign_position_group(row):
    pos = str(row.get("Position", "")).lower().strip()
    if pos in POSITION_MAP:
        return POSITION_MAP[pos]
    if pd.notna(row.get("stat_ERA")):
        return "pitcher"
    if pd.notna(row.get("stat_PA")):
        return "hitter"
    return "unknown"


def load_2026(path):
    df = pd.read_csv(path)
    stat_cols = [c for c in df.columns if c.startswith("stat_")]
    df["Position"] = df["Position"].astype(str).str.lower().str.strip()\
        .map(POSITION_MAP).fillna(df["Position"])
    for col in stat_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[\$,\s]", "", regex=True),
            errors="coerce"
        )
    for col in ["PlayerName", "Player"]:
        if col in df.columns and "Name" not in df.columns:
            df.rename(columns={col: "Name"}, inplace=True)
    df["group"] = df.apply(assign_position_group, axis=1)
    signed   = df[df["AAV"].notna() & (df["AAV"] > 0)].copy()
    unsigned = df[df["AAV"].isna()  | (df["AAV"] == 0)].copy()
    print(f"\n2026 FA class: {len(df)} total")
    print(f"  Signed:   {len(signed)}")
    print(f"  Unsigned: {len(unsigned)}")
    print("\n  Group breakdown:")
    for grp, cnt in df["group"].value_counts().sort_index().items():
        print(f"    {grp:<8} {cnt}")
    # Align with pipeline: map group -> pos_type and log-transform AAV
    signed["pos_type"] = signed["group"]
    signed = signed[signed["pos_type"].isin(["pitcher", "hitter"])].copy()
    signed["log_AAV"] = np.log1p(signed["AAV"])
    return df, signed, unsigned


# HELPERS


def load_and_prepare(path):
    df = pd.read_csv(path)
    stat_cols = [c for c in df.columns if c.startswith("stat_")]
    for col in stat_cols + ["Age", "Years"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[\$,\s]", "", regex=True),
                errors="coerce"
            )
    df[TARGET] = pd.to_numeric(
        df[TARGET].astype(str).str.replace(r"[\$,\s]", "", regex=True),
        errors="coerce"
    )
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()

    # Classify using stat presence — Position column has mixed/unreliable encoding
    df["pos_type"] = "unknown"
    df.loc[df["stat_ERA"].notna(), "pos_type"] = "pitcher"
    df.loc[df["stat_PA"].notna(),  "pos_type"] = "hitter"
    # Drop rows where no stats were found (can't predict without features)
    df = df[df["pos_type"] != "unknown"].copy()

    # Log-transform target — AAV is right-skewed
    df["log_AAV"] = np.log1p(df[TARGET])

    print(f"Loaded {len(df)} rows  |  "
          f"{(df['pos_type']=='pitcher').sum()} pitchers, "
          f"{(df['pos_type']=='hitter').sum()} hitters")
    print(f"FA years: {sorted(df['FA_Year'].unique())}")
    return df


def get_features(df, feature_list):
    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing columns (skipped): {missing}")
    return available


def prepare_xy(df, feature_list):
    cols = get_features(df, feature_list)
    X = df[cols].copy()
    y = df["log_AAV"].copy()
    mask = X.notna().any(axis=1)
    return X[mask], y[mask], df[mask]


def evaluate_cv(model, X, y, label=""):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_rmse = np.sqrt(-cross_val_score(model, X, y,
                                       scoring="neg_mean_squared_error", cv=kf))
    cv_r2   = cross_val_score(model, X, y, scoring="r2", cv=kf)
    rmse_dollars = np.mean(np.expm1(y.values) * np.abs(np.exp(cv_rmse.mean()) - 1))
    print(f"  {label}")
    print(f"    CV RMSE (log):  {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"    CV RMSE ($M):   ${rmse_dollars/1e6:.2f}M")
    print(f"    CV R²:          {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    return cv_rmse.mean(), cv_r2.mean()


def evaluate_test(model, X_train, y_train, X_test, y_test, label=""):
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, preds))
    r2     = r2_score(y_test, preds)
    rmse_dollars = np.expm1(rmse)
    mae_dollars  = np.mean(np.abs(np.expm1(y_test.values) - np.expm1(preds)))
    print(f"  {label} [HELD-OUT TEST]")
    print(f"    RMSE (log):  {rmse:.4f}")
    print(f"    RMSE ($M):   ${rmse_dollars/1e6:.2f}M")
    print(f"    MAE ($M):    ${mae_dollars/1e6:.2f}M")
    print(f"    R²:          {r2:.4f}")
    return preds


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


# PLOTS

def plot_feature_importance(model, feature_names, title, top_n=15, save_path=None):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    labels      = [feature_names[i].replace("stat_stat_", "").replace("stat_", "")
                   for i in indices]
    values      = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_ablation(results, title, save_path=None):
    labels = list(results.keys())
    r2s    = [results[l]["r2"]   for l in labels]
    rmses  = [results[l]["rmse"] for l in labels]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, r2s, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Test R²")
    ax1.set_title(f"{title} — R²")
    ax1.set_ylim(0, max(r2s) * 1.2)

    ax2.bar(x, [np.expm1(r)/1e6 for r in rmses], color="salmon")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Test RMSE ($M)")
    ax2.set_title(f"{title} — RMSE")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_predicted_vs_actual(y_true, y_pred, title, save_path=None):
    actual = np.expm1(y_true) / 1e6
    pred   = np.expm1(y_pred) / 1e6
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    coeffs = np.polyfit(actual, pred, 2)
    x_line = np.linspace(actual.min(), actual.max(), 200)
    ax.plot(x_line, np.polyval(coeffs, x_line), "b-", linewidth=1.5, label="Trendline (quadratic)")
    ax.set_xlabel("Actual AAV ($M)")
    ax.set_ylabel("Predicted AAV ($M)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


# EXPERIMENT 1 — Ablation


def run_ablation(train_df, test_df, pos_type, feature_groups):
    print(f"\n{'='*55}")
    print(f"EXPERIMENT 1: ABLATION — {pos_type.upper()}S")
    print(f"{'='*55}")

    sub      = train_df[train_df["pos_type"] == pos_type]
    test_sub = test_df[test_df["pos_type"]   == pos_type]
    results  = {}

    # Fit every feature group and collect metrics first
    for label, features in feature_groups.items():
        X_train, y_train, _ = prepare_xy(sub,      features)
        X_test,  y_test,  _ = prepare_xy(test_sub, features)
        if len(X_train) < 20:
            print(f"  {label}: not enough training rows, skipping")
            continue
        if len(X_test) < 5:
            print(f"  {label}: not enough test rows, skipping")
            continue
        model = make_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[label] = {
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2":   r2_score(y_test, preds),
        }

    # Print: full model first, then deltas for each ablation variant
    full_label = "Full Data Set"
    if full_label not in results:
        # Fallback: treat whichever has the best R² as the baseline
        full_label = max(results, key=lambda k: results[k]["r2"])

    full_r2      = results[full_label]["r2"]
    full_rmse    = results[full_label]["rmse"]
    full_rmse_m  = np.expm1(full_rmse)

    print(f"\n  Full model ({full_label}):")
    print(f"    R²:         {full_r2:.4f}")
    print(f"    RMSE (log): {full_rmse:.4f}  |  RMSE ($M): ${full_rmse_m/1e6:.2f}M")

    print(f"\n  Ablation deltas (vs full model):")
    for label, res in results.items():
        if label == full_label:
            continue
        delta_r2     = res["r2"]   - full_r2
        delta_rmse   = res["rmse"] - full_rmse
        delta_rmse_m = np.expm1(res["rmse"]) - full_rmse_m
        r2_sign   = "+" if delta_r2   >= 0 else ""
        rmse_sign = "+" if delta_rmse >= 0 else ""
        print(f"  {label:<22}  ΔR²: {r2_sign}{delta_r2:.4f}   "
              f"ΔRMSE (log): {rmse_sign}{delta_rmse:.4f}   "
              f"ΔRMSE ($M): {rmse_sign}{delta_rmse_m/1e6:.2f}M")

    return results



# EXPERIMENT 2 — Historical window


def run_historical_window(df, test_df, pos_type, feature_list):
    print(f"\n{'='*55}")
    print(f"EXPERIMENT 2: HISTORICAL WINDOW — {pos_type.upper()}S")
    print(f"{'='*55}")

    sub         = df[df["pos_type"] == pos_type]
    test_sub    = test_df[test_df["pos_type"] == pos_type]
    train_years = sorted(sub["FA_Year"].unique())

    X_test, y_test, _ = prepare_xy(test_sub, feature_list)

    windows = {
        "Last 3 yrs": train_years[-3:],
        "Last 5 yrs": train_years[-5:],
        "Last 7 yrs": train_years[-7:],
        "All years":  train_years,
    }

    results = {}
    for label, years in windows.items():
        train_sub = sub[sub["FA_Year"].isin(years)]
        X_train, y_train, _ = prepare_xy(train_sub, feature_list)
        if len(X_train) < 20 or len(X_test) < 5:
            print(f"  {label}: not enough data, skipping")
            continue
        model = make_model()
        preds = evaluate_test(model, X_train, y_train, X_test, y_test, label=label)
        results[label] = {
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2":   r2_score(y_test, preds),
            "n_train": len(X_train),
        }

    return results



# EXPERIMENT 3 — Unified vs split model


def run_unified_vs_split(train_df, test_df):
    print(f"\n{'='*55}")
    print("EXPERIMENT 3: UNIFIED vs SPLIT MODEL")
    print(f"{'='*55}")

    unified_features = list(set(HITTER_ALL + PITCHER_ALL))

    X_train_u, y_train_u, _ = prepare_xy(train_df, unified_features)
    X_test_u,  y_test_u,  _ = prepare_xy(test_df,  unified_features)

    p_train = train_df[train_df["pos_type"] == "pitcher"]
    h_train = train_df[train_df["pos_type"] == "hitter"]
    p_test  = test_df[test_df["pos_type"]  == "pitcher"]
    h_test  = test_df[test_df["pos_type"]  == "hitter"]

    X_pt,  y_pt,  _ = prepare_xy(p_train, PITCHER_ALL)
    X_ht,  y_ht,  _ = prepare_xy(h_train, HITTER_ALL)
    X_pte, y_pte, _ = prepare_xy(p_test,  PITCHER_ALL)
    X_hte, y_hte, _ = prepare_xy(h_test,  HITTER_ALL)

    # ── Cross-validation (all three) ──────────────────────────────────────
    print("\n--- Cross-Validation ---")
    evaluate_cv(make_model(), X_train_u, y_train_u, label="Unified CV")
    evaluate_cv(make_model(), X_pt,      y_pt,      label="Pitchers CV")
    evaluate_cv(make_model(), X_ht,      y_ht,      label="Hitters CV")

    # ── Test on 2026 data (all three) ─────────────────────────────────────
    print("\n--- Test on 2026 Data ---")
    evaluate_test(make_model(), X_train_u, y_train_u, X_test_u, y_test_u, label="Unified")
    p_model = make_model()
    h_model = make_model()
    preds_p = evaluate_test(p_model, X_pt, y_pt, X_pte, y_pte, label="Pitchers")
    preds_h = evaluate_test(h_model, X_ht, y_ht, X_hte, y_hte, label="Hitters")

    return p_model, h_model, X_pt, X_ht



# MAIN


def main():
    df       = load_and_prepare(DATA_PATH)
    _, test_df, _ = load_2026(TEST_PATH)
    train_df = df.copy()
    print(f"\nTrain rows: {len(train_df)} | Test rows: {len(test_df)}")

    # Experiment 1: Ablation
    hitter_ablation_groups = {
        "Full Data Set":        HITTER_ALL,
        "No Advanced Stats":    HITTER_TRADITIONAL + HITTER_CONTEXT,
        "No Traditional Stats": HITTER_ADVANCED    + HITTER_CONTEXT,
        "No Context Variables": HITTER_TRADITIONAL + HITTER_ADVANCED,
    }
    pitcher_ablation_groups = {
        "Full Data Set":        PITCHER_ALL,
        "No Advanced Stats":    PITCHER_TRADITIONAL + PITCHER_CONTEXT,
        "No Traditional Stats": PITCHER_ADVANCED    + PITCHER_CONTEXT,
        "No Context Variables": PITCHER_TRADITIONAL + PITCHER_ADVANCED,
    }

    h_ablation = run_ablation(train_df, test_df, "hitter",  hitter_ablation_groups)
    p_ablation = run_ablation(train_df, test_df, "pitcher", pitcher_ablation_groups)

    plot_ablation(h_ablation, "Hitter Ablation",  save_path="ablation_hitters.png")
    plot_ablation(p_ablation, "Pitcher Ablation", save_path="ablation_pitchers.png")

    # Experiment 2: Historical window
    run_historical_window(df, test_df, "hitter",  HITTER_ALL)
    run_historical_window(df, test_df, "pitcher", PITCHER_ALL)

    # Experiment 3: Unified vs split
    p_model, h_model, X_pt, X_ht = run_unified_vs_split(train_df, test_df)

    # Feature importance plots
    print("\nGenerating feature importance plots...")
    plot_feature_importance(p_model, list(X_pt.columns),
                            "Top Pitcher Features",
                            save_path="importance_pitchers.png")
    plot_feature_importance(h_model, list(X_ht.columns),
                            "Top Hitter Features",
                            save_path="importance_hitters.png")

    # Predicted vs actual (hitters, test set)
    X_hte, y_hte, _ = prepare_xy(
        test_df[test_df["pos_type"] == "hitter"], HITTER_ALL)
    preds = h_model.predict(X_hte)
    plot_predicted_vs_actual(y_hte, preds,
                             "Predicted vs Actual AAV — Hitters (Test Set)",
                             save_path="pred_vs_actual_hitters.png")

    X_pte, y_pte, _ = prepare_xy(
        test_df[test_df["pos_type"] == "pitcher"], PITCHER_ALL)
    preds_p = p_model.predict(X_pte)
    plot_predicted_vs_actual(y_pte, preds_p,
                             "Predicted vs Actual AAV — Pitchers (Test Set)",
                             save_path="pred_vs_actual_pitchers.png")

    print("\nDone. All plots saved.")


if __name__ == "__main__":
    main()
