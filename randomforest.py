import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match xgboost_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "mlb_fa_data_cleaned - mlb fa_training_data_v1.csv.csv"
TEST_PATH = "mlb_fa_2026.csv"
TARGET = "AAV"
RANDOM_SEED = 42

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
# POSITION MAPPING  (used by load_2026)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    df["pos_type"] = "unknown"
    df.loc[df["stat_ERA"].notna(), "pos_type"] = "pitcher"
    df.loc[df["stat_PA"].notna(), "pos_type"] = "hitter"
    df = df[df["pos_type"] != "unknown"].copy()
    df["log_AAV"] = np.log1p(df[TARGET])
    print(f"Loaded {len(df)} rows  |  "
          f"{(df['pos_type'] == 'pitcher').sum()} pitchers, "
          f"{(df['pos_type'] == 'hitter').sum()} hitters")
    return df


def get_features(df, feature_list):
    return [f for f in feature_list if f in df.columns]


def prepare_xy(df, feature_list):
    cols = get_features(df, feature_list)
    X = df[cols].copy()
    # Random Forest can't handle NaN — fill with column median
    X = X.fillna(X.median())
    y = df["log_AAV"].copy()
    mask = y.notna()
    return X[mask], y[mask], df[mask]


def rmse_to_dollars(log_rmse, y_true):
    """Convert log-scale RMSE to average dollar error."""
    actual = np.expm1(y_true)
    dollar_errors = actual * np.abs(np.exp(log_rmse) - 1)
    return np.mean(dollar_errors)


def evaluate_cv(model, X, y, label=""):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_rmse = np.sqrt(-cross_val_score(model, X, y,
                                       scoring="neg_mean_squared_error", cv=kf))
    cv_r2 = cross_val_score(model, X, y, scoring="r2", cv=kf)
    rmse_dollars = rmse_to_dollars(cv_rmse.mean(), y)
    print(f"  {label}")
    print(f"    CV RMSE (log):  {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"    CV RMSE ($M):   ${rmse_dollars / 1e6:.2f}M")
    print(f"    CV R²:          {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    return cv_rmse.mean(), cv_r2.mean()


def evaluate_test(model, X_train, y_train, X_test, y_test, label=""):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    rmse_dollars = rmse_to_dollars(rmse, y_test)
    print(f"  {label} [HELD-OUT TEST]")
    print(f"    RMSE (log):  {rmse:.4f}")
    print(f"    RMSE ($M):   ${rmse_dollars / 1e6:.2f}M")
    print(f"    R²:          {r2:.4f}")
    return preds, model


def make_model():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,  # let trees grow fully — RF is robust to this
        min_samples_leaf=3,  # prevents overfitting on small groups
        max_features="sqrt",  # standard for regression RF
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, title, top_n=15, save_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    labels = [feature_names[i].replace("stat_stat_", "").replace("stat_", "")
              for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1], color="forestgreen")
    ax.set_xlabel("Feature Importance (impurity)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")


def plot_ablation(results, title, save_path=None):
    labels = list(results.keys())
    r2s = [results[l]["r2"] for l in labels]
    rmses = [results[l]["rmse"] for l in labels]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, r2s, color="forestgreen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("CV R²")
    ax1.set_title(f"{title} — R²")
    ax1.set_ylim(0, max(r2s) * 1.2)

    ax2.bar(x, rmses, color="lightcoral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("CV RMSE (log)")
    ax2.set_title(f"{title} — RMSE")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")


def plot_predicted_vs_actual(y_true, y_pred, title, save_path=None):
    actual = np.expm1(y_true) / 1e6
    pred = np.expm1(y_pred) / 1e6
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.6, edgecolors="k", linewidths=0.3, color="forestgreen")
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual AAV ($M)")
    ax.set_ylabel("Predicted AAV ($M)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(train_df, pos_type, feature_groups):
    print(f"\n{'=' * 55}")
    print(f"RF EXPERIMENT 1: ABLATION — {pos_type.upper()}S")
    print(f"{'=' * 55}")

    sub = train_df[train_df["pos_type"] == pos_type]
    results = {}

    for label, features in feature_groups.items():
        X, y, _ = prepare_xy(sub, features)
        if len(X) < 20:
            print(f"  {label}: not enough rows, skipping")
            continue
        rmse, r2 = evaluate_cv(make_model(), X, y, label=label)
        results[label] = {"rmse": rmse, "r2": r2}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — Historical window
# ─────────────────────────────────────────────────────────────────────────────

def run_historical_window(df, test_df, pos_type, feature_list):
    print(f"\n{'=' * 55}")
    print(f"RF EXPERIMENT 2: HISTORICAL WINDOW — {pos_type.upper()}S")
    print(f"{'=' * 55}")

    sub = df[df["pos_type"] == pos_type]
    test_sub = test_df[test_df["pos_type"] == pos_type]
    train_years = sorted(sub["FA_Year"].unique())
    X_test, y_test, _ = prepare_xy(test_sub, feature_list)

    windows = {
        "Last 3 yrs": train_years[-3:],
        "Last 5 yrs": train_years[-5:],
        "Last 7 yrs": train_years[-7:],
        "All years": train_years,
    }

    results = {}
    for label, years in windows.items():
        train_sub = sub[sub["FA_Year"].isin(years)]
        X_train, y_train, _ = prepare_xy(train_sub, feature_list)
        if len(X_train) < 20 or len(X_test) < 5:
            print(f"  {label}: not enough data, skipping")
            continue
        preds, _ = evaluate_test(make_model(), X_train, y_train,
                                 X_test, y_test, label=label)
        results[label] = {
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds),
            "n_train": len(X_train),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 — Unified vs split
# ─────────────────────────────────────────────────────────────────────────────

def run_unified_vs_split(train_df, test_df):
    print(f"\n{'=' * 55}")
    print("RF EXPERIMENT 3: UNIFIED vs SPLIT MODEL")
    print(f"{'=' * 55}")

    unified_features = list(set(HITTER_ALL + PITCHER_ALL))

    X_train_u, y_train_u, _ = prepare_xy(train_df, unified_features)
    X_test_u, y_test_u, _ = prepare_xy(test_df, unified_features)
    print("\nUnified model (all positions):")
    evaluate_cv(make_model(), X_train_u, y_train_u, label="Unified CV")
    evaluate_test(make_model(), X_train_u, y_train_u,
                  X_test_u, y_test_u, label="Unified")

    print("\nSplit models (pitchers and hitters separately):")
    p_train = train_df[train_df["pos_type"] == "pitcher"]
    h_train = train_df[train_df["pos_type"] == "hitter"]
    p_test = test_df[test_df["pos_type"] == "pitcher"]
    h_test = test_df[test_df["pos_type"] == "hitter"]

    X_pt, y_pt, _ = prepare_xy(p_train, PITCHER_ALL)
    X_ht, y_ht, _ = prepare_xy(h_train, HITTER_ALL)
    X_pte, y_pte, _ = prepare_xy(p_test, PITCHER_ALL)
    X_hte, y_hte, _ = prepare_xy(h_test, HITTER_ALL)

    evaluate_cv(make_model(), X_pt, y_pt, label="Pitchers CV")
    evaluate_cv(make_model(), X_ht, y_ht, label="Hitters CV")

    preds_p, p_model = evaluate_test(make_model(), X_pt, y_pt, X_pte, y_pte, label="Pitchers")
    preds_h, h_model = evaluate_test(make_model(), X_ht, y_ht, X_hte, y_hte, label="Hitters")

    return p_model, h_model, X_pt, X_ht, X_pte, y_pte, X_hte, y_hte


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_and_prepare(DATA_PATH)
    _, test_df, _ = load_2026(TEST_PATH)
    train_df = df.copy()
    print(f"\nTrain rows: {len(train_df)} | Test rows: {len(test_df)}")

    # ── Experiment 1: Ablation ─────────────────────────────────────────────
    hitter_ablation_groups = {
        "Traditional only": HITTER_TRADITIONAL + HITTER_CONTEXT,
        "Advanced only": HITTER_ADVANCED + HITTER_CONTEXT,
        "Trad + Advanced": HITTER_ALL,
        "No age/years": HITTER_TRADITIONAL + HITTER_ADVANCED,
    }
    pitcher_ablation_groups = {
        "Traditional only": PITCHER_TRADITIONAL + PITCHER_CONTEXT,
        "Advanced only": PITCHER_ADVANCED + PITCHER_CONTEXT,
        "Trad + Advanced": PITCHER_ALL,
        "No age/years": PITCHER_TRADITIONAL + PITCHER_ADVANCED,
    }

    h_ablation = run_ablation(train_df, "hitter", hitter_ablation_groups)
    p_ablation = run_ablation(train_df, "pitcher", pitcher_ablation_groups)

    plot_ablation(h_ablation, "RF Hitter Ablation", save_path="rf_ablation_hitters.png")
    plot_ablation(p_ablation, "RF Pitcher Ablation", save_path="rf_ablation_pitchers.png")

    # ── Experiment 2: Historical window ───────────────────────────────────
    run_historical_window(df, test_df, "hitter", HITTER_ALL)
    run_historical_window(df, test_df, "pitcher", PITCHER_ALL)

    # ── Experiment 3: Unified vs split ────────────────────────────────────
    p_model, h_model, X_pt, X_ht, X_pte, y_pte, X_hte, y_hte = \
        run_unified_vs_split(train_df, test_df)

    # ── Feature importance ────────────────────────────────────────────────
    print("\nGenerating feature importance plots...")
    plot_feature_importance(p_model, list(X_pt.columns),
                            "RF Top Pitcher Features",
                            save_path="rf_importance_pitchers.png")
    plot_feature_importance(h_model, list(X_ht.columns),
                            "RF Top Hitter Features",
                            save_path="rf_importance_hitters.png")

    # ── Predicted vs actual ───────────────────────────────────────────────
    preds_h = h_model.predict(X_hte)
    plot_predicted_vs_actual(y_hte, preds_h,
                             "RF Predicted vs Actual AAV — Hitters (Test Set)",
                             save_path="rf_pred_vs_actual_hitters.png")

    preds_p = p_model.predict(X_pte)
    plot_predicted_vs_actual(y_pte, preds_p,
                             "RF Predicted vs Actual AAV — Pitchers (Test Set)",
                             save_path="rf_pred_vs_actual_pitchers.png")

    print("\nDone. All RF plots saved.")


if __name__ == "__main__":
    main()
