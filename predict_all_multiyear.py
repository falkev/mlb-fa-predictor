import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# CONFIG

TRAIN_DATA_PATH = "mlb_fa_training_data_v1.csv"
FA_2026_PATH    = "mlb_fa_2026.csv"
OUTPUT_PATH     = "2026_predictions_all_multiyear.csv"
RANDOM_SEED     = 42

# GS threshold: if a pitcher had more than this many starts, treat as SP
SP_GS_THRESHOLD = 10

# POSITION GROUPS
# Groups: SP, RP, C, MI (middle infield), CI (corner infield/DH), OF

POSITION_MAP = {
    "rhp-s": "17", "lhp-s": "20",           # SP
    "rhp-c": "16", "lhp-c": "19",           # closer
    "rhp":   "18", "lhp":   "15",           # RP
    "c":     "1",  "c-dh":  "1",            # catcher
    "1b":    "3",  "1b-dh": "3",  "1b-of": "3",  "1b-3b": "3",
    "dh":    "14", "dh-rf": "14", "dh-of": "14", "dh-lf": "14",
    "3b":    "5",  "3b-2b": "5",
    "2b":    "4",  "2b-of": "4",  "2b-3b": "4",
    "ss":    "6",  "ss-cf": "6",
    "lf":    "7",  "lf-dh": "7",  "lf-rf": "7",
    "cf":    "8",
    "rf":    "9",
    "of":    "9",
}
# Numeric codes from the B-Ref export
SP_CODES  = {"17", "20"}
RP_CODES  = {"15", "16", "18", "19"}
C_CODES   = {"1"}
CI_CODES  = {"3", "14"}
MI_CODES  = {"6", "11", "5", "4"}
OF_CODES  = {"7", "8", "9", "10"}


def assign_position_group(row):
    pos = str(row.get("Position", "")).strip().lower()
    gs  = pd.to_numeric(row.get("stat_GS", np.nan), errors="coerce")
    era = pd.to_numeric(row.get("stat_ERA", np.nan), errors="coerce")
    pa  = pd.to_numeric(row.get("stat_PA",  np.nan), errors="coerce")

    is_pitcher = (
        pd.notna(era) or
        any(x in pos for x in ["rhp", "lhp", "sp", "rp", "cp"]) or
        pos in SP_CODES | RP_CODES
    )

    if is_pitcher:
        if pd.notna(gs):
            return "SP" if gs > SP_GS_THRESHOLD else "RP"
        if any(x in pos for x in ["rhp-s", "lhp-s", "-s"]) or pos in SP_CODES:
            return "SP"
        return "RP"

    if pos in ["c", "c-dh"] or pos in C_CODES or pos.startswith("c-"):
        return "C"
    if any(x in pos for x in ["ss", "2b"]) or pos in MI_CODES:
        return "MI"
    if any(x in pos for x in ["1b", "3b", "dh"]) or pos in CI_CODES:
        return "CI"
    if any(x in pos for x in ["lf", "cf", "rf", "of"]) or pos in OF_CODES:
        return "OF"

    if pd.notna(pa):
        return "CI"

    return "UNKNOWN"


# FEATURE SETS PER GROUP
# All single-year stats + every available 2yr/3yr/trend column from the CSV

COMMON_CONTEXT = ["Age", "Years"]

# ---------- Pitchers ----------
# Single-year pitcher stats
_SP_BASE = [
    "stat_GS", "stat_IP", "stat_W", "stat_L", "stat_W-L%",
    "stat_ERA", "stat_FIP", "stat_WHIP", "stat_ERA+",
    "stat_H9", "stat_HR9", "stat_BB9", "stat_SO9", "stat_SO/BB",
    "stat_BB", "stat_SO", "stat_BF", "stat_WP", "stat_BK",
    "stat_ER", "stat_CG", "stat_SHO", "stat_WAR", "stat_stat_SO_W",
]
# All pitcher 2yr/3yr/trend columns available in the CSV
_SP_MULTI = [
    "stat_WAR_2yr", "stat_WAR_3yr", "stat_WAR_trend",
    "stat_ERA_2yr", "stat_ERA_3yr", "stat_ERA_trend",
    "stat_FIP_2yr", "stat_FIP_3yr",
    "stat_WHIP_2yr", "stat_WHIP_3yr",
    "stat_ERA+_2yr", "stat_ERA+_3yr",
    "stat_W_2yr", "stat_W_3yr",
    "stat_L_2yr", "stat_L_3yr",
    "stat_W-L%_2yr", "stat_W-L%_3yr",
    "stat_IP_2yr", "stat_IP_3yr",
    "stat_GS_2yr", "stat_GS_3yr",
    "stat_BB_2yr", "stat_BB_3yr",
    "stat_SO_2yr", "stat_SO_3yr",
    "stat_H9_2yr", "stat_H9_3yr",
    "stat_HR9_2yr", "stat_HR9_3yr",
    "stat_BB9_2yr", "stat_BB9_3yr",
    "stat_SO9_2yr", "stat_SO9_3yr",
    "stat_SO/BB_2yr", "stat_SO/BB_3yr",
    "stat_stat_SO_W_2yr", "stat_stat_SO_W_3yr",
    "stat_BF_2yr", "stat_BF_3yr",
    "stat_ER_2yr", "stat_ER_3yr",
    "stat_WP_2yr", "stat_WP_3yr",
    "stat_BK_2yr", "stat_BK_3yr",
]

SP_FEATURES = _SP_BASE + _SP_MULTI + COMMON_CONTEXT

_RP_BASE = [
    "stat_G", "stat_GF", "stat_SV", "stat_IP", "stat_W", "stat_L",
    "stat_ERA", "stat_FIP", "stat_WHIP", "stat_ERA+",
    "stat_H9", "stat_HR9", "stat_BB9", "stat_SO9", "stat_SO/BB",
    "stat_BB", "stat_SO", "stat_BF", "stat_WP", "stat_BK",
    "stat_ER", "stat_WAR", "stat_stat_SO_W",
]
_RP_MULTI = [
    "stat_WAR_2yr", "stat_WAR_3yr", "stat_WAR_trend",
    "stat_ERA_2yr", "stat_ERA_3yr", "stat_ERA_trend",
    "stat_FIP_2yr", "stat_FIP_3yr",
    "stat_WHIP_2yr", "stat_WHIP_3yr",
    "stat_ERA+_2yr", "stat_ERA+_3yr",
    "stat_W_2yr", "stat_W_3yr",
    "stat_L_2yr", "stat_L_3yr",
    "stat_W-L%_2yr", "stat_W-L%_3yr",
    "stat_IP_2yr", "stat_IP_3yr",
    "stat_GF_2yr", "stat_GF_3yr",
    "stat_SV_2yr", "stat_SV_3yr",
    "stat_G_2yr", "stat_G_3yr",
    "stat_BB_2yr", "stat_BB_3yr",
    "stat_SO_2yr", "stat_SO_3yr",
    "stat_H9_2yr", "stat_H9_3yr",
    "stat_HR9_2yr", "stat_HR9_3yr",
    "stat_BB9_2yr", "stat_BB9_3yr",
    "stat_SO9_2yr", "stat_SO9_3yr",
    "stat_SO/BB_2yr", "stat_SO/BB_3yr",
    "stat_stat_SO_W_2yr", "stat_stat_SO_W_3yr",
    "stat_BF_2yr", "stat_BF_3yr",
    "stat_ER_2yr", "stat_ER_3yr",
    "stat_WP_2yr", "stat_WP_3yr",
    "stat_BK_2yr", "stat_BK_3yr",
]

RP_FEATURES = _RP_BASE + _RP_MULTI + COMMON_CONTEXT

# ---------- Hitters ----------
# All hitter 2yr/3yr/trend columns available in the CSV
_HIT_MULTI = [
    "stat_WAR_2yr", "stat_WAR_3yr", "stat_WAR_trend",
    "stat_OPS_2yr", "stat_OPS_3yr", "stat_OPS_trend",
    "stat_OPS+_2yr", "stat_OPS+_3yr",
    "stat_rOBA_2yr", "stat_rOBA_3yr",
    "stat_Rbat+_2yr", "stat_Rbat+_3yr",
    "stat_BA_2yr", "stat_BA_3yr",
    "stat_OBP_2yr", "stat_OBP_3yr",
    "stat_SLG_2yr", "stat_SLG_3yr",
    "stat_HR_2yr", "stat_HR_3yr",
    "stat_R_2yr", "stat_R_3yr",
    "stat_RBI_2yr", "stat_RBI_3yr",
    "stat_H_2yr", "stat_H_3yr",
    "stat_2B_2yr", "stat_2B_3yr",
    "stat_3B_2yr", "stat_3B_3yr",
    "stat_BB_2yr", "stat_BB_3yr",
    "stat_SO_2yr", "stat_SO_3yr",
    "stat_SB_2yr", "stat_SB_3yr",
    "stat_CS_2yr", "stat_CS_3yr",
    "stat_TB_2yr", "stat_TB_3yr",
    "stat_PA_2yr", "stat_PA_3yr",
    "stat_AB_2yr", "stat_AB_3yr",
    "stat_G_2yr", "stat_G_3yr",
    "stat_GIDP_2yr", "stat_GIDP_3yr",
    "stat_HBP_2yr", "stat_HBP_3yr",
    "stat_SF_2yr", "stat_SF_3yr",
    "stat_SH_2yr", "stat_SH_3yr",
    "stat_IBB_2yr", "stat_IBB_3yr",
    "stat_stat_ISO_2yr", "stat_stat_ISO_3yr",
    "stat_stat_BB_K_2yr", "stat_stat_BB_K_3yr",
]

C_FEATURES = [
    "stat_G", "stat_PA", "stat_AB", "stat_R", "stat_H",
    "stat_2B", "stat_3B", "stat_HR", "stat_RBI", "stat_SB", "stat_CS",
    "stat_BB", "stat_SO", "stat_BA", "stat_OBP", "stat_SLG", "stat_OPS",
    "stat_OPS+", "stat_TB", "stat_GIDP", "stat_HBP", "stat_SF", "stat_IBB",
    "stat_WAR", "stat_rOBA", "stat_Rbat+", "stat_stat_ISO", "stat_stat_BB_K",
] + _HIT_MULTI + COMMON_CONTEXT

MI_FEATURES = [
    "stat_G", "stat_PA", "stat_AB", "stat_R", "stat_H",
    "stat_2B", "stat_3B", "stat_HR", "stat_RBI", "stat_SB", "stat_CS",
    "stat_BB", "stat_SO", "stat_BA", "stat_OBP", "stat_SLG", "stat_OPS",
    "stat_OPS+", "stat_TB", "stat_GIDP", "stat_HBP", "stat_SF", "stat_IBB",
    "stat_WAR", "stat_rOBA", "stat_Rbat+", "stat_stat_ISO", "stat_stat_BB_K",
] + _HIT_MULTI + COMMON_CONTEXT

CI_FEATURES = [
    "stat_G", "stat_PA", "stat_AB", "stat_R", "stat_H",
    "stat_2B", "stat_3B", "stat_HR", "stat_RBI", "stat_SB", "stat_CS",
    "stat_BB", "stat_SO", "stat_BA", "stat_OBP", "stat_SLG", "stat_OPS",
    "stat_OPS+", "stat_TB", "stat_GIDP", "stat_HBP", "stat_SF", "stat_IBB",
    "stat_WAR", "stat_rOBA", "stat_Rbat+", "stat_stat_ISO", "stat_stat_BB_K",
] + _HIT_MULTI + COMMON_CONTEXT

OF_FEATURES = [
    "stat_G", "stat_PA", "stat_AB", "stat_R", "stat_H",
    "stat_2B", "stat_3B", "stat_HR", "stat_RBI", "stat_SB", "stat_CS",
    "stat_BB", "stat_SO", "stat_BA", "stat_OBP", "stat_SLG", "stat_OPS",
    "stat_OPS+", "stat_TB", "stat_GIDP", "stat_HBP", "stat_SF", "stat_IBB",
    "stat_WAR", "stat_rOBA", "stat_Rbat+", "stat_stat_ISO", "stat_stat_BB_K",
] + _HIT_MULTI + COMMON_CONTEXT

GROUP_FEATURES = {
    "SP": SP_FEATURES,
    "RP": RP_FEATURES,
    "C":  C_FEATURES,
    "MI": MI_FEATURES,
    "CI": CI_FEATURES,
    "OF": OF_FEATURES,
}


# SALARY INFLATION

def compute_inflation_index(df):
    """
    Returns a dict {year: mean_AAV} and a scalar for 2026 (extrapolated).
    Normalizes each player's AAV by their year's mean so the model learns
    performance → relative value rather than performance → raw dollars.
    """
    yearly_mean = df.groupby("FA_Year")["AAV"].mean()
    print("\nMarket AAV by year (inflation index):")
    for yr, val in yearly_mean.items():
        print(f"  {yr}: ${val/1e6:.2f}M")

    recent = yearly_mean[yearly_mean.index >= 2022]
    years  = np.array(recent.index, dtype=float)
    vals   = recent.values
    slope, intercept = np.polyfit(years, vals, 1)
    mean_2026 = slope * 2026 + intercept
    print(f"  2026 (extrapolated): ${mean_2026/1e6:.2f}M")

    return yearly_mean.to_dict(), mean_2026


def normalize_aav(df, inflation_index):
    """Divide each player's AAV by their year's market mean."""
    df = df.copy()
    df["AAV_norm"] = df.apply(
        lambda r: r["AAV"] / inflation_index.get(r["FA_Year"], r["AAV"])
        if pd.notna(r["AAV"]) else np.nan,
        axis=1
    )
    return df


# HELPERS

def get_features(df, feature_list):
    return [f for f in feature_list if f in df.columns]


def prepare_xy(df, feature_list, target_col="AAV_norm"):
    cols = get_features(df, feature_list)
    X = df[cols].copy()
    y = np.log1p(df[target_col])
    mask = X.notna().any(axis=1) & y.notna()
    return X[mask], y[mask], df[mask]


def make_model():
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# STEP 1: Train one model per position group

def train_group_models(train_path):
    df = pd.read_csv(train_path)
    for col in ["AAV", "Guarantee", "Age", "Years", "FA_Year",
                "stat_GS", "stat_ERA", "stat_PA", "stat_G", "stat_IP"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[\$,\s]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )

    # Convert all stat_ columns to numeric
    stat_cols = [c for c in df.columns if c.startswith("stat_")]
    for col in stat_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[\$,\s]", "", regex=True),
            errors="coerce"
        )

    df = df[df["AAV"].notna() & (df["AAV"] > 0)].copy()

    df["group"] = df.apply(assign_position_group, axis=1)
    df = df[df["group"] != "UNKNOWN"].copy()

    inflation_index, mean_2026 = compute_inflation_index(df)
    df = normalize_aav(df, inflation_index)

    print(f"\nTraining dataset: {len(df)} rows")
    print(f"{'Group':<6} {'Count':>6}")
    print("-" * 14)
    for grp, cnt in df["group"].value_counts().sort_index().items():
        print(f"{grp:<6} {cnt:>6}")

    models    = {}
    feat_cols = {}

    for group, features in GROUP_FEATURES.items():
        sub = df[df["group"] == group]
        if len(sub) < 15:
            print(f"\n  {group}: only {len(sub)} rows — skipping")
            continue

        X, y, _ = prepare_xy(sub, features, target_col="AAV_norm")
        if len(X) < 15:
            print(f"\n  {group}: insufficient stat coverage — skipping")
            continue

        model = make_model()
        weights = np.where(sub.loc[X.index, "FA_Year"] >= 2022, 2.5, 1.0)
        model.fit(X, y, sample_weight=weights)
        models[group]    = model
        feat_cols[group] = list(X.columns)
        print(f"\n  {group}: trained on {len(X)} rows, "
              f"{len(X.columns)} features")

    return models, feat_cols, inflation_index, mean_2026


# STEP 2: Load 2026 FA data

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

    return df, signed, unsigned


# STEP 3: Predict

def predict_players(df, models, feat_cols, mean_2026):
    results = []

    for _, row in df.iterrows():
        group = row.get("group", "UNKNOWN")
        if group not in models:
            continue

        model    = models[group]
        features = feat_cols[group]

        x = pd.DataFrame([{f: row.get(f, np.nan) for f in features}])
        pred_norm = np.expm1(model.predict(x)[0])

        # Apply MLB minimum salary floor
        pred_aav = max(pred_norm * mean_2026, 600000)

        result = {
            "Name":          row.get("Name", row.get("PlayerName", "Unknown")),
            "Position":      row.get("Position", ""),
            "Group":         group,
            "Age":           row.get("Age", np.nan),
            "Actual_AAV":    row.get("AAV", np.nan),
            "Predicted_AAV": round(pred_aav, 0),
        }

        actual = row.get("AAV", np.nan)
        if pd.notna(actual) and actual > 0:
            diff = pred_aav - actual
            result["Difference"]  = round(diff, 0)
            result["Pct_Diff"]    = round(diff / actual * 100, 1)
            result["Over_Under"]  = "UNDERPAID" if diff > 0 else "OVERPAID"

        results.append(result)

    return pd.DataFrame(results)


# STEP 4: Evaluation metrics on signed players

def evaluate_predictions(signed_preds):
    df = signed_preds[signed_preds["Actual_AAV"].notna()].copy()
    if len(df) < 5:
        print("Not enough signed players to evaluate.")
        return

    y_true = np.log1p(df["Actual_AAV"])
    y_pred = np.log1p(df["Predicted_AAV"])
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    dollar_errors = (df["Predicted_AAV"] - df["Actual_AAV"]).abs()

    print(f"\n--- 2026 Signed Players: Model Performance ---")
    print(f"  N:            {len(df)}")
    print(f"  RMSE (log):   {rmse:.4f}")
    print(f"  Mean |error|: ${dollar_errors.mean()/1e6:.2f}M")
    print(f"  Median |err|: ${dollar_errors.median()/1e6:.2f}M")
    print(f"  R²:           {r2:.4f}")

    print(f"\n  Per-group R²:")
    for grp in sorted(df["Group"].unique()):
        sub = df[df["Group"] == grp]
        if len(sub) < 3:
            continue
        grp_r2 = r2_score(np.log1p(sub["Actual_AAV"]),
                          np.log1p(sub["Predicted_AAV"]))
        print(f"    {grp:<6} n={len(sub):>3}  R²={grp_r2:.3f}")


# STEP 5: Plots

GROUP_COLORS = {
    "SP": "steelblue", "RP": "cornflowerblue",
    "C":  "darkorange", "MI": "forestgreen",
    "CI": "mediumseagreen", "OF": "goldenrod",
}

def plot_pred_vs_actual(signed_preds, save_path="2026_pred_vs_actual_all_multiyear.png"):
    df = signed_preds[signed_preds["Actual_AAV"].notna()].copy()
    actual = df["Actual_AAV"] / 1e6
    pred   = df["Predicted_AAV"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 9))

    for grp in df["Group"].unique():
        mask = df["Group"] == grp
        ax.scatter(actual[mask], pred[mask],
                   color=GROUP_COLORS.get(grp, "gray"),
                   alpha=0.75, edgecolors="k", linewidths=0.3,
                   s=65, label=grp)

    lim = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="Perfect prediction")

    # Quadratic trendline (predicted ~ actual)
    coeffs = np.polyfit(actual, pred, 2)
    x_trend = np.linspace(0, lim, 300)
    y_trend = np.polyval(coeffs, x_trend)
    ax.plot(x_trend, y_trend, "r-", linewidth=1.5, label="Quadratic trendline")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual AAV ($M)", fontsize=12)
    ax.set_ylabel("Predicted AAV ($M)", fontsize=12)
    ax.set_title("2026 FA Class — Predicted vs Actual AAV\n(all multi-year stats)",
                 fontsize=13)
    ax.legend(fontsize=9)

    df["abs_diff"] = (df["Predicted_AAV"] - df["Actual_AAV"]).abs()
    for _, r in df.nlargest(10, "abs_diff").iterrows():
        ax.annotate(r["Name"],
                    xy=(r["Actual_AAV"]/1e6, r["Predicted_AAV"]/1e6),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_top_predictions(preds, title, save_path, n=20):
    df = preds.nlargest(n, "Predicted_AAV").copy()
    name_col = "Name" if "Name" in df.columns else df.columns[0]
    df["label"] = df[name_col] + "  (" + df["Group"] + ")"
    df = df.sort_values("Predicted_AAV")
    colors = [GROUP_COLORS.get(g, "gray") for g in df["Group"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df["label"], df["Predicted_AAV"] / 1e6, color=colors)
    ax.set_xlabel("Predicted AAV ($M)")
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g)
                       for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_feature_importance(models, feat_cols, top_n=12):
    for group, model in models.items():
        features = feat_cols[group]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        labels  = [features[i].replace("stat_stat_", "").replace("stat_", "")
                   for i in indices]
        values  = importances[indices]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(labels[::-1], values[::-1],
                color=GROUP_COLORS.get(group, "steelblue"))
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title(f"Feature Importance — {group}")
        plt.tight_layout()
        save_path = f"importance_2026_{group}_all_multiyear.png"
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")


# MAIN

def main():
    print("=== 2026 MLB FA Salary Predictor (Position-Group Models, All Multi-Year Stats) ===\n")

    # Step 1
    print("[Step 1] Training per-group models on 2015-2025 data...")
    models, feat_cols, inflation_index, mean_2026 = \
        train_group_models(TRAIN_DATA_PATH)

    # Step 2
    print(f"\n[Step 2] Loading 2026 FA class from {FA_2026_PATH}...")
    df_2026, signed, unsigned = load_2026(FA_2026_PATH)

    # Step 3a — signed players
    print("\n[Step 3] Predicting signed players...")
    signed_preds = predict_players(signed, models, feat_cols, mean_2026)
    if "Difference" in signed_preds.columns:
        signed_preds = signed_preds.sort_values("Difference",
                                                key=abs, ascending=False)

    print("\n--- TOP 15 MOST MISPRICED (signed players) ---")
    top = signed_preds.head(15).copy()
    for col in ["Actual_AAV", "Predicted_AAV"]:
        if col in top.columns:
            top[col] = top[col].apply(lambda x: f"${x/1e6:.1f}M"
                                      if pd.notna(x) else "N/A")
    if "Difference" in top.columns:
        top["Difference"] = top["Difference"].apply(
            lambda x: f"${x/1e6:+.1f}M" if pd.notna(x) else "N/A")
    disp = [c for c in ["Name", "Group", "Age", "Actual_AAV",
                         "Predicted_AAV", "Difference",
                         "Pct_Diff", "Over_Under"] if c in top.columns]
    print(top[disp].to_string(index=False))

    evaluate_predictions(signed_preds)

    # Step 3b — unsigned players
    print("\n[Step 4] Predicting unsigned players...")
    unsigned_preds = predict_players(unsigned, models, feat_cols, mean_2026)
    unsigned_preds = unsigned_preds.sort_values("Predicted_AAV", ascending=False)

    if len(unsigned_preds) > 0:
        print("\n--- TOP 15 UNSIGNED PLAYERS BY PREDICTED AAV ---")
        top_u = unsigned_preds.head(15).copy()
        top_u["Predicted_AAV"] = top_u["Predicted_AAV"].apply(
            lambda x: f"${x/1e6:.1f}M")
        disp_u = [c for c in ["Name", "Group", "Age", "Predicted_AAV"]
                  if c in top_u.columns]
        print(top_u[disp_u].to_string(index=False))

    # Save CSV
    all_preds = pd.concat([signed_preds, unsigned_preds], ignore_index=True)
    all_preds.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll predictions saved → {OUTPUT_PATH}")

    print("\n--- ACCURACY BY GROUP ---")
    for grp in sorted(signed_preds["Group"].unique()):
        sub = signed_preds[signed_preds["Group"] == grp].copy()
        sub = sub[sub["Actual_AAV"].notna()]
        if len(sub) < 3:
            continue
        mae = (sub["Predicted_AAV"] - sub["Actual_AAV"]).abs().mean()
        print(f"{grp}: n={len(sub)}, mean error=${mae/1e6:.1f}M")
        print(sub[["Name", "Actual_AAV", "Predicted_AAV", "Pct_Diff"]].head(5).to_string(index=False))
        print()

    # Plots
    print("\n[Step 5] Generating plots...")
    if len(signed_preds) > 5:
        plot_pred_vs_actual(signed_preds)
    if len(signed_preds) > 0:
        plot_top_predictions(signed_preds,
                             "Top Signed Players — Predicted vs Actual AAV",
                             "2026_signed_predictions_all_multiyear.png")
    if len(unsigned_preds) > 0:
        plot_top_predictions(unsigned_preds,
                             "Top Unsigned 2026 FAs — Predicted AAV",
                             "2026_unsigned_predictions_all_multiyear.png")
    plot_feature_importance(models, feat_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
