import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATA_PATH = "mlb_fa_training_data_v1.csv"
PRED_DATA_PATH  = "2026_predictions.csv"    # output from predict_second_try.py
OUTPUT_PATH     = "team_efficiency.csv"
MIN_SIGNINGS    = 3     # minimum signings for a team to be included
N_COMPONENTS    = 3     # latent factors for matrix factorization

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def assign_position_group(pos, era, gs, pa):
    """Simplified position classifier matching predict_second_try.py logic."""
    pos = str(pos).lower().strip()
    gs  = pd.to_numeric(gs,  errors="coerce")
    era = pd.to_numeric(era, errors="coerce")
    pa  = pd.to_numeric(pa,  errors="coerce")

    SP_CODES = {"17", "20"}
    RP_CODES = {"15", "16", "18", "19"}

    is_pitcher = (
        pd.notna(era) or
        any(x in pos for x in ["rhp", "lhp", "sp", "rp", "cp"]) or
        pos in SP_CODES | RP_CODES
    )

    if is_pitcher:
        if pd.notna(gs):
            return "SP" if gs > 10 else "RP"
        if any(x in pos for x in ["rhp-s", "lhp-s", "-s"]) or pos in SP_CODES:
            return "SP"
        return "RP"

    if pos in ["c", "c-dh"] or pos == "1" or pos.startswith("c-"):
        return "C"
    if any(x in pos for x in ["ss", "2b"]) or pos in {"4","5","6","11"}:
        return "MI"
    if any(x in pos for x in ["1b", "3b", "dh"]) or pos in {"3","14"}:
        return "CI"
    if any(x in pos for x in ["lf", "cf", "rf", "of"]) or pos in {"7","8","9","10"}:
        return "OF"
    if pd.notna(pa):
        return "CI"
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and prepare training data
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(path):
    df = pd.read_csv(path)

    # Force numeric
    for col in ["AAV", "Age", "FA_Year", "stat_GS", "stat_ERA", "stat_PA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[\$,\s]", "", regex=True),
                errors="coerce"
            )

    df = df[df["AAV"].notna() & (df["AAV"] > 0)].copy()

    # Assign groups
    df["group"] = df.apply(
        lambda r: assign_position_group(
            r.get("Position", ""), r.get("stat_ERA", np.nan),
            r.get("stat_GS", np.nan), r.get("stat_PA", np.nan)
        ), axis=1
    )

    # Find team column
    team_col = next((c for c in ["NewClub", "New Club", "Team", "NewTeam"]
                     if c in df.columns), None)
    if team_col is None:
        print("WARNING: No team column found. Columns:", list(df.columns))
        return pd.DataFrame()

    df.rename(columns={team_col: "Team"}, inplace=True)
    df = df[df["Team"].notna() & (df["group"] != "UNKNOWN")].copy()

    print(f"Loaded {len(df)} contracts across "
          f"{df['Team'].nunique()} teams ({df['FA_Year'].min():.0f}"
          f"–{df['FA_Year'].max():.0f})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load predictions and merge to get efficiency ratings
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiency(train_df, pred_path):
    """
    Merge predicted AAV onto historical contracts.
    Efficiency = Predicted / Actual
      > 1 → team underpaid (got a bargain)
      < 1 → team overpaid
      = 1 → fair market value
    """
    # We re-run a simple mean-based prediction per group/year as a proxy
    # since we need efficiency across ALL years not just 2026.
    # Use within-year group median as the "market rate" baseline.
    df = train_df.copy()

    group_median = (
        df.groupby("group")["AAV"]
        .median()
        .reset_index()
        .rename(columns={"AAV": "market_median_AAV"})
    )
    df = df.merge(group_median, on="group", how="left")
    # Efficiency: how does actual AAV compare to market median?
    # < 1 = underpaid relative to peers (team got value)
    # > 1 = overpaid relative to peers
    df["efficiency"] = df["AAV"] / df["market_median_AAV"]

    # Invert so higher = better deal for the team
    # "value_score" > 1 means team got a bargain
    df["value_score"] = 1 / df["efficiency"]

    print(f"\nEfficiency stats:")
    print(f"  Mean:   {df['efficiency'].mean():.3f}")
    print(f"  Median: {df['efficiency'].median():.3f}")
    print(f"  Std:    {df['efficiency'].std():.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build the Team × Group matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_matrix(df, min_signings=MIN_SIGNINGS):
    # Require minimum signings per team-group combination
    # to avoid unstable ratios from single observations
    counts = df.groupby(["Team", "group"]).size().reset_index(name="n")
    df = df.merge(counts, on=["Team", "group"])
    df = df[df["n"] >= 2].copy()  # at least 2 signings per cell

    # Cap value score to prevent explosion from outliers
    df["value_score"] = df["value_score"].clip(lower=0.2, upper=3.0)

    # Filter teams with enough total signings
    team_counts = df.groupby("Team").size()
    valid_teams = team_counts[team_counts >= min_signings].index
    df = df[df["Team"].isin(valid_teams)].copy()

    matrix = df.pivot_table(
        index="Team",
        columns="group",
        values="value_score",
        aggfunc="mean"
    )

    # Fill missing cells with 1.0 (neutral)
    matrix = matrix.fillna(1.0)

    print(f"\nTeam x Group matrix: {matrix.shape[0]} teams x "
          f"{matrix.shape[1]} groups")
    return matrix, df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Matrix Factorization with NMF
# ─────────────────────────────────────────────────────────────────────────────

def factorize_matrix(matrix, n_components=N_COMPONENTS):
    """
    Non-negative Matrix Factorization:
    Matrix ≈ W × H
    W = team "spending style" in latent space
    H = how each latent style relates to position groups
    """
    scaler = MinMaxScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
    W = nmf.fit_transform(matrix_scaled)   # Teams × latent factors
    H = nmf.components_                    # latent factors × Groups

    reconstruction_err = nmf.reconstruction_err_
    print(f"\nNMF reconstruction error: {reconstruction_err:.4f}")
    print(f"(Lower = better fit to the data)")

    # Reconstruct matrix to get smoothed predictions
    reconstructed_scaled = W @ H
    reconstructed = scaler.inverse_transform(reconstructed_scaled)
    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=matrix.index,
        columns=matrix.columns
    )

    return W, H, reconstructed_df, nmf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Analysis — which teams are most/least efficient?
# ─────────────────────────────────────────────────────────────────────────────

def analyze_teams(matrix, reconstructed_df, df):
    """Print team rankings and interesting findings."""
    # Overall efficiency per team (mean across all groups)
    team_overall = matrix.mean(axis=1).sort_values(ascending=False)

    print("\n--- TOP 10 MOST EFFICIENT TEAMS (best bargain hunters) ---")
    print(team_overall.head(10).apply(lambda x: f"{x:.3f}").to_string())

    print("\n--- TOP 10 LEAST EFFICIENT TEAMS (biggest overpayers) ---")
    print(team_overall.tail(10).apply(lambda x: f"{x:.3f}").to_string())

    # Best/worst per position group
    print("\n--- BIGGEST SPENDERS BY POSITION GROUP ---")
    for grp in matrix.columns:
        col = matrix[grp].sort_values()
        print(f"\n  {grp}:")
        print(f"    Best value:  {col.index[-1]} ({col.iloc[-1]:.3f})")
        print(f"    Most excess: {col.index[0]}  ({col.iloc[0]:.3f})")

    return team_overall


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(matrix, save_path="team_efficiency_heatmap.png"):
    """
    Heatmap of team spending efficiency by position group.
    Red = overpaying, Green = getting bargains.
    """
    fig, ax = plt.subplots(figsize=(12, max(8, len(matrix) * 0.3)))

    # Center colormap at 1.0 (fair value)
    vmin = max(0.5, matrix.values.min())
    vmax = min(2.0, matrix.values.max())

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlGn",
        center=1.0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Value Score (>1 = bargain, <1 = overpaid)"}
    )

    ax.set_title("MLB Team Salary Efficiency by Position Group\n"
                 "(2015–2025 Free Agent Signings)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Position Group")
    ax.set_ylabel("Team")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")


def plot_team_rankings(team_overall, save_path="team_overall_efficiency.png", n=15):
    """Bar chart of top and bottom teams by overall efficiency."""
    top    = team_overall.head(n)
    bottom = team_overall.tail(n)
    combined = pd.concat([top, bottom]).drop_duplicates()
    combined = combined.sort_values()

    colors = ["salmon" if v < 1.0 else "steelblue" for v in combined.values]

    fig, ax = plt.subplots(figsize=(8, max(6, len(combined) * 0.35)))
    ax.barh(combined.index, combined.values, color=colors)
    ax.axvline(x=1.0, color="black", linestyle="--",
               linewidth=1.5, label="Fair market value")
    ax.set_xlabel("Mean Value Score")
    ax.set_title(f"Top & Bottom {n} Teams — FA Salary Efficiency\n"
                 "(Blue = bargain hunters, Red = overspenders)",
                 fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_latent_factors(H, groups, save_path="nmf_latent_factors.png"):
    """
    Shows what each latent "spending style" means in terms of position groups.
    Useful for explaining what patterns NMF found.
    """
    n_components = H.shape[0]
    fig, axes = plt.subplots(1, n_components,
                             figsize=(5 * n_components, 4), sharey=True)

    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.barh(groups, H[i], color="steelblue")
        ax.set_title(f"Latent Factor {i+1}", fontsize=11)
        ax.set_xlabel("Factor Loading")
        ax.axvline(x=0, color="black", linewidth=0.8)

    plt.suptitle("NMF Latent Spending Styles\n"
                 "(Each factor = a pattern of team spending behavior)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def plot_team_styles(W, matrix, save_path="team_spending_styles.png", n=20):
    """
    Scatter plot of teams in latent factor space (Factor 1 vs Factor 2).
    Teams clustered together have similar spending behaviors.
    """
    if W.shape[1] < 2:
        print("  Need at least 2 latent factors for style plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(W[:, 0], W[:, 1], alpha=0.7,
               s=80, edgecolors="k", linewidths=0.5)

    # Label all teams
    for i, team in enumerate(matrix.index):
        ax.annotate(team, (W[i, 0], W[i, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)

    ax.set_xlabel("Latent Factor 1", fontsize=11)
    ax.set_ylabel("Latent Factor 2", fontsize=11)
    ax.set_title("Team Spending Styles in Latent Space\n"
                 "(Teams near each other have similar FA spending patterns)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== MLB FA Collaborative Filtering — Team Spending Efficiency ===\n")

    # Step 1: Load training data
    print("[Step 1] Loading training data...")
    train_df = load_training_data(TRAIN_DATA_PATH)
    if train_df.empty:
        print("ERROR: Could not load training data.")
        return

    # Step 2: Compute efficiency ratings
    print("\n[Step 2] Computing salary efficiency ratings...")
    df = compute_efficiency(train_df, PRED_DATA_PATH)

    # Step 3: Build matrix
    print("\n[Step 3] Building Team × Position Group matrix...")
    matrix, df_filtered = build_matrix(df)

    # Step 4: Matrix factorization
    print("\n[Step 4] Running NMF matrix factorization...")
    W, H, reconstructed_df, nmf = factorize_matrix(matrix)

    # Step 5: Analysis
    print("\n[Step 5] Analyzing team patterns...")
    team_overall = analyze_teams(matrix, reconstructed_df, df_filtered)

    # Save results
    matrix.to_csv("team_efficiency_matrix.csv")
    reconstructed_df.to_csv("team_efficiency_reconstructed.csv")
    team_overall.to_frame("overall_value_score").to_csv(OUTPUT_PATH)
    print(f"\nSaved: team_efficiency_matrix.csv")
    print(f"Saved: team_efficiency_reconstructed.csv")
    print(f"Saved: {OUTPUT_PATH}")

    # Step 6: Plots
    print("\n[Step 6] Generating plots...")
    plot_heatmap(matrix)
    plot_team_rankings(team_overall)
    plot_latent_factors(H, list(matrix.columns))
    plot_team_styles(W, matrix)

    print("\nDone.")


if __name__ == "__main__":
    main()