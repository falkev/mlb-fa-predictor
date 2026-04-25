import pandas as pd
import numpy as np
import re
import os

def clean_currency(value):
    if pd.isna(value) or str(value).strip() in ["", "nan", "NaN"]:
        return 0.0
    clean_val = re.sub(r'[^\d.]', '', str(value))
    return float(clean_val) if clean_val else 0.0

def normalize_name_fa(name):
    if pd.isna(name): return ""
    parts = str(name).split(',')
    if len(parts) == 2:
        return f"{parts[1].strip()} {parts[0].strip()}"
    return str(name).strip()


def normalize_name_stats(name):
    if pd.isna(name): return ""
    # Remove * and # symbols B-Ref uses for All-Stars/Postseason
    return re.sub(r'[*#]', '', str(name)).strip()


def detect_name_column(df, label=""):
    for candidate in ["Name", "Player", "name", "player"]:
        if candidate in df.columns:
            return candidate
    print(f"  WARNING: Could not find a name column in {label}. Columns: {list(df.columns)}")
    return None

def dedup_stats(df):
    TOTAL_TAGS = {"2TM", "3TM", "4TM", "5TM"}

    totals = df[df["Team"].isin(TOTAL_TAGS)].copy()
    singles = df[~df["Team"].isin(TOTAL_TAGS)].copy()

    players_with_totals = set(totals["CleanName"])
    singles_only = singles[~singles["CleanName"].isin(players_with_totals)]

    combined = pd.concat([totals, singles_only], ignore_index=True)
    return combined

STAT_COLS_TO_DROP = [
    "stat_Player", "stat_player",
    "stat_Rk", "stat_rk",
    "stat_Age",
    "stat_Team", "stat_team",
    "stat_Lg", "stat_lg",
    "stat_Awards", "stat_awards",
    "stat_Player-additional",
    "stat_Pos",
    "stat_Pos'n",
]

def process_year(fa_file, batting_file, pitching_file, year):
    print(f"--- Processing {year} FA with stats from {year - 1} ---")

    # 1. Load FA Data
    try:
        temp = pd.read_csv(fa_file, nrows=20, header=None, encoding='latin-1')
        header_idx = next(
            i for i, row in temp.iterrows() if row.astype(str).str.contains('Player', case=False).any())
        df_fa = pd.read_csv(fa_file, skiprows=header_idx, encoding='latin-1')
    except (UnicodeDecodeError, StopIteration):
        df_fa = pd.read_csv(fa_file, skiprows=12, encoding='cp1252')

    df_fa.columns = df_fa.columns.str.strip().str.replace('\n', ' ')
    df_fa = df_fa.dropna(subset=['Player'])
    df_fa = df_fa[df_fa['Player'].str.contains(',', na=False)]
        # Normalize position column name
    for col in ["Pos'n", "Position", "Pos", "POS"]:
        if col in df_fa.columns:
            df_fa.rename(columns={col: "Position"}, inplace=True)
            break

    df_fa['CleanName'] = df_fa['Player'].apply(normalize_name_fa)
    df_fa['Guarantee'] = df_fa['Guarantee'].apply(clean_currency)
    df_fa['AAV'] = df_fa['AAV'].apply(clean_currency)
    df_fa['FA_Year'] = year

    # 2. Load Stats
    b_stats = pd.read_csv(batting_file, encoding='latin-1')
    p_stats = pd.read_csv(pitching_file, encoding='latin-1')

    # Detect name column (B-Ref uses 'Player', some exports use 'Name')
    b_name_col = detect_name_column(b_stats, label=batting_file)
    p_name_col = detect_name_column(p_stats, label=pitching_file)

    if b_name_col is None or p_name_col is None:
        raise ValueError(f"Could not find name column in stats files for {year}.")

    # Build CleanName BEFORE dedup and add_prefix
    b_stats['CleanName'] = b_stats[b_name_col].apply(normalize_name_stats)
    p_stats['CleanName'] = p_stats[p_name_col].apply(normalize_name_stats)

    # 3. Deduplicates: one row per player (keep 2TM/3TM total row for traded players)
    b_before = len(b_stats)
    p_before = len(p_stats)
    b_stats = dedup_stats(b_stats)
    p_stats = dedup_stats(p_stats)
    print(f"    Batting:  {b_before} rows -> {len(b_stats)} after dedup")
    print(f"    Pitching: {p_before} rows -> {len(p_stats)} after dedup")

    # 4. Add Predictive Features (after dedup, before prefix)
    b_stats['stat_ISO']  = b_stats['SLG'] - b_stats['BA']
    b_stats['stat_BB_K'] = b_stats['BB'] / b_stats['SO'].replace(0, 1)
    p_stats['stat_SO_W'] = p_stats['SO'] / p_stats['BB'].replace(0, 1)

    # 5. Prefix all stat columns, then restore CleanName as the merge key
    b_stats = b_stats.add_prefix('stat_').rename(columns={'stat_CleanName': 'CleanName'})
    p_stats = p_stats.add_prefix('stat_').rename(columns={'stat_CleanName': 'CleanName'})

    # 6. Drop redundant metadata columns that would clutter the final dataset
    b_stats.drop(columns=[c for c in STAT_COLS_TO_DROP if c in b_stats.columns], inplace=True)
    p_stats.drop(columns=[c for c in STAT_COLS_TO_DROP if c in p_stats.columns], inplace=True)

    # 7. Merge
    pos_col = next((c for c in ["Pos'n", "Position", "Pos", "POS"] if c in df_fa.columns), None)
    if pos_col is None:
        print(f"  WARNING: No position column found. Columns: {list(df_fa.columns)}")
        is_pitcher = pd.Series(False, index=df_fa.index)
    else:
        is_pitcher = df_fa[pos_col].str.contains('p', case=False, na=False)

    df_hitters  = df_fa[~is_pitcher].merge(b_stats, on='CleanName', how='left')
    df_pitchers = df_fa[is_pitcher].merge(p_stats,  on='CleanName', how='left')

    final_year_df = pd.concat([df_hitters, df_pitchers], ignore_index=True)
    print(f"    -> {len(final_year_df)} rows ({is_pitcher.sum()} pitchers, {(~is_pitcher).sum()} hitters)")
    return final_year_df

# DIAGNOSTIC
_b = pd.read_csv('2014_batting.csv', encoding='latin-1', nrows=3)
_p = pd.read_csv('2014_pitching.csv', encoding='latin-1', nrows=3)
print("Batting columns: ", list(_b.columns))
print("Pitching columns:", list(_p.columns))
del _b, _p
# --- EXECUTION LOOP ---

def add_multi_year_stats(df):
    """
    For each player-year, add their 2-year and 3-year average stats.
    df must have PlayerName, FA_Year, and all stat_ columns.
    """
    stat_cols = [c for c in df.columns if c.startswith("stat_")]
    df = df.sort_values(["CleanName", "FA_Year"])
    
    for col in stat_cols:
        # 2-year average (current + prior year)
        df[f"{col}_2yr"] = (
            df.groupby("CleanName")[col]
            .transform(lambda x: x.rolling(2, min_periods=1).mean())
        )
        # 3-year average
        df[f"{col}_3yr"] = (
            df.groupby("CleanName")[col]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
    
    # add year-over-year change as a trend signal
    for col in ["stat_WAR", "stat_ERA", "stat_OPS"]:
        if col in df.columns:
            df[f"{col}_trend"] = df.groupby("CleanName")[col].diff()
    
    return df

all_years_data = []



for year in range(2015, 2027):
    fa_path = f'MLB-Free Agency 1991-2026.xls - {year}.csv'
    b_path  = f'{year - 1}_batting.csv'
    p_path  = f'{year - 1}_pitching.csv'

    if os.path.exists(fa_path) and os.path.exists(b_path):
        try:
            df = process_year(fa_path, b_path, p_path, year)
            all_years_data.append(df)
        except Exception as e:
            print(f"Error processing {year}: {e}")
    else:
        if not os.path.exists(fa_path):
            print(f"  Skipping {year}: FA file not found ({fa_path})")
        if not os.path.exists(b_path):
            print(f"  Skipping {year}: batting file not found ({b_path})")

if all_years_data:
    full_dataset = pd.concat(all_years_data, ignore_index=True)
    full_dataset = add_multi_year_stats(full_dataset)
    # AFTER
    train = full_dataset[full_dataset["FA_Year"] != 2026].copy()
    predict = full_dataset[full_dataset["FA_Year"] == 2026].copy()

    train.to_csv("mlb_fa_training_data_v1.csv", index=False)
    predict.to_csv("mlb_fa_2026.csv", index=False)

    print(f"Training set: {len(train)} rows (2015-2025) → mlb_fa_training_data_v1.csv")
    print(f"Prediction set: {len(predict)} rows (2026) → mlb_fa_2026.csv")
else:
    print("No data found. Check your file naming (e.g., '2022_batting.csv').")


