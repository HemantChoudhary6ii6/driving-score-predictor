"""
step0_prepare.py
=================
Prepares data for LSTM / GRU / Transformer training.

Uses the EXACT same restructure and clean logic as the ML project:
  - Reads data/raw_real/1 star/ ... 5 star/
  - Drops ID/Name column, adds Rating from folder name
  - Runs all 11 cleaning steps (same as step1_clean.py in ML project)
  - Saves all intermediate files to data/ exactly like the ML project
  - Then builds windowed sequences for deep learning

Input:
    data/raw_real/
    ├── 1 star/   ← Rating 1 (most dangerous → risk score 100)
    ├── 2 star/
    ├── 3 star/
    ├── 4 star/
    └── 5 star/   ← Rating 5 (safest → risk score 0)

Each CSV must have:
    ID, SrNo, Timestamp, X_Acc, Y_Acc, Z_Acc, X_Gyro, Y_Gyro, Z_Gyro

Output:
    data/raw/all_sessions.csv          (same as ML project step0)
    data/clean/all_sessions_clean.csv  (same as ML project step1)
    data/sequences_X.npy               shape (N, 100, 6)
    data/sequences_y.npy               shape (N,)  labels 1-5
    data/dl_scaler.pkl                 normalisation params
    data/split_indices.pkl             train/test index arrays
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ── Same constants as ML project ──────────────────────────────────────────────
RAW_REAL_DIR = os.path.join('data', 'raw_real')
RATING_MAP   = {'1 star': 1, '2 star': 2, '3 star': 3, '4 star': 4, '5 star': 5}
SENSOR_COLS  = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']
ACC_LIMIT    = 40.0
GYRO_LIMIT   = 300.0
GAP_MS       = 5000

# ── Window config ─────────────────────────────────────────────────────────────
WINDOW = 100
STEP   = 50


# ══════════════════════════════════════════════════════════════════════════════
# STEP A — RESTRUCTURE  (identical to step0_restructure.py in ML project)
# ══════════════════════════════════════════════════════════════════════════════
def restructure() -> str:
    """
    Reads star folders, drops ID column, adds Rating, merges all CSVs.
    Returns path to the merged CSV.
    Identical logic to ML project step0_restructure.py.
    """
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    output_path = os.path.join('data', 'raw', 'all_sessions.csv')

    if not os.path.exists(RAW_REAL_DIR):
        print(f"  ERROR: Folder '{RAW_REAL_DIR}' not found.")
        print("  Create it and put your '1 star' to '5 star' folders inside.")
        raise SystemExit(1)

    all_frames  = []
    global_srno = 1

    for folder_name, rating in RATING_MAP.items():
        folder_path = os.path.join(RAW_REAL_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"  [SKIP] '{folder_name}' not found — rating {rating} skipped")
            continue

        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        print(f"\n  [{folder_name}]  Rating={rating}  |  {len(csv_files)} files")

        for fpath in sorted(csv_files):
            fname = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath, low_memory=False)

                # Drop ID / Name column — not a sensor feature
                drop_cols = [c for c in df.columns if c.strip().upper() in ('ID', 'NAME')]
                df = df.drop(columns=drop_cols, errors='ignore')

                # Check required sensor columns
                missing = [c for c in ['Timestamp'] + SENSOR_COLS if c not in df.columns]
                if missing:
                    print(f"    [SKIP] {fname} — missing: {missing}")
                    continue

                # Rating from folder name
                df['Rating'] = rating

                # Globally unique SrNo
                df['SrNo'] = range(global_srno, global_srno + len(df))
                global_srno += len(df)

                all_frames.append(df)
                print(f"    OK  {fname}  →  {len(df):,} rows")

            except Exception as e:
                print(f"    [ERROR] {fname}: {e}")

    if not all_frames:
        print("\n  ERROR: No data loaded. Check your folder structure.")
        raise SystemExit(1)

    merged = pd.concat(all_frames, ignore_index=True)
    merged['SrNo'] = range(1, len(merged) + 1)

    keep   = ['SrNo', 'Timestamp', 'X_Acc', 'Y_Acc', 'Z_Acc',
              'X_Gyro', 'Y_Gyro', 'Z_Gyro', 'Rating']
    merged = merged[[c for c in keep if c in merged.columns]]
    merged.to_csv(output_path, index=False)

    print(f"\n  {'='*50}")
    print(f"  RESTRUCTURE COMPLETE")
    print(f"  Total rows : {len(merged):,}")
    print(f"  Output     : {output_path}")
    print(f"  Rating distribution:")
    for r, cnt in merged['Rating'].value_counts().sort_index().items():
        pct = 100 * cnt / len(merged)
        bar = '█' * int(pct / 2)
        print(f"    Rating {r}  :  {cnt:>7,} rows  ({pct:5.1f}%)  {bar}")
    print(f"  {'='*50}")

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP B — CLEAN  (identical to step1_clean.py in ML project)
# ══════════════════════════════════════════════════════════════════════════════
def clean(filepath: str) -> pd.DataFrame:
    """
    Runs all 11 cleaning steps.
    Identical logic to ML project step1_clean.py.
    """
    df     = pd.read_csv(filepath, low_memory=False)
    report = {'original_rows': len(df)}

    # 1. Exact duplicate rows
    n = len(df); df = df.drop_duplicates()
    report['exact_dupes_removed'] = n - len(df)

    # 2. Duplicate SrNo
    if 'SrNo' in df.columns:
        n = len(df)
        df = df.drop_duplicates(subset=['SrNo'], keep='first').reset_index(drop=True)
        report['srno_dupes_removed'] = n - len(df)

    # 3. NaN — forward-fill, then drop remaining
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill()
    n = len(df)
    df = df.dropna(subset=SENSOR_COLS + ['Rating', 'Timestamp'])
    report['nan_rows_removed'] = n - len(df)

    # 4. All-zero sensor rows
    n = len(df)
    df = df[~(df[SENSOR_COLS] == 0).all(axis=1)].reset_index(drop=True)
    report['all_zero_removed'] = n - len(df)

    # 5. Z_Acc == 0 (gravity makes this impossible)
    n = len(df)
    df = df[df['Z_Acc'] != 0].reset_index(drop=True)
    report['z_acc_zero_removed'] = n - len(df)

    # 6. All gyro axes zero simultaneously
    n = len(df)
    df = df[~((df['X_Gyro']==0) & (df['Y_Gyro']==0) & (df['Z_Gyro']==0))].reset_index(drop=True)
    report['all_gyro_zero_removed'] = n - len(df)

    # 7. Frozen/stuck sensor (rolling std = 0 over 10 rows)
    n = len(df)
    for col in SENSOR_COLS:
        rstd = df[col].rolling(window=10, min_periods=10).std()
        df   = df[~((rstd == 0) & rstd.notna())].reset_index(drop=True)
    report['frozen_rows_removed'] = n - len(df)

    # 8. Hard physical limits
    for col in ['X_Acc', 'Y_Acc', 'Z_Acc']:
        df[col] = df[col].clip(-ACC_LIMIT, ACC_LIMIT)
    for col in ['X_Gyro', 'Y_Gyro', 'Z_Gyro']:
        df[col] = df[col].clip(-GYRO_LIMIT, GYRO_LIMIT)

    # 9. Statistical clip (1st–99th percentile per axis)
    for col in SENSOR_COLS:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # 10. Sort by timestamp, assign session_id
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    df['session_id'] = (df['Timestamp'].diff().fillna(0) > GAP_MS).cumsum()
    report['session_segments'] = int(df['session_id'].nunique())

    # 11. Rating validation
    n = len(df)
    df = df[df['Rating'].between(1, 5)].reset_index(drop=True)
    df['Rating'] = df['Rating'].astype(int)
    report['invalid_rating_removed'] = n - len(df)

    df['SrNo'] = range(1, len(df) + 1)

    report['final_rows']    = len(df)
    report['rows_removed']  = report['original_rows'] - len(df)
    report['retention_pct'] = round(100 * len(df) / report['original_rows'], 2)

    print('\n  ======= CLEANING AUDIT REPORT =======')
    for k, v in report.items():
        print(f'    {k:<32}: {v}')
    print('  ======================================\n')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP C — BUILD SEQUENCES  (DL-specific windowing)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_mode(series):
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else series.iloc[0]


def build_sequences(df: pd.DataFrame):
    """Slide a 100-row window over each session, step 50."""
    X_list, y_list = [], []

    for _, sdf in df.groupby('session_id'):
        sdf  = sdf.reset_index(drop=True)
        vals = sdf[SENSOR_COLS].values.astype(np.float32)
        rats = sdf['Rating'].astype(int).values

        for i in range(0, len(sdf) - WINDOW + 1, STEP):
            X_list.append(vals[i: i + WINDOW])
            y_list.append(int(_safe_mode(pd.Series(rats[i: i + WINDOW]))))

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('\n' + '='*55)
    print('  Step 0 — Data preparation')
    print('='*55)

    # A — Restructure (same as ML project step0_restructure.py)
    print('\n  [A] Restructuring star folders …')
    raw_path = restructure()

    # B — Clean (same as ML project step1_clean.py)
    print('\n  [B] Cleaning …')
    os.makedirs(os.path.join('data', 'clean'), exist_ok=True)
    clean_df = clean(raw_path)
    clean_path = os.path.join('data', 'clean', 'all_sessions_clean.csv')
    clean_df.to_csv(clean_path, index=False)
    print(f'  Clean data saved → {clean_path}  ({len(clean_df):,} rows)')

    # C — Build sequences for deep learning
    print('\n  [C] Building sequences …')
    X, y = build_sequences(clean_df)

    print(f'  Windows : {X.shape[0]:,}')
    print(f'  Shape   : {X.shape}  (windows × timesteps × sensors)')
    print(f'\n  Window label distribution:')
    for r, cnt in zip(*np.unique(y, return_counts=True)):
        pct = 100 * cnt / len(y)
        print(f'    Rating {r}: {cnt:>6,}  ({pct:.1f}%)')

    # Normalise
    mu  = X.mean(axis=(0, 1), keepdims=True)
    sig = X.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_n = (X - mu) / sig

    # Train / test split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_n))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    # Save
    np.save('data/sequences_X.npy',    X_n)
    np.save('data/sequences_y.npy',    y)
    joblib.dump({'mean': mu, 'std': sig},          'data/dl_scaler.pkl')
    joblib.dump({'train': tr_idx, 'test': te_idx}, 'data/split_indices.pkl')

    print(f'\n  Train windows : {len(tr_idx):,}')
    print(f'  Test  windows : {len(te_idx):,}')
    print(f'\n  Saved → data/sequences_X.npy')
    print(f'  Saved → data/sequences_y.npy')
    print(f'  Saved → data/dl_scaler.pkl')
    print(f'  Saved → data/split_indices.pkl')
    print(f'\n  Next: python step1_train.py')


if __name__ == '__main__':
    main()