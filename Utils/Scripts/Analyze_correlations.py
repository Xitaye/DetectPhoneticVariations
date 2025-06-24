#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def fisher_aggregate(r_values, weights=None):
    """
    Aggregate correlation coefficients via Fisher z-transform.
    If weights provided, they should correspond to each r (e.g., n_i - 3);
    otherwise unweighted average is used.
    """
    r_vals = np.array(r_values, dtype=float)
    # Clip to avoid exactly ±1
    eps = 1e-7
    r_clipped = np.clip(r_vals, -1 + eps, 1 - eps)
    z_vals = np.arctanh(r_clipped)
    if weights is not None:
        w = np.array(weights, dtype=float)
        w = np.where(w > 0, w, 0)
        if np.sum(w) > 0:
            z_mean = np.sum(w * z_vals) / np.sum(w)
        else:
            z_mean = np.mean(z_vals)
    else:
        z_mean = np.mean(z_vals)
    return np.tanh(z_mean), z_mean

def analyze_csv_overall(csv_path):
    """
    Read CSV, identify correlation columns, and aggregate each column via Fisher transform overall.
    Outputs a single-row CSV with aggregated_r and mean_z for each correlation column.
    """
    df = pd.read_csv(csv_path)
    # Identify correlation columns: those containing 'pearson' or 'spearman'
    corr_cols = [col for col in df.columns if 'pearson' in col.lower() or 'spearman' in col.lower()]
    if not corr_cols:
        raise ValueError("No correlation columns found containing 'Pearson' or 'Spearman'.")
    
    result = {}
    for col in corr_cols:
        vals = df[col].dropna().tolist()
        if len(vals) == 0:
            agg_r, mean_z = np.nan, np.nan
        else:
            agg_r, mean_z = fisher_aggregate(vals, weights=None)
        # Use safe column names for output
        safe_col = col.replace(" ", "_").replace("/", "_")
        result[f'{safe_col}_aggregated_r'] = agg_r
        result[f'{safe_col}_mean_z'] = mean_z
    
    result_df = pd.DataFrame([result])
    # Print to console
    print("Overall aggregated correlation results:")
    print(result_df.to_string(index=False))
    # Save to CSV
    out_path = csv_path.rsplit('.', 1)[0] + '_overall_aggregated.csv'
    result_df.to_csv(out_path, index=False)
    print(f"Saved overall aggregated results to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate each correlation column in a CSV into a single overall result via Fisher z-transform"
    )
    parser.add_argument('csv_path', help="Path to the CSV file")
    args = parser.parse_args()
    analyze_csv_overall(args.csv_path)

if __name__ == "__main__":
    main()
