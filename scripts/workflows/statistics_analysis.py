#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os

def fisher_aggregate(r_values, weights=None):
    """
    Aggregate correlation coefficients via Fisher z-transform.
    If weights provided, they should correspond to each r (e.g., n_i - 3);
    otherwise unweighted average is used.
    r_values: list or array of correlation coefficients (r).
    weights: optional list or array of weights for each r value.
    Returns the aggregated correlation coefficient after applying Fisher's z-transform.
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
    return np.tanh(z_mean)

def correlation_analysis(results_df, speaker):
    """
    Compute Pearson and Spearman correlations between projected distances
    and phoneme cosine similarity for a given speaker in results_df.
    Adds new columns to results_df and returns the Pearson r.
    results_df: DataFrame containing columns:
        - 'Distance Projected on {speaker}': distances projected onto the speaker's trajectory.
        - 'Phonemes Cosine Similarity': cosine similarity of phoneme vectors.
        - 'speaker': string indicating the speaker's name (e.g., 'L1', 'L2').
    Returns a tuple of (Pearson correlation, Spearman correlation).
    """
    pearson_correlation = results_df[f'Distance Projected on {speaker}'].corr(results_df['Phonemes Cosine Similarity'])
    spearman_correlation = results_df[f'Distance Projected on {speaker}'].corr(results_df['Phonemes Cosine Similarity'], method='spearman')

    return pearson_correlation, spearman_correlation

def save_correlations(correlation_results, speaker_path, correlations_csv_file):
    """
    Append a summary row to the CSV file for a given speaker.
    correlation_results: dict containing correlation results with keys:
        - 'Pearson Correlation'
        - 'Spearman Correlation'
    speaker_path: path to the speaker's directory, used to extract speaker and sentence ID.
    correlations_csv_file: path to the CSV file where results will be saved.
    The CSV will have columns:
        - 'Speaker'
        - 'Sentence ID'
        - 'Pearson Correlation'
        - 'Spearman Correlation'
    If the file does not exist, it will be created with a header.
    If it exists, a new row will be appended without rewriting the header.
    """
    # Extract folder name and sentence ID
    speaker = os.path.basename(os.path.dirname(speaker_path))
    sentence = os.path.basename(speaker_path)

    # Build the row to append
    row = {
        'Speaker': speaker,
        'Sentence ID': sentence
    }
    # Merge in correlation values
    row.update(correlation_results)

    # Create DataFrame and append
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(correlations_csv_file)
    df_row.to_csv(correlations_csv_file, mode='a', header=write_header, index=False)
    print(f"Appended summary for {speaker}/{sentence} to {correlations_csv_file}")

def analyze_overall(csv_path):
    """
    Read CSV, identify correlation columns, and aggregate each column via Fisher transform overall.
    Outputs a single-row CSV with aggregated_r for each correlation column.
    csv_path: path to the input CSV file containing correlation results.
    The CSV should contain columns with names that include 'Pearson' or 'Spearman'.
    Outputs a new CSV with aggregated results, named as <original_name>_overall_aggreg
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
            agg_r = np.nan
        else:
            agg_r = fisher_aggregate(vals, weights=None)
        # Use safe column names for output
        safe_col = col.replace(" ", "_").replace("/", "_")
        result[f'{safe_col}_aggregated_r'] = agg_r
    
    result_df = pd.DataFrame([result])
    # Print to console
    print("Overall aggregated correlation results:")
    print(result_df.to_string(index=False))
    # Save to CSV
    out_path = csv_path.rsplit('.', 1)[0] + '_overall_aggregated.csv'
    result_df.to_csv(out_path, index=False)
    print(f"Saved overall aggregated results to: {out_path}")
