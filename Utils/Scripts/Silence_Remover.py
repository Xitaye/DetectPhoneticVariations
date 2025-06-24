#!/usr/bin/env python3
"""
Recursively remove silences from all .wav files under a fixed root folder
and write the cleaned files into a parallel, hard-coded output folder,
while printing progress information.

Improved threshold logic: uses RMS and mean-norm, with smaller factor and tighter caps.
Requires: sox installed and in your PATH.
"""

import os
import subprocess
import re
import sys

# === Hard-coded directories ===
INPUT_DIR  = "/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings"
OUTPUT_DIR = "/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings_clean"

# === Threshold settings ===
# Aggressiveness: lower factor → more aggressive (lower threshold)
TARGET_FACTOR = 0.2   # try 0.3; lower for more aggressive trimming
MIN_PCT = 0.5         # minimum threshold percent
MAX_PCT = 10.0        # maximum threshold percent

def gather_wav_paths(input_dir):
    """Walk input_dir and return list of (input_path, relative_dir)."""
    pairs = []
    for root, _, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        for fname in files:
            if fname.lower().endswith('.wav'):
                pairs.append((os.path.join(root, fname), rel_dir))
    return pairs

def get_sox_stats(wav_path):
    """
    Run `sox wav_path -n stat` and parse stderr for:
      - max amplitude
      - RMS amplitude
      - mean norm amplitude
    Returns (max_amp, rms_amp, mean_norm).
    """
    try:
        result = subprocess.run(
            ['sox', wav_path, '-n', 'stat'],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        # sox stat often returns non-zero but writes stats to stderr
        output = e.stderr
    else:
        output = result.stderr

    max_amp = None
    rms_amp = None
    mean_norm = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Maximum amplitude"):
            try:
                max_amp = float(line.split(":",1)[1].strip())
            except:
                pass
        elif re.search(r'RMS\s+amplitude', line):
            try:
                rms_amp = float(line.split(":",1)[1].strip())
            except:
                pass
        elif re.search(r'Mean\s+norm', line):
            # line like "Mean    norm:          0.030037"
            try:
                mean_norm = float(line.split(":",1)[1].strip())
            except:
                pass
    return max_amp, rms_amp, mean_norm

def compute_threshold_pct(max_amp, rms_amp, mean_norm,
                          target_factor=TARGET_FACTOR,
                          min_pct=MIN_PCT, max_pct=MAX_PCT):
    """
    Compute two candidate thresholds:
      - from RMS: (rms_amp / max_amp) * 100 * target_factor
      - from mean norm: (mean_norm / max_amp) * 100 * target_factor_mean
    Then pick the smaller, bounded to [min_pct, max_pct].
    """
    if max_amp is None or max_amp <= 0:
        return None

    candidates = []

    if rms_amp is not None:
        raw_rms_pct = (rms_amp / max_amp) * 100 * target_factor
        candidates.append(raw_rms_pct)
    if mean_norm is not None:
        # use slightly different factor for mean_norm if desired
        raw_mean_pct = (mean_norm / max_amp) * 100 * target_factor
        candidates.append(raw_mean_pct)

    if not candidates:
        return None

    # pick the smallest candidate (more aggressive trimming)
    raw_pct = min(candidates)
    # bound into [min_pct, max_pct]
    pct = max(min_pct, min(raw_pct, max_pct))
    return pct

def clean_silence():
    wav_list = gather_wav_paths(INPUT_DIR)
    total = len(wav_list)
    if total == 0:
        print(f"⚠️  No .wav files found under '{INPUT_DIR}'")
        return

    print(f"👣 Found {total} .wav files — starting processing...\n")

    for idx, (in_path, rel_dir) in enumerate(wav_list, start=1):
        # Prepare target folder
        target_dir = os.path.join(OUTPUT_DIR, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        fname   = os.path.basename(in_path)
        out_path = os.path.join(target_dir, fname)

        # Progress log
        print(f"[{idx:>3}/{total}] → {rel_dir}/{fname}")

        # Get sox stats
        max_amp, rms_amp, mean_norm = get_sox_stats(in_path)
        if max_amp is None:
            print("   ⚠️ Could not parse max amplitude, using default threshold 2%")
            threshold_pct = 2.0
        else:
            threshold_pct = compute_threshold_pct(max_amp, rms_amp, mean_norm)
            if threshold_pct is None:
                print("   ⚠️ Could not compute threshold from stats, using default 2%")
                threshold_pct = 2.0
            else:
                print_str = f"   • Max: {max_amp:.4f}"
                if rms_amp is not None:
                    print_str += f", RMS: {rms_amp:.4f}"
                if mean_norm is not None:
                    print_str += f", MeanNorm: {mean_norm:.4f}"
                print_str += f" -> threshold ~{threshold_pct:.1f}%"
                print(print_str)

        # Build sox silence command using computed threshold
        thresh_str = f"{threshold_pct:.1f}%"
        cmd = [
            'sox', in_path, out_path,
            'silence', '-l',
              '1', '0.02', thresh_str,
             '-1', '0.02', thresh_str
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"   ❌ sox silence failed: {e}", file=sys.stderr)

    print(f"\n✅ Done! Cleaned files are in: '{OUTPUT_DIR}'")

if __name__ == '__main__':
    clean_silence()
