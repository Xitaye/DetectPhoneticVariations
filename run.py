import os
import re
import logging
from datetime import datetime
import shutil
import pandas as pd
import subprocess
import argparse
from collections import defaultdict
from scripts.workflows.statistics_analysis import analyze_overall


# === Debug ===
success_count_by_folder = defaultdict(int)
folder_level_textgrid_errors = set()
file_level_textgrid_errors = defaultdict(int)
folder_level_wav_errors = set()
file_level_wav_errors = defaultdict(int)

log_dir = None
log_file = None
processed_dir = None # (if set, copy output CSV and L2 results into this folder)

# === Utility function ===
def log(msg):
    print(msg)
    logging.info(msg)

def extract_id(filename):
    match = re.search(r'(\d{2})[-_](\d{2})', filename)
    return match.group(0) if match else None

def process_pair(l1_wav, l2_wav, l2_folder, path_to_projection_analysis, path_to_correlations_csv):
    log(f"\nProcessing pair:\n      - L1 Speaker: {l1_wav}\n      - L2 Speaker:   {l2_wav}")
    success_count_by_folder[l2_folder] += 1

    # Build a per-pair log directory under the main log_dir
    speaker_folder = os.path.basename(os.path.dirname(l2_wav))
    pair_log_dir = os.path.join(log_dir, speaker_folder)
    os.makedirs(pair_log_dir, exist_ok=True)

    # Name the log after the WAV file (change .wav → .log)
    wav_base = os.path.basename(l2_wav)
    log_filename = os.path.splitext(wav_base)[0] + ".log"
    log_path = os.path.join(pair_log_dir, log_filename)

    # Run the projection script, capture its output
    proc = subprocess.Popen(
        ["python", path_to_projection_analysis, l1_wav, l2_wav, path_to_correlations_csv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True  # get strings not bytes
    )

    # Stream its output into the per-pair log, prefixing each line with a timestamp
    with open(log_path, "w") as lf:
        for line in proc.stdout:
            ts_line = f"{datetime.now():%Y-%m-%d %H:%M:%S} - {line.rstrip()}\n"
            lf.write(ts_line)

    proc.wait()
    if proc.returncode != 0:
        log(f"\n⚠️ Child process for {wav_base} exited with {proc.returncode}", level="warning")
    else:
        log(f"\n✅ Finished {wav_base}; logged output to {log_path}")

    if globals().get('processed_dir'):
        # Build the destination folder under processed_dir/<SpeakerFolder>
        dest_folder = os.path.join(processed_dir, speaker_folder)
        os.makedirs(dest_folder, exist_ok=True)

        # The projection script creates a CSV next to the WAV by replacing .wav with .csv
        pair_csv = os.path.splitext(l2_wav)[0] + "_results.csv"
        csv_basename = os.path.basename(pair_csv)
        dest_csv = os.path.join(dest_folder, csv_basename)

        try:
            shutil.copy2(pair_csv, dest_csv)
            log(f"\nCopied {pair_csv} to {dest_csv}")
        except Exception as e:
            log(f"\n⚠️ Failed to copy {pair_csv} to {dest_csv}: {e}", level="warning")





def main():
    parser = argparse.ArgumentParser(description="Run full phonetic analysis pipeline")
    parser.add_argument('--base_data_path', '-b', required=True,
                        help="Root path of Phase_1_recordings_clean")
    parser.add_argument('--projection_script', '-p', required=True,
                        help="Path to projection_analysis.py")
    parser.add_argument('--output_csv', '-o', required=True,
                        help="Path to summary correlations CSV file")
    parser.add_argument('--log_dir', '-l', default='./logs',
                        help="Directory for the process log")
    parser.add_argument('--clean_logs', '-c',
                        action='store_true',
                        help='If set, delete existing logs before running')
    parser.add_argument('--processed_dir', '-cp',
                        help='If set, copy output CSV (and L2 results) into this folder')

    args = parser.parse_args()

    # === Set globals ===
    global log_dir, log_file, processed_dir

    # === Setup of clean logs ===
    log_dir = args.log_dir

    log_file = os.path.join(log_dir, "full_process.log")
    with open(log_file, "w"): pass  # clear old log
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    if args.clean_logs and os.path.exists(log_dir):
        log(f"\n🧹 Cleaning old logs in {log_dir}")
        # Only remove directories and their contents, leave files in log_dir intact
        for entry in os.listdir(log_dir):
            path = os.path.join(log_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)

    os.makedirs(log_dir, exist_ok=True)

    
    # === Setup variables ===
    base_data_path = args.base_data_path
    path_to_projection_analysis = args.projection_script
    path_to_correlations_csv = args.output_csv
    log(f"\n📂 Base data path: {base_data_path}"
        f"\n📂 projection analysis script: {path_to_projection_analysis}"
        f"\n📂 Output correlations CSV: {path_to_correlations_csv}"
        )
    l1_dirs = ["MidlandFemale", "MidlandMale"]
    
    # === Remove old summary CSV if exists ===
    if os.path.exists(path_to_correlations_csv):
        os.remove(path_to_correlations_csv)

    # === Setup processed_dir if requested ===
    processed_dir = args.processed_dir
    if processed_dir:
        # Clean only the contents of the folder, leave the folder itself
        if os.path.exists(processed_dir):
            for entry in os.listdir(processed_dir):
                path = os.path.join(processed_dir, entry)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        os.makedirs(processed_dir, exist_ok=True)
    
    # === Main processing ===
    for l1_dir in l1_dirs:
        log(f"\n📂 Scanning folder: {l1_dir}")
        l1_folder = os.path.join(base_data_path, l1_dir)
        is_female = "Female" in l1_dir
        gender = "Female" if is_female else "Male"


        # Get all valid wavs with TextGrid
        l1_sentences = [
            sentence for sentence in os.listdir(l1_folder)
            if sentence.endswith(".wav") and os.path.exists(os.path.join(l1_folder, sentence.replace(".wav", ".TextGrid")))
        ]
        log(f"\n🔢 Found {len(l1_sentences)} Midland{gender}.wav files in {l1_dir}")


        # Count .wav files missing .TextGrid
        l1_missing_textgrids = [
            f for f in os.listdir(l1_folder)
            if f.endswith(".wav") and not os.path.exists(os.path.join(l1_folder, f.replace(".wav", ".TextGrid")))
        ]
        log(f"\n⚠️ Found {len(l1_missing_textgrids)} Midland{gender}.wav files **missing** TextGrid in {l1_dir}")

        # Get all L2 folders excluding the current L1 one with the correct gender
        l2_dirs = [
            diractory for diractory in os.listdir(base_data_path)
                if os.path.isdir(os.path.join(base_data_path, diractory)) 
                and diractory != l1_dir
                and gender in diractory]
        

        for l2_dir in l2_dirs:
            l2_path = os.path.join(base_data_path, l2_dir)
            l2_wavs = [f for f in os.listdir(l2_path) if f.endswith(".wav")]
            l2_textgrids = [f for f in os.listdir(l2_path) if f.endswith(".TextGrid")]

            # 🔴 Entire folder missing WAVs
            if not l2_wavs:
                log(f"\n⛔ Folder {l2_dir} has no WAV files at all.")
                folder_level_wav_errors.add(l2_dir)
                continue

            # 🔴 Entire folder missing TextGrids
            if not l2_textgrids:
                log(f"\n⛔ Folder {l2_dir} has no TextGrids at all.")
                folder_level_textgrid_errors.add(l2_dir)
                continue 

            for l1_sentence in l1_sentences:
                l1_file_path = os.path.join(l1_folder, l1_sentence)
                l1_id = extract_id(l1_sentence)

                if not l1_id:
                    log(f"\n⚠️ Skipping {l1_sentence} — could not extract ID.")
                    continue

                for l2_file in l2_wavs:
                    if extract_id(l2_file) == l1_id:
                        full_l2_path = os.path.join(l2_path, l2_file)
                        textgrid_path = full_l2_path.replace(".wav", ".TextGrid")

                        if not os.path.exists(full_l2_path):
                            log(f"\n⛔ Skipping {l2_file} in {l2_dir} — missing WAV file")
                            file_level_wav_errors[l2_dir] += 1
                            continue

                        if not os.path.exists(textgrid_path):
                            log(f"\n⛔ Skipping {l2_file} in {l2_dir} — missing TextGrid")
                            file_level_textgrid_errors[l2_dir] += 1
                            continue

                        process_pair(l1_file_path, full_l2_path, l2_dir, path_to_projection_analysis, path_to_correlations_csv)

    # === Summary logs ===
    log("\n📊 Summary of successful comparisons by folder:")
    for folder, count in sorted(success_count_by_folder.items()):
        log(f"  {folder}: {count} successful comparisons")


    log("\n❌ Summary of missing TextGrids by folder:")
    for folder in sorted(folder_level_textgrid_errors):
        log(f"  {folder}: no TextGrid files at all")
    for folder, count in sorted(file_level_textgrid_errors.items()):
        log(f"  {folder}: {count} missing TextGrid files")
    total_missing_tg = len(folder_level_textgrid_errors)
    total_missing_tg += sum(file_level_textgrid_errors.values())
    log(f"🔴 Total missing TextGrids: {len(folder_level_textgrid_errors)} folders with none + {sum(file_level_textgrid_errors.values())} individual missing TextGrids")


    log("\n❌ Summary of missing WAV files by folder:")
    for folder in sorted(folder_level_wav_errors):
        log(f"  {folder}: no WAV files at all")
    for folder, count in sorted(file_level_wav_errors.items()):
        log(f"  {folder}: {count} missing WAV files")
    log(f"🔴 Total missing WAVs: {len(folder_level_wav_errors)} folders with none + {sum(file_level_wav_errors.values())} individual missing WAVs")


    # Sort the correlations file by sentence ID (L2 speaker)
    correlations_df = pd.read_csv(path_to_correlations_csv)
    correlations_df_sorted = correlations_df.sort_values(by="Sentence ID", ascending=True)
    correlations_df_sorted.to_csv(path_to_correlations_csv, index=False)
    log(f"\n✅ All processing complete! Correlations saved to {path_to_correlations_csv}")

    analyze_overall(path_to_correlations_csv)
    log("\n📂 All processing complete! Check the log files for details.")

    if processed_dir:
        # Copy the final CSV into processed_dir
        correlations_csv_dest = os.path.join(processed_dir, os.path.basename(path_to_correlations_csv))
        shutil.copy(path_to_correlations_csv, correlations_csv_dest)

        agg_correlations_csv = path_to_correlations_csv.rsplit('.', 1)[0] + '_overall_aggregated.csv'
        agg_correlations_csv_dest = os.path.join(processed_dir, os.path.basename(agg_correlations_csv))
        shutil.copy(agg_correlations_csv, agg_correlations_csv_dest)

        log(f"\nCopied CSVs to processed_dir: {correlations_csv_dest, agg_correlations_csv_dest}")

if __name__ == "__main__":
    main()       