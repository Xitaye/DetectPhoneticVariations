import os
import re
import logging
import pandas as pd
import subprocess  # Or replace with your own function

# === Debug ===
from collections import defaultdict
success_count_by_folder = defaultdict(int)
folder_level_textgrid_errors = set()
file_level_textgrid_errors = defaultdict(int)
folder_level_wav_errors = set()
file_level_wav_errors = defaultdict(int)


# === Setup logging ===
log_dir = "/home/itayefrat/Util"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "Run_all_log.txt")

# Clear previous log before initializing logging
with open(log_file, "w"):
    pass

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log(msg):
    print(msg)
    logging.info(msg)


# === Setup variables ===
base_path = "/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings_clean/"
midland_dirs = ["MidlandFemale", "MidlandMale"]
path_to_statistics_csv = "/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings_clean/Statistics_results.csv"
# Check if the means CSV exists
if os.path.exists(path_to_statistics_csv):
    os.remove(path_to_statistics_csv)


# === Utility function ===
def extract_id(filename):
    match = re.search(r'(\d{2})[-_](\d{2})', filename)
    return match.group(0) if match else None

def process_pair(midland_wav, other_wav, other_folder):
    log(f"\n✅ Processing pair:\n      - Midland: {midland_wav}\n      - Other:   {other_wav}")
    success_count_by_folder[other_folder] += 1

    Script_path = "/home/itayefrat/Roni_percept_sim/percept_sim/trajectory_analysis_4_HF_Changes.py"
    subprocess.run(["python", Script_path, midland_wav, other_wav, path_to_statistics_csv])


# === Main processing ===
for midland_dir in midland_dirs:
    log(f"\n📂 Scanning folder: {midland_dir}")
    midland_folder = os.path.join(base_path, midland_dir)
    is_female = "Female" in midland_dir
    gender = "Female" if is_female else "Male"


    # Get all valid wavs with TextGrid
    midland_sentences = [
        sentence for sentence in os.listdir(midland_folder)
        if sentence.endswith(".wav") and os.path.exists(os.path.join(midland_folder, sentence.replace(".wav", ".TextGrid")))
    ]
    log(f"🔢 Found {len(midland_sentences)} Midland{gender}.wav files in {midland_dir}")


    # Count .wav files missing .TextGrid
    midland_missing_textgrids = [
        f for f in os.listdir(midland_folder)
        if f.endswith(".wav") and not os.path.exists(os.path.join(midland_folder, f.replace(".wav", ".TextGrid")))
    ]
    log(f"⚠️ Found {len(midland_missing_textgrids)} Midland{gender}.wav files **missing** TextGrid in {midland_dir}")

    # Get all other folders excluding the current Midland one with the correct gender
    other_dirs = [
        diractory for diractory in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, diractory)) 
            and diractory != midland_dir
            and gender in diractory]
    

    for other_dir in other_dirs:
        other_path = os.path.join(base_path, other_dir)
        other_wavs = [f for f in os.listdir(other_path) if f.endswith(".wav")]
        other_textgrids = [f for f in os.listdir(other_path) if f.endswith(".TextGrid")]

        # 🔴 Entire folder missing WAVs
        if not other_wavs:
            log(f"\n⛔ Folder {other_dir} has no WAV files at all.")
            folder_level_wav_errors.add(other_dir)
            continue

        # 🔴 Entire folder missing TextGrids
        if not other_textgrids:
            log(f"\n⛔ Folder {other_dir} has no TextGrids at all.")
            folder_level_textgrid_errors.add(other_dir)
            continue 

        for midland_sentence in midland_sentences:
            midland_file_path = os.path.join(midland_folder, midland_sentence)
            midland_id = extract_id(midland_sentence)

            if not midland_id:
                log(f"\n⚠️ Skipping {midland_sentence} — could not extract ID.")
                continue

            for other_file in other_wavs:
                if extract_id(other_file) == midland_id:
                    full_other_path = os.path.join(other_path, other_file)
                    textgrid_path = full_other_path.replace(".wav", ".TextGrid")

                    if not os.path.exists(full_other_path):
                        log(f"\n⛔ Skipping {other_file} in {other_dir} — missing WAV file")
                        file_level_wav_errors[other_dir] += 1
                        continue

                    if not os.path.exists(textgrid_path):
                        log(f"\n⛔ Skipping {other_file} in {other_dir} — missing TextGrid")
                        file_level_textgrid_errors[other_dir] += 1
                        continue

                    process_pair(midland_file_path, full_other_path, other_dir)

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


# Sort the Means.csv file by sentence ID (L2 speaker)
df = pd.read_csv(path_to_statistics_csv)
df_sorted = df.sort_values(by="Sentence ID", ascending=True)
df_sorted.to_csv(path_to_statistics_csv, index=False)
log(f"\n✅ All processing complete! Means saved to {path_to_statistics_csv}")

for col in df_sorted.columns:
    if col != "Speaker" and col != "Sentence ID":
        log(f"Column: {col} - Mean: {df_sorted[col].mean()}")
        print(f"Column: {col} - Mean: {df_sorted[col].mean()}")
log("\n📂 All processing complete! Check the log file for details.")
            
