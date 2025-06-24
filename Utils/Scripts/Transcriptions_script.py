#!/usr/bin/env python3
"""
Process a phonetic CSV to produce .txt files under language+gender directories
in a given output root. Deletes existing .txt files recursively before writing new ones.

Usage:
    python process_phonetic_csv.py --csv "/path/to/Phonetic Transcription.csv" \
                                   --out_root "/desired/output/base/path"
"""

import os
import csv
import re
import argparse
import sys

# Diphthong replacements only (after removing the tie bar)
diphthong_replacements = {
    "aɪ": "aj", "ai": "aj",
    "aʊ": "aw", "au": "aw",
    "eɪ": "ej", "ei": "ej",
    "oʊ": "ow", "ou": "ow"
}
# Compile regex to match diphthongs, longest first
diphthong_pattern = re.compile(
    "|".join(sorted(map(re.escape, diphthong_replacements.keys()), key=len, reverse=True))
)

def format_sentence(sentence):
    """
    1. Remove tie-bar "͡"
    2. Replace diphthongs per diphthong_replacements
    3. Insert spaces between each resulting character or replaced unit.
    """
    # Step 1: remove the tie bar
    sentence = sentence.replace("͡", "")

    # Step 2: replace diphthongs
    result = []
    i = 0
    while i < len(sentence):
        match = diphthong_pattern.match(sentence, i)
        if match:
            rep = diphthong_replacements[match.group()]
            result.append(rep)
            i += len(match.group())
        else:
            result.append(sentence[i])
            i += 1

    # Step 3: add spaces between phonemes/tokens
    return " ".join(result)

def gather_target_dirs_from_csv(csv_file):
    """
    Read CSV once to collect all target directory names (language+gender).
    Returns a set of directory names (strings).
    """
    target_dirs = set()
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            language = row.get('Language', '').strip()
            speaker = row.get('Speaker', '').strip()
            if not language or not speaker:
                continue
            gender = 'Female' if speaker[-1].lower() == 'f' else 'Male'
            dir_name = f"{language}{gender}"
            target_dirs.add(dir_name)
    return target_dirs

def delete_existing_txt_files_recursively(base_root, target_dirs):
    """
    For each target directory name in target_dirs (relative names like "SpanishFemale"),
    walk recursively under base_root/<dir> and delete any .txt files found.
    """
    for rel in target_dirs:
        full_dir = os.path.join(base_root, rel)
        if not os.path.exists(full_dir):
            continue
        for dirpath, _, filenames in os.walk(full_dir):
            for fname in filenames:
                if fname.lower().endswith('.txt'):
                    path = os.path.join(dirpath, fname)
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"⚠️ Failed to remove {path}: {e}", file=sys.stderr)

def process_csv_and_write_txt(csv_file, out_root):
    """
    Read CSV again, and for each line create the formatted .txt under out_root/<language+gender>/filename.txt
    """
    # Combinations requiring "__leveled" suffix
    leveled_languages = {
        ("spanish", "Female"), ("spanish", "Male"),
        ("midland", "Male"), ("mandarin", "Male"),
        ("japanese", "Male"), ("german", "Male"),
        ("french", "Female"), ("french", "Male"),
        ("british", "Male")
    }
    # Special naming overrides: mapping (language.lower(), gender) to a function of sentence_number
    special_naming = {
        ("scottish", "Female"): lambda sn: f"SC_{'_'.join(f'{int(p):02}' for p in sn.split('_'))}.txt",
        ("german", "Female"):   lambda sn: f"G_{'_'.join(f'{int(p):02}' for p in sn.split('_'))}.txt",
        ("mandarin", "Female"): lambda sn: f"C_{'_'.join(f'{int(p):02}' for p in sn.split('_'))}.txt"
    }

    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            language = row.get('Language', '').strip()
            speaker = row.get('Speaker', '').strip()
            sentence_number = row.get('Sentence number', '').strip()
            sentence = row.get('Sentence', '').strip()
            if not language or not speaker or not sentence_number:
                continue

            gender = 'Female' if speaker[-1].lower() == 'f' else 'Male'
            dir_name = f"{language}{gender}"
            target_dir = os.path.join(out_root, dir_name)
            os.makedirs(target_dir, exist_ok=True)

            key = (language.lower(), gender)
            # Determine file_name
            if key in special_naming:
                try:
                    file_name = special_naming[key](sentence_number)
                except Exception:
                    # fallback if sentence_number format unexpected
                    file_name = f"{speaker}-HINT-{sentence_number}.txt"
            else:
                # generic: ensure two digits per segment
                parts = sentence_number.split('_')
                try:
                    parts2 = [f"{int(p):02}" for p in parts]
                except Exception:
                    parts2 = parts
                joined = '-'.join(parts2)
                leveled_suffix = "__leveled" if (language.lower(), gender) in leveled_languages else ""
                file_name = f"{speaker}-HINT-{joined}{leveled_suffix}.txt"

            full_path = os.path.join(target_dir, file_name)
            formatted = format_sentence(sentence)
            try:
                with open(full_path, mode='w', encoding='utf-8') as outf:
                    outf.write(formatted)
            except Exception as e:
                print(f"⚠️ Failed to write {full_path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Process a phonetic CSV to produce formatted .txt files under language+gender dirs."
    )
    parser.add_argument(
        '--csv', '-c',
        required=True,
        help="Path to the CSV file (e.g. '/path/to/Phonetic Transcription.csv')."
    )
    parser.add_argument(
        '--out_root', '-o',
        required=True,
        help="Base output directory under which <Language><Gender> subfolders will be created."
    )
    args = parser.parse_args()

    csv_file = args.csv
    out_root = args.out_root

    if not os.path.isfile(csv_file):
        print(f"❌ CSV file not found: {csv_file}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_root, exist_ok=True)

    # First pass: collect target dirs
    target_dirs = gather_target_dirs_from_csv(csv_file)

    # Delete existing .txt files recursively under each target dir
    delete_existing_txt_files_recursively(out_root, target_dirs)

    # Second pass: write new .txt files
    process_csv_and_write_txt(csv_file, out_root)

    print("✅ Processing complete. Files have been created under:", out_root)

if __name__ == '__main__':
    main()
