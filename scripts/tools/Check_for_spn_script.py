import os
from textgrid import TextGrid

# === CONFIGURATION ===
name = "/SpanishMale"
input_folder = "/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings" + name
#input_folder = "/home/itayefrat/MFA_Test/Testing19-05"                                      # <-- Change this to your folder
output_file = input_folder+"/spn_words_output.txt"                                          # Output file (will be overwritten first)


# === MAIN LOGIC ===
def extract_spn_words_from_textgrid(filepath):
    tg = TextGrid.fromFile(filepath)

    # Try to identify tiers (assume 2-tier structure: phonemes and words)
    phoneme_tier = None
    word_tier = None

    for tier in tg.tiers:
        name = tier.name.lower()
        if "phoneme" in name or "phone" in name:
            phoneme_tier = tier
        elif "word" in name:
            word_tier = tier

    # Fallback: assume 2 tiers if names aren't helpful
    if not phoneme_tier or not word_tier:
        if len(tg.tiers) >= 2:
            phoneme_tier = tg.tiers[1]
            word_tier = tg.tiers[0]
        else:
            print(f"[!] Skipping {filepath} - insufficient tiers")
            return []

    matched = []

    for phoneme_interval in phoneme_tier.intervals:
        if phoneme_interval.mark.strip().lower() == "spn":
            spn_start = phoneme_interval.minTime
            spn_end = phoneme_interval.maxTime

            for word_interval in word_tier.intervals:
                word = word_interval.mark.strip()
                if not word:
                    continue

                # Check if overlapping
                if (word_interval.minTime <= spn_start < word_interval.maxTime) or \
                   (word_interval.minTime < spn_end <= word_interval.maxTime) or \
                   (spn_start <= word_interval.minTime and spn_end >= word_interval.maxTime):
                    matched.append((os.path.basename(filepath), word))
                    break

    return matched

# Clear the output file
with open(output_file, "w") as f:
    f.write("")

# Walk through files
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".TextGrid") or file.endswith(".textgrid"):
            full_path = os.path.join(root, file)
            print(f"Processing: {full_path}")
            results = extract_spn_words_from_textgrid(full_path)

            with open(output_file, "a") as out_f:
                for filename, word in results:
                    out_f.write(f"{filename}\t{word}\n")  # tab-separated

print(f"\nDone! Output written to: {output_file}")