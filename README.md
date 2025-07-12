# Detecting Phonetic Variations in Accented Speech Using HuBERT

This repository contains the code and pipeline developed as part of a final-year research project at the Technion. The project investigates whether HuBERT's internal representations can detect and quantify phonetic variation between native (L1) and non-native (L2) speakers more effectively than phoneme transcriptions.

---

## Full Pipeline Overview

The data consists of `.wav` audio files of different speakers—both native American English (L1) and various non-native (L2) accents-reading the same English sentences. Accompanying these are phoneme transcriptions stored in Excel files, which annotate each sentence’s phonetic content. This parallel data allows direct comparison of phonetic variation across accents.

The pipeline aligns phonemes with speech using MFA, extracts HuBERT embeddings, calculates distance projections using DTW, and evaluates the correlation with phonological similarity.

Each component of the process is modular and documented below.

---

## 1. Installation

- Follow the installation instructions of fairseq:  <br>
https://github.com/facebookresearch/fairseq/tree/main#requirements-and-installation

- Go back to the DetectPhoneticVariations repo

- Clone the repository:
    ```bash
    git clone https://github.com/Xitaye/DetectPhoneticVariations.git
    cd DetectPhoneticVariations
    ```

- Set up your environment:
    
    ### Recommended: Conda 
    ```bash
    conda env create -n phon_var -f environment.yml
    conda activate phon_var
    ```

    ### Or: pip + venv 
    ```bash
    python -m venv env
    source env/bin/activate  # Windows: .\env\Scripts\activate
    pip install -r requirements.txt
    ```

- **External tools** (must be on your `PATH`): <br>
    >    • [SoX](http://sox.sourceforge.net/) 
    >    • [MFA ≥ 3.2](https://montreal-forced-aligner.readthedocs.io)

---

## 2. Data preparation

- ### Silence Removal (optional but recommended)

    The following script use SoX to trim silences and normalize WAV files.

    ```bash
    python scripts/tools/silence_remover.py --input_dir raw_wavs --output_dir clean_wavs
    ```

    - **Input**: raw `.wav` files arranged in speaker subfolders under `--input_dir`
    - **Output**: trimmed `.wav` files in the same relative subfolder structure under `--output_dir`

    ---

    - ### Transcription Conversion

    Convert IPA transcriptions (from Excel) into MFA-compatible `.txt` files.

    ```bash
    python scripts/tools/transcriptions_script.py --csv phonetic_transcription --out_root output_root 
    ```
    - **Input**: `.csv` file containing phonetic transcriptions with columns like Language, Speaker, Sentence number, and Sentence.
    - **Output**: `.txt` phoneme files per sentence Organizes

    Process a phonetic `.csv` to produce .txt files under language+gender directories
    in a given output root. <br>
    **Deletes existing `.txt` files recursively before writing new ones.** 
    
    ---

 - ### Forced Alignment using MFA

    Align WAV+TXT pairs with Montreal Forced Aligner to generate phoneme-aligned `.TextGrid` files.

    ```bash
    python scripts/tools/mfa_alignment.py --base_data_path clean_data_root --dictionary ipa_phone_mapping --acoustic_model english_mfa
    ```

    - **Input**: 
        - `base_data_path`: Root directory containing language+gender subfolders (e.g. SpanishFemale/,  GermanMale/, …). Each subfolder must hold matching `.wav` and `.txt` files (one TXT per sentence).
        - `dictionary`: Pronunciation dictionary (e.g. `ipa_phone_mapping`)
        - `acoustic_model`: Acoustic model name as required by MFA. (e.g. `english_mfa`)
    - **Output**: new `.TextGrid` files, one per WAV+TXT pair are generated and saved alongside the originals in their respective subfolder.

    Process a folder of WAV+TXT pairs to produce `.TextGrid` files under each language+gender subdirectory in `base_data_path`. <br>
    **Deletes any existing `.TextGrid` files recursively before writing new ones.**

    ---

 - ### Sanity Check for MFA Outputs

    Scan `.TextGrid` files to identify any phoneme alignment issues (e.g. `spn` phoneme intervals) and extract the overlapping words into a summary file.

    ```bash
    python scripts/tools/check_for_spn.py --folder TextGrid_directory_root
    ```

    - **Input**: root directory containing `.TextGrid` files
    - **Output**: printed report of problematic phonemes

    ---

## 3. Run Full Phonetic Analysis Pipeline

Run the end-to-end phonetic comparison workflow across all speakers to generate per-pair logs and a summary correlations CSV. <br>
This script extracts HuBERT features, calculates DTW, projects distances, aligns phonemes, and correlates phonological similarity.

```bash
python run.py --base_data_path prepared_data_root --projection_script scripts/workflows/projection_analysis.py --output_csv correlations_summary.csv --log_dir logs [--clean_logs] [--processed_dir processed_results]
```
**Arguments**:
- `--base_data_path`: root directory with language+gender subfolders (e.g. MidlandFemale/, SpanishMale/), each containing matching .wav & .TextGrid files.
- `--projection_script`: path to `projection_analysis.py` which computes per-pair feature projections and writes `<wav_name>_results.csv`.
- `--output_csv`: target CSV file for aggregated correlations **(overwrites if existing)**.
- `--log_dir`: directory where `full_process.log` and per-pair `<wav_name>.log` files will be saved.
- `--clean_logs` (optional): if set, deletes any existing logs under `--log_dir` before running.
- `--processed_dir` (optional): folder to copy the final correlations_summary.csv (and its aggregated variant) plus all individual `<wav_name>_results.csv` for easy access.

**What It Does**:
- (Optionally) cleans out old logs.
- Scans the L1 folders (MidlandFemale, MidlandMale) for WAV–TextGrid pairs and reports any mismatches.
- For each matching L1–L2 file pair, runs the projection script, logging stdout/stderr to `log_dir/<speaker>/<wav>.log`.
- Tracks success and missing-file counts by language/gender.
- Merges all per-pair CSV outputs into `--output_csv`, sorted by sentence ID, then runs the overall aggregation step.
- If `--processed_dir` is provided, copies all result CSVs there.
- Prints a summary of completed analyses and the locations of outputs.

**Output Files:**

- **Logs** (in `--log_dir`):  
  - `full_process.log` — end-to-end pipeline log with timestamps  
  - `<LanguageGender>/<wav_basename>.log` — stdout/stderr for each L1–L2 pair

- **Correlations Summary** (in `--output_csv`):  
  - `correlations_summary.csv` — per-pair Pearson & Spearman correlations  
  - `correlations_summary_overall_aggregated.csv` — aggregated metrics across all pairs  

- **Per-pair Results**:  
  - `<wav_basename>_results.csv` in each `<LanguageGender>` subfolder (frame- and phoneme-level projections & similarity)  

- **Processed Directory** (`--processed_dir`, optional):  
  - Copies of `correlations_summary.csv`, `*_overall_aggregated.csv`, and all `<wav_basename>_results.csv` for easy access

### Projection Analysis Script - The Core 
Run the core feature‐extraction and similarity‐analysis on one native/non-native WAV pair to produce a per‐pair results CSV.

```bash
python scripts/workflows/projection_analysis.py path/to/L1.wav path/to/L2.wav output_results.csv
```

**Arguments**:
- `path/to/L1.wav`: cleaned native speaker audio
- `path/to/L2.wav`: cleaned non-native speaker audio
- `output_results.csv`: target CSV to store summary correlations


**What It Does**:
- Loads WAV files, aligns with corresponding `.TextGrid`
- Extracts HuBERT embeddings (layer 12)
- Computes DTW warping path
- Projects distance signature per speaker
- Aligns with phonemes via `TextGrid` + `ipa2spe.csv`
- Calculates cosine similarity between phoneme vectors
- Outputs Pearson/Spearman correlations and saves results

> Set flags inside the script to enable/disable plots and outputs.

**Feature Vectors and Mapping** <br>
HuBERT embeddings are aligned against IPA-based phonological vectors from `ipa2spe.csv`. You can customize this file if using another phoneme set.

---

## 4. Project Structure

```
scripts/
├── tools/
│   ├── silence_remover.py         # Trims silence using SoX
│   ├── transcriptions_script.py   # Converts phonetic CSV into MFA‐compatible .txt files
│   ├── mfa_alignment.py           # Batch alignment with MFA to produce .TextGrid files
│   └── check_for_spn.py           # Scans TextGrids for “spn” intervals and extracts the words
│
├── workflows/
│   ├── projection_analysis.py     # Extracts HuBERT features, computes DTW & projections, and measures similarity
│   └── statistics_analysis.py     # Aggregates per‐pair correlations
│
└── run.py                        # Full pipeline runner: orchestrates tools/workflows end to end
```

### Before you start, arrange your raw data like this:

```
data/
└── config/...
├── raw_audio/
│   ├── MidlandFemale/ # L1 reference WAVs
│   ├── MidlandMale/
│   ├── SpanishFemale/ # L2 comparison WAVs
│   ├── SpanishMale/
│   └── … # other L2 subfolders
└── Phonetic_Transcription.csv # IPA transcriptions of all the sentences
```

- **`data/raw_audio/<Language><Gender>/`** subfolders contain your raw `.wav` files (one per sentence).  
- **`Phonetic_Transcription.csv`** must have columns: `Language`, `Speaker`, `Sentence number`, `Sentence`.  
- **`ipa2spe.csv`** defines phonological feature vectors for each IPA symbol.


---

## 5. Reproducing the Results

The full methodology and interpretation are described in `FinalReport - Reut and Itay.pdf`. To replicate a typical experiment:

```bash
python scripts/workflows/projection_analysis.py  data/L1/E1M-HINT-06-04.wav data/L2/S2M-HINT-06-04.wav S2M-HINT-06-04.wav results/E1M-S2M_results.csv
```

---

## 6. Plotting and Visualization Guide

The project includes several plot types to visualize phonetic variation and HuBERT's internal representations.

### How to Enable Plots

In `scripts/workflows/projection_analysis.py`, activate visualizations by setting the following flags:

```python
plot_DTW_cost_matrix = True     # Show heatmap of accumulated DTW cost + warping path
plot_projections     = True     # Line plots of distance signature projection per frame for each speaker
plot_results         = True     # Overlay of L1/L2 projected costs with DTW alignment links; hover shows phoneme & cosine similarity
```

---

### Plot Types and What They Show

#### 1. DTW Cost Matrix with Warping Path
- **Flag:** `plot_DTW_cost_matrix`

- **Shows:** Heatmap of the accumulated DTW cost between L1 & L2 embeddings, with the optimal warping path overlaid.

- **Use:** Inspect alignment quality and see where the algorithm matches frames.

#### 2. Projected Distance Signatures
- **Flag:** plot_projections

- **Shows:** Line plots of the per-frame “distance signature” for each speaker, derived from the DTW path.

- **Use:** Identify time regions where the two speakers diverge acoustically.

#### 3. Cosine Similarity vs. Distance Signature
- **Flag:** plot_results

- **Shows:** Overlaid line-and-marker plots of projected costs for L1 and L2 connected by semi-transparent DTW alignment links, with hover labels displaying phoneme identities and their cosine-similarity scores.

- **Use:** It lets you visualize how acoustic distance trajectories align across speakers and examine the relationship between projected cost and phonemic similarity.

### Tip

All plots are generated with `plotly`, which opens them interactively in your browser. To export them:

```python
fig.write_html("plot.html")           # Interactive HTML plot
# or
plotly.io.write_image("plot.png")     # Static image (requires `kaleido`)
```

## 7. Credits

This project is based on concepts from:
- [HuBERT: Hsu et al., 2021](https://arxiv.org/abs/2106.07447)
- [Chernyak et al., 2024](https://doi.org/10.1121/10.0026358)

It was completed as part of a final-year research seminar at the Technion.

---

## 👩‍🔬 Authors

- Reut Vitzner  
- Itay Efrat  
- Supervisor: Prof. Joseph Keshet  
- Technion, Winter 2025  

---

## License

For academic and non-commercial use only.
