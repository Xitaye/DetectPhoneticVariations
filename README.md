# Detecting Phonetic Variations in Accented Speech Using HuBERT

This repository contains the code and pipeline developed as part of a final-year research project at the Technion. The project investigates whether HuBERT's internal representations can detect and quantify phonetic variation between native (L1) and non-native (L2) speakers more effectively than phoneme transcriptions.

---

## 📊 Full Pipeline Overview

The data consists of `.wav` audio files of different speakers—both native American English (L1) and various non-native (L2) accents—reading the same English sentences. Accompanying these are phoneme transcriptions stored in Excel files, which annotate each sentence’s phonetic content. This parallel data allows direct comparison of phonetic variation across accents.

The pipeline aligns phonemes with speech using MFA, extracts HuBERT embeddings, calculates distance projections using DTW, and evaluates the correlation with phonological similarity.

Each component of the process is modular and documented below.

---

## 1. Installation

- Clone the repository:
```bash
git clone https://github.com/Xitaye/DetectPhoneticVariations.git
cd DetectPhoneticVariations
```

- Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # Windows: .\env\Scripts\activate
pip install -r requirements.txt
```

> If `requirements.txt` is not present, manually install:
```bash
pip install torch torchaudio numpy scipy pandas plotly librosa scikit-learn tgt
```

---

## 2. Step-by-Step Usage Instructions

### Silence Removal (optional but recommended)

Use SoX to trim silences and normalize WAV files.

```bash
python Scripts/Silence_Remover.py
```

- **Input**: raw `.wav` files in speaker folders
- **Output**: cleaned `.wav` files in a parallel folder structure

---

### Transcription Conversion

Convert IPA transcriptions (from Excel) into MFA-compatible `.txt` files.

```bash
python Scripts/Transcriptions_script.py
```

- **Input**: Excel CSV + IPA-to-feature mapping (`ipa2spe.csv`)
- **Output**: `.txt` phoneme files per sentence

---

### Forced Alignment using MFA

Align WAV + TXT pairs with Montreal Forced Aligner to generate phoneme-aligned `.TextGrid` files.

```bash
python Scripts/MFA_Alignment_script.py
```

- Ensure you have a working MFA installation.
- Input and output formats must match MFA guidelines.

---

### Sanity Check for MFA Outputs

Identify any phoneme alignment issues (e.g., `spn`).

```bash
python Scripts/Check_for_spn_script.py
```

- **Input**: `.TextGrid` files
- **Output**: Printed report of problematic phonemes

---

### Run Full Trajectory Analysis Pipeline

This script extracts HuBERT features, calculates DTW, projects distances, aligns phonemes, and correlates phonological similarity.

```bash
python Scripts/Trajectory_analysis_for_all_scripts.py path/to/L1.wav path/to/L2.wav output_results.csv
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

---

## 3. Output Files

- `S2_results.csv`: detailed phoneme-by-phoneme similarity with projected cost
- `means.csv`: summary of Pearson/Spearman correlations
- Optional: `.html` or interactive 3D TSNE plots (if enabled)

---

## 4. Feature Vectors and Mapping

HuBERT embeddings are aligned against IPA-based phonological vectors from `ipa2spe.csv`. You can customize this file if using another phoneme set.

---

## 5. Project Structure

```
Scripts/
├── Silence_Remover.py               # Trims silence using SoX
├── Transcriptions_script.py        # Converts IPA to MFA transcription format
├── MFA_Alignment_script.py         # Batch alignment with MFA
├── Check_for_spn_script.py         # Finds unaligned phonemes (spn)
├── Trajectory_analysis_for_all_scripts.py   # Main pipeline: features, DTW, correlation
├── Analyze_correlations.py         # (optional) aggregates and visualizes correlation summaries
```

---

## 6. Reproducing the Results

The full methodology and interpretation are described in `FinalReportV2.docx`. To replicate a typical experiment:

```bash
python Scripts/Trajectory_analysis_for_all_scripts.py     data/L1/E1M-HINT-06-04.wav     data/L2/S2M-HINT-06-04.wav     results/means.csv
```

---

## 7. Plotting and Visualization Guide

The project includes several plot types to visualize phonetic variation and HuBERT's internal representations.

### How to Enable Plots

In `Trajectory_analysis_for_all_scripts.py`, activate visualizations by setting the following flags:

```python
run_projection = True            # Enables distance signature projections
plot_results = True              # Enables cosine vs. cost plots
plot_graph = True                # Enables 3D t-SNE visualization
run_peak_analyze = True          # Enables peak mismatch analysis
run_our_WP = True                # Visualizes DTW cost matrix and warping path
```

---

### 📈 Plot Types and What They Show

#### 1. DTW Cost Matrix with Warping Path
- Displays the accumulated DTW cost between HuBERT features of L1 and L2 speakers.
- Overlays the optimal alignment path.
- Useful for understanding alignment quality and trajectory structure.

#### 2. Projected Distance Signatures (Cost Projection per Frame)

Once the DTW warping path is calculated between the two speaker sequences, the accumulated cost is decomposed and projected back onto each speaker’s timeline to form a “distance signature.”

- Enabled by: `run_projection = True`

#### 3. Cosine Similarity vs. Distance Signature
- Aligns phoneme identity with DTW-derived distances.
- Compares acoustic distance with phonological similarity (cosine between phoneme vectors).
- Useful for verifying if HuBERT captures linguistically meaningful differences.

- Enabled by: `plot_results = True`

#### 4. Peak Phoneme Analysis
- Detects sharp spikes in the distance signature.
- Maps peaks to phonemes and compares L1–L2 pronunciation.
- Classifies mismatches and saves a detailed CSV report.

- Enabled by: `run_peak_analyze = True`

#### 5. 3D Trajectory Plot (TSNE)
- Visualizes HuBERT features of both speakers using t-SNE dimensionality reduction.
- Displays time-embedded speech trajectories in 3D.
- Helps explore structural alignment between speaker sequences.

- Enabled by: `plot_graph = True` and `apply_TSNE = True`

---

### 💡 Exporting Plots

All visualizations are created using Plotly. To export:
```python
fig.write_html("filename.html")           # Interactive HTML plot
# or
plotly.io.write_image("filename.png")     # Static image (requires kaleido)
```
### 💡 Tip

Most plots are generated with `plotly`, which opens them interactively in your browser. To export them:
```python
fig.write_html("output.html")
```
Or save static versions using `plotly.io.write_image()`.
## 8. Credits

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
