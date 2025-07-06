import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d

import torch

from transformers import HubertModel
import torchaudio

from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
import plotly.graph_objs as go

import tgt
import sys

from statistics_analysis import correlation_analysis, save_correlations


def mut_normalize_sequences(sq1, sq2, normalize: bool):
    """
    Normalize the sequences together by z-scoring each dimension.
    sq1: numpy array of shape (t1, d)
    sq2: numpy array of shape (t2, d)
    normalize: if True, normalize the sequences together
    """
    if normalize:
        sq1 = np.copy(sq1)
        sq2 = np.copy(sq2)
        len_sq1 = sq1.shape[0]

        arr = np.concatenate((sq1, sq2), axis=0)
        for dim in range(sq1.shape[1]):
            arr[:, dim] = zscore(arr[:, dim])
        sq1 = arr[:len_sq1, :]
        sq2 = arr[len_sq1:, :]
    return sq1, sq2

def time_txt(time, time_frame=5):
    """
    Convert time in seconds to a text representation.
    If time is a multiple of time_frame, return the time in seconds rounded to 2 decimal places.
    Otherwise, return an empty string.
    """
    if time % time_frame == 0:
        return f"{round(time * 0.02, 2)}"
    return ""

def create_hubert_df(feats, speaker_len, names):
    """ 
    Create a DataFrame from the features and speaker lengths.
    feats: numpy array of shape (n_samples, n_features)
    speaker_len: list of lengths of each speaker's features
    names: list of speaker names
    """
    cols = [f"val {i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=cols)
    df['idx'] = df.index
    time_index = {i: speaker_len[i] for i in range(len(speaker_len))}
    com_time_index = {i: sum(speaker_len[:i]) for i in range(len(speaker_len))}
    df_speaker_count = pd.Series(time_index)
    df_speaker_count = df_speaker_count.reindex(df_speaker_count.index.repeat(df_speaker_count.to_numpy())).rename_axis(
        'speaker_id').reset_index()
    df['speaker_id'] = df_speaker_count['speaker_id']
    df['speaker_len'] = df['speaker_id'].apply(lambda row: speaker_len[row])
    df['com_sum'] = df['speaker_id'].apply(lambda i: com_time_index[i])
    df['speaker'] = df['speaker_id'].apply(lambda i: names[i])
    df['time'] = df['idx'] - df['com_sum']
    df['time_txt'] = df[['time', 'speaker_len']].apply(lambda row: time_txt(row['time'], time_frame), axis=1)
    assert len(df.loc[df['speaker'] == -1]) == 0
    assert len(df_speaker_count) == len(df)
    df_subset = df.copy()
    data_subset = df_subset[cols].values
    return data_subset, df_subset, cols

def calculate_dtw_path(df_subset, speaker1, speaker2, cols):
    """ 
    Calculate DTW path between two speakers' features.
    df_subset: DataFrame containing the features and speaker information
    speaker1: name of the first speaker
    speaker2: name of the second speaker
    cols: list of feature columns to use for DTW
    """
    features_speaker1 = df_subset[df_subset['speaker'] == speaker1][cols].to_numpy()
    features_speaker2 = df_subset[df_subset['speaker'] == speaker2][cols].to_numpy()

    features_speaker1, features_speaker2 = mut_normalize_sequences(features_speaker1, features_speaker2, True)  
    optimal_distances, warping_path = lib_dtw(features_speaker1.transpose(), features_speaker2.transpose(), backtrack=True)   
    out_wp_cols = [speaker1+" index", "frame_range","frame_range_ms", 
                   speaker2+" index", 'frame_range',"frame_range_ms", "cost"]  
    samples_per_chunk = 0.02*16000  
    out_wp = []    
    for i, j in warping_path[::-1]:   
        match_cost = optimal_distances[i, j]/(len(features_speaker1)+len(features_speaker2))    
        out_wp.append((i, (i*samples_per_chunk, i*samples_per_chunk+samples_per_chunk), (i*0.02, i*0.02+0.02),
                       j, (j*samples_per_chunk, j*samples_per_chunk+samples_per_chunk), (j*0.02, j*0.02+0.02),
                       match_cost))

    df = pd.DataFrame(out_wp, columns=out_wp_cols)
    return df , optimal_distances, warping_path

def wp_projection_calc(speaker_frames, costs):
    """ 
    Calculate the projection of the warping path for a speaker.
    speaker_frames: list of frames for the speaker
    costs: list of costs for each frame
    """
    updated_costs = np.zeros_like(costs)
    updated_costs[0] = costs[0]
    for frame in range(len(speaker_frames)-1):
        updated_costs[frame+1] = costs[frame+1] - costs[frame]
    frame_cost_map = {}
    
    # Accumulate costs and count occurrences
    for frame, cost in zip(speaker_frames, updated_costs):
        if frame not in frame_cost_map:
            frame_cost_map[frame] = {'total_cost': 0, 'count': 0}
        frame_cost_map[frame]['total_cost'] += cost
        frame_cost_map[frame]['count'] += 1
    
    # Calculate the average cost for each frame
    average_costs = [[frame, data['total_cost'] / data['count']] for frame, data in frame_cost_map.items()]
    sum_costs = [[frame, data['total_cost']] for frame, data in frame_cost_map.items()]
    
    # Convert to a numpy array for easy manipulation
    return np.array(sum_costs)

def wp_projection_plot(projection_costs, speaker, max_cost):
    """
    Create a scatter plot of the projected costs for a speaker.
    projection_costs: numpy array of shape (n_frames, 2) where the first column is the frame index and the second column is the average cost.
    speaker: name of the speaker
    max_cost: maximum cost to set the y-axis range
    """
    # Create the scatter plot
    scatter = go.Scatter(
        x=projection_costs[:, 0], 
        y=projection_costs[:, 1],
        mode='lines',
        marker=dict(size=10, color='blue'),
        name='Average Cost'
    )

    # Create the figure
    fig = go.Figure(data=[scatter])

    # Update layout
    fig.update_layout(
        title='Average Cost Per Frame Of '+speaker,
        xaxis_title='Frame Index',
        yaxis=dict(title='Average Cost', range=[0, max_cost]),  # Set y-axis range
        template='plotly_white'
    )

    # Display the plot
    fig.show()

def textgrid2intervals(textgrid_path_1, textgrid_path_2, tier_name="phones"):
    """
    Read two TextGrid files and extract intervals from a specific tier.
    textgrid_path_1: path to the first TextGrid file
    textgrid_path_2: path to the second TextGrid file
    tier_name: name of the tier to extract intervals from (default is "phones")
    Returns two lists of intervals - for each TextGrid file.
    """
    # Read a TextGrid files
    l1_tg = tgt.io.read_textgrid(textgrid_path_1)
    l2_tg = tgt.io.read_textgrid(textgrid_path_2)

    # Access a specific tier
    l1_tier = l1_tg.get_tier_by_name(tier_name)
    l2_tier = l2_tg.get_tier_by_name(tier_name)

    return l1_tier.intervals, l2_tier.intervals

def create_results_df(l1_intervals, l2_intervals, warping_path, projected_costs_1, projected_costs_2):
    """
    Create a DataFrame with the results of the DTW alignment and phoneme cosine similarity.
    l1_intervals: list of intervals for the first TextGrid file
    l2_intervals: list of intervals for the second TextGrid file
    warping_path: list of tuples representing the warping path
    projected_costs_1: numpy array of projected costs for the first speaker
    projected_costs_2: numpy array of projected costs for the second speaker
    Returns a DataFrame with the results.
    """
    l1_first_phoneme_found = False
    l2_first_phoneme_found = False
    cosine_similarities = []
    results_to_csv = []
    for time_step in warping_path[::-1]:
        l1_current_phoneme = None
        l2_current_phoneme = None
        for interval in l1_intervals:
            if interval.start_time <= time_step[0]*0.02 <= interval.end_time:
                l1_current_phoneme = interval.text
                l1_first_phoneme_found = True
                break
        for interval in l2_intervals:
            if interval.start_time <= time_step[1]*0.02 <= interval.end_time:
                l2_current_phoneme = interval.text
                l2_first_phoneme_found = True
                break
        

        # Correct first and last phonemes
        # If the phoneme is not found in the intervals, we take the first or last phoneme
        if l1_current_phoneme is None:
            if not l1_first_phoneme_found:
                l1_current_phoneme = l1_intervals[0].text
            else:
                l1_current_phoneme = l1_intervals[-1].text

        if l2_current_phoneme is None:
            if not l2_first_phoneme_found:
                l2_current_phoneme = l2_intervals[0].text
            else:
                l2_current_phoneme = l2_intervals[-1].text

        l1_phoneme_vec = (phoneme_features_mapping[phoneme_features_mapping['ipa'].values == l1_current_phoneme]).drop(columns=['ipa']).values.flatten()
        l2_phoneme_vec = (phoneme_features_mapping[phoneme_features_mapping['ipa'].values == l2_current_phoneme]).drop(columns=['ipa']).values.flatten()

        # Calculate cosine similarity between phoneme vectors
        cos_sim = cosine_similarity(l1_phoneme_vec.reshape(1, -1), l2_phoneme_vec.reshape(1, -1))
        cosine_similarities.append(cos_sim.item())

        results_to_csv.append((time_step[0]*0.02, projected_costs_1[time_step[0]][1],l1_current_phoneme , time_step[1]*0.02, projected_costs_2[time_step[1]][1] ,l2_current_phoneme))

    cols = ["L1 Timestep", "Distance Projected on L1","L1 Phoneme" , "L2 Matching Timestep", "Distance Projected on L2","L2 Phoneme"]
    results_df = pd.DataFrame(results_to_csv, columns=cols)

    smoothed_cosine_sim = gaussian_filter1d(np.array(cosine_similarities), sigma=9)
    results_df["Phonemes Cosine Similarity"] = smoothed_cosine_sim

    return results_df

def create_phonological_feature_vectors(path_to_phoneme_dict="data/raw/ipa_phone_mapping.dict", path_to_mapping_csv="data/raw/ipa2spe.csv"):
    """
    Create a DataFrame mapping phonemes to their phonological feature vectors.
    path_to_phoneme_dict: path to the phoneme dictionary file
    path_to_mapping_csv: path to the CSV file containing the phoneme mapping
    Returns a DataFrame with the phoneme vectors.
    """
    df = pd.read_csv(path_to_mapping_csv, delimiter=',')

    ipa_col = df['ipa']
    mapping = {'+': 1, 'n': 0, '-': -1, '.':0}
    df_mapped = df.drop(columns=['ipa']).replace(mapping)
    phoneme_vec_mapping = pd.concat([ipa_col, df_mapped], axis=1)

    # Track missing phonemes
    missing_phonemes = []

    with open(path_to_phoneme_dict, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phoneme = parts[1]  # 2nd column

                if phoneme in phoneme_vec_mapping['ipa'].values:
                    row = phoneme_vec_mapping[phoneme_vec_mapping['ipa'].values == phoneme]
                    if row.empty:
                        missing_phonemes.append(phoneme)
                        raise ValueError(f"Phoneme '{phoneme}' not found in the DataFrame.")
                else:
                    missing_phonemes.append(phoneme)

    if len(missing_phonemes) > 0:
        with open("/home/itayefrat/Util/Missing_phonems_vector.txt", "a") as f:
            f.write(missing_phonemes)


    return phoneme_vec_mapping


# The Code Start Here
#~~~~~~~~~~~~~~~~~~~~~

# Model's label rate is 0.02 seconds. To not overflow the plot, time is shown every 5 samples (0.1 seconds).
# To change that, change "time_frame" below.
l1_speaker = sys.argv[1]
l2_speaker = sys.argv[2]
correlations_csv_file = sys.argv[3]

wav_paths = [l1_speaker, l2_speaker]

# Extract speaker names
names = [f.split(".")[0] for f in wav_paths]
S1 = names[0]
S2 = names[1]

textgrid_path_1 = S1 + ".TextGrid"
textgrid_path_2 = S2 + ".TextGrid"

time_frame = 5
expected_sr = 16000

# Load and resample wav files
wavs = []
for wav_path in wav_paths:
    wav, sr = torchaudio.load(wav_path)
    if sr != expected_sr:
        print(f"Sampling rate of {wav_path} is not {expected_sr} -> Resampling the file")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=expected_sr)
        wav = resampler(wav)
        wav.squeeze()
    wavs.append(wav)

# Generate Features
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(f'Running on {device_name}')


model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
features = None
speaker_len = []
layer = 12

# Not batched to know the actual seqence shape
for wav in wavs:
    wav_features = model(wav, return_dict=True, output_hidden_states=True).hidden_states[
        layer].squeeze().detach().numpy()
    features = wav_features if features is None else np.concatenate([features, wav_features], axis=0)
    speaker_len.append(wav_features.shape[0])

# Create & Fill a dataframe with the details
data_subset, df_subset, hubert_feature_columns = create_hubert_df(features, speaker_len, names)





# Calculate DTW distances and warping path
df_alignment, optimal_distances, warping_path = calculate_dtw_path(df_subset, S1, S2, hubert_feature_columns)
path_cost = df_alignment['cost']


# Calculate the projected costs for each speaker
name_1 = df_alignment.columns[0]
frames_1 = df_alignment[name_1]
projected_costs_1 = wp_projection_calc(frames_1, path_cost)

name_2 = df_alignment.columns[3]
frames_2 = df_alignment[name_2]
projected_costs_2 = wp_projection_calc(frames_2, path_cost)

# Analyze the TextGrid files and extract intervals
l1_intervals, l2_intervals = textgrid2intervals(textgrid_path_1, textgrid_path_2)

# Create phonological feature vectors
phoneme_features_mapping = create_phonological_feature_vectors()


# Compare the textgrid intervals with the projected costs
results_df = create_results_df(l1_intervals, l2_intervals, warping_path, projected_costs_1, projected_costs_2)

# Save results to CSV in the same folder as L2 speaker
csv_name = S2 + "_results.csv"
results_df.to_csv(csv_name, index=False)
print(f'Comparison results saved to {csv_name}')

# Calculate correlations with L1 and L2 projections
pearson_correlation_L1, spearman_correlation_L1 = correlation_analysis(results_df, "L1")
pearson_correlation_L2, spearman_correlation_L2 = correlation_analysis(results_df, "L2")

correlation_results = {
    "Pearson correlation with L1": pearson_correlation_L1,
    "Spearman correlation with L1": spearman_correlation_L1,
    "Pearson correlation with L2": pearson_correlation_L2,
    "Spearman correlation with L2": spearman_correlation_L2
}

save_correlations(correlation_results, S2, correlations_csv_file)
print(f'Finished analyzing {S2}')


# === DTW Cost Matrix Plotting ===
plot_DTW_cost_matrix = False
if plot_DTW_cost_matrix:  

    fig = go.Figure()

    # Add the accumulated cost matrix as a heatmap
    fig.add_trace(go.Heatmap(
        z=optimal_distances,
        x=np.arange(optimal_distances.shape[1]),  # x-axis represents sequence X
        y=np.arange(optimal_distances.shape[0]),  # y-axis represents sequence Y
        colorscale='Blues',
        colorbar=dict(title='Accumulated Cost'),
        showscale=True
    ))

    # Extract the warping path coordinates
    wp_x = [p[1] for p in warping_path]  # Extract x indices (m)
    wp_y = [p[0] for p in warping_path]  # Extract y indices (n)

    # Add the warping path as a line
    fig.add_trace(go.Scatter(
        x=wp_x,  # x positions from warping path
        y=wp_y,  # y positions from warping path
        mode='lines+markers',
        line=dict(color='red', width=3),
        name='Warping Path'
    ))

    # Customize the layout
    fig.update_layout(
        title='DTW Cost Matrix with Warping Path',
        xaxis=dict(title='English L2'),
        yaxis=dict(title='English L1'),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        template='plotly_white',
    )

    fig.show()

# === Plot the projected costs ===
plot_projections = False
if plot_projections:
    # Plot the projections
    max_cost = max(np.max(projected_costs_1[:, 1]), np.max(projected_costs_2[:, 1]))
    wp_projection_plot(projected_costs_1, name_1.replace(" index", ""), max_cost)
    wp_projection_plot(projected_costs_2, name_2.replace(" index", ""), max_cost)

# === Plot the results ===
plot_results = False
if plot_results:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['L1 Timestep'],
        y=results_df['Distance Projected on L1'],
        mode='lines+markers',
        name='Projected Cost on L1',
        line=dict(color='blue', width=2),
        marker=dict(size=5),
        text=results_df['L1 Phoneme'],
        hovertemplate='Time: %{x}<br>Cost: %{y}<br>L1 Phoneme: %{text}<br>Match: %{customdata}<extra></extra>',
        customdata=results_df['True/False']
    ))


    fig.add_trace(go.Scatter(
        x=results_df['L2 Matching Timestep'],
        y=results_df['Distance Projected on L2'],
        mode='lines+markers',
        name='Projected Cost on L2',
        line=dict(color='red', width=2),
        marker=dict(size=5),
        text=results_df['L2 Phoneme'],
        hovertemplate='Time: %{x}<br>Cost: %{y}<br>L2 Phoneme: %{text}<br>Match: %{customdata}<extra></extra>',
        customdata=results_df['True/False']
    ))

    fig.update_layout(
        title='Projected Costs on L1 and L2',
        xaxis_title='Time (s)',
        yaxis_title='Projected Cost',
        legend_title='Legend',
        template='plotly_white'
    )

    fig.show()
