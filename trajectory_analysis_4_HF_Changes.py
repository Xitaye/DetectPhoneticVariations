import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d

import torch

from transformers import HubertModel
import torchaudio

from scipy.stats import zscore
from librosa.sequence import dtw as lib_dtw
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo

import tgt
from pathlib import Path
import sys
import os

tsne_1 = 'tsne-3d-one'
tsne_2 = 'tsne-3d-two'
tsne_3 = 'tsne-3d-thr'


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


def librosa_dtw(sq1, sq2):
    """
    Compute the Dynamic Time Warping distance between two sequences.
    sq1: numpy array of shape (t1, d)
    sq2: numpy array of shape (t2, d)
    """
    return lib_dtw(sq1.transpose(), sq2.transpose())[0][-1, -1]

#~~~~~~~~~~~~~~~~~~~librosa dtw backtrack~~~~~~~~~~~~~~~~~~~~~~~~~~~
def librosa_dtw_backtrack(sq1, sq2):
    """
    Compute the Dynamic Time Warping distance between two sequences.
    sq1: numpy array of shape (t1, d)
    sq2: numpy array of shape (t2, d)
    """
    
    optimal_distances, warping_path = lib_dtw(sq1.transpose(), sq2.transpose() ,backtrack=True)
    return optimal_distances, warping_path
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def time_txt(time, time_frame=5):
    if time % time_frame == 0:
        return f"{round(time * 0.02, 2)}"
    return ""


def create_df(feats, speaker_len, names):
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


def tsne(data_subset, init='pca', early_exaggeration=12.0, lr='auto', n_comp=3, perplexity=40, iters=1000,
         random_state=None):
    tsne = TSNE(n_components=n_comp, verbose=1, perplexity=perplexity, n_iter=iters, init=init,
                early_exaggeration=early_exaggeration,
                learning_rate=lr, random_state=random_state)
    tsne_results = tsne.fit_transform(data_subset)
    return tsne_results


def fill_tsne(df_subset, tsne_results):
    print(tsne_results[:, 0].shape)
    df_subset[tsne_1] = tsne_results[:, 0]
    df_subset[tsne_2] = tsne_results[:, 1]
    if tsne_results.shape[1] == 3:
        df_subset[tsne_3] = tsne_results[:, 2]
    return df_subset


def plot_tsne(df_subset):
    pyo.init_notebook_mode()
    fig = px.scatter_3d(df_subset, x=tsne_1, y=tsne_2, z=tsne_3,
                        color='speaker')
    fig.update_traces(mode='lines+markers+text')
    pyo.iplot(fig, filename='jupyter-styled_bar')


def calc_distance(df_subset, speaker1, speaker2, cols):
    features_speaker1 = df_subset[df_subset['speaker'] == speaker1][cols].to_numpy()
    features_speaker2 = df_subset[df_subset['speaker'] == speaker2][cols].to_numpy()
    features_speaker1, features_speaker2 = mut_normalize_sequences(features_speaker1, features_speaker2, True)
    distance = librosa_dtw(features_speaker1, features_speaker2)
    distance = distance / (len(features_speaker1) + len(features_speaker2))
    return distance

#~~~~~~~~~~~~~partial distances function~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calc_partial_distances(df_subset, speaker1, speaker2, cols):
    features_speaker1 = df_subset[df_subset['speaker'] == speaker1][cols].to_numpy()
    features_speaker2 = df_subset[df_subset['speaker'] == speaker2][cols].to_numpy()
    features_speaker1, features_speaker2 = mut_normalize_sequences(features_speaker1, features_speaker2, True)
    optimal_distances, warping_path = librosa_dtw_backtrack(features_speaker1, features_speaker2)
    optimal_distances = optimal_distances / (len(features_speaker1) + len(features_speaker2))
    return optimal_distances, warping_path, features_speaker1, features_speaker2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~Normalize and Calculate DTW accotding to Wraping Path for Projection~~~~~~~~~~~~~~~~~~~~~~~~
def calculate_dtw_path(df_subset, speaker1, speaker2, cols):
    features_speaker1 = df_subset[df_subset['speaker'] == speaker1][cols].to_numpy()
    features_speaker2 = df_subset[df_subset['speaker'] == speaker2][cols].to_numpy()

    features_speaker1, features_speaker2 = mut_normalize_sequences(features_speaker1, features_speaker2, True)  
    D, wp = lib_dtw(features_speaker1.transpose(), features_speaker2.transpose(), backtrack=True)   
    out_wp_cols = [speaker1+" index", "frame_range","frame_range_ms", 
                   speaker2+" index", 'frame_range',"frame_range_ms", "cost"]  
    samples_per_chunk = 0.02*16000  
    out_wp = []    
    for i, j in wp[::-1]:   
        match_cost = D[i, j]/(len(features_speaker1)+len(features_speaker2))    
        out_wp.append((i, (i*samples_per_chunk, i*samples_per_chunk+samples_per_chunk), (i*0.02, i*0.02+0.02),
                       j, (j*samples_per_chunk, j*samples_per_chunk+samples_per_chunk), (j*0.02, j*0.02+0.02),
                       match_cost))

    df = pd.DataFrame(out_wp, columns=out_wp_cols)
    return df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Calculate Projection of WP~~~~~~~~~~~~~~~~~~ 
def wp_projection_calc(speaker_frames, costs):
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Plot Projection of WP~~~~~~~~~~~~~~~~~~
def wp_projection_plot(projection_costs, speaker, max_cost):
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Find Peaks Of Projection ~~~~~~~~~~~~~~~~~~
def find_peaks(projection_costs, gradient=0.1, threshold=0.35):
    # Find the peaks in the projection costs
    peaks = []
    for i in range(1, len(projection_costs) - 1):
        if projection_costs[i][1] - projection_costs[i - 1][1] > gradient and projection_costs[i][1] - projection_costs[i+1][1] > gradient:
            if projection_costs[i][1] > threshold:      #TODO: Check threshold
                peaks.append(projection_costs[i])
    peaks = np.array(peaks)

    peaks_dict = {}
    for peak in peaks:
        peaks_dict[peak[0]] = peak[1]

    return peaks , peaks_dict
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~L1 to L2 peak timesteps convertor~~~~~~~~~~~~~~~~~~
def convert_l1_to_l2_timesteps(l1_peak_timesteps, wp_matrix):
    # L1 is x axis, L2 is y axis in this wp_matrix
    l2_peak_timesteps = []
    for i in range(len(wp_matrix)):
        if(wp_matrix[i][0] in l1_peak_timesteps):
            l2_peak_timesteps.append( (wp_matrix[i][0], wp_matrix[i][1]) )

    l2_peak_timesteps = np.array(l2_peak_timesteps)
    return l2_peak_timesteps[::-1]  # Reverse the order
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~ Import intervals from textgrid ~~~~~~~~~~~~~~~~~~
def textgrid2intervals(textgrid_path_1, textgrid_path_2):

    # Read a TextGrid files
    l1_tg = tgt.io.read_textgrid(textgrid_path_1)
    l2_tg = tgt.io.read_textgrid(textgrid_path_2)

    # Access a specific tier
    l1_tier = l1_tg.get_tier_by_name("phones")  # Change to the correct tier name if needed
    l2_tier = l2_tg.get_tier_by_name("phones")

    return l1_tier.intervals, l2_tier.intervals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~ Summerize results to CSV ~~~~~~~~~~~~~~~~~~
def summerize_results_to_csv(l1_intervals, l2_intervals, warping_path, projected_costs_1, projected_costs_2):   
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

    cosine_sim_array = np.array(cosine_similarities)
    smoothed_cosine_sim = gaussian_filter1d(cosine_sim_array, sigma=9)

    cols = ["L1 Timestep", "Distance Projected on L1","L1 Phoneme" , "L2 Matching Timestep", "Distance Projected on L2","L2 Phoneme"]
    results_df = pd.DataFrame(results_to_csv, columns=cols)
    results_df["Phonemes Cosine Similarity"] = smoothed_cosine_sim

    return results_df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~ Map L1 and L2 peak timesteps ~~~~~~~~~~~~~~~~~
def map_l1_l2_peak_timesteps(peaks_correlation):
    mapping = {}
    for l1, l2 in peaks_correlation:
        if l1*0.02 in mapping:
            mapping[l1*0.02].append(l2*0.02)
        else:
            mapping[l1*0.02] = [l2*0.02]
    return mapping
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~Match time to phonems~~~~~~~~~~~~~~~~~~~~~
def match_time2phonems(peaks_in_time, tier, error_margin=0.03):
    peak_phonemes = []
    for peak in peaks_in_time:
        for interval in tier.intervals:
            if(interval.start_time - error_margin <= peak <= interval.end_time + error_margin):
                peak_phonemes.append( (peak, interval.text) )
            ## TODO: Check if we want error margin only for exactly between two phonems
    peak_phonemes = np.array(peak_phonemes)
    return peak_phonemes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~Clean results~~~~~~~~~~~~~~~~~~~~~
def clean_result(data):
    seen = set()
    unique_data = []
    for row in data:
        t = tuple(row)
        if t not in seen:
            seen.add(t)
            unique_data.append(row)
    unique_data = np.array(unique_data)
    return unique_data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~Analyze peaks~~~~~~~~~~~~~~~~~~~~~
def analyze_peaks(l1_projected_costs, warping_path, textgrid_path_1=None, textgrid_path_2=None, l2_name=None, gradient=0.1, threshold=0.35,error_margin=0.001):

    peaks, peak_dict = find_peaks(l1_projected_costs, gradient, threshold)
    print("Peaks:", peaks)

    peaks_correlation = convert_l1_to_l2_timesteps(peaks[:, 0], warping_path)
    print("Peaks correlation:", peaks_correlation)

    l1_l2_mapping = map_l1_l2_peak_timesteps(peaks_correlation)
    print("L1 to L2 mapping:", l1_l2_mapping)

    l1_peaks_in_time = peaks[:, 0] * 0.02
    l2_peaks_in_time = peaks_correlation[:, 1] * 0.02
    print("L1 Peaks in time:", l1_peaks_in_time)
    print("L2 Peaks in time:", l2_peaks_in_time)

    # Read a TextGrid files
    l1_tg = tgt.io.read_textgrid(textgrid_path_1)
    l2_tg = tgt.io.read_textgrid(textgrid_path_2)

    # Access a specific tier
    l1_tier = l1_tg.get_tier_by_name("phones")
    l2_tier = l2_tg.get_tier_by_name("phones")
    
    # Iterate through intervals and get phoneme in peaks
    l1_peak_phonemes = match_time2phonems(l1_peaks_in_time, l1_tier, error_margin)
    l2_peak_phonemes = match_time2phonems(l2_peaks_in_time, l2_tier, error_margin)
    print("L1 Peak phonemes:", l1_peak_phonemes)
    print("L2 Peak phonemes:", l2_peak_phonemes)

    results = []
    for l1_peak in l1_peak_phonemes:
        l1_time = float(l1_peak[0])
        l1_phoneme = l1_peak[1]
        matches = np.where(l2_peak_phonemes[:, 0] == str(l1_l2_mapping[l1_time][0]))[0]
        i = matches[0] if len(matches) > 0 else None
        for l2_peak_time in l1_l2_mapping[l1_time]:
            while (i < len(l2_peak_phonemes) and float(l2_peak_phonemes[i][0]) == l2_peak_time):
                prediction_result = (l2_peak_phonemes[i][1] != l1_phoneme)    # Wrong phonemes = True prediction
                results.append([l2_peak_time, l1_phoneme, peak_dict[l1_time/0.02], prediction_result])
                i += 1

    clean_res = clean_result(results)

    true_count = sum(row[3] == 'True' for row in clean_res)
    false_count = sum(row[3] == 'False' for row in clean_res)
    grade = true_count / (true_count + false_count) * 100
    print(f"True count: {true_count}, False count: {false_count}, Grade: {grade:.2f}%")
    # Save the results to a CSV file                    # TODO: Check if needed
    df_results = pd.DataFrame(clean_res, columns=['L2 Peak Time', 'L1 Phoneme', 'Cost', 'Prediction'])
    csv_name = l2_name + "_experiment_results.csv"
    df_results.to_csv(csv_name, index=False)
    print("Experiment results saved to ", csv_name)

    return clean_res, grade
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Plot two speakers trajectory~~~~~~~~~~~~~~~~~~~~~~~~
def plot_two_speakers(speaker1, speaker2, max_s1=None, max_s2=None, save=False, show=True, plot_output="tsne_plot"):
    def axes_style3d(bgcolor = "rgb(20, 20, 20)", gridcolor="rgb(255, 255, 255)"): 
        return dict(showbackground =True, backgroundcolor=bgcolor, gridcolor=gridcolor, zeroline=False)
    dcp = df_subset.loc[df_subset['speaker'].isin([speaker1, speaker2])].copy().rename(
        columns={tsne_1: "x", tsne_2: 'y', tsne_3: 'z'})
    dcp1 = dcp.loc[(dcp['speaker'] == speaker1)].copy()
    dcp2 = dcp.loc[(dcp['speaker'] == speaker2)].copy()
    dcp1['clr'] = np.linspace(0, 1, dcp.loc[(dcp['speaker'] == speaker1)].shape[0])
    dcp2['clr'] = np.linspace(1, 0, dcp.loc[(dcp['speaker'] == speaker2)].shape[0])

    if max_s1 is not None:
        dcp1 = dcp1[:max_s1]

    if max_s2 is not None:
        dcp2 = dcp2[:max_s2]
    # S1
    fig = px.scatter_3d(dcp1, x='x', y='y', z='z',
                        color='clr', symbol='speaker',
                        text='time_txt',
                        labels={'x': 't-SNE-dim1', 'y': 't-SNE-dim2', 'z': 't-SNE-dim3'})
    fig.update_traces(marker_symbol='diamond', marker_coloraxis=None, marker_colorscale='burg',
                      mode='lines+markers+text', line_color='lightgray')
    fig.for_each_trace(lambda t: t.update(textfont_color='darkred'))

    # S2
    fig2 = px.scatter_3d(dcp2, x='x', y='y', z='z',
                         color='clr', symbol='speaker',
                         text='time_txt',
                         labels={'x': 't-SNE-dim1', 'y': 't-SNE-dim2', 'z': 't-SNE-dim3'})
    fig2.update_traces(marker_coloraxis=None, marker_colorscale='ice', mode='lines+markers+text', line_color='lightgray')
    fig2.for_each_trace(lambda t: t.update(textfont_color='blue'))

    axis_style = axes_style3d(bgcolor='rgb(245, 249, 252)',) #transparent background color
    fig3 = go.Figure(data=fig.data + fig2.data)
    fig3.update_layout(scene=dict(
        xaxis = axis_style,
        yaxis = axis_style,
        zaxis = axis_style,
        xaxis_title='dimension 1 (t-SNE)',
        yaxis_title='dimension 2 (t-SNE)',
        zaxis_title='dimension 3 (t-SNE)',
    ),

        margin=dict(r=20, b=10, l=10, t=10),
        legend_title="Speaker", )

    if show:
        fig3.show()
    if save:
        fig3.write_html(f"{plot_output}.html")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Create phonological feature vectors~~~~~~~~~~~~~~~~~~~~~~~~
def create_phonological_feature_vectors(path_to_phoneme_dict="/home/itayefrat/Documents/MFA/pretrained_models/dictionary/ipa_phone_mapping.dict",
                                        path_to_mapping_csv="/home/itayefrat/Util/ipa2spe.csv"):
    
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def correlation_analysis(results_df, speaker):

    pearson_correlation = results_df['Distance Projected on ' + speaker].corr(results_df['Phonemes Cosine Similarity'])
    spearman_correlation = results_df['Distance Projected on ' + speaker].corr(results_df['Phonemes Cosine Similarity'], method='spearman')

    results_df["Pearson correlation with " + speaker] = pearson_correlation
    results_df["Spearman correlation with " + speaker] = spearman_correlation

    return pearson_correlation, results_df


# Model's label rate is 0.02 seconds. To not overflow the plot, time is shown every 5 samples (0.1 seconds).
# To change that, change "time_frame" below.
seed = 31415
time_frame = 5

# Load wav files
expected_sr = 16000

# wav_paths = [
# #    '/mlspeech/data/ronich/Speechbox_related/Speechbox_KR_dataset/Korean-EnglishIntelligibility/KEI_KF04_EN038_cnv.wav',
# #    '/mlspeech/data/ronich/Speechbox_related/Speechbox_KR_dataset/Korean-EnglishIntelligibility/KEI_EF08_EN038_cnv.wav'
# #    '/mlspeech/data/ronich/Speechbox_related/Speechbox_KR_dataset/Korean-EnglishIntelligibility/KEI_KF04_EN038_cnv.wav'

# #    '/mlspeech/data/ronich/Speechbox_related/Speechbox_KR_dataset/Korean-EnglishIntelligibility/KEI_EM03_EN038_cnv.wav'
# #     '/mlspeech/data/ronich/Speechbox_related/Speechbox_KR_dataset/Korean-EnglishIntelligibility/KEI_KM03_EN038_cnv.wav'
#     '/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings/MidlandMale/E1M-HINT-06-04__leveled.wav',
#     '/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings/SpanishMale/S2M-HINT-06-04__leveled.wav'
# ]

l1_speaker = sys.argv[1]
l2_speaker = sys.argv[2]
means_csv_file = sys.argv[3]

wav_paths = [l1_speaker, l2_speaker]



print(len(wav_paths))
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
names = [f.split(".")[0] for f in wav_paths]
# Not batched to know the actual seqence shape
for wav in wavs:
    wav_features = model(wav, return_dict=True, output_hidden_states=True).hidden_states[
        layer].squeeze().detach().numpy()
    features = wav_features if features is None else np.concatenate([features, wav_features], axis=0)
    speaker_len.append(wav_features.shape[0])

# Create & Fill a dataframe with the details
data_subset, df_subset, hubert_feature_columns = create_df(features, speaker_len, names)

df_subset_orig = df_subset.copy()
data_subset_orig = data_subset.copy()

tsne_results = tsne(data_subset, init='pca', early_exaggeration=2.0, lr=100.0, n_comp=3, perplexity=40, iters=1000,
                    random_state=seed)
#df_subset = fill_tsne(df_subset, tsne_results)

# Evaluate Distance of Two Speakers
S1 = names[0]
S2 = names[1]

# FULL DIMENSIONALITY
distance = calc_distance(df_subset, S1, S2, hubert_feature_columns)
print(f"Full Dim. Distance: {distance}")

apply_TSNE = False
if apply_TSNE:  
    # TSNE DIMENSIONALITY
    cols = [tsne_1, tsne_2, tsne_3]
    distance = calc_distance(df_subset, S1, S2, cols)
    print(f"TSNE Dim. Distance: {distance}")




#~~~~~~~~~~~~~~~~~~ DTW Backtrack Calculation~~~~~~~~~~~~~~~~~~~~~~~
if apply_TSNE:  
    # Partial distances - TSNE
    optimal_distances, warping_path ,features_speaker1, features_speaker2 = calc_partial_distances(df_subset, S1, S2, cols)
else:
    # Partial distances without TSNE
    optimal_distances, warping_path ,features_speaker1, features_speaker2 = calc_partial_distances(df_subset, S1, S2, hubert_feature_columns)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~Our WP graph~~~~~~~~~~~~~~~~~~~~~~~~
run_our_WP = False
if run_our_WP:  
    # Create a Plotly figure
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
        template='plotly_dark'
    )

    # Show the plot
    fig.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~Execute Projections~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_alignment = calculate_dtw_path(df_subset, S1, S2, hubert_feature_columns)
costs = df_alignment['cost']

name_1 = df_alignment.columns[0]
frames_1 = df_alignment[name_1]
projected_costs_1 = wp_projection_calc(frames_1, costs)

name_2 = df_alignment.columns[3]
frames_2 = df_alignment[name_2]
projected_costs_2 = wp_projection_calc(frames_2, costs)

run_projection = False  
if run_projection:  
# Plot the projections
    max_cost = max(np.max(projected_costs_1[:, 1]), np.max(projected_costs_2[:, 1]))
    wp_projection_plot(projected_costs_1, name_1.replace(" index", ""), max_cost)
    wp_projection_plot(projected_costs_2, name_2.replace(" index", ""), max_cost)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

phoneme_features_mapping = create_phonological_feature_vectors()

textgrid_path_1 = S1 + ".TextGrid"
textgrid_path_2 = S2 + ".TextGrid"

#~~~~~~~~~~~~~~~~~~~~~~~Find and analyze Peaks~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_peak_analyze = False
if run_peak_analyze:
    
    results, grade = analyze_peaks(projected_costs_1, warping_path, textgrid_path_1, textgrid_path_2, S2, gradient=0.1, threshold=0.35,error_margin=0.001)        #TODO: Check if this is the right values

    print("Experiment Grade:", grade)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Analyze Textgrid~~~~~~~~~~~~~~~~~~~~~~~~~~~~
analyze_textgrid = True
if analyze_textgrid:

    l1_intervals, l2_intervals = textgrid2intervals(textgrid_path_1, textgrid_path_2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Compare Textgrid and Projections~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compare_textgrid_and_projections = True
if compare_textgrid_and_projections:

    results_df = summerize_results_to_csv(l1_intervals, l2_intervals, warping_path, projected_costs_1, projected_costs_2)

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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


l1_correlation, results_df = correlation_analysis(results_df, "L1")
l2_correlation, results_df = correlation_analysis(results_df, "L2")


csv_name = S2 + "_results.csv"
results_df.to_csv(csv_name, index=False)
print("Results saved to ", csv_name)


speaker_folder = os.path.basename(os.path.dirname(S2))
sentence = os.path.basename(S2)
print(speaker_folder, sentence)


new_row = {
    "Speaker": speaker_folder,
    "Sentence ID": sentence,
    "Pearson correlation with L1": results_df["Pearson correlation with L1"][0],
    "Spearman correlation with L1": results_df["Spearman correlation with L1"][0],
    "Pearson correlation with L2": results_df["Pearson correlation with L2"][0],
    "Spearman correlation with L2": results_df["Spearman correlation with L2"][0],
}
df = pd.DataFrame([new_row])


write_header = not os.path.exists(means_csv_file)
df.to_csv(means_csv_file, mode='a', header=write_header, index=False)
print("Mean results saved to ", means_csv_file)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~First try of projection~~~~~~~~~~~~~~~~~~~~
"""
s1_graph = np.ones((optimal_distances.shape[0],2))
s2_graph = np.ones((optimal_distances.shape[1],2))

s1_x_length = warping_path.shape[0]
i=0
j=0
while(i<optimal_distances.shape[0]):
    if(j<s1_x_length-1 and warping_path[j][0]==warping_path[j+1][0]):
        s1_graph[i][1]+=1
    else:
        s1_graph[i][0] = i+1
        s1_graph[i][1] /= s1_x_length
        i+=1
    j+=1

s2_x_length = warping_path.shape[0]
i=0
j=0
while(i<optimal_distances.shape[1]):
    if(j<s2_x_length-1 and warping_path[j][1]==warping_path[j+1][1]):
        s2_graph[i][1]+=1
    else:
        s2_graph[i][0] = i+1
        s2_graph[i][1] /= s2_x_length
        i+=1
    j+=1

plot_wp = False
if plot_wp:
    #~~~~~~~~~~~~~~~~~~~~~~ plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assuming s1_graph is a list of points or a 2D array
    df = pd.DataFrame(s1_graph, columns=['X', 'Y'])

    # Create a basic figure
    fig = go.Figure()

    # Add the continuous line with markers
    fig.add_trace(go.Scatter(
        x=df['X'],  # x positions
        y=df['Y'],  # y positions
        mode='lines+markers',  # Line with markers
        line=dict(color='blue', width=3),  # Customize line color and width
        marker=dict(size=6),  # Customize marker size
        name='Continuous Line'
    ))

    # Customize the layout
    fig.update_layout(
        title='English L1',
        xaxis=dict(title='X-axis (Column 1)', showgrid=False),
        yaxis=dict(title='Y-axis (Column 2)', showgrid=False),
    )

    # Show the plot
    fig.show()

    # Assuming s2_graph is a list of points or a 2D array
    df = pd.DataFrame(s2_graph, columns=['X', 'Y'])

    # Create a basic figure
    fig = go.Figure()

    # Add the continuous line with markers
    fig.add_trace(go.Scatter(
        x=df['X'],  # x positions
        y=df['Y'],  # y positions
        mode='lines+markers',  # Line with markers
        line=dict(color='red', width=3),  # Customize line color and width
        marker=dict(size=6),  # Customize marker size
        name='Continuous Line'
    ))

    # Customize the layout
    fig.update_layout(
        title='English L2',
        xaxis=dict(title='X-axis (Column 1)', showgrid=False),
        yaxis=dict(title='Y-axis (Column 2)', showgrid=False),
    )

    # Show the plot
    fig.show()
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~Plot trajectory~~~~~~~~~~~~~~~~~~~~~~~~~
plot_graph = False
if plot_graph and apply_TSNE:
    # TSNE plot of the 2 speakers 
    plot_two_speakers(S1, S2, save=False, show=True)
    
    # TSNE plot of the 2 speakers - Cutted areas
    max_em = 33
    max_ef = 36
    max_kf = 58
    max_km = 46

    #max_s1 = 73 # end of "dripped"
    max_s1 = max_ef
    max_s2 = max_kf
    plot_two_speakers(S1, S2, max_s1, max_s2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~