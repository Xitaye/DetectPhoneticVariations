import pandas as pd
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go

# Load the CSV file
csv_path = '/mlspeech/data/itayefrat/Similarities_Project/Phase_1_recordings/SpanishFemale/S3F-HINT-06-01__leveled_results.csv'  
df = pd.read_csv(csv_path)  # Replace with your actual filename

# Column names
col_L1 = 'Distance Projected on L1'
col_L2 = 'Distance Projected on L2'
col_label = 'True/False'

# Convert 0/1 to boolean for easier filtering
df[col_label] = df[col_label].astype(bool)

# Split values by group (label True/False)
group_L1_true = df[df[col_label]][col_L1]
group_L1_false = df[~df[col_label]][col_L1]

group_L2_true = df[df[col_label]][col_L2]
group_L2_false = df[~df[col_label]][col_L2]

# Mann–Whitney U tests
stat_L1, p_L1 = mannwhitneyu(group_L1_true, group_L1_false, alternative='two-sided')
stat_L2, p_L2 = mannwhitneyu(group_L2_true, group_L2_false, alternative='two-sided')

# Print results
print("Mann–Whitney U Test Results:\n")

print(f"📌 L1 Distance:")
print(f"U-statistic: {stat_L1:.3f}")
print(f"P-value: {p_L1:.5f}")
print("Significant difference" if p_L1 < 0.05 else "No significant difference")

print(f"\n📌 L2 Distance:")
print(f"U-statistic: {stat_L2:.3f}")
print(f"P-value: {p_L2:.5f}")
print("Significant difference" if p_L2 < 0.05 else "No significant difference")

plot_results = True

if plot_results:
    fig = go.Figure()

    # Plot L1 projection
    fig.add_trace(go.Scatter(
        x=df['L1 Timestep'],  # Replace with your actual column name for L1 time
        y=df['Distance Projected on L1'],
        mode='lines+markers',
        name='Projected Cost on L1',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        text=df['L1 Phoneme'],  # Optional text for hover
        hovertemplate='Time: %{x}<br>Cost: %{y}<br>L1 Phoneme: %{text}<br>Match: %{customdata}<extra></extra>',
        customdata=df['True/False']
    ))

    # Plot L2 projection
    fig.add_trace(go.Scatter(
        x=df['L2 Matching Timestep'],  # Replace with your actual column name for L2 time
        y=df['Distance Projected on L2'],
        mode='lines+markers',
        name='Projected Cost on L2',
        line=dict(color='red', width=2),
        marker=dict(size=6),
        text=df['L2 Phoneme'],  # Optional text for hover
        hovertemplate='Time: %{x}<br>Cost: %{y}<br>L2 Phoneme: %{text}<br>Match: %{customdata}<extra></extra>',
        customdata=df['True/False']
    ))

    # Layout
    fig.update_layout(
        title='Projected Costs on L1 and L2',
        xaxis_title='Time (s)',
        yaxis_title='Projected Distance',
        legend_title='Projection Line',
        template='plotly_white'
    )

    fig.show()