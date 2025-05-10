import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import mne
from mne.datasets import eegbci
import joblib
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import signal

# Set page configuration
st.set_page_config(
    page_title="BCI Motor Imagery Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme
COLORS = {
    'background': '#f0f0f0',      # Light gray background
    'panel_bg': '#ffffff',        # White panel background
    'text': '#333333',            # Dark gray text
    'primary': '#2196F3',         # Material Blue
    'secondary': '#4CAF50',       # Material Green
    'accent': '#FF9800',          # Material Orange
    'error': '#F44336',           # Material Red
    'grid': '#dddddd',            # Light gray grid
    'states': {
        'rest': '#4CAF50',        # Green
        'left': '#2196F3',        # Blue
        'right': '#FF9800'        # Orange
    },
    'highlight': {
        'rest': 'rgba(76, 175, 80, 0.2)',     # Light green
        'left': 'rgba(33, 150, 243, 0.2)',    # Light blue
        'right': 'rgba(255, 152, 0, 0.2)'     # Light orange
    }
}

# Try to load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/high_acc_gb_model.pkl')
        return model, True
    except Exception as e:
        st.warning(f"Could not load model: {e}. Running in simulation mode.")
        return None, False

model, have_model = load_model()

# Simulated EEG data parameters
sampling_rate = 250  # Hz
signal_length = 5    # seconds
buffer_size = sampling_rate * signal_length
time_axis = np.linspace(0, signal_length, buffer_size)

# Key EEG channels for motor imagery
default_channels = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']

# Class names
class_names = ['rest', 'left', 'right']

# Define runs for task 4 and task 5
task_runs = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

# Signal patterns for different states (for simulation mode)
def generate_eeg_signal(state, noise_level=0.5):
    """
    Generate simulated EEG for the given mental state.
    
    Parameters:
    -----------
    state : str
        Mental state ('rest', 'left', or 'right')
    noise_level : float
        Amount of noise to add (0.0-1.0)
        
    Returns:
    --------
    signals : list of ndarrays
        Simulated EEG signals for C3, Cz, and C4 channels
    """
    # Base signal (alpha rhythm at 10Hz)
    base_signal = np.sin(2 * np.pi * 10 * time_axis)
    
    # Add noise
    noise = np.random.normal(0, noise_level, buffer_size)
    
    # Different patterns for each channel and each state
    signals = []
    
    if state == 'rest':
        # REST: Strong alpha (10Hz) in all channels
        c3 = base_signal * 1.5 + noise * 0.4  # Strong alpha
        cz = base_signal * 1.2 + noise * 0.3  # Strong alpha
        c4 = base_signal * 1.5 + noise * 0.4  # Strong alpha
    
    elif state == 'left':
        # LEFT hand: Reduced alpha in right motor cortex (C4)
        c3 = base_signal * 1.3 + noise * 0.3  # Normal alpha
        cz = base_signal * 1.0 + noise * 0.4  # Slightly reduced alpha
        c4 = base_signal * 0.6 + noise * 0.8  # Strongly reduced alpha (ERD on right side)
    
    elif state == 'right':
        # RIGHT hand: Reduced alpha in left motor cortex (C3)
        c3 = base_signal * 0.6 + noise * 0.8  # Strongly reduced alpha (ERD on left side)
        cz = base_signal * 1.0 + noise * 0.4  # Slightly reduced alpha
        c4 = base_signal * 1.3 + noise * 0.3  # Normal alpha
    
    signals = [c3, cz, c4]
    return signals

# Extract simulated features
def extract_simple_features(signals):
    """
    Extract features from EEG signals for motor imagery classification.
    """
    features = []
    
    # For each channel
    for signal in signals:
        # Calculate band powers (simplified)
        # Theta (4-8 Hz)
        theta_power = np.mean(np.abs(np.fft.rfft(signal))[4:8])
        
        # Alpha (8-13 Hz)
        alpha_power = np.mean(np.abs(np.fft.rfft(signal))[8:13])
        
        # Beta (13-30 Hz)
        beta_power = np.mean(np.abs(np.fft.rfft(signal))[13:30])
        
        # Temporal features
        variance = np.var(signal)
        
        # Add to feature vector
        features.extend([theta_power, alpha_power, beta_power, variance])
    
    # Add simulated connectivity features
    features.extend([random.random() for _ in range(10)])
    
    return np.array(features).reshape(1, -1)

# Function to create spectrogram for visualization
def create_spectrogram(signal_data, fs=250):
    """Create a spectrogram for the given signal data"""
    f, t, Sxx = signal.spectrogram(
        signal_data, 
        fs=fs, 
        nperseg=min(64, len(signal_data)//8),
        noverlap=16,
        scaling='density', 
        mode='magnitude'
    )
    
    # Trim frequency to 0-30Hz for display
    freq_mask = f <= 30
    f = f[freq_mask]
    Sxx = Sxx[freq_mask]
    
    # Normalize for display
    if Sxx.max() > 0:
        Sxx = Sxx / Sxx.max()
    
    return f, t, Sxx

# Initialize session state
if 'current_classification' not in st.session_state:
    st.session_state.current_classification = 'rest'
if 'current_signals' not in st.session_state:
    st.session_state.current_signals = generate_eeg_signal('rest')
if 'use_real_data' not in st.session_state:
    st.session_state.use_real_data = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_subject' not in st.session_state:
    st.session_state.current_subject = 1
if 'current_task' not in st.session_state:
    st.session_state.current_task = 'Task4'
if 'epochs' not in st.session_state:
    st.session_state.epochs = None
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'selected_channels' not in st.session_state:
    st.session_state.selected_channels = default_channels.copy()
if 'available_channels' not in st.session_state:
    st.session_state.available_channels = []
if 'time_series_data' not in st.session_state:
    st.session_state.time_series_data = {
        'C3': np.zeros(100),
        'Cz': np.zeros(100),
        'C4': np.zeros(100),
    }
if 'paused' not in st.session_state:
    st.session_state.paused = False

# Function to load EEG data
def load_eeg_data(subject_num, task_name):
    """Load EEG data from MNE dataset"""
    try:
        runs = task_runs[task_name]
        
        # Create simplified cache path
        cache_file = f"cache_S{subject_num}_{task_name}.pkl"
        
        # Try loading from cache first
        if os.path.exists(cache_file):
            with st.spinner('Loading cached data...'):
                cached_data = joblib.load(cache_file)
                epochs = cached_data['epochs']  
                channel_names = cached_data['channels']
                available_channels = cached_data['channels']
                time_axis = cached_data['times']
                fs = cached_data['fs']
                
                return epochs, channel_names, available_channels, time_axis, fs, True
        
        # Process from raw files
        raw_list = []
        
        # Download only if needed
        for run in runs:
            with st.spinner(f'Checking files for subject {subject_num}, run {run}...'):
                file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                                       f'S{str(subject_num).zfill(3)}', 
                                       f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
                
                if not os.path.exists(file_path):
                    with st.spinner(f'Downloading data for subject {subject_num}, run {run}...'):
                        try:
                            eegbci.load_data(subject_num, runs=[run], path='files/')
                        except Exception as e:
                            st.error(f"Download failed: {e}")
                            continue
                            
                # Load the file with minimal settings
                try:
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    raw_list.append(raw)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    continue
        
        if not raw_list:
            st.error("No data found for the selected subject and task.")
            return None, [], [], None, None, False
        
        # Very basic preprocessing - just extract events
        with st.spinner('Processing EEG data...'):
            raw_concat = mne.concatenate_raws(raw_list)
            events, event_id = mne.events_from_annotations(raw_concat)
            
            # Create epochs with minimal settings
            event_id_selected = {k: v for k, v in event_id.items() if any(marker in k for marker in ['T0', 'T1', 'T2', 'T3', 'T4'])}
            epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                               tmin=0.5, tmax=3.5, baseline=None, preload=True)
            
            # Save to simple cache
            cache_data = {
                'epochs': epochs,
                'channels': epochs.ch_names,
                'times': epochs.times,
                'fs': int(epochs.info['sfreq'])
            }
            joblib.dump(cache_data, cache_file)
            
            return epochs, epochs.ch_names, epochs.ch_names, epochs.times, int(epochs.info['sfreq']), True
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, [], [], None, None, False

# Function to update state randomly for simulation
def update_state_randomly():
    """Randomly change the brain state for simulation"""
    if not st.session_state.paused and not st.session_state.use_real_data:
        if random.random() < 0.15:  # 15% chance to change state
            st.session_state.current_classification = random.choice(class_names)
            st.session_state.current_signals = generate_eeg_signal(st.session_state.current_classification)

# Function to plot EEG signals
def plot_eeg_signals(signals, channel_names, current_classification):
    """Plot EEG signals in matplotlib"""
    # Set the number of subplots based on available signals
    n_channels = min(len(signals), len(channel_names))
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2*n_channels))
    fig.patch.set_facecolor(COLORS['panel_bg'])
    
    # Handle case where there's only one channel (axes is not a list)
    if n_channels == 1:
        axes = [axes]
    
    # Highlight colors based on classification state
    if current_classification == 'rest':
        highlight_colors = [COLORS['states']['rest']] * n_channels
        highlight_opacity = [0.2] * n_channels
    elif current_classification == 'left':
        highlight_colors = [COLORS['states']['left']] * n_channels
        highlight_opacity = [0.1] * n_channels
        # Highlight C4 more if it exists
        c4_indices = [i for i, ch in enumerate(channel_names[:n_channels]) if 'C4' in ch or 'c4' in ch]
        for idx in c4_indices:
            highlight_opacity[idx] = 0.4
    elif current_classification == 'right':
        highlight_colors = [COLORS['states']['right']] * n_channels
        highlight_opacity = [0.1] * n_channels
        # Highlight C3 more if it exists
        c3_indices = [i for i, ch in enumerate(channel_names[:n_channels]) if 'C3' in ch or 'c3' in ch]
        for idx in c3_indices:
            highlight_opacity[idx] = 0.4
    
    for i, ax in enumerate(axes):
        if i < n_channels:
            ax.set_facecolor(COLORS['panel_bg'])
            ax.set_title(channel_names[i], fontsize=12, color=COLORS['text'], fontweight='bold')
            ax.set_ylabel('Amplitude (μV)', fontsize=10, color=COLORS['text'])
            
            # Plot the signal - Create appropriate time axis for this signal
            signal_data = signals[i]
            # Create an appropriate time axis for this specific signal
            signal_time_axis = np.linspace(0, signal_length, len(signal_data))
            
            line_color = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], '#9C27B0', '#FF5722'][i % 5]
            ax.plot(signal_time_axis, signal_data, color=line_color, linewidth=1.5, alpha=0.8)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
            
            # Add highlight box
            ylim = ax.get_ylim()
            height = ylim[1] - ylim[0]
            
            # Add highlight based on state and channel
            special_highlight = False
            channel_name_lower = channel_names[i].lower()
            
            if current_classification == 'left' and ('c4' in channel_name_lower or channel_name_lower.startswith('c4')):
                ax.add_patch(Rectangle((0, ylim[0]), signal_length, height, 
                                      alpha=highlight_opacity[i], 
                                      color=highlight_colors[i],
                                      zorder=0))
                # Add ERD annotation
                ax.text(signal_length/2, (ylim[0] + ylim[1])/2, 'ERD', ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.5', fc=COLORS['states']['left'], alpha=0.2, ec=COLORS['states']['left']),
                      fontsize=9)
                special_highlight = True
                
            elif current_classification == 'right' and ('c3' in channel_name_lower or channel_name_lower.startswith('c3')):
                ax.add_patch(Rectangle((0, ylim[0]), signal_length, height, 
                                      alpha=highlight_opacity[i], 
                                      color=highlight_colors[i],
                                      zorder=0))
                # Add ERD annotation
                ax.text(signal_length/2, (ylim[0] + ylim[1])/2, 'ERD', ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.5', fc=COLORS['states']['right'], alpha=0.2, ec=COLORS['states']['right']),
                      fontsize=9)
                special_highlight = True
                
            if not special_highlight:
                ax.add_patch(Rectangle((0, ylim[0]), signal_length, height, 
                                      alpha=0.1, 
                                      color=highlight_colors[i],
                                      zorder=0))
            
            # Add frequency band annotation
            ax.annotate('Alpha band (8-13 Hz)', xy=(0.85, 0.9), xycoords='axes fraction',
                       fontsize=8, color=COLORS['text'],
                       bbox=dict(boxstyle="round,pad=0.3", fc=COLORS['panel_bg'], ec=line_color, alpha=0.3))
            
            # Set xlim to full range
            ax.set_xlim(0, signal_length)
            
            # Only add x-label for the bottom plot
            if i == n_channels - 1:
                ax.set_xlabel('Time (seconds)', fontsize=10, color=COLORS['text'])
    
    plt.tight_layout()
    return fig


# Function to plot spectrogram
def plot_spectrogram(signals):
    """Plot spectrogram for the first channel"""
    if not signals or len(signals) == 0:
        return None
    
    # Just use the first channel for the spectrogram
    signal = signals[0]
    
    # Create spectrogram
    f, t, Sxx = create_spectrogram(signal)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_facecolor(COLORS['panel_bg'])
    
    # Plot spectrogram
    im = ax.pcolormesh(t, f, Sxx, cmap='viridis', shading='gouraud')
    
    # Add frequency band markers
    ax.axhline(y=4, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=8, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=13, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=30, color='white', linestyle='--', alpha=0.5)
    
    # Add band labels
    ax.text(max(t)*1.05, 2, 'Delta', fontsize=8, ha='left', va='center', color='white')
    ax.text(max(t)*1.05, 6, 'Theta', fontsize=8, ha='left', va='center', color='white')
    ax.text(max(t)*1.05, 10, 'Alpha', fontsize=8, ha='left', va='center', color='white')
    ax.text(max(t)*1.05, 20, 'Beta', fontsize=8, ha='left', va='center', color='white')
    
    # Set labels
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Spectrogram (C3 Channel)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Power')
    
    plt.tight_layout()
    return fig

# Function to plot feature analysis
def plot_feature_analysis(signals):
    """Create feature analysis plots"""
    # Extract features
    features = extract_simple_features(signals)
    radar_data = features[0, :9]  # First 9 features (band powers)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # 1. Radar chart for band powers
    ax1 = fig.add_subplot(221, polar=True)
    ax1.set_facecolor(COLORS['panel_bg'])
    ax1.set_title('EEG Band Power Distribution', fontsize=12, color=COLORS['text'])
    
    # Create radar chart categories and data
    radar_categories = ['C3-Theta', 'C3-Alpha', 'C3-Beta', 
                        'Cz-Theta', 'Cz-Alpha', 'Cz-Beta',
                        'C4-Theta', 'C4-Alpha', 'C4-Beta']
    
    # Set up the angles for the radar chart
    radar_num_vars = len(radar_categories)
    radar_angles = np.linspace(0, 2*np.pi, radar_num_vars, endpoint=False).tolist()
    radar_angles += radar_angles[:1]  # Close the loop
    
    # Prepare radar values
    radar_values = radar_data.tolist()
    radar_values += radar_values[:1]  # Close the loop
    
    # Plot radar chart
    ax1.plot(radar_angles, radar_values, linewidth=2, linestyle='solid', color=COLORS['primary'])
    ax1.fill(radar_angles, radar_values, color=COLORS['primary'], alpha=0.25)
    
    # Set radar chart labels
    ax1.set_xticks(radar_angles[:-1])
    ax1.set_xticklabels(radar_categories, color=COLORS['text'], fontsize=8)
    
    # 2. Bar chart for band powers
    ax2 = fig.add_subplot(222)
    ax2.set_facecolor(COLORS['panel_bg'])
    ax2.set_title('Band Power Comparison', fontsize=12, color=COLORS['text'])
    ax2.set_xlabel('Channel-Band', fontsize=10, color=COLORS['text'])
    ax2.set_ylabel('Power', fontsize=10, color=COLORS['text'])
    
    # Create bar positions
    bar_x = np.arange(len(radar_categories))
    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']] * 3
    
    # Plot bars
    ax2.bar(bar_x, radar_data, color=bar_colors, alpha=0.7)
    
    # Set bar chart labels
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(radar_categories, rotation=45, ha='right', fontsize=8, color=COLORS['text'])
    
    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # 3. Time series of feature changes - use session state to persist data
    ax3 = fig.add_subplot(223)
    ax3.set_facecolor(COLORS['panel_bg'])
    ax3.set_title('Alpha Power Over Time', fontsize=12, color=COLORS['text'])
    ax3.set_xlabel('Time (last 10 seconds)', fontsize=10, color=COLORS['text'])
    ax3.set_ylabel('Alpha Power', fontsize=10, color=COLORS['text'])
    
    # Update time series data
    st.session_state.time_series_data['C3'] = np.roll(st.session_state.time_series_data['C3'], -1)
    st.session_state.time_series_data['Cz'] = np.roll(st.session_state.time_series_data['Cz'], -1)
    st.session_state.time_series_data['C4'] = np.roll(st.session_state.time_series_data['C4'], -1)
    
    # Add new data points (alpha band power)
    st.session_state.time_series_data['C3'][-1] = features[0, 1]  # C3 Alpha
    st.session_state.time_series_data['Cz'][-1] = features[0, 4]  # Cz Alpha
    st.session_state.time_series_data['C4'][-1] = features[0, 7]  # C4 Alpha
    
    # Time axis for the time series plot
    ts_time = np.linspace(-10, 0, len(st.session_state.time_series_data['C3']))
    
    # Plot time series
    ax3.plot(ts_time, st.session_state.time_series_data['C3'], linewidth=1.5, color=COLORS['primary'], label='C3 (Left)')
    ax3.plot(ts_time, st.session_state.time_series_data['Cz'], linewidth=1.5, color=COLORS['secondary'], label='Cz (Center)')
    ax3.plot(ts_time, st.session_state.time_series_data['C4'], linewidth=1.5, color=COLORS['accent'], label='C4 (Right)')
    
    # Add legend
    ax3.legend(fontsize=8)
    
    # Add grid
    ax3.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # 4. Feature description/ERD visualization
    ax4 = fig.add_subplot(224)
    ax4.set_facecolor(COLORS['panel_bg'])
    ax4.set_title('Event-Related Desynchronization (ERD)', fontsize=12, color=COLORS['text'])
    
    # Create a schematic of ERD
    erd_x = np.linspace(0, 5, 500)
    baseline = np.sin(2*np.pi*10*erd_x) * 0.8 + np.random.normal(0, 0.1, 500)
    erd_signal = np.copy(baseline)
    
    # Create ERD in the middle
    erd_idx = np.logical_and(erd_x >= 2, erd_x <= 3)
    erd_signal[erd_idx] = erd_signal[erd_idx] * 0.4  # Reduce amplitude during ERD
    
    # Plot ERD demonstration
    ax4.plot(erd_x, baseline, color='gray', alpha=0.5, linewidth=1, label='Baseline')
    ax4.plot(erd_x, erd_signal, color=COLORS['primary'], linewidth=1.5, label='ERD')
    
    # Highlight ERD period
    ax4.axvspan(2, 3, alpha=0.2, color=COLORS['primary'])
    ax4.text(2.5, 0, 'ERD', ha='center', fontsize=10, color=COLORS['text'],
           bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=COLORS['primary'], alpha=0.7))
    
    # Add description text
    ax4.text(0.5, -1.5, 
           "ERD is a decrease in neural oscillations (particularly in the alpha band)\n"
           "when a brain region becomes active during motor imagery or actual movement.",
           ha='center', fontsize=8, color=COLORS['text'])
    
    # Add legend
    ax4.legend(fontsize=8)
    
    # Set axis limits
    ax4.set_xlim(0, 5)
    ax4.set_ylim(-2, 2)
    
    # Remove y ticks for cleaner appearance
    ax4.set_yticks([])
    
    # Add x label
    ax4.set_xlabel('Time (s)', fontsize=10, color=COLORS['text'])
    
    plt.tight_layout()
    return fig

# Function to plot brain topography
def plot_brain_topography(current_classification):
    """Plot brain topography visualization"""
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Set up grid layout
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top view
    ax2 = fig.add_subplot(gs[0, 1])  # Side view
    ax3 = fig.add_subplot(gs[1, 0])  # Alpha band
    ax4 = fig.add_subplot(gs[1, 1])  # Beta band
    
    # Set backgrounds
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(COLORS['panel_bg'])
    
    # Set titles
    ax1.set_title('Top View', fontsize=12, color=COLORS['text'])
    ax2.set_title('Side View', fontsize=12, color=COLORS['text'])
    ax3.set_title('Alpha Band (8-13 Hz)', fontsize=12, color=COLORS['text'])
    ax4.set_title('Beta Band (13-30 Hz)', fontsize=12, color=COLORS['text'])
    
    # Draw head outlines
    # Top view (ax1 and ax3, ax4)
    for ax in [ax1, ax3, ax4]:
        # Draw head circle
        circle = plt.Circle((0, 0), 1, fill=False, color=COLORS['text'], linewidth=2)
        ax.add_patch(circle)
        
        # Draw nose
        ax.plot([0, 0], [0.9, 1.1], color=COLORS['text'], linewidth=2)
        
        # Draw ears
        ax.plot([-1.1, -0.9], [0, 0], color=COLORS['text'], linewidth=2)
        ax.plot([0.9, 1.1], [0, 0], color=COLORS['text'], linewidth=2)
    
    # Side view (ax2)
    # Draw head profile
    x = np.linspace(-1, 1, 100)
    y_top = np.sqrt(1 - x**2)
    
    # Head outline
    ax2.plot(x, y_top, color=COLORS['text'], linewidth=2)
    ax2.plot([-1, 1], [0, 0], color=COLORS['text'], linewidth=2)
    
    # Nose
    ax2.plot([0.9, 1.1], [0.2, 0.2], color=COLORS['text'], linewidth=2)
    
    # Ear
    ax2.plot([-1.05, -1.05], [-0.2, 0.2], color=COLORS['text'], linewidth=2)
    
    # Mark electrode positions
    # Top view electrodes
    top_positions = {
        'C3': (-0.5, 0),
        'Cz': (0, 0),
        'C4': (0.5, 0),
        'F3': (-0.5, 0.5),
        'Fz': (0, 0.5),
        'F4': (0.5, 0.5),
        'P3': (-0.5, -0.5),
        'Pz': (0, -0.5),
        'P4': (0.5, -0.5)
    }
    
    for ax in [ax1, ax3, ax4]:
        for name, (x, y) in top_positions.items():
            ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
            ax.text(x, y+0.1, name, ha='center', va='center', fontsize=8, color=COLORS['text'])
            
            # Highlight motor cortex electrodes
            if name in ['C3', 'Cz', 'C4']:
                ax.plot(x, y, 'o', markersize=10, mfc='none', mec=COLORS['accent'], linewidth=2)
    
    # Side view electrodes
    side_positions = {
        'C3': (-0.5, 0.75),
        'Cz': (0, 1),
        'C4': (0.5, 0.75),
        'F3': (-0.75, 0.5),
        'Fz': (0, 0.75),
        'F4': (0.75, 0.5),
    }
    
    for name, (x, y) in side_positions.items():
        if x >= -0.2:  # Only show electrodes visible from this side
            ax2.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
            ax2.text(x, y+0.1, name, ha='center', va='center', fontsize=8, color=COLORS['text'])
            
            # Highlight motor cortex electrodes
            if name in ['Cz', 'C4']:
                ax2.plot(x, y, 'o', markersize=10, mfc='none', mec=COLORS['accent'], linewidth=2)
    
    # Create topography heatmaps
    # Create a grid for the heatmap
    n = 20
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Create a circular mask
    mask = X**2 + Y**2 <= 1
    
    # Initialize data
    alpha_data = np.zeros((n, n))
    beta_data = np.zeros((n, n))
    
    # Add baseline activity
    alpha_data[mask] = np.random.normal(0.5, 0.1, np.sum(mask))
    beta_data[mask] = np.random.normal(0.2, 0.1, np.sum(mask))
    
    # Add state-specific patterns
    if current_classification == 'rest':
        # Rest: Strong alpha everywhere
        alpha_data[mask] = np.random.normal(0.8, 0.1, np.sum(mask))
        
    elif current_classification == 'left':
        # Left: ERD in right motor cortex (C4)
        # Create a right-sided focus
        right_mask = np.logical_and(mask, X > 0.2)
        alpha_data[right_mask] = np.random.normal(-0.5, 0.2, np.sum(right_mask))
        beta_data[right_mask] = np.random.normal(0.6, 0.2, np.sum(right_mask))
        
    elif current_classification == 'right':
        # Right: ERD in left motor cortex (C3)
        # Create a left-sided focus
        left_mask = np.logical_and(mask, X < -0.2)
        alpha_data[left_mask] = np.random.normal(-0.5, 0.2, np.sum(left_mask))
        beta_data[left_mask] = np.random.normal(0.6, 0.2, np.sum(left_mask))
    
    # Create a custom colormap (blue=low, red=high)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('ERD_ERS', colors, N=100)
    
    # Plot the heatmaps
    im3 = ax3.imshow(
        alpha_data, 
        extent=[-1.2, 1.2, -1.2, 1.2],
        origin='lower',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation='bilinear'
    )
    
    im4 = ax4.imshow(
        beta_data, 
        extent=[-1.2, 1.2, -1.2, 1.2],
        origin='lower',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation='bilinear'
    )
    
    # Add colorbars
    plt.colorbar(im3, ax=ax3, shrink=0.7, label='Alpha Activity')
    plt.colorbar(im4, ax=ax4, shrink=0.7, label='Beta Activity')
    
    # Remove axis ticks for cleaner appearance
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

# Function to extract real data from MNE epochs
# Modify the get_epoch_data function to also return the event type
def get_epoch_data(epochs, epoch_index, selected_channels, channel_names):
    """Extract data from the specified epoch and determine its state with improved event handling"""
    if epochs is None or epoch_index >= len(epochs):
        return None, None
    
    try:
        # Get selected channel indices
        channel_indices = []
        for ch in selected_channels:
            if ch in channel_names:
                channel_indices.append(channel_names.index(ch))
            elif ch.split(' ')[0] in channel_names:  # Try without the description
                channel_indices.append(channel_names.index(ch.split(' ')[0]))
        
        # If no valid channels, return None
        if not channel_indices:
            return None, None
        
        # Get data for the selected epoch and channels
        epoch_data = epochs.get_data()[epoch_index]
        
        # Extract selected channels
        selected_data = [epoch_data[idx] for idx in channel_indices[:3]]  # Get up to 3 channels
        
        # Determine the state from the event code
        event_code = epochs.events[epoch_index, 2]
        
        # Create a more comprehensive mapping system with debug info
        event_id_map = epochs.event_id
        
        # Print debug info to help diagnose the issue
        print(f"Event code for epoch {epoch_index}: {event_code}")
        print(f"Available event ids: {event_id_map}")
        
        # Default to rest
        state = 'rest'
        
        # First check for direct matches
        for event_name, code in event_id_map.items():
            if code == event_code:
                event_name_lower = event_name.lower()
                
                # More flexible matching for left hand
                if 'left' in event_name_lower or 't1' in event_name_lower or event_name_lower.endswith('1'):
                    state = 'left'
                    print(f"Mapped event {event_name} to LEFT")
                    break
                    
                # More flexible matching for right hand
                elif 'right' in event_name_lower or 't2' in event_name_lower or event_name_lower.endswith('2'):
                    state = 'right'
                    print(f"Mapped event {event_name} to RIGHT")
                    break
                    
                # More flexible matching for feet/both hands (some datasets use T3/T4)
                elif 'feet' in event_name_lower or 't3' in event_name_lower or event_name_lower.endswith('3'):
                    state = 'right'  # Map feet to right for visualization
                    print(f"Mapped event {event_name} (feet) to RIGHT")
                    break
                
                elif 'hands' in event_name_lower or 't4' in event_name_lower or event_name_lower.endswith('4'):
                    state = 'left'  # Map both hands to left for visualization
                    print(f"Mapped event {event_name} (hands) to LEFT")
                    break
        
        print(f"Final state determination: {state}")
        return selected_data, state
    except Exception as e:
        st.error(f"Error extracting epoch data: {e}")
        return None, None  
    
# Main application layout
st.title("🧠 Brain-Computer Interface: Motor Imagery Analysis")

# Main container for the application content
main_container = st.container()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Data mode selection
    st.subheader("Data Source")
    data_mode = st.radio(
        "Select data mode:",
        ["Simulation", "Real Data"],
        index=0 if not st.session_state.use_real_data else 1,
        key="data_mode"
    )
    
    # Update session state based on selection
    st.session_state.use_real_data = (data_mode == "Real Data")
    
    # Real data controls (conditionally displayed)
    if st.session_state.use_real_data:
        st.subheader("Data Selection")
        
        # Subject selection
        subject_num = st.selectbox(
            "Subject:",
            range(1, 11),
            index=st.session_state.current_subject - 1,
            key="subject_selector"
        )
        st.session_state.current_subject = subject_num
        
        # Task selection
        task_name = st.selectbox(
            "Task:",
            list(task_runs.keys()),
            index=0 if st.session_state.current_task == 'Task4' else 1,
            key="task_selector"
        )
        st.session_state.current_task = task_name
        
        # Load data button
        if st.button("Load Data", key="load_data_btn"):
            with st.spinner("Loading EEG data..."):
                # Load the data
                epochs, channel_names, available_channels, time_axis, fs, success = load_eeg_data(
                    st.session_state.current_subject, 
                    st.session_state.current_task
                )
                
                if success:
                    # Store in session state
                    st.session_state.epochs = epochs
                    st.session_state.channel_names = channel_names
                    st.session_state.available_channels = available_channels
                    st.session_state.time_axis = time_axis
                    st.session_state.fs = fs
                    st.session_state.data_loaded = True
                    st.session_state.current_epoch = 0
                    
                    # Select motor channels by default
                    st.session_state.selected_channels = []
                    for ch in ['C3', 'Cz', 'C4']:
                        matching = [name for name in available_channels if ch in name]
                        if matching:
                            st.session_state.selected_channels.append(matching[0])
                    
                    # If we don't have enough channels, use the first available ones
                    while len(st.session_state.selected_channels) < 3 and len(available_channels) > len(st.session_state.selected_channels):
                        next_ch = available_channels[len(st.session_state.selected_channels)]
                        if next_ch not in st.session_state.selected_channels:
                            st.session_state.selected_channels.append(next_ch)
                    
                    st.success(f"Loaded data for Subject {subject_num}, {task_name}")
                    
                    # Extract data for the current epoch and get state
                    signals, state = get_epoch_data(
                        st.session_state.epochs,
                        st.session_state.current_epoch,
                        st.session_state.selected_channels,
                        st.session_state.channel_names
                    )
                    
                    if signals:
                        st.session_state.current_signals = signals
                        # Update the state if available
                        if state:
                            st.session_state.current_classification = state
        
        # # Data visualization controls (only if data is loaded)
        # if st.session_state.data_loaded:
        #     # Divider for better organization
        #     st.markdown("---")
            
            # Channel selection section
            st.subheader("Channel Selection")
            
            # Available channels dropdown
            if st.session_state.available_channels:
                available_ch_list = [ch for ch in st.session_state.available_channels if ch not in st.session_state.selected_channels]
                if available_ch_list:
                    selected_channel = st.selectbox(
                        "Add channel:",
                        available_ch_list,
                        key="channel_selector"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Add", key="add_channel_btn"):
                            if selected_channel and selected_channel not in st.session_state.selected_channels:
                                st.session_state.selected_channels.append(selected_channel)
                                
                                # Immediately update the signals with new channel selection
                                signals, state = get_epoch_data(
                                    st.session_state.epochs,
                                    st.session_state.current_epoch,
                                    st.session_state.selected_channels,
                                    st.session_state.channel_names
                                )
                                
                                if signals:
                                    st.session_state.current_signals = signals
                                    # Update state if available
                                    if state:
                                        st.session_state.current_classification = state
                                        
                                # Force rerun to update interface
                                st.experimental_rerun()
                    
                    with col2:
                        if st.button("Clear", key="clear_channels_btn"):
                            # Reset to default channels
                            st.session_state.selected_channels = []
                            st.experimental_rerun()
                else:
                    st.info("All channels are already selected")
            
            # Show currently selected channels
            st.write("**Selected channels:**")
            if st.session_state.selected_channels:
                st.code(", ".join(st.session_state.selected_channels))
            else:
                st.warning("No channels selected")
            
            # Divider for better organization
            st.markdown("---")
            
            # Navigation and epoch controls
            st.subheader("Navigation")
            
            # Epoch navigation
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⏮ Prev", key="prev_epoch_btn", disabled=st.session_state.current_epoch <= 0):
                    if st.session_state.current_epoch > 0:
                        st.session_state.current_epoch -= 1
                        
                        # Update signals based on new epoch
                        signals, state = get_epoch_data(
                            st.session_state.epochs,
                            st.session_state.current_epoch,
                            st.session_state.selected_channels,
                            st.session_state.channel_names
                        )
                        
                        if signals:
                            st.session_state.current_signals = signals
                            # Update the state if available
                            if state:
                                st.session_state.current_classification = state
            
            with col2:
                pause_label = "▶ Play" if st.session_state.paused else "⏸ Pause"
                if st.button(pause_label, key="pause_play_btn"):
                    st.session_state.paused = not st.session_state.paused
            
            with col3:
                # Disable next button if at last epoch
                last_epoch = False
                if st.session_state.epochs:
                    last_epoch = st.session_state.current_epoch >= len(st.session_state.epochs) - 1
                    
                if st.button("⏭ Next", key="next_epoch_btn", disabled=last_epoch):
                    if st.session_state.epochs and st.session_state.current_epoch < len(st.session_state.epochs) - 1:
                        st.session_state.current_epoch += 1
                        
                        # Update signals based on new epoch
                        signals, state = get_epoch_data(
                            st.session_state.epochs,
                            st.session_state.current_epoch,
                            st.session_state.selected_channels,
                            st.session_state.channel_names
                        )
                        
                        if signals:
                            st.session_state.current_signals = signals
                            # Update the state if available
                            if state:
                                st.session_state.current_classification = state
            
            # Show current epoch and event information
            if st.session_state.epochs:
                # Current epoch number
                total_epochs = len(st.session_state.epochs)
                st.write(f"**Epoch:** {st.session_state.current_epoch + 1}/{total_epochs}")
                
                # Add event code info for debugging
                event_code = st.session_state.epochs.events[st.session_state.current_epoch, 2]
                event_id_map = st.session_state.epochs.event_id
                event_name = "Unknown"
                
                # Find the name for this event code
                for name, code in event_id_map.items():
                    if code == event_code:
                        event_name = name
                        break
                
                # Display event information
                st.write(f"**Event:** {event_name} (code: {event_code})")
                
                # Show event mapping explanation
                with st.expander("About Event Codes"):
                    st.markdown("""
                    **Event Code Mapping:**
                    - **T0/Rest**: Relaxed state, no movement
                    - **T1/Left**: Left hand motor imagery
                    - **T2/Right**: Right hand motor imagery
                    - **T3/Feet**: Feet motor imagery (in Task 5)
                    - **T4/Hands**: Both hands motor imagery (in Task 5)
                    """)
    
    # Divider before status section
    st.markdown("---")
    
    # Current Mode info
    mode_text = "Simulation" if not st.session_state.use_real_data else "Real Data"
    st.info(f"**Current mode:** {mode_text}")
    
    # Display current classification with style
    st.subheader("Current Mental Command")
    
    # Get state color from the color map
    state_color = COLORS['states'][st.session_state.current_classification]
    
    # Display mental command in styled container
    st.markdown(
        f"""
        <div style="
            background-color: {state_color}; 
            color: white; 
            padding: 10px; 
            border-radius: 5px; 
            text-align: center;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 15px;
        ">
            {st.session_state.current_classification.upper()}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # EEG pattern reference guide
    st.markdown("""
    ### EEG Patterns Reference:
    - **REST**: Strong alpha (8-13 Hz) rhythms in all channels
    - **LEFT Hand**: ERD in right motor cortex (C4)
    - **RIGHT Hand**: ERD in left motor cortex (C3)
    """)
    
    # Credits section at bottom
    st.markdown("---")
    st.markdown("#### About")
    st.markdown("Brain-Computer Interface visualization and analysis tool.")
    st.markdown("Data source: [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)")
    
    # Version info
    st.caption("v1.0.0 - BCI Motor Imagery Analysis")

# Main content
with main_container:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["EEG Signals", "Feature Analysis", "Brain Topography"])
    
    # Update state
    if not st.session_state.use_real_data:
        update_state_randomly()
    
    # Tab 1: EEG Signals
    with tab1:
        # Status/info bar
        mode_text = "SIMULATION MODE" if not st.session_state.use_real_data else "REAL DATA MODE"
        
        # Get annotation based on state
        if st.session_state.current_classification == 'rest':
            annotation = "REST state: Strong alpha waves visible in all channels"
        elif st.session_state.current_classification == 'left':
            annotation = "LEFT hand imagery: ERD visible in right motor cortex (C4)"
        elif st.session_state.current_classification == 'right':
            annotation = "RIGHT hand imagery: ERD visible in left motor cortex (C3)"
        else:
            annotation = f"Current state: {st.session_state.current_classification.upper()}"
        
        st.info(f"{mode_text} - {annotation}")
        
        # Plot EEG signals
        channel_names = st.session_state.selected_channels if st.session_state.use_real_data and st.session_state.data_loaded else default_channels
        eeg_fig = plot_eeg_signals(st.session_state.current_signals, channel_names, st.session_state.current_classification)
        st.pyplot(eeg_fig)
        
        # Plot spectrogram
        st.subheader("Time-Frequency Analysis")
        spec_fig = plot_spectrogram(st.session_state.current_signals)
        if spec_fig:
            st.pyplot(spec_fig)
        
        # Explanation text
        st.markdown("""
        **EEG Motor Imagery Patterns:**
        * **REST**: Strong alpha (10Hz) rhythms in all channels
        * **LEFT Hand**: Event-Related Desynchronization (ERD) in right motor cortex (C4)
        * **RIGHT Hand**: Event-Related Desynchronization (ERD) in left motor cortex (C3)
        """)
    
    # Tab 2: Feature Analysis
    with tab2:
        st.subheader("EEG Feature Analysis")
        
        # Feature analysis plots
        feature_fig = plot_feature_analysis(st.session_state.current_signals)
        st.pyplot(feature_fig)
        
        # Explanation
        st.markdown("""
        This analysis shows various features extracted from the EEG signals that can be used for motor imagery classification:
        
        1. **Band Powers**: Power in the theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) frequency bands for each channel
        2. **Power Distribution**: Radar chart showing relative distribution of band powers
        3. **Alpha Power Tracking**: Time series of alpha power changes over time
        4. **Event-Related Desynchronization (ERD)**: Visualization of the pattern that occurs during motor imagery
        """)
    
    # Tab 3: Brain Topography
    with tab3:
        st.subheader("Brain Activity Topography")
        
        # Topography visualization
        topo_fig = plot_brain_topography(st.session_state.current_classification)
        st.pyplot(topo_fig)
        
        # Explanation
        st.markdown("""
        **Brain topography shows the spatial distribution of EEG activity across the scalp.**
        
        For motor imagery, we focus on activity over the sensorimotor cortex (areas C3, Cz, and C4).
        - **RED** areas indicate higher activity
        - **BLUE** areas indicate lower activity (event-related desynchronization)
        
        The alpha and beta band topographies show how activity changes during different mental commands:
        - **REST**: Strong alpha activity across the motor cortex
        - **LEFT** hand imagery: Decreased activity in right motor cortex (C4)
        - **RIGHT** hand imagery: Decreased activity in left motor cortex (C3)
        """)

# Use Streamlit's rerun mechanism to simulate animation if not paused
if not st.session_state.paused:
    time.sleep(0.1)  # Short delay to prevent excessive reruns
    st.experimental_rerun()