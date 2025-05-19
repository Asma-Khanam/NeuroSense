import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import joblib
import mne
from mne.datasets import eegbci
import argparse
from datetime import datetime

# Professional color scheme
COLORS = {
    'background': '#222222',      # Dark background
    'text': '#f0f0f0',            # Light text
    'primary': '#2196F3',         # Blue
    'secondary': '#4CAF50',       # Green
    'accent': '#FF9800',          # Orange
    'grid': '#444444',            # Grid lines
    'states': {
        'rest': '#4CAF50',        # Green
        'left': '#2196F3',        # Blue
        'right': '#FF9800'        # Orange
    }
}

# Define runs for task 4 and task 5
TASK_RUNS = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

def generate_simulated_data(state, duration=5, fs=250):
    """Generate simulated EEG data for visualization"""
    time_axis = np.linspace(0, duration, int(duration * fs))
    
    # Base signal (alpha rhythm at 10Hz)
    base_signal = np.sin(2 * np.pi * 10 * time_axis)
    noise_level = 0.5
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(time_axis))
    
    # Different patterns for each channel and each state
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
    
    signals = np.array([c3, cz, c4])
    return time_axis, signals

def extract_simple_features(signals):
    """Extract features from EEG signals for visualization"""
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
    
    return np.array(features)

def load_subject_data(subject_num, task_name, cache_dir='./cache', download_if_missing=True):
    """Load EEG data for a specific subject and task, with caching for efficiency"""
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file path
    cache_file = os.path.join(cache_dir, f"subject_{subject_num}_{task_name}.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading cached data for Subject {subject_num}, {task_name}...")
        cached_data = joblib.load(cache_file)
        return cached_data
    
    print(f"Processing data for Subject {subject_num}, {task_name}...")
    
    # Get runs for the selected task
    runs = TASK_RUNS[task_name]
    
    # Process subject data
    raw_list = []
    
    for run in runs:
        # File path for the EEG data
        file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                               f'S{str(subject_num).zfill(3)}', 
                               f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
        
        if not os.path.exists(file_path) and download_if_missing:
            print(f"Downloading data for Subject {subject_num}, Run {run}...")
            eegbci.load_data(subject_num, runs=[run], path='files/')
        
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw_list.append(raw)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not raw_list:
        print(f"No data found for Subject {subject_num}, {task_name}")
        return None
    
    # Concatenate all runs
    raw_concat = mne.concatenate_raws(raw_list)
    
    # Set EEG reference to average
    raw_concat.set_eeg_reference('average', projection=False)
    
    # Basic preprocessing
    raw_concat.filter(l_freq=1.0, h_freq=45.0)
    
    # Extract events
    events, event_id = mne.events_from_annotations(raw_concat)
    
    # Select relevant events for motor imagery
    event_id_selected = {}
    for key, value in event_id.items():
        if 'T0' in key:  # Rest
            event_id_selected['rest'] = value
        elif 'T1' in key:  # Left hand
            event_id_selected['left'] = value
        elif 'T2' in key:  # Right hand
            event_id_selected['right'] = value
    
    # Create epochs - use a shorter time window to avoid excessive data
    epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                       tmin=0.5, tmax=3.5, baseline=None, preload=True)
    
    # Prepare the data structure
    data = {
        'epochs': epochs,
        'event_id': event_id_selected,
        'channel_names': epochs.ch_names,
        'times': epochs.times,
        'fs': int(epochs.info['sfreq']),
        'subject': subject_num,
        'task': task_name
    }
    
    # Cache the processed data
    print(f"Saving processed data to cache...")
    joblib.dump(data, cache_file)
    
    return data

def generate_spectrogram(signal, fs):
    """Generate a spectrogram for visualization"""
    from scipy import signal as sg
    
    # Calculate spectrogram
    f, t, Sxx = sg.spectrogram(
        signal, 
        fs=fs, 
        nperseg=min(64, len(signal)//8),
        noverlap=32,
        scaling='density', 
        mode='magnitude'
    )
    
    # Limit to frequencies of interest (0-30 Hz)
    freq_mask = f <= 30
    f = f[freq_mask]
    Sxx = Sxx[freq_mask]
    
    # Normalize
    if Sxx.max() > 0:
        Sxx = Sxx / Sxx.max()
    
    return f, t, Sxx

def draw_head_outline(ax, view='top'):
    """Draw a simple head outline on the given axis"""
    if view == 'top':
        # Draw head circle
        circle = plt.Circle((0, 0), 1, fill=False, color=COLORS['text'], linewidth=2)
        ax.add_patch(circle)
        
        # Draw nose
        ax.plot([0, 0], [0.9, 1.1], color=COLORS['text'], linewidth=2)
        
        # Draw ears
        ax.plot([-1.1, -0.9], [0, 0], color=COLORS['text'], linewidth=2)
        ax.plot([0.9, 1.1], [0, 0], color=COLORS['text'], linewidth=2)
        
    elif view == 'side':
        # Draw head profile
        x = np.linspace(-1, 1, 100)
        y_top = np.sqrt(1 - x**2)
        
        # Head outline
        ax.plot(x, y_top, color=COLORS['text'], linewidth=2)
        ax.plot([-1, 1], [0, 0], color=COLORS['text'], linewidth=2)
        
        # Nose
        ax.plot([0.9, 1.1], [0.2, 0.2], color=COLORS['text'], linewidth=2)
        
        # Ear
        ax.plot([-1.05, -1.05], [-0.2, 0.2], color=COLORS['text'], linewidth=2)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Set limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

def mark_electrodes(ax, view='top'):
    """Mark electrode positions on the given axis"""
    if view == 'top':
        # Define positions for key electrodes (simplified)
        positions = {
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
        
        # Mark each electrode
        for name, (x, y) in positions.items():
            ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
            ax.text(x, y+0.1, name, ha='center', va='center', fontsize=8, 
                    color=COLORS['text'])
            
            # Highlight motor cortex electrodes
            if name in ['C3', 'Cz', 'C4']:
                ax.plot(x, y, 'o', markersize=10, mfc='none', 
                        mec=COLORS['accent'], linewidth=2)
    
    elif view == 'side':
        # Define positions for side view (simplified)
        positions = {
            'C3': (-0.5, 0.75),
            'Cz': (0, 1),
            'C4': (0.5, 0.75),
            'F3': (-0.75, 0.5),
            'Fz': (0, 0.75),
            'F4': (0.75, 0.5),
        }
        
        # Mark each electrode
        for name, (x, y) in positions.items():
            if x >= -0.2:  # Only show electrodes visible from this side
                ax.plot(x, y, 'o', markersize=8, color=COLORS['primary'])
                ax.text(x, y+0.1, name, ha='center', va='center', 
                        fontsize=8, color=COLORS['text'])
                
                # Highlight motor cortex electrodes
                if name in ['Cz', 'C4']:
                    ax.plot(x, y, 'o', markersize=10, mfc='none', 
                            mec=COLORS['accent'], linewidth=2)

def create_topo_heatmap(ax, data, title):
    """Create a topographic heatmap on the given axis"""
    # Create a grid for the heatmap
    n = 20
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Create a circular mask
    mask = X**2 + Y**2 <= 1
    
    # Create a custom colormap (blue=low, red=high)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('ERD_ERS', colors, N=100)
    
    # Apply mask to data
    display_data = np.zeros((n, n))
    display_data[mask] = data[mask] if data is not None else np.random.normal(0, 0.5, np.sum(mask))
    
    # Plot the heatmap
    im = ax.imshow(
        display_data, 
        extent=[-1.2, 1.2, -1.2, 1.2],
        origin='lower',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation='bilinear'
    )
    
    # Set title
    ax.set_title(title, fontsize=10, color=COLORS['text'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Activity', fontsize=8, color=COLORS['text'])
    
    return im

def generate_alpha_beta_topography(state):
    """Generate simulated topography data based on mental state"""
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
    if state == 'rest':
        # Rest: Strong alpha everywhere
        alpha_data[mask] = np.random.normal(0.8, 0.1, np.sum(mask))
        
    elif state == 'left':
        # Left: ERD in right motor cortex (C4)
        # Create a right-sided focus
        right_mask = np.logical_and(mask, X > 0.2)
        alpha_data[right_mask] = np.random.normal(-0.5, 0.2, np.sum(right_mask))
        beta_data[right_mask] = np.random.normal(0.6, 0.2, np.sum(right_mask))
        
    elif state == 'right':
        # Right: ERD in left motor cortex (C3)
        # Create a left-sided focus
        left_mask = np.logical_and(mask, X < -0.2)
        alpha_data[left_mask] = np.random.normal(-0.5, 0.2, np.sum(left_mask))
        beta_data[left_mask] = np.random.normal(0.6, 0.2, np.sum(left_mask))
    
    return alpha_data, beta_data

def visualize_subject_state(subject_data=None, state='rest', epoch_idx=0, use_real_data=False,
                           output_dir='./visualizations'):
    """Create a comprehensive visualization of EEG data for a specific subject and state"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Set up the figure with explicit constraints
        plt.rcParams['figure.max_open_warning'] = 0
        fig = plt.figure(figsize=(15, 10), dpi=100, facecolor=COLORS['background'])
        
        # Use constrained_layout instead of tight_layout
        fig.set_constrained_layout(True)
        
        # Create a more compact and well-structured grid
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Add title with subject info
        subject_info = f"Subject {subject_data['subject']}" if use_real_data and subject_data else "Simulated Data"
        plt.suptitle(f"Brain-Computer Interface: {state.upper()} State - {subject_info}", 
                    color=COLORS['text'], fontsize=16, y=0.98)
        
        # EEG Signal Visualization (Row 1)
        # -------------------------------
        
        # Get data based on mode
        if use_real_data and subject_data:
            # Select the specified epoch
            epochs = subject_data['epochs']
            
            # Find epochs with the specified state
            try:
                state_epochs = epochs[state]
                
                if len(state_epochs) == 0:
                    print(f"No epochs found for state '{state}'")
                    return None
                
                # Select the specific epoch index (or first if index is out of range)
                epoch_idx = min(epoch_idx, len(state_epochs) - 1)
                
                # Get EEG data for the selected epoch
                # IMPORTANT: Limit the amount of data to avoid huge visualizations
                # We'll sample the data to reduce points if needed
                epoch_data = state_epochs[epoch_idx].get_data()[0]
                
                # If epoch data is too large, downsample it
                max_samples = 1000  # Maximum number of samples to display
                if epoch_data.shape[1] > max_samples:
                    # Downsample by taking every nth point
                    step = epoch_data.shape[1] // max_samples
                    epoch_data = epoch_data[:, ::step]
                    time_axis = subject_data['times'][::step]
                else:
                    time_axis = subject_data['times']
                
                fs = subject_data['fs']
                
                # Select 3 channels (C3, Cz, C4 or closest available)
                channels = ['C3', 'Cz', 'C4']
                channel_indices = []
                channel_names = []
                
                for ch in channels:
                    # Find channels that match the pattern
                    matches = [i for i, name in enumerate(subject_data['channel_names']) if ch in name]
                    if matches:
                        channel_indices.append(matches[0])
                        channel_names.append(subject_data['channel_names'][matches[0]])
                
                # If we don't have 3 channels, use the first available ones
                while len(channel_indices) < 3 and len(channel_indices) < len(subject_data['channel_names']):
                    if len(subject_data['channel_names']) > len(channel_indices):
                        next_idx = len(channel_indices)
                        if next_idx not in channel_indices:
                            channel_indices.append(next_idx)
                            channel_names.append(subject_data['channel_names'][next_idx])
                
                # Extract data for selected channels
                signals = epoch_data[channel_indices] if channel_indices else np.zeros((3, len(time_axis)))
            
            except Exception as e:
                print(f"Error processing epoch data: {e}")
                # Fallback to simulated data
                print("Falling back to simulated data")
                time_axis, signals = generate_simulated_data(state)
                fs = 250  # default for simulated data
                channel_names = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']
        else:
            # Generate simulated data
            time_axis, signals = generate_simulated_data(state)
            fs = 250  # default for simulated data
            channel_names = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']
        
        # Plot EEG signals
        channel_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        for i, (ch_name, signal) in enumerate(zip(channel_names, signals)):
            ax = fig.add_subplot(gs[i, 0:2])
            ax.set_facecolor(COLORS['background'])
            
            # Plot the EEG signal
            ax.plot(time_axis, signal, color=channel_colors[i], linewidth=1.5)
            
            # Customize appearance
            ax.set_title(ch_name, color=COLORS['text'], fontsize=12)
            ax.set_ylabel('Amplitude (μV)', color=COLORS['text'], fontsize=10)
            
            if i == len(signals) - 1:
                ax.set_xlabel('Time (s)', color=COLORS['text'], fontsize=10)
                
            ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
            ax.tick_params(colors=COLORS['text'])
            
            for spine in ax.spines.values():
                spine.set_color(COLORS['grid'])
        
        # Feature Analysis (Row 2, Left)
        # ------------------------------
        
        # Extract features
        features = extract_simple_features(signals)
        
        # Set up radar chart for band powers
        ax_radar = fig.add_subplot(gs[0, 2], polar=True)
        ax_radar.set_facecolor(COLORS['background'])
        ax_radar.set_title('Band Power Distribution', color=COLORS['text'], fontsize=12)
        
        # Prepare radar chart data
        categories = ['C3-Theta', 'C3-Alpha', 'C3-Beta', 
                     'Cz-Theta', 'Cz-Alpha', 'Cz-Beta',
                     'C4-Theta', 'C4-Alpha', 'C4-Beta']
        
        # Set up the angles for the radar chart
        num_vars = len(categories)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Use the first 9 features (band powers)
        values = features[:9].tolist()
        values += values[:1]  # Close the loop
        
        # Plot radar chart
        ax_radar.plot(angles, values, color=COLORS['primary'], linewidth=2)
        ax_radar.fill(angles, values, color=COLORS['primary'], alpha=0.25)
        
        # Set radar chart ticks and labels
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, color=COLORS['text'], fontsize=8)
        ax_radar.tick_params(colors=COLORS['text'])
        
        # Bar chart for feature comparison
        ax_bar = fig.add_subplot(gs[1, 0:2])
        ax_bar.set_facecolor(COLORS['background'])
        ax_bar.set_title('Band Power by Channel', color=COLORS['text'], fontsize=12)
        
        # Simplify features for clarity - show just alpha and beta for each channel
        simple_features = [
            features[1], features[7],  # Alpha: C3, C4
            features[2], features[8]   # Beta: C3, C4
        ]
        labels = ['C3 Alpha', 'C4 Alpha', 'C3 Beta', 'C4 Beta']
        colors = [COLORS['primary'], COLORS['accent'], COLORS['primary'], COLORS['accent']]
        
        # Create bars
        bars = ax_bar.bar(range(len(simple_features)), simple_features, color=colors, alpha=0.7)
        
        # Add comparison arrows/annotations for alpha and beta comparisons
        if state != 'rest':
            # For left imagery: C4 Alpha should be lower than C3 Alpha (ERD in contralateral hemisphere)
            # For right imagery: C3 Alpha should be lower than C4 Alpha (ERD in contralateral hemisphere)
            if state == 'left' and simple_features[0] > simple_features[1]:
                ax_bar.annotate('ERD', xy=(1, simple_features[1]), 
                              xytext=(1, simple_features[1] + 0.5),
                              arrowprops=dict(arrowstyle='->'), 
                              color=COLORS['text'], ha='center')
            
            elif state == 'right' and simple_features[1] > simple_features[0]:
                ax_bar.annotate('ERD', xy=(0, simple_features[0]), 
                              xytext=(0, simple_features[0] + 0.5),
                              arrowprops=dict(arrowstyle='->'), 
                              color=COLORS['text'], ha='center')
        
        # Customize appearance
        ax_bar.set_xticks(range(len(simple_features)))
        ax_bar.set_xticklabels(labels, color=COLORS['text'], fontsize=10)
        ax_bar.set_ylabel('Power', color=COLORS['text'], fontsize=10)
        ax_bar.tick_params(colors=COLORS['text'])
        ax_bar.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        # Spectrogram (Row 2, Right)
        # --------------------------
        
        # Create spectrogram for C3 channel (index 0)
        ax_spec = fig.add_subplot(gs[1, 2])
        ax_spec.set_facecolor(COLORS['background'])
        ax_spec.set_title('Spectrogram (C3)', color=COLORS['text'], fontsize=12)
        
        # Generate spectrogram data
        _, _, Sxx = generate_spectrogram(signals[0], fs)
        
        # Plot spectrogram
        spec = ax_spec.imshow(Sxx, aspect='auto', origin='lower', cmap='viridis',
                             extent=[0, time_axis[-1], 0, 30])
        
        # Add frequency band markers
        ax_spec.axhline(y=4, color='white', linestyle='--', alpha=0.5)
        ax_spec.axhline(y=8, color='white', linestyle='--', alpha=0.5)
        ax_spec.axhline(y=13, color='white', linestyle='--', alpha=0.5)
        
        # Add band labels
        ax_spec.text(time_axis[-1] + 0.1, 6, 'Theta', fontsize=8, va='center', color=COLORS['text'])
        ax_spec.text(time_axis[-1] + 0.1, 10, 'Alpha', fontsize=8, va='center', color=COLORS['text'])
        ax_spec.text(time_axis[-1] + 0.1, 20, 'Beta', fontsize=8, va='center', color=COLORS['text'])
        
        # Customize appearance
        ax_spec.set_xlabel('Time (s)', color=COLORS['text'], fontsize=10)
        ax_spec.set_ylabel('Frequency (Hz)', color=COLORS['text'], fontsize=10)
        ax_spec.tick_params(colors=COLORS['text'])
        
        # Add colorbar
        cbar = plt.colorbar(spec, ax=ax_spec, shrink=0.8)
        cbar.set_label('Power', color=COLORS['text'], fontsize=8)
        cbar.ax.tick_params(colors=COLORS['text'])
        
        # Brain Topography (Row 3)
        # ------------------------
        
        # Generate alpha and beta topography data
        alpha_data, beta_data = generate_alpha_beta_topography(state)
        
        # Alpha band topography
        ax_topo_alpha = fig.add_subplot(gs[2, 0])
        ax_topo_alpha.set_facecolor(COLORS['background'])
        
        # Draw head outline and electrodes
        draw_head_outline(ax_topo_alpha)
        mark_electrodes(ax_topo_alpha)
        
        # Create heatmap
        create_topo_heatmap(ax_topo_alpha, alpha_data, 'Alpha Band (8-13 Hz)')
        
        # Beta band topography
        ax_topo_beta = fig.add_subplot(gs[2, 1])
        ax_topo_beta.set_facecolor(COLORS['background'])
        
        # Draw head outline and electrodes
        draw_head_outline(ax_topo_beta)
        mark_electrodes(ax_topo_beta)
        
        # Create heatmap
        create_topo_heatmap(ax_topo_beta, beta_data, 'Beta Band (13-30 Hz)')
        
        # ERD Explanation (Row 3, Right)
        # ------------------------------
        
        ax_erd = fig.add_subplot(gs[2, 2])
        ax_erd.set_facecolor(COLORS['background'])
        ax_erd.set_title('Event-Related Desynchronization (ERD)', color=COLORS['text'], fontsize=12)
        
        # Create a schematic of ERD
        t = np.linspace(0, 5, 500)
        baseline = np.sin(2*np.pi*10*t) * 0.8 + np.random.normal(0, 0.1, 500)
        erd_signal = np.copy(baseline)
        
        # Create ERD in the middle
        erd_idx = np.logical_and(t >= 2, t <= 3)
        erd_signal[erd_idx] = erd_signal[erd_idx] * 0.4  # Reduce amplitude during ERD
        
        # Plot ERD demonstration
        ax_erd.plot(t, baseline, color='gray', alpha=0.5, linewidth=1, label='Baseline')
        ax_erd.plot(t, erd_signal, color=COLORS['primary'], linewidth=1.5, label='ERD')
        
        # Highlight ERD period
        ax_erd.axvspan(2, 3, alpha=0.2, color=COLORS['primary'])
        ax_erd.text(2.5, 0, 'ERD', ha='center', fontsize=10, color=COLORS['text'],
                  bbox=dict(boxstyle="round,pad=0.3", fc='black', ec=COLORS['primary'], alpha=0.7))
        
        # Add explanation text
        ax_erd.text(2.5, -1.5, 
                  "ERD is a decrease in neural oscillations\n"
                  "when a brain region becomes active during\n"
                  "motor imagery or actual movement.",
                  ha='center', fontsize=8, color=COLORS['text'])
        
        # Customize appearance
        ax_erd.set_xlim(0, 5)
        ax_erd.set_ylim(-2, 2)
        ax_erd.set_xticks([])
        ax_erd.set_yticks([])
        ax_erd.legend(fontsize=8, loc='upper right', framealpha=0.5)
        
        # Add state-specific annotations
        if state == 'rest':
            info_text = "REST state: Strong alpha rhythms present in both hemispheres"
        elif state == 'left':
            info_text = "LEFT hand imagery: ERD in right motor cortex (C4)"
        elif state == 'right':
            info_text = "RIGHT hand imagery: ERD in left motor cortex (C3)"
        
        # Add information text at the bottom
        fig.text(0.5, 0.01, info_text, color=COLORS['states'][state], 
                 ha='center', fontsize=12, weight='bold')
        
        # Save the figure
        subject_str = f"subject_{subject_data['subject']}" if use_real_data and subject_data else "simulated"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if use_real_data and subject_data:
            filename = f"{output_dir}/{subject_str}_{subject_data['task']}_{state}_epoch{epoch_idx}_{timestamp}.png"
        else:
            filename = f"{output_dir}/{subject_str}_{state}_{timestamp}.png"
        
        # Use a more controlled approach to save the figure
        # Limit the DPI to avoid huge image files
        plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close(fig)
        
        print(f"Visualization saved to {filename}")
        return filename
    
    except Exception as e:
        print(f"Error generating visualization for {state}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate EEG visualizations for BCI data')
    parser.add_argument('--subject', type=int, default=None, 
                        help='Subject number (1-10). If not provided, simulated data will be used')
    parser.add_argument('--task', type=str, choices=list(TASK_RUNS.keys()), default='Task4',
                        help='Task to visualize')
    parser.add_argument('--state', type=str, choices=['rest', 'left', 'right', 'all'], default='all',
                        help='Mental state to visualize. Use "all" for all states')
    parser.add_argument('--output', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--cache', type=str, default='./cache',
                        help='Cache directory for processed data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine states to visualize
    states = ['rest', 'left', 'right'] if args.state == 'all' else [args.state]
    
    # Check if using real data
    use_real_data = args.subject is not None
    
    if use_real_data:
        # Load subject data
        subject_data = load_subject_data(args.subject, args.task, cache_dir=args.cache)
        
        if subject_data is None:
            print(f"No data available for Subject {args.subject}, {args.task}")
            use_real_data = False
            subject_data = None
    else:
        subject_data = None
    
    # Generate visualizations for each state
    for state in states:
        try:
            # Generate visualization
            visualize_subject_state(
                subject_data=subject_data,
                state=state,
                epoch_idx=0,  # Use first available epoch
                use_real_data=use_real_data,
                output_dir=args.output
            )
        except Exception as e:
            print(f"Error generating visualization for {state}: {e}")
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()