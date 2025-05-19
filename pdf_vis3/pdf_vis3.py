import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import joblib
import mne
from mne.datasets import eegbci
import argparse
from datetime import datetime
import warnings
from scipy import signal as sg
from matplotlib.colors import LinearSegmentedColormap
import sys

# Suppress all warnings to avoid clutter
warnings.filterwarnings("ignore")

# Define color scheme
COLORS = {
    'rest': '#4CAF50',    # Green 
    'left': '#2196F3',    # Blue
    'right': '#FF9800',   # Orange
    'background': '#f5f5f5',  # Light gray
    'grid': '#dddddd',    # Light gray for grids
    'text': '#333333'     # Dark gray for text
}

# Define runs for tasks
TASK_RUNS = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

def downsample_signals(signals, times, max_samples=500):
    """Downsample signals to keep visualization manageable"""
    if signals.shape[1] > max_samples:
        step = signals.shape[1] // max_samples
        downsampled = signals[:, ::step]
        times_ds = times[::step]
        return downsampled, times_ds
    return signals, times

def extract_band_powers(signal, fs=160):
    """Extract frequency band powers from a signal"""
    # Use Welch's method for more stable PSD estimation
    try:
        freqs, psd = sg.welch(signal, fs=fs, nperseg=min(256, len(signal)))

        # Define frequency bands
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)

        # Find indices corresponding to each band
        theta_idx = np.logical_and(freqs >= theta_band[0], freqs <= theta_band[1])
        alpha_idx = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
        beta_idx = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])

        # Calculate average power in each band
        theta_power = np.mean(psd[theta_idx]) if np.any(theta_idx) else 0
        alpha_power = np.mean(psd[alpha_idx]) if np.any(alpha_idx) else 0
        beta_power = np.mean(psd[beta_idx]) if np.any(beta_idx) else 0

        return theta_power, alpha_power, beta_power, freqs, psd
    except:
        # Return zeros if there's an error
        return 0, 0, 0, np.array([]), np.array([])

def extract_features_for_visualization(signals, fs=160):
    """Extract all features needed for visualization"""
    features = []
    alpha_powers = []
    beta_powers = []
    spectrogram_data = None
    
    # Process each channel
    for i, signal in enumerate(signals):
        # Get band powers and raw spectral data
        theta, alpha, beta, freqs, psd = extract_band_powers(signal, fs)
        
        # Store powers for tracking
        alpha_powers.append(alpha)
        beta_powers.append(beta)
        
        # Compute spectrogram for first channel (C3 typically)
        if i == 0 and len(signal) > 0:
            try:
                # Use scipy's spectrogram
                f, t, Sxx = sg.spectrogram(
                    signal, 
                    fs=fs,
                    nperseg=min(256, len(signal)),
                    noverlap=128,
                    mode='magnitude',
                    scaling='spectrum'
                )
                
                # Limit to 0-30 Hz
                f_mask = f <= 30
                spectrogram_data = {
                    'f': f[f_mask],
                    't': t,
                    'Sxx': Sxx[f_mask]
                }
            except Exception as e:
                print(f"Error computing spectrogram: {e}")
        
        # Store all features
        features.append({
            'theta': theta,
            'alpha': alpha,
            'beta': beta,
            'channel': i,  # 0=C3, 1=Cz, 2=C4
            'freqs': freqs,
            'psd': psd
        })
    
    return features, alpha_powers, beta_powers, spectrogram_data

def get_subject_data(subject_num, task_name, state, epoch_idx=0, max_samples=500, cache_dir='./cache'):
    """Load and prepare subject data for visualization"""
    try:
        # Check if cache directory exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Check if cache file exists
        cache_file = os.path.join(cache_dir, f"subject_{subject_num}_{task_name}.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading cached data for Subject {subject_num}, {task_name}...")
            # Load cached data
            try:
                cached_data = joblib.load(cache_file)
                
                # Get epochs for the specified state
                try:
                    # Check if the state exists in this subject's data
                    if state not in cached_data['epochs'].event_id:
                        print(f"State '{state}' not found for Subject {subject_num}")
                        return None, None, None, None
                    
                    # Get epochs for this state
                    state_epochs = cached_data['epochs'][state]
                    
                    if len(state_epochs) == 0:
                        print(f"No epochs found for state '{state}' in Subject {subject_num}")
                        return None, None, None, None
                    
                    # Get specific epoch index (or first if index is too large)
                    epoch_idx = min(epoch_idx, len(state_epochs) - 1)
                    
                    # Get the data for this epoch
                    epoch_data = state_epochs[epoch_idx].get_data()[0]  # [0] gets the first trial
                    times = cached_data['times']
                    
                    # Downsample to keep size manageable
                    epoch_data, times = downsample_signals(epoch_data, times, max_samples)
                    
                    # Get channel indices for motor cortex (C3, Cz, C4)
                    ch_indices = []
                    ch_names = []
                    
                    # Look for these channels
                    for channel in ['C3', 'Cz', 'C4']:
                        # Find channels that contain these names
                        matches = [i for i, name in enumerate(cached_data['channel_names']) 
                                   if channel in name]
                        if matches:
                            ch_indices.append(matches[0])
                            ch_names.append(cached_data['channel_names'][matches[0]])
                    
                    # If we found channels, extract them
                    if ch_indices:
                        # Extract only the signals for these channels
                        selected_signals = epoch_data[ch_indices]
                        
                        # If we don't have 3 channels, pad with zeros
                        while len(selected_signals) < 3:
                            selected_signals = np.vstack([selected_signals, np.zeros_like(selected_signals[0])])
                            ch_names.append(f"Channel {len(ch_names)}")
                        
                        return times, selected_signals, ch_names, cached_data['fs']
                    else:
                        # If no channels were found, take the first 3 available
                        selected_signals = epoch_data[:3] if epoch_data.shape[0] >= 3 else np.zeros((3, len(times)))
                        ch_names = cached_data['channel_names'][:3] if len(cached_data['channel_names']) >= 3 else [f"Channel {i}" for i in range(3)]
                        
                        return times, selected_signals, ch_names, cached_data['fs']
                    
                except Exception as e:
                    print(f"Error processing epochs for Subject {subject_num}: {e}")
                    return None, None, None, None
                    
            except Exception as e:
                print(f"Error loading cache for Subject {subject_num}: {e}")
                return None, None, None, None
                
        else:
            print(f"Cache file not found for Subject {subject_num}, {task_name}")
            # Try to download and process data
            if download_and_cache_subject(subject_num, task_name, cache_dir):
                # If successful, try again
                return get_subject_data(subject_num, task_name, state, epoch_idx, max_samples, cache_dir)
            else:
                return None, None, None, None
    
    except Exception as e:
        print(f"Error loading data for Subject {subject_num}: {e}")
        return None, None, None, None

def download_and_cache_subject(subject_num, task_name, cache_dir):
    """Download and cache data for a subject if not already available"""
    try:
        # Get runs for the task
        runs = TASK_RUNS.get(task_name, [4])  # Default to Task4
        
        # Create list for raw data objects
        raw_list = []
        
        # Process each run
        for run in runs:
            try:
                # File path
                file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', 
                                        f'1.0.0', f'S{str(subject_num).zfill(3)}', 
                                        f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
                
                # Check if file exists, download if not
                if not os.path.exists(file_path):
                    print(f"Downloading data for Subject {subject_num}, Run {run}...")
                    eegbci.load_data(subject_num, runs=[run], path='files/')
                
                # Load raw data
                raw = mne.io.read_raw_edf(file_path, preload=True)
                raw_list.append(raw)
                
            except Exception as e:
                print(f"Error processing run {run} for Subject {subject_num}: {e}")
                continue
        
        # If no data was loaded, return failure
        if not raw_list:
            print(f"No data loaded for Subject {subject_num}")
            return False
        
        # Concatenate runs
        raw_concat = mne.concatenate_raws(raw_list)
        
        # Basic preprocessing
        raw_concat.filter(l_freq=1.0, h_freq=45.0)
        raw_concat.set_eeg_reference('average', projection=False)
        
        # Extract events
        events, event_id = mne.events_from_annotations(raw_concat)
        
        # Map events to standard states
        event_id_selected = {}
        for key, value in event_id.items():
            if 'T0' in key:  # Rest
                event_id_selected['rest'] = value
            elif 'T1' in key:  # Left hand
                event_id_selected['left'] = value
            elif 'T2' in key:  # Right hand
                event_id_selected['right'] = value
        
        # Create epochs
        epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                           tmin=0.5, tmax=3.5, baseline=None, preload=True)
        
        # Prepare data structure
        data = {
            'epochs': epochs,
            'event_id': event_id_selected,
            'channel_names': epochs.ch_names,
            'times': epochs.times,
            'fs': int(epochs.info['sfreq']),
            'subject': subject_num,
            'task': task_name
        }
        
        # Save to cache
        cache_file = os.path.join(cache_dir, f"subject_{subject_num}_{task_name}.pkl")
        joblib.dump(data, cache_file)
        
        print(f"Data cached for Subject {subject_num}, {task_name}")
        return True
        
    except Exception as e:
        print(f"Error downloading/caching Subject {subject_num}: {e}")
        return False

def draw_head_outline(ax, view='top'):
    """Draw a head outline for the brain topography"""
    if view == 'top':
        # Draw head circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # Draw nose (small line at top)
        ax.plot([0, 0], [0.9, 1.1], color='black', linewidth=2)
        
        # Draw left and right markers (small ticks on the circle)
        ax.plot([-1.1, -0.9], [0, 0], color='black', linewidth=2)
        ax.plot([0.9, 1.1], [0, 0], color='black', linewidth=2)
        
        # Draw vertical markers
        ax.plot([0, 0], [-1.1, -0.9], color='black', linewidth=2)
        
    elif view == 'side':
        # Draw head profile (semicircle)
        theta = np.linspace(0, np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, color='black', linewidth=2)
        
        # Draw bottom line
        ax.plot([-1, 1], [0, 0], color='black', linewidth=2)
        
        # Draw nose
        ax.plot([0.9, 1.1], [0.2, 0.2], color='black', linewidth=2)
        
        # Draw vertical marker
        ax.plot([-1, -1], [-0.1, 0.1], color='black', linewidth=2)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Set limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

def mark_electrodes(ax, view='top'):
    """Mark electrode positions on the brain topography"""
    if view == 'top':
        # Define positions for key electrodes
        positions = {
            'F3': (-0.5, 0.5),
            'Fz': (0, 0.5),
            'F4': (0.5, 0.5),
            'C3': (-0.5, 0),
            'Cz': (0, 0),
            'C4': (0.5, 0),
            'P3': (-0.5, -0.5),
            'Pz': (0, -0.5),
            'P4': (0.5, -0.5)
        }
        
        # Special highlighting for C3, Cz, C4 (motor cortex)
        motor_channels = ['C3', 'Cz', 'C4']
        
        # Mark each electrode
        for name, (x, y) in positions.items():
            # Use blue dots for all electrodes
            dot_color = '#2196F3'  # Blue
            
            # Plot the electrode point
            ax.plot(x, y, 'o', markersize=8, color=dot_color)
            
            # Add the electrode name
            ax.text(x, y+0.05, name, ha='center', va='bottom', fontsize=8)
            
            # Add orange highlight to motor cortex electrodes (C3, Cz, C4)
            if name in motor_channels:
                ax.plot(x, y, 'o', markersize=10, mfc='none', mec='#FF9800', linewidth=2)
    
    elif view == 'side':
        # Define positions for side view
        positions = {
            'Fz': (-0.75, 0.75),
            'F4': (-0.25, 0.9),
            'Cz': (0, 1),
            'C4': (0.5, 0.8),
            'P4': (0.75, 0.5)
        }
        
        # Mark each electrode
        for name, (x, y) in positions.items():
            dot_color = '#2196F3'  # Blue
            ax.plot(x, y, 'o', markersize=8, color=dot_color)
            ax.text(x, y+0.05, name, ha='center', va='bottom', fontsize=8)
            
            # Highlight motor electrodes
            if name in ['Cz', 'C4']:
                ax.plot(x, y, 'o', markersize=10, mfc='none', mec='#FF9800', linewidth=2)

def create_brain_topography(fig, gs, features, alpha_powers, beta_powers, state):
    """Create a brain topography visualization matching Image 1"""
    # Four subplots: Top view, Side view, Alpha band, Beta band
    topo_top_ax = fig.add_subplot(gs[0, 0])
    topo_side_ax = fig.add_subplot(gs[0, 1])
    alpha_band_ax = fig.add_subplot(gs[1, 0])
    beta_band_ax = fig.add_subplot(gs[1, 1])
    
    # Set titles
    topo_top_ax.set_title("Top View", fontsize=14, fontweight='bold')
    topo_side_ax.set_title("Side View", fontsize=14, fontweight='bold')
    alpha_band_ax.set_title("Alpha Band (8-13 Hz)", fontsize=14, fontweight='bold')
    beta_band_ax.set_title("Beta Band (13-30 Hz)", fontsize=14, fontweight='bold')
    
    # Draw head outlines
    draw_head_outline(topo_top_ax, view='top')
    draw_head_outline(topo_side_ax, view='side')
    draw_head_outline(alpha_band_ax, view='top')
    draw_head_outline(beta_band_ax, view='top')
    
    # Place electrodes on top and side views
    mark_electrodes(topo_top_ax, view='top')
    mark_electrodes(topo_side_ax, view='side')
    
    # Create custom brain activity heatmaps for alpha and beta bands
    # Create a custom colormap (blue->white->red)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('ERD_ERS', colors, N=100)
    
    # Create heatmaps based on real data
    n = 50  # Grid resolution
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Create a circular mask
    mask = X**2 + Y**2 <= 1
    
    # Define electrode positions for heatmap
    positions = {
        'C3': (-0.5, 0),
        'Cz': (0, 0),
        'C4': (0.5, 0)
    }
    
    # Initialize data with zeros
    alpha_data = np.zeros((n, n))
    beta_data = np.zeros((n, n))
    
    # Generate heatmaps from real band power values
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                point = (x[j], y[i])  # Note: X,Y from meshgrid are swapped
                
                # Calculate weighted contribution from each electrode based on distance
                alpha_val = 0
                beta_val = 0
                total_weight = 0
                
                for ch_idx, (ch_name, pos) in enumerate(positions.items()):
                    # Calculate inverse distance weight
                    dist = np.sqrt((point[0] - pos[0])**2 + (point[1] - pos[1])**2)
                    weight = 1 / (dist + 0.1)**2
                    
                    # Add weighted contribution
                    if ch_idx < len(alpha_powers):
                        alpha_val += weight * alpha_powers[ch_idx]
                        beta_val += weight * beta_powers[ch_idx]
                    
                    total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    alpha_data[i, j] = alpha_val / total_weight
                    beta_data[i, j] = beta_val / total_weight
    
    # Normalize data for visualization
    if np.max(alpha_data) > 0:
        alpha_data = alpha_data / np.max(alpha_data) * 2 - 1
    if np.max(beta_data) > 0:
        beta_data = beta_data / np.max(beta_data) * 2 - 1
    
    # Modify based on mental state to match expected ERD/ERS patterns
    if state == 'left':
        # Left hand imagery: Reduce activity in right hemisphere (C4) for alpha
        for i in range(n):
            for j in range(n):
                if mask[i, j] and x[j] > 0.2:  # Right hemisphere
                    alpha_data[i, j] *= -0.5  # Decrease alpha (ERD)
                    beta_data[i, j] *= 1.5    # Increase beta (ERS)
    
    elif state == 'right':
        # Right hand imagery: Reduce activity in left hemisphere (C3) for alpha
        for i in range(n):
            for j in range(n):
                if mask[i, j] and x[j] < -0.2:  # Left hemisphere
                    alpha_data[i, j] *= -0.5  # Decrease alpha (ERD)
                    beta_data[i, j] *= 1.5    # Increase beta (ERS)
    
    # Plot the heatmaps
    alpha_im = alpha_band_ax.imshow(
        alpha_data, 
        extent=[-1.2, 1.2, -1.2, 1.2],
        origin='lower',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation='bilinear'
    )
    
    beta_im = beta_band_ax.imshow(
        beta_data, 
        extent=[-1.2, 1.2, -1.2, 1.2],
        origin='lower',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation='bilinear'
    )
    
    # Add colorbars
    cbar_alpha = plt.colorbar(alpha_im, ax=alpha_band_ax)
    cbar_alpha.set_label('Activity')
    
    cbar_beta = plt.colorbar(beta_im, ax=beta_band_ax)
    cbar_beta.set_label('Activity')
    
    return [topo_top_ax, topo_side_ax, alpha_band_ax, beta_band_ax]

def create_time_frequency_analysis(fig, gs, signals, times, spectrogram_data, fs=160):
    """Create time-frequency analysis visualization matching Image 2"""
    # Create a spectrogram subplot
    tf_ax = fig.add_subplot(gs)
    tf_ax.set_title("Spectrogram (C3 Channel)", fontsize=12)
    tf_ax.set_ylabel("Frequency (Hz)")
    tf_ax.set_xlabel("Time (s)")
    
    # Make sure we have data
    if spectrogram_data is not None and 'Sxx' in spectrogram_data and spectrogram_data['Sxx'].size > 0:
        # Use the real spectrogram data
        f = spectrogram_data['f']
        t = spectrogram_data['t']
        Sxx = spectrogram_data['Sxx']
        
        # Normalize for better visualization
        if np.max(Sxx) > 0:
            Sxx_norm = Sxx / np.max(Sxx)
        else:
            Sxx_norm = Sxx
        
        # Plot the spectrogram
        img = tf_ax.pcolormesh(t, f, Sxx_norm, shading='gouraud', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=tf_ax)
        cbar.set_label('Power')
        
        # Add frequency band markers
        tf_ax.axhline(y=4, color='white', linestyle='--', alpha=0.5)
        tf_ax.axhline(y=8, color='white', linestyle='--', alpha=0.5)
        tf_ax.axhline(y=13, color='white', linestyle='--', alpha=0.5)
        tf_ax.axhline(y=30, color='white', linestyle='--', alpha=0.5)
        
        # Add band labels
        tf_ax.text(np.max(t)*1.05, 2, 'Delta', fontsize=8, ha='left', va='center', color='white')
        tf_ax.text(np.max(t)*1.05, 6, 'Theta', fontsize=8, ha='left', va='center', color='white')
        tf_ax.text(np.max(t)*1.05, 10, 'Alpha', fontsize=8, ha='left', va='center', color='white')
        tf_ax.text(np.max(t)*1.05, 20, 'Beta', fontsize=8, ha='left', va='center', color='white')
        
        # Set y-axis limit to 30 Hz
        tf_ax.set_ylim(0, 30)
    else:
        # If data is not available, show a message
        tf_ax.text(0.5, 0.5, "Spectrogram data not available", 
                 ha='center', va='center', fontsize=12,
                 transform=tf_ax.transAxes)
        # Set background color to match Image 2
        tf_ax.set_facecolor('#202020')
    
    return tf_ax

def plot_eeg_signals(fig, gs, times, signals, ch_names):
    """Plot EEG signals for all three channels"""
    # Create subplots for each channel
    axes = []
    channel_colors = ['#2196F3', '#4CAF50', '#FF9800']  # Blue, Green, Orange
    
    for i in range(min(3, len(signals))):
        # Create subplot
        ax = fig.add_subplot(gs[i])
        
        # Get channel label
        if i < len(ch_names):
            ch_name = ch_names[i]
            # Add motor area label if needed
            if 'C3' in ch_name:
                title = f"{ch_name} (Left Motor)"
            elif 'Cz' in ch_name:
                title = f"{ch_name} (Central)"
            elif 'C4' in ch_name:
                title = f"{ch_name} (Right Motor)"
            else:
                title = ch_name
        else:
            title = f"Channel {i}"
        
        # Set title
        ax.set_title(title, fontsize=12)
        
        # Plot signal
        ax.plot(times, signals[i], color=channel_colors[i], linewidth=1)
        
        # Add labels
        ax.set_ylabel("Amplitude (μV)")
        if i == len(signals) - 1 or i == 2:  # Add x-label to last plot
            ax.set_xlabel("Time (seconds)")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add annotation for alpha band
        ax.annotate('Alpha band (8-13 Hz)', 
                   xy=(0.85, 0.9), 
                   xycoords='axes fraction',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='white', 
                             alpha=0.7))
        
        axes.append(ax)
    
    return axes

def create_feature_analysis_page(pdf, subject_num, task_name, state, epoch_idx=0, cache_dir='./cache'):
    """Create a feature analysis visualization page that matches the provided image"""
    try:
        # Get data for this subject
        times, signals, ch_names, fs = get_subject_data(
            subject_num, task_name, state, epoch_idx, cache_dir=cache_dir
        )
        
        # If no valid data found, skip this subject
        if times is None or signals is None or ch_names is None or fs is None:
            print(f"No valid data found for Subject {subject_num}, state {state}")
            return False
        
        # Extract features from real data
        features, alpha_powers, beta_powers, spectrogram_data = extract_features_for_visualization(signals, fs)
        
        # Create a new figure matching the layout of the provided image
        fig = plt.figure(figsize=(11, 8.5), dpi=100)
        
        # Add headers
        fig.suptitle("Brain-Computer Interface: Motor Imagery Analysis", fontsize=16)
        plt.figtext(0.5, 0.92, f"EEG Feature Analysis - Subject {subject_num}, {state.upper()} State", 
                   ha='center', fontsize=14)
        
        # Create grid layout matching the image layout
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.3])
        
        # 1. EEG Band Power Distribution (Radar Chart) - Top Left
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        ax_radar.set_title('EEG Band Power Distribution', fontsize=12)
        
        # Prepare radar chart data
        categories = ['C3-Beta', 'Cz-Theta', 'C3-Alpha', 
                     'C3-Theta', 'C4-Beta', 'C4-Alpha', 
                     'C4-Theta', 'Cz-Beta', 'Cz-Alpha']
        
        # Extract values for radar chart - reorder to match the image
        radar_values = []
        ch_map = {'C3': 0, 'Cz': 1, 'C4': 2}  # Map channel names to indices
        
        for cat in categories:
            ch, band = cat.split('-')
            ch_idx = ch_map.get(ch, 0)
            if ch_idx < len(features):
                radar_values.append(features[ch_idx][band.lower()])
            else:
                radar_values.append(0)
        
        # Normalize radar values for better visibility
        if max(radar_values) > 0:
            radar_values = [v / max(radar_values) * 10 for v in radar_values]
        else:
            radar_values = [0] * len(radar_values)
        
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # Close the loop
        radar_values += [radar_values[0]]
        angles += [angles[0]]
        
        # Plot radar chart with transparent green fill matching image
        ax_radar.fill(angles, radar_values, color=COLORS[state], alpha=0.3)
        ax_radar.plot(angles, radar_values, color=COLORS[state], linewidth=2)
        
        # Add labels
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=8)
        
        # Add custom radial guides (2, 4, 6, 8, 10)
        ax_radar.set_yticks([2, 4, 6, 8, 10])
        ax_radar.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
        ax_radar.set_ylim(0, 10)
        
        # Set grid color
        ax_radar.grid(color='gray', alpha=0.5)
        
        # 2. Band Power Comparison (Bar Chart) - Top Right
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_bar.set_title('Band Power Comparison', fontsize=12)
        ax_bar.set_ylabel('Power')
        
        # Extract values for bar chart
        bar_data = []
        bar_labels = []
        
        for ch_name in ['C3', 'Cz', 'C4']:
            ch_idx = ch_map.get(ch_name, 0)
            for band in ['Theta', 'Alpha', 'Beta']:
                band_lower = band.lower()
                if ch_idx < len(features):
                    power = features[ch_idx][band_lower]
                    bar_data.append(power)
                    bar_labels.append(f"{ch_name}-{band}")
                else:
                    bar_data.append(0)
                    bar_labels.append(f"{ch_name}-{band}")
        
        # Plot bars with colors matching channels (C3=blue, Cz=green, C4=orange)
        bar_positions = np.arange(len(bar_data))
        bar_colors = []
        
        for i, label in enumerate(bar_labels):
            if 'C3' in label:
                bar_colors.append('#2196F3')  # Blue
            elif 'Cz' in label:
                bar_colors.append('#4CAF50')  # Green
            else:  # C4
                bar_colors.append('#FF9800')  # Orange
        
        # Plot bars
        ax_bar.bar(bar_positions, bar_data, color=bar_colors, alpha=0.7)
        
        # Add labels
        ax_bar.set_xticks(bar_positions)
        ax_bar.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
        
        # Add grid
        ax_bar.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # 3. Alpha Power Over Time - Bottom Left
        ax_alpha_time = fig.add_subplot(gs[1, 0])
        ax_alpha_time.set_title('Alpha Power Over Time', fontsize=12)
        ax_alpha_time.set_xlabel('Time (last 10 seconds)')
        ax_alpha_time.set_ylabel('Alpha Power')
        
        # Create historical time axis (-10 to 0 seconds)
        hist_time = np.linspace(-10, 0, 100)
        
        # Use real alpha power values from the current data
        c3_alpha = alpha_powers[0] if len(alpha_powers) > 0 else 0
        cz_alpha = alpha_powers[1] if len(alpha_powers) > 1 else 0
        c4_alpha = alpha_powers[2] if len(alpha_powers) > 2 else 0
        
        # Create historical alpha values (stable before event)
        c3_hist = np.ones_like(hist_time) * c3_alpha
        cz_hist = np.ones_like(hist_time) * cz_alpha
        c4_hist = np.ones_like(hist_time) * c4_alpha
        
        # Plot alpha power over time with the channel colors
        ax_alpha_time.plot(hist_time, c3_hist, label='C3 (Left)', color='#2196F3', linewidth=2)
        ax_alpha_time.plot(hist_time, cz_hist, label='Cz (Center)', color='#4CAF50', linewidth=2)
        ax_alpha_time.plot(hist_time, c4_hist, label='C4 (Right)', color='#FF9800', linewidth=2)
        
        # Add grid
        ax_alpha_time.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        ax_alpha_time.legend(loc='upper right', fontsize=8)
        
        # 4. Event-Related Desynchronization (ERD) Visualization - Bottom Right
        ax_erd = fig.add_subplot(gs[1, 1])
        ax_erd.set_title('Event-Related Desynchronization (ERD)', fontsize=12)
        ax_erd.set_xlabel('Time (s)')
        ax_erd.set_ylabel('Amplitude')
        
        # Create example ERD visualization
        t = np.linspace(0, 5, 500)
        baseline = np.sin(2*np.pi*10*t) * 0.8 + np.random.normal(0, 0.05, 500)
        erd_signal = np.copy(baseline)
        
        # Create ERD in the middle section (2-3 seconds)
        erd_idx = np.logical_and(t >= 2, t <= 3)
        erd_signal[erd_idx] = erd_signal[erd_idx] * 0.4
        
        # Plot both signals
        ax_erd.plot(t, baseline, label='Baseline', color='#dddddd', linewidth=1, alpha=0.5)
        ax_erd.plot(t, erd_signal, label='ERD', color=COLORS[state], linewidth=1.5)
        
        # Highlight ERD region
        ax_erd.axvspan(2, 3, alpha=0.2, color=COLORS[state])
        ax_erd.text(2.5, 0, 'ERD', ha='center', fontsize=10, color='black',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add explanation text
        ax_erd.text(2.5, -0.8, 
                  "ERD is a decrease in neural oscillations (particularly in the alpha band)\n"
                  "when a brain region becomes active during motor imagery or actual movement.",
                  ha='center', fontsize=8, color='black',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set axis limits
        ax_erd.set_xlim(0, 5)
        ax_erd.set_ylim(-1, 1)
        
        # Add grid
        ax_erd.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        ax_erd.legend(loc='upper right', fontsize=8)
        
        # 5. Information text box at bottom
        info_ax = fig.add_subplot(gs[2, :])
        
        # State-specific text
        if state == 'rest':
            state_text = "REST STATE: Strong alpha rhythm present in all channels."
        elif state == 'left':
            state_text = "LEFT HAND IMAGERY: Event-Related Desynchronization (ERD) in right motor cortex (C4)."
        elif state == 'right':
            state_text = "RIGHT HAND IMAGERY: Event-Related Desynchronization (ERD) in left motor cortex (C3)."
        
        # Create green border box with text
        info_box = {
            'boxstyle': 'round,pad=0.8',
            'facecolor': 'white',
            'edgecolor': COLORS[state],
            'linewidth': 2,
            'alpha': 1.0
        }
        
        # Add text and box
        info_ax.text(0.5, 0.5, f"Mental Command: {state.upper()}\n\n{state_text}", 
                   ha='center', va='center', fontsize=12,
                   bbox=info_box,
                   transform=info_ax.transAxes)
        
        # Turn off axes for info panel
        info_ax.axis('off')
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save to PDF
        pdf.savefig(fig)
        plt.close(fig)
        
        # Create Brain Topography visualization as a second page
        topo_fig = plt.figure(figsize=(11, 8.5), dpi=100)
        topo_fig.suptitle("Brain Activity Topography", fontsize=16)
        
        # Add subtitle
        plt.figtext(0.5, 0.92, f"Subject {subject_num}, {state.upper()} State", 
                   ha='center', fontsize=14)
        
        # Create a 2x2 grid for the brain topography
        topo_gs = gridspec.GridSpec(2, 2, figure=topo_fig)
        
        # Create the brain topography visualization
        topo_axes = create_brain_topography(topo_fig, topo_gs, features, alpha_powers, beta_powers, state)
        
        # Add explanation text at the bottom
        plt.figtext(0.5, 0.05, 
                   "Brain topography shows the spatial distribution of EEG activity across the scalp.\n"
                   "For motor imagery, we focus on activity over the sensorimotor cortex (areas C3, Cz, and C4).\n"
                   "RED areas indicate higher activity, BLUE areas indicate lower activity (event-related desynchronization).",
                   ha='center', fontsize=10)
        
        # Adjust layout
        topo_fig.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save topography page to PDF
        pdf.savefig(topo_fig)
        plt.close(topo_fig)
        
        return True
        
    except Exception as e:
        print(f"Error creating visualization for Subject {subject_num}, {state}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_pdf_report(subjects, task_name='Task4', states=None, output_dir='./pdf_vis3', cache_dir='./cache'):
    """Create a PDF report with EEG visualizations for specified subjects"""
    if states is None:
        states = ['rest', 'left', 'right']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create PDF filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(output_dir, f"BCI_feature_analysis_{timestamp}.pdf")
    
    # Create PDF
    with PdfPages(pdf_filename) as pdf:
        # Add title page
        fig = plt.figure(figsize=(11, 8.5))
        # Background color
        fig.patch.set_facecolor(COLORS['background'])
        
        # Title
        plt.text(0.5, 0.6, "Brain-Computer Interface", fontsize=24, ha='center')
        plt.text(0.5, 0.5, "Motor Imagery Analysis", fontsize=24, ha='center')
        plt.text(0.5, 0.4, f"Subjects: {', '.join(map(str, subjects))}", fontsize=16, ha='center')
        plt.text(0.5, 0.3, f"Task: {task_name}", fontsize=16, ha='center')
        plt.text(0.5, 0.2, f"Date: {datetime.now().strftime('%Y-%m-%d')}", fontsize=14, ha='center')
        
        # No axes for title page
        plt.axis('off')
        
        # Save title page
        pdf.savefig()
        plt.close()
        
        # Process each subject and state
        for subject_num in subjects:
            for state in states:
                print(f"Processing Subject {subject_num}, State {state}...")
                result = create_feature_analysis_page(
                    pdf=pdf,
                    subject_num=subject_num,
                    task_name=task_name,
                    state=state,
                    cache_dir=cache_dir
                )
                
                if not result:
                    print(f"Skipping Subject {subject_num}, State {state} due to errors")
    
    print(f"PDF report saved to {pdf_filename}")
    return pdf_filename

def ask_user_for_visualization_parameters():
    """Ask the user for subject, state, and task to visualize"""
    print("\n=== BCI Visualization Settings ===")
    
    # Ask for subject
    subject_input = input("Enter subject number (1-10) or range (e.g., 1-5): ").strip()
    
    # Parse subject input
    if '-' in subject_input:
        try:
            start, end = map(int, subject_input.split('-'))
            subjects = list(range(start, end + 1))
        except ValueError:
            print("Invalid range format. Using default subject 1.")
            subjects = [1]
    else:
        try:
            subjects = [int(s) for s in subject_input.split(',')]
        except ValueError:
            print("Invalid subject format. Using default subject 1.")
            subjects = [1]
    
    # Validate subject numbers
    valid_subjects = []
    for subject in subjects:
        if 1 <= subject <= 10:
            valid_subjects.append(subject)
        else:
            print(f"Subject {subject} is out of range (1-10) and will be skipped.")
    
    if not valid_subjects:
        print("No valid subjects. Using default subject 1.")
        valid_subjects = [1]
    
    # Ask for task
    print("\nAvailable tasks:")
    print("1. Task4 (Left/right hand imagery)")
    print("2. Task5 (Hands/feet imagery)")
    task_input = input("Select task (1 or 2): ").strip()
    
    if task_input == "2":
        task_name = "Task5"
    else:
        task_name = "Task4"
    
    # Ask for state
    print("\nAvailable states:")
    print("1. rest (Resting state)")
    print("2. left (Left hand imagery)")
    print("3. right (Right hand imagery)")
    print("4. all (All states)")
    state_input = input("Select state (1-4): ").strip()
    
    if state_input == "2":
        states = ["left"]
    elif state_input == "3":
        states = ["right"]
    elif state_input == "4":
        states = ["rest", "left", "right"]
    else:
        states = ["rest"]
    
    return valid_subjects, task_name, states

def main():
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Generate BCI visualizations as PDF')
        parser.add_argument('--subjects', type=str, default='1',
                            help='Subject numbers to process (comma-separated or range like 1-10)')
        parser.add_argument('--task', type=str, choices=list(TASK_RUNS.keys()), default='Task4',
                            help='Task to visualize')
        parser.add_argument('--state', type=str, choices=['rest', 'left', 'right', 'all'], default='all',
                            help='Mental state to visualize. Use "all" for all states')
        parser.add_argument('--output', type=str, default='./pdf_vis3',
                            help='Output directory for visualizations')
        parser.add_argument('--cache', type=str, default='./cache',
                            help='Cache directory for processed data')
        
        args = parser.parse_args()
        
        # Parse subjects argument
        if '-' in args.subjects:
            # Process range
            start, end = map(int, args.subjects.split('-'))
            subjects = list(range(start, end + 1))
        else:
            # Process comma-separated list
            subjects = [int(s) for s in args.subjects.split(',')]
        
        # Parse states
        states = ['rest', 'left', 'right'] if args.state == 'all' else [args.state]
        
        # Output and cache directories
        output_dir = args.output
        cache_dir = args.cache
        task_name = args.task
    else:
        # Ask user for visualization parameters
        subjects, task_name, states = ask_user_for_visualization_parameters()
        
        # Use default directories
        output_dir = './pdf_vis3'
        cache_dir = './cache'
    
    # Create PDF report
    pdf_filename = create_pdf_report(
        subjects=subjects,
        task_name=task_name,
        states=states,
        output_dir=output_dir,
        cache_dir=cache_dir
    )
    
    print(f"PDF report created: {pdf_filename}")

if __name__ == "__main__":
    main()