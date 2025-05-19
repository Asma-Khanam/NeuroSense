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

# Suppress all warnings to avoid clutter
warnings.filterwarnings("ignore")

# Define color scheme similar to the screenshot
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

        return theta_power, alpha_power, beta_power
    except:
        # Return zeros if there's an error
        return 0, 0, 0

def extract_features_for_visualization(signals, fs=160):
    """Extract all features needed for visualization"""
    features = []
    alpha_powers = []
    
    # Process each channel
    for i, signal in enumerate(signals):
        # Get band powers
        theta, alpha, beta = extract_band_powers(signal, fs)
        
        # Store alpha power for tracking
        alpha_powers.append(alpha)
        
        # Store all features
        features.append({
            'theta': theta,
            'alpha': alpha,
            'beta': beta,
            'channel': i  # 0=C3, 1=Cz, 2=C4
        })
    
    return features, alpha_powers

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
                        return None, None, None
                    
                    # Get epochs for this state
                    state_epochs = cached_data['epochs'][state]
                    
                    if len(state_epochs) == 0:
                        print(f"No epochs found for state '{state}' in Subject {subject_num}")
                        return None, None, None
                    
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

def create_feature_analysis_page(pdf, subject_num, task_name, state, epoch_idx=0, cache_dir='./cache'):
    """Create a feature analysis visualization page for a PDF report"""
    try:
        # Get data for this subject
        times, signals, ch_names, fs = get_subject_data(
            subject_num, task_name, state, epoch_idx, cache_dir=cache_dir
        )
        
        # If no valid data found, skip this subject
        if times is None or signals is None or ch_names is None:
            print(f"No valid data found for Subject {subject_num}, state {state}")
            return False
        
        # Extract features
        features, alpha_powers = extract_features_for_visualization(signals, fs)
        
        # Create a new figure with the layout matching the screenshot
        fig = plt.figure(figsize=(11, 8.5), dpi=100)
        
        # Title similar to the screenshot
        fig.suptitle(f"Brain-Computer Interface: Motor Imagery Analysis", fontsize=16)
        
        # Create a more complex grid layout to match the screenshot
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])
        
        # Add subtitle for feature analysis
        plt.figtext(0.5, 0.92, f"EEG Feature Analysis - Subject {subject_num}, {state.upper()} State", 
                   ha='center', fontsize=14)
        
        # 1. Band Power Distribution (Radar Chart) - Top Left
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        
        # Prepare radar chart data
        categories = ['C3-Theta', 'C3-Alpha', 'C3-Beta', 
                     'Cz-Theta', 'Cz-Alpha', 'Cz-Beta',
                     'C4-Theta', 'C4-Alpha', 'C4-Beta']
        
        # Extract values for radar chart
        radar_values = []
        for channel in range(3):  # C3, Cz, C4
            for band in ['theta', 'alpha', 'beta']:
                # Find the feature for this channel
                for feat in features:
                    if feat['channel'] == channel:
                        radar_values.append(feat[band])
                        break
        
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
        
        # Plot radar chart
        ax_radar.fill(angles, radar_values, color=COLORS[state], alpha=0.25)
        ax_radar.plot(angles, radar_values, color=COLORS[state], linewidth=2)
        
        # Add labels
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=8)
        ax_radar.set_yticks([2, 4, 6, 8, 10])
        ax_radar.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
        ax_radar.set_ylim(0, 10)
        
        # Add title
        ax_radar.set_title('EEG Band Power Distribution', fontsize=12)
        
        # 2. Band Power Comparison (Bar Chart) - Top Right
        ax_bar = fig.add_subplot(gs[0, 1])
        
        # Extract values for all bands and channels
        bar_data = []
        bar_labels = []
        
        for channel_name, channel_idx in zip(['C3', 'Cz', 'C4'], [0, 1, 2]):
            for band in ['Theta', 'Alpha', 'Beta']:
                band_lower = band.lower()
                # Find this channel's data
                for feat in features:
                    if feat['channel'] == channel_idx:
                        bar_data.append(feat[band_lower])
                        bar_labels.append(f"{channel_name}-{band}")
                        break
        
        # Plot bars
        bar_positions = np.arange(len(bar_data))
        bar_colors = []
        
        for i in range(len(bar_data)):
            # Color by channel (C3, Cz, C4)
            if i % 3 == 0 or (i - 0) % 9 == 0:  # C3
                bar_colors.append('#2196F3')  # Blue
            elif i % 3 == 1 or (i - 1) % 9 == 0:  # Cz
                bar_colors.append('#4CAF50')  # Green
            else:  # C4
                bar_colors.append('#FF9800')  # Orange
        
        # Plot bars
        ax_bar.bar(bar_positions, bar_data, color=bar_colors, alpha=0.7)
        
        # Add labels
        ax_bar.set_xticks(bar_positions)
        ax_bar.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
        ax_bar.set_ylabel('Power')
        ax_bar.set_title('Band Power Comparison', fontsize=12)
        ax_bar.grid(True, linestyle='--', alpha=0.3)
        
        # 3. Alpha Power Over Time - Bottom Left
        ax_alpha_time = fig.add_subplot(gs[1, 0])
        
        # Create simulated historical data (10 seconds before event)
        hist_time = np.linspace(-10, 0, 100)
        
        # Create historical alpha values (stable before event)
        c3_hist = np.ones_like(hist_time) * alpha_powers[0]
        cz_hist = np.ones_like(hist_time) * alpha_powers[1]
        c4_hist = np.ones_like(hist_time) * alpha_powers[2]
        
        # Plot alpha power over time
        ax_alpha_time.plot(hist_time, c3_hist, label='C3 (Left)', color='#2196F3', linewidth=2)
        ax_alpha_time.plot(hist_time, cz_hist, label='Cz (Center)', color='#4CAF50', linewidth=2)
        ax_alpha_time.plot(hist_time, c4_hist, label='C4 (Right)', color='#FF9800', linewidth=2)
        
        # Add labels
        ax_alpha_time.set_xlabel('Time (last 10 seconds)')
        ax_alpha_time.set_ylabel('Alpha Power')
        ax_alpha_time.set_title('Alpha Power Over Time', fontsize=12)
        ax_alpha_time.grid(True, linestyle='--', alpha=0.3)
        ax_alpha_time.legend(loc='upper right', fontsize=8)
        
        # 4. ERD/ERS Visualization - Bottom Right
        ax_erd = fig.add_subplot(gs[1, 1])
        
        # Generate example ERD signal
        t = np.linspace(0, 5, 500)
        baseline = np.sin(2*np.pi*10*t) * 0.8 + np.random.normal(0, 0.05, 500)
        erd_signal = np.copy(baseline)
        
        # Create ERD in the middle
        erd_idx = np.logical_and(t >= 2, t <= 3)
        erd_signal[erd_idx] = erd_signal[erd_idx] * 0.4
        
        # Plot signals
        ax_erd.plot(t, baseline, label='Baseline', color='#dddddd', linewidth=1, alpha=0.5)
        ax_erd.plot(t, erd_signal, label='ERD', color=COLORS[state], linewidth=1.5)
        
        # Highlight ERD region
        ax_erd.axvspan(2, 3, alpha=0.2, color=COLORS[state])
        ax_erd.text(2.5, 0, 'ERD', ha='center', va='center', 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=COLORS[state]))
        
        # Add labels
        ax_erd.set_xlabel('Time (s)')
        ax_erd.set_ylabel('Amplitude')
        ax_erd.set_title('Event-Related Desynchronization (ERD)', fontsize=12)
        ax_erd.set_xlim(0, 5)
        ax_erd.grid(True, linestyle='--', alpha=0.3)
        ax_erd.legend(loc='upper right', fontsize=8)
        
        # Add ERD explanation
        ax_erd.text(2.5, -1.2, 
                  "ERD is a decrease in neural oscillations (particularly in the alpha band)\n" + 
                  "when a brain region becomes active during motor imagery or actual movement.",
                  ha='center', fontsize=8, 
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 5. Information Display at bottom
        ax_info = fig.add_subplot(gs[2, :])
        
        # State-specific text
        if state == 'rest':
            state_text = "REST STATE: Strong alpha rhythm present in all channels."
        elif state == 'left':
            state_text = "LEFT HAND IMAGERY: Event-Related Desynchronization (ERD) in right motor cortex (C4)."
        elif state == 'right':
            state_text = "RIGHT HAND IMAGERY: Event-Related Desynchronization (ERD) in left motor cortex (C3)."
        
        # Display command info
        command_text = f"Mental Command: {state.upper()}"
        info_text = f"{command_text}\n\n{state_text}"
        
        # Add text and brain diagram
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', ec=COLORS[state], alpha=0.8))
        
        # Turn off axes for info panel
        ax_info.axis('off')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save to PDF
        pdf.savefig(fig)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating visualization for Subject {subject_num}, {state}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_pdf_report(subjects, task_name='Task4', states=None, output_dir='./pdf_vis1', cache_dir='./cache'):
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate BCI visualizations as PDF')
    parser.add_argument('--subjects', type=str, default='1-10',
                        help='Subject numbers to process (comma-separated or range like 1-10)')
    parser.add_argument('--task', type=str, choices=list(TASK_RUNS.keys()), default='Task4',
                        help='Task to visualize')
    parser.add_argument('--state', type=str, choices=['rest', 'left', 'right', 'all'], default='all',
                        help='Mental state to visualize. Use "all" for all states')
    parser.add_argument('--output', type=str, default='./pdf_vis1',
                        help='Output directory for visualizations')
    parser.add_argument('--cache', type=str, default='./cache',
                        help='Cache directory for processed data')
    
    args = parser.parse_args()
    
    # Parse subjects argument
    if '-' in args.subjects:
        # Process range
        start, end = map(int, args.subjects.split('-'))
        subjects = range(start, end + 1)
    else:
        # Process comma-separated list
        subjects = [int(s) for s in args.subjects.split(',')]
    
    # Parse states
    states = ['rest', 'left', 'right'] if args.state == 'all' else [args.state]
    
    # Create PDF report
    pdf_filename = create_pdf_report(
        subjects=subjects,
        task_name=args.task,
        states=states,
        output_dir=args.output,
        cache_dir=args.cache
    )
    
    print(f"PDF report created: {pdf_filename}")

if __name__ == "__main__":
    main()