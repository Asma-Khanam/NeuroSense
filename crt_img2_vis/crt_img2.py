import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import mne
from mne.datasets import eegbci
import argparse
from datetime import datetime
import warnings
import concurrent.futures
from scipy import signal as sg

# Suppress all warnings to avoid clutter
warnings.filterwarnings("ignore")

# Define color scheme
COLORS = {
    'rest': '#4CAF50',    # Green
    'left': '#2196F3',    # Blue
    'right': '#FF9800'    # Orange
}

# Define runs for tasks
TASK_RUNS = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

def downsample_signals(signals, times, max_samples=300):
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

def get_subject_data(subject_num, task_name, state, epoch_idx=0, max_samples=300, cache_dir='./cache'):
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
                        
                        return times, selected_signals, ch_names
                    else:
                        # If no channels were found, take the first 3 available
                        selected_signals = epoch_data[:3] if epoch_data.shape[0] >= 3 else np.zeros((3, len(times)))
                        ch_names = cached_data['channel_names'][:3] if len(cached_data['channel_names']) >= 3 else [f"Channel {i}" for i in range(3)]
                        
                        return times, selected_signals, ch_names
                    
                except Exception as e:
                    print(f"Error processing epochs for Subject {subject_num}: {e}")
                    return None, None, None
                    
            except Exception as e:
                print(f"Error loading cache for Subject {subject_num}: {e}")
                return None, None, None
                
        else:
            print(f"Cache file not found for Subject {subject_num}, {task_name}")
            # Try to download and process data
            if download_and_cache_subject(subject_num, task_name, cache_dir):
                # If successful, try again
                return get_subject_data(subject_num, task_name, state, epoch_idx, max_samples, cache_dir)
            else:
                return None, None, None
    
    except Exception as e:
        print(f"Error loading data for Subject {subject_num}: {e}")
        return None, None, None

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

def create_visualization(subject_num, task_name, state, epoch_idx=0, output_dir='./crt_img2_vis', cache_dir='./cache'):
    """Create visualization for a subject's data"""
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Get data for this subject
        times, signals, ch_names = get_subject_data(
            subject_num, task_name, state, epoch_idx, cache_dir=cache_dir
        )
        
        # If no valid data found, skip this subject
        if times is None or signals is None or ch_names is None:
            print(f"No valid data found for Subject {subject_num}, state {state}")
            return None
        
        # Create the figure
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        
        # Define a grid layout
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # Title with subject info
        plt.suptitle(f"Subject {subject_num} - {state.upper()} State", fontsize=14)
        
        # Plot EEG signals (left column)
        for i in range(min(3, len(signals))):
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(times, signals[i], linewidth=1, color=COLORS[state])
            ax.set_title(ch_names[i] if i < len(ch_names) else f"Channel {i}")
            ax.set_ylabel('μV')
            
            if i == 2:  # Only add x-label to bottom plot
                ax.set_xlabel('Time (s)')
                
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # Extract band powers for each channel
        features = []
        for signal in signals:
            theta, alpha, beta = extract_band_powers(signal)
            features.extend([theta, alpha, beta])
        
        # Plot bar chart of key features
        ax_bar = fig.add_subplot(gs[0, 1])
        
        # Focus on alpha band powers for C3 and C4 (index 1 and 7)
        alpha_powers = [features[1], features[7]]  # C3 Alpha, C4 Alpha
        
        # Create labels and bars
        labels = ['C3 Alpha', 'C4 Alpha']
        alpha_colors = [COLORS['left'], COLORS['right']]
        
        # Create bars
        ax_bar.bar(range(len(alpha_powers)), alpha_powers, color=alpha_colors, alpha=0.7)
        ax_bar.set_xticks(range(len(alpha_powers)))
        ax_bar.set_xticklabels(labels)
        ax_bar.set_title('Alpha Band Power')
        
        # Add ERD indicators if appropriate
        if state == 'left' and alpha_powers[0] > alpha_powers[1]:
            ax_bar.annotate('ERD', xy=(1, alpha_powers[1]), 
                          xytext=(1, alpha_powers[1] + max(alpha_powers)/5),
                          arrowprops=dict(arrowstyle='->'), 
                          ha='center', fontsize=10)
        elif state == 'right' and alpha_powers[1] > alpha_powers[0]:
            ax_bar.annotate('ERD', xy=(0, alpha_powers[0]), 
                          xytext=(0, alpha_powers[0] + max(alpha_powers)/5),
                          arrowprops=dict(arrowstyle='->'), 
                          ha='center', fontsize=10)
        
        # Draw a schematic head with electrode positions
        ax_head = fig.add_subplot(gs[1, 1])
        
        # Draw a circular head
        circle = plt.Circle((0, 0), 1, fill=False, color='black')
        ax_head.add_patch(circle)
        ax_head.plot([0, 0], [0.8, 1.1], 'k-')  # Nose
        
        # Add electrodes
        ax_head.plot(-0.5, 0, 'o', markersize=12, color='gray')  # C3
        ax_head.plot(0, 0, 'o', markersize=12, color='gray')     # Cz
        ax_head.plot(0.5, 0, 'o', markersize=12, color='gray')   # C4
        
        # Label electrodes
        ax_head.text(-0.5, -0.2, 'C3', ha='center')
        ax_head.text(0, -0.2, 'Cz', ha='center')
        ax_head.text(0.5, -0.2, 'C4', ha='center')
        
        # Highlight based on imagery state
        if state == 'left':
            # Highlight C4 (right motor cortex)
            ax_head.plot(0.5, 0, 'o', markersize=15, mfc='none', mec=COLORS['left'], linewidth=2)
            ax_head.text(0.5, 0.2, 'ERD', color=COLORS['left'], ha='center')
        elif state == 'right':
            # Highlight C3 (left motor cortex)
            ax_head.plot(-0.5, 0, 'o', markersize=15, mfc='none', mec=COLORS['right'], linewidth=2)
            ax_head.text(-0.5, 0.2, 'ERD', color=COLORS['right'], ha='center')
            
        ax_head.set_xlim(-1.2, 1.2)
        ax_head.set_ylim(-1.2, 1.2)
        ax_head.set_aspect('equal')
        ax_head.axis('off')
        ax_head.set_title('ERD Location')
        
        # Plot frequency spectrum (FFT)
        ax_fft = fig.add_subplot(gs[2, 1])
        
        # Create FFT for the signal with most evident ERD
        signal_idx = 2 if state == 'left' else 0 if state == 'right' else 1
        
        # Compute frequency spectrum
        signal = signals[min(signal_idx, len(signals)-1)]
        freqs = np.fft.rfftfreq(len(signal), d=1.0/160)
        fft_vals = np.abs(np.fft.rfft(signal))
        
        # Plot only up to 45 Hz
        freq_mask = freqs <= 45
        ax_fft.plot(freqs[freq_mask], fft_vals[freq_mask], color=COLORS[state])
        
        # Add frequency band markers
        ax_fft.axvspan(4, 8, color=COLORS[state], alpha=0.1)    # Theta
        ax_fft.axvspan(8, 13, color=COLORS[state], alpha=0.2)   # Alpha
        ax_fft.axvspan(13, 30, color=COLORS[state], alpha=0.1)  # Beta
        
        # Add labels
        ax_fft.text(6, max(fft_vals[freq_mask])*0.8, 'θ', fontsize=10, ha='center')
        ax_fft.text(10.5, max(fft_vals[freq_mask])*0.8, 'α', fontsize=10, ha='center')
        ax_fft.text(21.5, max(fft_vals[freq_mask])*0.8, 'β', fontsize=10, ha='center')
        
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Amplitude')
        ax_fft.set_title('Frequency Spectrum')
        
        # Add explanation of the state
        explanation_text = {
            'rest': "REST STATE\nStrong alpha rhythm (8-13 Hz)\npresent in both hemispheres",
            'left': "LEFT HAND IMAGERY\nEvent-Related Desynchronization (ERD)\nin right motor cortex (C4)",
            'right': "RIGHT HAND IMAGERY\nEvent-Related Desynchronization (ERD)\nin left motor cortex (C3)"
        }
        
        # Add textbox with explanation
        ax_text = fig.add_subplot(gs[2, 0])
        ax_text.text(0.5, 0.5, explanation_text[state], 
                   ha='center', va='center', 
                   fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   color=COLORS[state])
        ax_text.axis('off')
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/subject_{subject_num}_{task_name}_{state}_epoch{epoch_idx}_{timestamp}.png"
        
        # Save figure
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Visualization saved to {filename}")
        return filename
    
    except Exception as e:
        print(f"Error creating visualization for Subject {subject_num}, {state}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_subjects(subjects=range(1, 11), task_name='Task4', states=None, output_dir='./crt_img2_vis', cache_dir='./cache'):
    """Process and visualize data for all subjects"""
    if states is None:
        states = ['rest', 'left', 'right']
    
    total = len(subjects) * len(states)
    count = 0
    
    # Create a progress tracker
    print(f"Processing {len(subjects)} subjects with {len(states)} states each...")
    
    for subject_num in subjects:
        for state in states:
            count += 1
            print(f"Progress: {count}/{total} - Subject {subject_num}, State: {state}")
            
            # Process this subject and state
            create_visualization(
                subject_num=subject_num,
                task_name=task_name,
                state=state,
                output_dir=output_dir,
                cache_dir=cache_dir
            )
            
    print(f"\nVisualization completed for {len(subjects)} subjects!")
    print(f"Results saved to {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate visualizations for BCI data from multiple subjects')
    parser.add_argument('--subjects', type=str, default='1-10',
                        help='Subject numbers to process (comma-separated or range like 1-10)')
    parser.add_argument('--task', type=str, choices=list(TASK_RUNS.keys()), default='Task4',
                        help='Task to visualize')
    parser.add_argument('--state', type=str, choices=['rest', 'left', 'right', 'all'], default='all',
                        help='Mental state to visualize. Use "all" for all states')
    parser.add_argument('--output', type=str, default='./crt_img2_vis',
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
    
    # Process all specified subjects
    process_all_subjects(
        subjects=subjects,
        task_name=args.task,
        states=states,
        output_dir=args.output,
        cache_dir=args.cache
    )

if __name__ == "__main__":
    main()