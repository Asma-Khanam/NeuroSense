import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse
from datetime import datetime
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Simple color scheme
COLORS = {
    'rest': '#4CAF50',    # Green
    'left': '#2196F3',    # Blue
    'right': '#FF9800'    # Orange
}

# Define runs for task 4 and task 5
TASK_RUNS = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

def generate_simulated_data(state, n_samples=300):
    """Generate simulated EEG data for visualization with controlled size"""
    # Create time points
    times = np.linspace(0, 3, n_samples)
    
    # Base signal (alpha rhythm at 10Hz)
    base_signal = np.sin(2 * np.pi * 10 * times)
    noise_level = 0.5
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Create state-specific patterns
    if state == 'rest':
        # REST: Strong alpha in all channels
        c3 = base_signal * 1.5 + noise * 0.4
        cz = base_signal * 1.2 + noise * 0.3
        c4 = base_signal * 1.5 + noise * 0.4
    
    elif state == 'left':
        # LEFT hand: Reduced alpha in right motor cortex (C4)
        c3 = base_signal * 1.3 + noise * 0.3
        cz = base_signal * 1.0 + noise * 0.4
        c4 = base_signal * 0.6 + noise * 0.8  # ERD in C4
    
    elif state == 'right':
        # RIGHT hand: Reduced alpha in left motor cortex (C3)
        c3 = base_signal * 0.6 + noise * 0.8  # ERD in C3
        cz = base_signal * 1.0 + noise * 0.4
        c4 = base_signal * 1.3 + noise * 0.3
    
    signals = np.array([c3, cz, c4])
    return times, signals

def extract_simple_features(signals):
    """Extract very basic features from signals"""
    features = []
    
    # For each channel
    for signal in signals:
        # Simple FFT
        fft_vals = np.abs(np.fft.rfft(signal, n=100))
        
        # Basic band powers
        theta_idx = slice(4, 8)
        alpha_idx = slice(8, 13)
        beta_idx = slice(13, 30)
        
        if theta_idx.stop <= len(fft_vals):
            theta = np.mean(fft_vals[theta_idx])
            features.append(theta)
        else:
            features.append(0)
            
        if alpha_idx.stop <= len(fft_vals):
            alpha = np.mean(fft_vals[alpha_idx])
            features.append(alpha)
        else:
            features.append(0)
            
        if beta_idx.stop <= len(fft_vals):
            beta = np.mean(fft_vals[beta_idx])
            features.append(beta)
        else:
            features.append(0)
    
    return np.array(features)

def downsample_signals(epoch_data, times, max_samples=300):
    """Downsample signals to keep size manageable"""
    if epoch_data.shape[1] > max_samples:
        # Calculate step size to get desired number of samples
        step = epoch_data.shape[1] // max_samples
        downsampled = epoch_data[:, ::step]
        times_ds = times[::step]
        return downsampled, times_ds
    return epoch_data, times

def get_subject_data(subject_num, task_name, state, max_samples=300, cache_dir='./cache'):
    """Load subject data and extract the relevant signals"""
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Try to load from cache
    cache_file = os.path.join(cache_dir, f"subject_{subject_num}_{task_name}.pkl")
    
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached data for Subject {subject_num}, {task_name}...")
            cached_data = joblib.load(cache_file)
            
            # Extract data for the specified state
            try:
                # Get epochs for this state
                state_epochs = cached_data['epochs'][state]
                
                if len(state_epochs) == 0:
                    print(f"No epochs found for state '{state}'")
                    return None, None, None, None
                
                # Get first epoch
                epoch_data = state_epochs[0].get_data()[0]
                times = cached_data['times']
                
                # Downsample
                epoch_data, times = downsample_signals(epoch_data, times, max_samples)
                
                # Get channel names
                channels = ['C3', 'Cz', 'C4']
                ch_indices = []
                ch_names = []
                
                # Find channels
                for ch in channels:
                    matches = [i for i, name in enumerate(cached_data['channel_names']) if ch in name]
                    if matches:
                        ch_indices.append(matches[0])
                        ch_names.append(cached_data['channel_names'][matches[0]])
                
                # Fill with available channels if needed
                while len(ch_indices) < 3 and len(ch_indices) < len(cached_data['channel_names']):
                    next_idx = len(ch_indices)
                    if next_idx not in ch_indices:
                        ch_indices.append(next_idx)
                        ch_names.append(cached_data['channel_names'][next_idx])
                
                # Get signals
                if ch_indices:
                    signals = epoch_data[ch_indices]
                    return times, signals, ch_names, subject_num
                else:
                    return None, None, None, None
                    
            except Exception as e:
                print(f"Error processing epochs: {e}")
                return None, None, None, None
                
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None, None, None, None
    
    else:
        print(f"No cache file found for Subject {subject_num}, {task_name}")
        return None, None, None, None

def create_simple_visualization(state='rest', times=None, signals=None, ch_names=None, subject_id=None, 
                               output_dir='./crt_img1_vis'):
    """Create a very simple visualization focusing on just the essentials"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If no data provided, use simulated data
    if times is None or signals is None or ch_names is None:
        print("Using simulated data")
        times, signals = generate_simulated_data(state)
        ch_names = ['C3 (Left Motor)', 'Cz (Central)', 'C4 (Right Motor)']
        subject_id = 'simulated'
    
    # Create a simple figure
    fig, axs = plt.subplots(3, 2, figsize=(8, 6), dpi=100)
    
    # Add title
    subject_info = f"Subject {subject_id}" if subject_id and subject_id != 'simulated' else "Simulated Data"
    fig.suptitle(f"BCI {state.upper()} State - {subject_info}", fontsize=12)
    
    # EEG signals - left column
    for i in range(3):
        ax = axs[i, 0]
        ax.plot(times, signals[i], linewidth=1, color=COLORS[state])
        ax.set_title(ch_names[i], fontsize=10)
        ax.set_ylabel('μV', fontsize=8)
        if i == 2:  # Only add x-label to bottom plot
            ax.set_xlabel('Time (s)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Extract features
    features = extract_simple_features(signals)
    
    # Bar chart of key features - top right
    ax_bar = axs[0, 1]
    bands = ['Alpha', 'Beta']
    ch_short = ['C3', 'C4']
    
    # Extract just alpha and beta for C3 and C4
    feature_data = [
        features[1],  # C3 Alpha (index 1)
        features[7],  # C4 Alpha (index 7)
        features[2],  # C3 Beta (index 2)
        features[8]   # C4 Beta (index 8)
    ]
    
    # Create labels
    labels = [f"{ch}_{band}" for ch in ch_short for band in bands]
    
    # Plot bars
    colors = [COLORS['left'], COLORS['right'], 
              COLORS['left'], COLORS['right']]
    ax_bar.bar(range(len(feature_data)), feature_data, color=colors, alpha=0.7)
    ax_bar.set_xticks(range(len(feature_data)))
    ax_bar.set_xticklabels(labels, rotation=45, fontsize=8)
    ax_bar.set_title("Band Power", fontsize=10)
    
    # Add ERD indicators
    if state == 'left' and feature_data[0] > feature_data[1]:
        ax_bar.annotate('ERD', xy=(1, feature_data[1]), 
                        xytext=(1, feature_data[1] + max(feature_data)/5),
                        arrowprops=dict(arrowstyle='->'), 
                        ha='center', fontsize=8)
    elif state == 'right' and feature_data[1] > feature_data[0]:
        ax_bar.annotate('ERD', xy=(0, feature_data[0]), 
                        xytext=(0, feature_data[0] + max(feature_data)/5),
                        arrowprops=dict(arrowstyle='->'), 
                        ha='center', fontsize=8)
    
    # Brain illustration - middle right
    ax_brain = axs[1, 1]
    # Clear the axis
    ax_brain.clear()
    # Draw a simple head outline
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax_brain.add_patch(circle)
    ax_brain.plot([0, 0], [0.8, 1.1], 'k-')  # Nose
    
    # Add simplified C3, Cz, C4 locations
    ax_brain.plot(-0.5, 0, 'o', markersize=12, color='gray')
    ax_brain.plot(0, 0, 'o', markersize=12, color='gray')
    ax_brain.plot(0.5, 0, 'o', markersize=12, color='gray')
    
    # Label electrodes
    ax_brain.text(-0.5, -0.2, 'C3', ha='center')
    ax_brain.text(0, -0.2, 'Cz', ha='center')
    ax_brain.text(0.5, -0.2, 'C4', ha='center')
    
    # Highlight active regions based on state
    if state == 'rest':
        # No specific highlighting for rest
        pass
    elif state == 'left':
        # Highlight C4 (right hemisphere)
        ax_brain.plot(0.5, 0, 'o', markersize=15, mfc='none', mec='blue', linewidth=2)
        ax_brain.text(0.5, 0.2, 'ERD', color='blue', ha='center')
    elif state == 'right':
        # Highlight C3 (left hemisphere)
        ax_brain.plot(-0.5, 0, 'o', markersize=15, mfc='none', mec='orange', linewidth=2)
        ax_brain.text(-0.5, 0.2, 'ERD', color='orange', ha='center')
    
    ax_brain.set_xlim(-1.2, 1.2)
    ax_brain.set_ylim(-1.2, 1.2)
    ax_brain.set_aspect('equal')
    ax_brain.axis('off')
    ax_brain.set_title('ERD Location', fontsize=10)
    
    # ERD explanation - bottom right
    ax_erd = axs[2, 1]
    # Simple explanation text
    explanation = {
        'rest': "REST STATE:\nStrong alpha rhythm in\nboth hemispheres.",
        'left': "LEFT HAND IMAGERY:\nEvent-Related Desynchronization\nin right motor cortex (C4).",
        'right': "RIGHT HAND IMAGERY:\nEvent-Related Desynchronization\nin left motor cortex (C3)."
    }
    
    ax_erd.text(0.5, 0.5, explanation[state], 
                ha='center', va='center', 
                fontsize=8, color=COLORS[state],
                bbox=dict(boxstyle='round,pad=0.5', 
                         fc='white', ec=COLORS[state], alpha=0.8))
    ax_erd.axis('off')
    ax_erd.set_title("BCI Pattern", fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if subject_id != 'simulated':
        filename = f"{output_dir}/subject_{subject_id}_{state}_{timestamp}.png"
    else:
        filename = f"{output_dir}/{subject_id}_{state}_{timestamp}.png"
    
    # Save figure
    plt.savefig(filename, dpi=100)
    plt.close()
    
    print(f"Visualization saved to {filename}")
    return filename

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate simple EEG visualizations')
    parser.add_argument('--subject', type=int, default=None, 
                        help='Subject number (1-10). If not provided, simulated data will be used')
    parser.add_argument('--task', type=str, choices=list(TASK_RUNS.keys()), default='Task4',
                        help='Task to visualize')
    parser.add_argument('--state', type=str, choices=['rest', 'left', 'right', 'all'], default='all',
                        help='Mental state to visualize. Use "all" for all states')
    parser.add_argument('--output', type=str, default='./crt_img1_vis',
                        help='Output directory for visualizations')
    parser.add_argument('--cache', type=str, default='./cache',
                        help='Cache directory for processed data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine states to visualize
    states = ['rest', 'left', 'right'] if args.state == 'all' else [args.state]
    
    # Iterate through states
    for state in states:
        try:
            if args.subject is not None:
                # Try to get real data
                times, signals, ch_names, subject_id = get_subject_data(
                    args.subject, args.task, state, cache_dir=args.cache)
                
                # Create visualization
                create_simple_visualization(
                    state=state,
                    times=times,
                    signals=signals,
                    ch_names=ch_names,
                    subject_id=subject_id,
                    output_dir=args.output
                )
            else:
                # Create visualization with simulated data
                create_simple_visualization(
                    state=state,
                    output_dir=args.output
                )
                
        except Exception as e:
            print(f"Error generating visualization for {state}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()