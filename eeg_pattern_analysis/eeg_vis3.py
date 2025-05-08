import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mne
import os
import pandas as pd
from scipy import signal

def generate_sample_eeg_patterns():
    """Generate representative EEG patterns for motor imagery states"""
    # Parameters
    sampling_rate = 250  # Hz
    duration = 3  # seconds
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Generate baseline alpha rhythm (8-13 Hz)
    alpha_freq = 10  # Hz
    alpha_wave = np.sin(2 * np.pi * alpha_freq * t)
    
    # Generate beta rhythm (13-30 Hz)
    beta_freq = 20  # Hz
    beta_wave = 0.5 * np.sin(2 * np.pi * beta_freq * t)
    
    # Add some noise
    noise_level = 0.3
    noise = np.random.normal(0, noise_level, len(t))
    
    # Create state-specific patterns
    patterns = {}
    
    # REST state: Strong alpha rhythm in all channels
    patterns['rest'] = {
        'C3': 1.5 * alpha_wave + 0.3 * beta_wave + 0.3 * noise,
        'Cz': 1.3 * alpha_wave + 0.2 * beta_wave + 0.3 * noise,
        'C4': 1.5 * alpha_wave + 0.3 * beta_wave + 0.3 * noise
    }
    
    # LEFT hand MI: ERD in right motor cortex (C4)
    patterns['left'] = {
        'C3': 1.2 * alpha_wave + 0.4 * beta_wave + 0.3 * noise,
        'Cz': 1.0 * alpha_wave + 0.3 * beta_wave + 0.3 * noise,
        'C4': 0.6 * alpha_wave + 0.8 * beta_wave + 0.5 * noise  # ERD on contralateral side
    }
    
    # RIGHT hand MI: ERD in left motor cortex (C3)
    patterns['right'] = {
        'C3': 0.6 * alpha_wave + 0.8 * beta_wave + 0.5 * noise,  # ERD on contralateral side
        'Cz': 1.0 * alpha_wave + 0.3 * beta_wave + 0.3 * noise,
        'C4': 1.2 * alpha_wave + 0.4 * beta_wave + 0.3 * noise
    }
    
    return patterns, t

def plot_eeg_patterns(patterns, t, save_dir='eeg_pattern_analysis'):
    """Plot the EEG patterns with annotations"""
    # Create directory to save plots if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    states = ['rest', 'left', 'right']
    channels = ['C3', 'Cz', 'C4']
    
    # Plot individual state patterns (one plot per state)
    for state in states:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'EEG Pattern Analysis: {state.upper()} State', fontsize=16)
        
        for i, channel in enumerate(channels):
            ax = axes[i]
            eeg_data = patterns[state][channel]
            
            # Plot time domain signal
            ax.plot(t, eeg_data, linewidth=1.5)
            
            # Highlight key areas of interest based on state and channel
            if state == 'left' and channel == 'C4':
                # Highlight the ERD in right motor cortex during left MI
                ax.add_patch(Rectangle((0.5, -2), 2, 4, 
                                       alpha=0.2, color='red', 
                                       label='ERD (decreased alpha)'))
                ax.annotate('Event-Related Desynchronization (ERD)', 
                            xy=(1.5, -1), xytext=(1.5, -1.5),
                            arrowprops=dict(arrowstyle='->'),
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                
            elif state == 'right' and channel == 'C3':
                # Highlight the ERD in left motor cortex during right MI
                ax.add_patch(Rectangle((0.5, -2), 2, 4, 
                                       alpha=0.2, color='red',
                                       label='ERD (decreased alpha)'))
                ax.annotate('Event-Related Desynchronization (ERD)', 
                            xy=(1.5, -1), xytext=(1.5, -1.5),
                            arrowprops=dict(arrowstyle='->'),
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                
            elif state == 'rest' and channel == 'Cz':
                # Highlight the strong alpha rhythm during rest
                ax.add_patch(Rectangle((0.5, -2), 2, 4, 
                                       alpha=0.2, color='green',
                                       label='Strong alpha rhythm'))
                ax.annotate('Strong Alpha Rhythm (8-13 Hz)', 
                            xy=(1.5, 1), xytext=(1.5, 1.5),
                            arrowprops=dict(arrowstyle='->'),
                            bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.3))
            
            # Add spectral analysis (power spectral density)
            freqs, psd = signal.welch(eeg_data, fs=len(t)/t[-1], nperseg=256)
            
            # Create an inset for the PSD
            ax_inset = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
            ax_inset.plot(freqs, psd, color='purple')
            ax_inset.set_xlim(0, 40)  # Show frequencies up to 40 Hz
            ax_inset.set_title('Power Spectrum', fontsize=8)
            ax_inset.set_xlabel('Frequency (Hz)', fontsize=8)
            ax_inset.tick_params(labelsize=6)
            
            # Highlight alpha (8-13 Hz) and beta (13-30 Hz) bands
            ax_inset.axvspan(8, 13, alpha=0.3, color='green', label='Alpha band')
            ax_inset.axvspan(13, 30, alpha=0.3, color='blue', label='Beta band')
            
            # Annotate the PSD based on state and channel
            if (state == 'left' and channel == 'C4') or (state == 'right' and channel == 'C3'):
                # Decreased alpha power with ERD
                max_alpha_idx = np.argmax(psd[(freqs >= 8) & (freqs <= 13)])
                alpha_freq_idx = np.where((freqs >= 8) & (freqs <= 13))[0][max_alpha_idx]
                ax_inset.annotate('Reduced\nAlpha Power', 
                                xy=(10, psd[alpha_freq_idx]), 
                                xytext=(20, psd[alpha_freq_idx] + 0.2),
                                arrowprops=dict(arrowstyle='->'),
                                fontsize=6)
            
            ax.set_title(f'{channel} Channel', fontsize=14)
            ax.set_ylabel('Amplitude (μV)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add channel-specific annotations
            if channel == 'C3':
                ax.text(0.02, 0.95, 'Left Motor Cortex', transform=ax.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            elif channel == 'C4':
                ax.text(0.02, 0.95, 'Right Motor Cortex', transform=ax.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            elif channel == 'Cz':
                ax.text(0.02, 0.95, 'Central Motor Cortex', transform=ax.transAxes, 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Add overall state description
        if state == 'rest':
            description = """
REST STATE CHARACTERISTICS:
• Strong alpha rhythm (8-13 Hz) present in all channels
• Regular, high-amplitude oscillations
• Symmetrical patterns between left (C3) and right (C4) hemispheres
• Low beta activity relative to alpha activity
• This pattern represents the baseline neural state (no motor imagery)
            """
        elif state == 'left':
            description = """
LEFT HAND MOTOR IMAGERY CHARACTERISTICS:
• Event-Related Desynchronization (ERD) in RIGHT motor cortex (C4)
• Decreased alpha power in C4 compared to rest state
• Relatively normal alpha rhythm in LEFT motor cortex (C3)
• Increased beta activity in C4
• This hemisphere-specific desynchronization is the key feature for classification
            """
        else:  # right
            description = """
RIGHT HAND MOTOR IMAGERY CHARACTERISTICS:
• Event-Related Desynchronization (ERD) in LEFT motor cortex (C3)
• Decreased alpha power in C3 compared to rest state
• Relatively normal alpha rhythm in RIGHT motor cortex (C4)
• Increased beta activity in C3
• Mirror image pattern compared to left hand motor imagery
            """
        
        plt.figtext(0.1, 0.01, description, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        plt.savefig(os.path.join(save_dir, f'{state}_pattern_analysis.png'), dpi=300)
        plt.close()
    
    # Create comparison plot with all states
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Comparison of EEG Patterns Across Mental States', fontsize=18)
    
    line_styles = {'rest': '-', 'left': '--', 'right': '-.'}
    colors = {'rest': 'green', 'left': 'blue', 'right': 'red'}
    
    for i, channel in enumerate(channels):
        ax = axes[i]
        
        for state in states:
            ax.plot(t, patterns[state][channel], 
                    label=f'{state.upper()}', 
                    linestyle=line_styles[state],
                    color=colors[state],
                    linewidth=1.5)
            
        ax.set_title(f'{channel} Channel', fontsize=16)
        ax.set_ylabel('Amplitude (μV)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add channel-specific annotations
        if channel == 'C3':
            ax.text(0.02, 0.95, 'Left Motor Cortex', transform=ax.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            # Highlight right hand MI ERD
            ax.annotate('ERD during RIGHT hand MI', 
                        xy=(1.5, -1), xytext=(1.5, -1.5),
                        arrowprops=dict(arrowstyle='->'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3))
            
        elif channel == 'C4':
            ax.text(0.02, 0.95, 'Right Motor Cortex', transform=ax.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            # Highlight left hand MI ERD
            ax.annotate('ERD during LEFT hand MI', 
                        xy=(1.5, -1), xytext=(1.5, -1.5),
                        arrowprops=dict(arrowstyle='->'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3))
            
        elif channel == 'Cz':
            ax.text(0.02, 0.95, 'Central Motor Cortex', transform=ax.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel('Time (s)', fontsize=14)
    
    # Add key findings text box
    findings = """
KEY FINDINGS FOR MOTOR IMAGERY CLASSIFICATION:
• LEFT vs. RIGHT discrimination depends primarily on which hemisphere shows Event-Related Desynchronization (ERD)
• LEFT hand MI: ERD occurs in RIGHT motor cortex (C4 channel)
• RIGHT hand MI: ERD occurs in LEFT motor cortex (C3 channel)
• REST state: Strong alpha rhythm in all channels with no hemispheric asymmetry
• The contralateral organization of motor cortex creates these characteristic patterns
• Classification accuracy depends on detecting these hemisphere-specific patterns amid noise and variability
"""
    plt.figtext(0.1, 0.01, findings, fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(os.path.join(save_dir, 'comparison_analysis.png'), dpi=300)
    plt.close()
    
    print(f"Analysis complete! Images saved to {save_dir} directory")
    return os.path.abspath(save_dir)

def analyze_real_data(file_path=None):
    """Function to analyze real EEG data if available"""
    if file_path is None or not os.path.exists(file_path):
        print("No real data file provided or file does not exist. Using simulated data instead.")
        patterns, t = generate_sample_eeg_patterns()
        return plot_eeg_patterns(patterns, t)
    
    try:
        # Try to load as CSV
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Extract relevant channels if they exist
            channels = ['C3', 'Cz', 'C4']
            if all(ch in df.columns for ch in channels):
                data = {ch: df[ch].values for ch in channels}
                t = np.arange(len(df)) / 250  # Assuming 250 Hz
                
                # Need to separate into states - this depends on how data is labeled
                # For simplicity, we'll just use the first segment for each state
                # This would need to be adjusted based on your data format
                segment_length = min(750, len(df) // 3)  # 3 seconds at 250 Hz
                
                patterns = {
                    'rest': {ch: data[ch][:segment_length] for ch in channels},
                    'left': {ch: data[ch][segment_length:2*segment_length] for ch in channels},
                    'right': {ch: data[ch][2*segment_length:3*segment_length] for ch in channels}
                }
                
                t = t[:segment_length]
                return plot_eeg_patterns(patterns, t, save_dir='real_data_analysis')
            else:
                print(f"Required channels {channels} not found in CSV. Using simulated data instead.")
                patterns, t = generate_sample_eeg_patterns()
                return plot_eeg_patterns(patterns, t)
                
        # Try to load as EDF (requires MNE)
        elif file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True)
            # Extract motor cortex channels
            picks = mne.pick_channels(raw.ch_names, include=['C3', 'Cz', 'C4'], ordered=True)
            if len(picks) == 3:
                data = raw.get_data(picks=picks)
                
                # Need to separate into states - this depends on events in the data
                # For simplicity, we'll just use the first segment for each state
                segment_length = min(750, data.shape[1] // 3)  # 3 seconds at 250 Hz
                
                patterns = {
                    'rest': {ch: data[i, :segment_length] for i, ch in enumerate(['C3', 'Cz', 'C4'])},
                    'left': {ch: data[i, segment_length:2*segment_length] for i, ch in enumerate(['C3', 'Cz', 'C4'])},
                    'right': {ch: data[i, 2*segment_length:3*segment_length] for i, ch in enumerate(['C3', 'Cz', 'C4'])}
                }
                
                t = np.arange(segment_length) / raw.info['sfreq']
                return plot_eeg_patterns(patterns, t, save_dir='real_data_analysis')
            else:
                print("Required channels not found in EDF. Using simulated data instead.")
                patterns, t = generate_sample_eeg_patterns()
                return plot_eeg_patterns(patterns, t)
        else:
            print("Unsupported file format. Using simulated data instead.")
            patterns, t = generate_sample_eeg_patterns()
            return plot_eeg_patterns(patterns, t)
    
    except Exception as e:
        print(f"Error loading data file: {e}")
        print("Falling back to simulated data.")
        patterns, t = generate_sample_eeg_patterns()
        return plot_eeg_patterns(patterns, t)

if __name__ == "__main__":
    # You can specify your real data file here if available
    # analyze_real_data("path/to/your/eeg_data.csv")
    
    # Or use simulated data for demonstration
    patterns, t = generate_sample_eeg_patterns()
    save_dir = plot_eeg_patterns(patterns, t)
    print(f"Pattern analysis complete! Please check {save_dir} folder for visualizations.")