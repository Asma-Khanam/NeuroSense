import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import mne
from scipy import signal
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

class EEGPatternVisualizer:
    def __init__(self):
        self.model = None
        self.class_names = ['rest', 'left', 'right']
        self.data_samples = {state: None for state in self.class_names}
        self.features = {state: None for state in self.class_names}
        self.channels = ['C3', 'Cz', 'C4']  # Standard motor imagery channels
        
    def load_model(self, model_path='high_acc_gb_model.pkl'):
        """Load the trained model"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_samples_from_raw_file(self, file_path, sfreq=250):
        """Extract EEG samples from raw data file"""
        if file_path.endswith('.edf'):
            try:
                # Load EDF file
                raw = mne.io.read_raw_edf(file_path, preload=True)
                
                # Filter the data
                raw.filter(l_freq=1.0, h_freq=45.0)
                
                # Try to find motor imagery events
                events, event_id = mne.events_from_annotations(raw)
                
                # Print available events to help debugging
                print(f"Available events: {event_id}")
                
                # Map event IDs to class names if possible
                state_event_map = {}
                for state in self.class_names:
                    for key, value in event_id.items():
                        if state.lower() in key.lower() or state.lower() in str(value).lower():
                            state_event_map[state] = value
                            break
                        # Common alternative mappings
                        if state == 'rest' and ('T0' in key or 'baseline' in key.lower()):
                            state_event_map[state] = value
                            break
                        if state == 'left' and ('T1' in key or 'left_hand' in key.lower() or 'left_mi' in key.lower()):
                            state_event_map[state] = value
                            break
                        if state == 'right' and ('T2' in key or 'right_hand' in key.lower() or 'right_mi' in key.lower()):
                            state_event_map[state] = value
                            break
                
                print(f"State to event mapping: {state_event_map}")
                
                # If mapping found, extract epochs
                if state_event_map:
                    # Extract motor channels
                    motor_channels = [ch for ch in raw.ch_names if ch in self.channels]
                    if not motor_channels:
                        motor_channels = [ch for ch in raw.ch_names if any(motor in ch for motor in ['C3', 'Cz', 'C4'])]
                    
                    if not motor_channels:
                        print(f"Could not find motor channels in {raw.ch_names}")
                        return False
                    
                    raw.pick_channels(motor_channels)
                    
                    # Create epochs for each state
                    for state, event_id in state_event_map.items():
                        try:
                            epochs = mne.Epochs(raw, events, event_id={state: event_id}, 
                                               tmin=0.5, tmax=3.5, baseline=(0.5, 1.0), preload=True)
                            
                            if len(epochs) > 0:
                                # Get the first epoch as sample
                                self.data_samples[state] = epochs.get_data()[0]
                                print(f"Extracted {len(epochs)} samples for state '{state}'")
                            else:
                                print(f"No epochs found for state '{state}'")
                        except Exception as e:
                            print(f"Error extracting epochs for state '{state}': {e}")
                    
                    # Check if samples were extracted
                    has_samples = False
                    for sample in self.data_samples.values():
                        if sample is not None:
                            has_samples = True
                            break
                    
                    if has_samples:
                        return True
                    else:
                        # Try an alternative approach if no events found
                        return self._extract_samples_from_continuous(raw)
                else:
                    # Try an alternative approach if no events found
                    return self._extract_samples_from_continuous(raw)
                    
            except Exception as e:
                print(f"Error processing EDF file: {e}")
                return False
        
        elif file_path.endswith('.csv'):
            try:
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Check for common column naming patterns for motor channels
                motor_cols = []
                for channel in self.channels:
                    matching_cols = [col for col in df.columns if channel.lower() in col.lower()]
                    if matching_cols:
                        motor_cols.extend(matching_cols[:1])  # Take the first match for each channel
                
                if not motor_cols:
                    print(f"Could not find motor channels in {df.columns}")
                    return False
                
                # Check for event/label column
                label_col = None
                for col in df.columns:
                    if col.lower() in ['event', 'label', 'class', 'state', 'type']:
                        label_col = col
                        break
                
                if label_col:
                    # Extract samples for each state
                    for state in self.class_names:
                        # Find rows corresponding to this state
                        rows = df.loc[df[label_col].str.lower().str.contains(state.lower(), na=False)].index.tolist()
                        if not rows:
                            # Try numerical mapping (0=rest, 1=left, 2=right is common)
                            if state == 'rest':
                                rows = df.loc[df[label_col] == 0].index.tolist()
                            elif state == 'left':
                                rows = df.loc[df[label_col] == 1].index.tolist()
                            elif state == 'right':
                                rows = df.loc[df[label_col] == 2].index.tolist()
                        
                        if rows:
                            # Take a continuous segment of 3 seconds (assuming 250 Hz)
                            start_idx = rows[0]
                            end_idx = min(start_idx + int(3 * sfreq), len(df))
                            sample = df.loc[start_idx:end_idx, motor_cols].values.T
                            self.data_samples[state] = sample
                            print(f"Extracted sample for state '{state}' from rows {start_idx}-{end_idx}")
                        else:
                            print(f"No data found for state '{state}'")
                else:
                    # No label column found, try to segment the data
                    total_rows = len(df)
                    segment_size = int(3 * sfreq)  # 3 seconds at sfreq
                    
                    # Divide data into three equal parts
                    self.data_samples['rest'] = df.iloc[0:segment_size, motor_cols].values.T
                    self.data_samples['left'] = df.iloc[segment_size:2*segment_size, motor_cols].values.T
                    self.data_samples['right'] = df.iloc[2*segment_size:3*segment_size, motor_cols].values.T
                    
                    print("No label column found. Extracted samples by dividing data into three equal parts.")
                
                # Check if samples were extracted
                has_samples = False
                for sample in self.data_samples.values():
                    if sample is not None:
                        has_samples = True
                        break
                
                return has_samples
                
            except Exception as e:
                print(f"Error processing CSV file: {e}")
                return False
        
        else:
            print(f"Unsupported file format: {file_path}")
            return False
    
    def _extract_samples_from_continuous(self, raw):
        """Extract samples from continuous data by dividing into segments"""
        try:
            # Get data array
            data = raw.get_data()
            
            # Divide data into three equal parts
            total_samples = data.shape[1]
            segment_size = total_samples // 3
            
            # Assign segments to states
            self.data_samples['rest'] = data[:, 0:segment_size]
            self.data_samples['left'] = data[:, segment_size:2*segment_size]
            self.data_samples['right'] = data[:, 2*segment_size:3*segment_size]
            
            print("Extracted samples by dividing continuous data into three equal parts.")
            return True
        except Exception as e:
            print(f"Error extracting samples from continuous data: {e}")
            return False
    
    def extract_features_from_samples(self):
        """Extract features from the EEG samples for visualization"""
        try:
            for state, sample in self.data_samples.items():
                if sample is not None:
                    # Calculate frequency domain features
                    freqs = {}
                    psds = {}
                    
                    for i in range(sample.shape[0]):
                        # Calculate power spectral density
                        f, psd = signal.welch(sample[i], fs=250, nperseg=256)
                        freqs[i] = f
                        psds[i] = psd
                    
                    # Store features
                    self.features[state] = {
                        'freqs': freqs,
                        'psds': psds
                    }
            
            return True
        except Exception as e:
            print(f"Error extracting features: {e}")
            return False
    
    def visualize_patterns(self, save_dir='eeg_pattern_analysis'):
        """Visualize the EEG patterns"""
        # Create directory to save plots if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Check if samples were extracted
        has_samples = False
        for sample in self.data_samples.values():
            if sample is not None:
                has_samples = True
                break
                
        if not has_samples:
            print("No data samples available for visualization.")
            return False
        
        # Plot individual state patterns
        for state in self.class_names:
            if self.data_samples[state] is None:
                print(f"No data for state: {state}")
                continue
                
            fig, axes = plt.subplots(len(self.data_samples[state]), 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'EEG Pattern Analysis: {state.upper()} State', fontsize=16)
            
            # If only one channel, wrap axes in a list
            if len(self.data_samples[state]) == 1:
                axes = [axes]
            
            # Time axis
            t = np.linspace(0, 3, self.data_samples[state].shape[1])
            
            for i in range(len(self.data_samples[state])):
                ax = axes[i]
                
                # Get channel name (if available, otherwise use index)
                channel_name = self.channels[i] if i < len(self.channels) else f"Channel {i}"
                
                # Plot time domain signal
                ax.plot(t, self.data_samples[state][i], linewidth=1.5)
                
                # Highlight key areas of interest based on state and channel
                y_min = np.min(self.data_samples[state][i])
                y_max = np.max(self.data_samples[state][i])
                height = y_max - y_min
                
                if state == 'left' and ('C4' in channel_name or i == 2):  # Right motor cortex
                    # Highlight the ERD in right motor cortex during left MI
                    ax.add_patch(plt.Rectangle((0.5, y_min), 2, height, 
                                           alpha=0.2, color='red'))
                    ax.annotate('Potential ERD', 
                                xy=(1.5, np.mean([y_min, y_max])), 
                                xytext=(1.5, y_min + 0.3 * height),
                                arrowprops=dict(arrowstyle='->'),
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                    
                elif state == 'right' and ('C3' in channel_name or i == 0):  # Left motor cortex
                    # Highlight the ERD in left motor cortex during right MI
                    ax.add_patch(plt.Rectangle((0.5, y_min), 2, height,
                                           alpha=0.2, color='red'))
                    ax.annotate('Potential ERD', 
                                xy=(1.5, np.mean([y_min, y_max])), 
                                xytext=(1.5, y_min + 0.3 * height),
                                arrowprops=dict(arrowstyle='->'),
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                
                elif state == 'rest' and ('Cz' in channel_name or i == 1):  # Central channel
                    # Highlight the alpha rhythm during rest
                    ax.add_patch(plt.Rectangle((0.5, y_min), 2, height,
                                           alpha=0.2, color='green'))
                    ax.annotate('Alpha Rhythm', 
                                xy=(1.5, np.mean([y_min, y_max])), 
                                xytext=(1.5, y_max - 0.3 * height),
                                arrowprops=dict(arrowstyle='->'),
                                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.3))
                
                # Add spectral analysis if features were extracted
                if (self.features.get(state) is not None and 
                    self.features[state]['freqs'].get(i) is not None):
                    # Create an inset for the PSD
                    ax_inset = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
                    ax_inset.plot(self.features[state]['freqs'][i], self.features[state]['psds'][i], color='purple')
                    ax_inset.set_xlim(0, 40)  # Show frequencies up to 40 Hz
                    ax_inset.set_title('Power Spectrum', fontsize=8)
                    ax_inset.set_xlabel('Frequency (Hz)', fontsize=8)
                    ax_inset.tick_params(labelsize=6)
                    
                    # Highlight alpha (8-13 Hz) and beta (13-30 Hz) bands
                    ax_inset.axvspan(8, 13, alpha=0.3, color='green', label='Alpha band')
                    ax_inset.axvspan(13, 30, alpha=0.3, color='blue', label='Beta band')
                
                ax.set_title(f'{channel_name}', fontsize=14)
                ax.set_ylabel('Amplitude (μV)', fontsize=12)
                ax.grid(True, alpha=0.3)
                
            # Add overall state description
            if state == 'rest':
                description = """
REST STATE CHARACTERISTICS:
• Look for strong alpha rhythm (8-13 Hz) across all channels
• Regular oscillations, particularly in central channels
• Symmetrical patterns between left (C3) and right (C4) hemispheres
• Minimal event-related desynchronization (ERD)
                """
            elif state == 'left':
                description = """
LEFT HAND MOTOR IMAGERY CHARACTERISTICS:
• Look for Event-Related Desynchronization (ERD) in RIGHT motor cortex (C4)
• Decreased signal power in C4 compared to rest state
• Relatively normal rhythm in LEFT motor cortex (C3)
• This hemisphere-specific desynchronization is the key feature for classification
                """
            else:  # right
                description = """
RIGHT HAND MOTOR IMAGERY CHARACTERISTICS:
• Look for Event-Related Desynchronization (ERD) in LEFT motor cortex (C3)
• Decreased signal power in C3 compared to rest state
• Relatively normal rhythm in RIGHT motor cortex (C4)
• Mirror image pattern compared to left hand motor imagery
                """
            
            plt.figtext(0.1, 0.01, description, fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            axes[-1].set_xlabel('Time (s)', fontsize=12)
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])
            plt.savefig(os.path.join(save_dir, f'{state}_pattern_analysis.png'), dpi=300)
            plt.close()
        
        # Create comparison plot with all states
        try:
            # Get available samples
            available_samples = [s for s, sample in self.data_samples.items() if sample is not None]
            
            if not available_samples:
                print("No samples available for comparison plot")
                return True
                
            # Get max number of channels across all states
            max_channels = max(self.data_samples[s].shape[0] for s in available_samples)
            
            fig, axes = plt.subplots(max_channels, 1, figsize=(14, 12), sharex=True)
            fig.suptitle('Comparison of EEG Patterns Across Mental States', fontsize=18)
            
            # If only one channel, wrap axes in a list
            if max_channels == 1:
                axes = [axes]
            
            # Time axis (use maximum length across samples)
            max_length = max(self.data_samples[s].shape[1] for s in available_samples)
            t = np.linspace(0, 3, max_length)
            
            line_styles = {'rest': '-', 'left': '--', 'right': '-.'}
            colors = {'rest': 'green', 'left': 'blue', 'right': 'red'}
            
            for i in range(max_channels):
                ax = axes[i]
                
                # Get channel name (if available, otherwise use index)
                channel_name = self.channels[i] if i < len(self.channels) else f"Channel {i}"
                
                # Plot each state for this channel
                for state in available_samples:
                    if i < self.data_samples[state].shape[0]:
                        # Use only the length of time axis that matches the data
                        t_use = t[:self.data_samples[state].shape[1]]
                        ax.plot(t_use, self.data_samples[state][i], 
                                label=f'{state.upper()}', 
                                linestyle=line_styles[state],
                                color=colors[state],
                                linewidth=1.5)
                
                ax.set_title(f'{channel_name}', fontsize=16)
                ax.set_ylabel('Amplitude (μV)', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Add channel-specific annotations
                if i < len(self.channels):
                    if 'C3' in channel_name or i == 0:  # Left motor cortex
                        ax.text(0.02, 0.95, 'Left Motor Cortex', transform=ax.transAxes, 
                                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
                        y_range = ax.get_ylim()
                        annotation_y = y_range[0] + 0.2 * (y_range[1] - y_range[0])
                        ax.annotate('Look for ERD during RIGHT hand MI', 
                                    xy=(1.5, annotation_y),
                                    xytext=(1.5, annotation_y - 0.1 * (y_range[1] - y_range[0])),
                                    arrowprops=dict(arrowstyle='->'),
                                    bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3))
                        
                    elif 'C4' in channel_name or i == 2:  # Right motor cortex
                        ax.text(0.02, 0.95, 'Right Motor Cortex', transform=ax.transAxes, 
                                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
                        y_range = ax.get_ylim()
                        annotation_y = y_range[0] + 0.2 * (y_range[1] - y_range[0])
                        ax.annotate('Look for ERD during LEFT hand MI', 
                                    xy=(1.5, annotation_y),
                                    xytext=(1.5, annotation_y - 0.1 * (y_range[1] - y_range[0])),
                                    arrowprops=dict(arrowstyle='->'),
                                    bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3))
                        
                    elif 'Cz' in channel_name or i == 1:  # Central channel
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
• REST state: Alpha rhythm in all channels with no hemispheric asymmetry
• The contralateral organization of motor cortex creates these characteristic patterns
• Classification accuracy depends on detecting these hemisphere-specific patterns
            """
            plt.figtext(0.1, 0.01, findings, fontsize=14, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])
            plt.savefig(os.path.join(save_dir, 'comparison_analysis.png'), dpi=300)
            plt.close()
            
            print(f"Visualizations saved to {save_dir} directory")
            return True
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
            return False
    
    def analyze_model_predictions(self, data_file=None):
        """Analyze model predictions on sample data"""
        if self.model is None:
            print("No model loaded.")
            return False
        
        # Check if we have data samples
        has_samples = False
        for sample in self.data_samples.values():
            if sample is not None:
                has_samples = True
                break
                
        if not has_samples:
            if data_file:
                self.extract_samples_from_raw_file(data_file)
                # Check again after extraction
                has_samples = False
                for sample in self.data_samples.values():
                    if sample is not None:
                        has_samples = True
                        break
                        
                if not has_samples:
                    print("No data samples available and no data file provided.")
                    return False
            else:
                print("No data samples available and no data file provided.")
                return False
        
        try:
            # Create a directory for results
            result_dir = 'model_analysis'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # Extract features from samples
            features = {}
            for state, sample in self.data_samples.items():
                if sample is None:
                    continue
                
                # Extract frequency domain features for each channel
                state_features = []
                for channel in range(sample.shape[0]):
                    # Calculate power in frequency bands
                    f, psd = signal.welch(sample[channel], fs=250, nperseg=256)
                    
                    # Alpha power (8-13 Hz)
                    alpha_idx = np.logical_and(f >= 8, f <= 13)
                    alpha_power = np.mean(psd[alpha_idx])
                    
                    # Beta power (13-30 Hz)
                    beta_idx = np.logical_and(f >= 13, f <= 30)
                    beta_power = np.mean(psd[beta_idx])
                    
                    # Add features
                    state_features.extend([alpha_power, beta_power])
                
                # Add some additional features to match model input size
                # Number of features depends on your model's input size
                current_features = len(state_features)
                expected_features = 84  # Default value, adjust based on your model
                
                if hasattr(self.model, 'n_features_in_'):
                    expected_features = self.model.n_features_in_
                
                # If missing features, pad with zeros
                if current_features < expected_features:
                    state_features.extend([0] * (expected_features - current_features))
                # If too many features, truncate
                elif current_features > expected_features:
                    state_features = state_features[:expected_features]
                
                # Store features for this state
                features[state] = np.array(state_features).reshape(1, -1)
            
            # Predict on each sample
            predictions = {}
            for state, X in features.items():
                try:
                    # Make prediction
                    pred = self.model.predict(X)
                    pred_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
                    
                    # Map predicted index to class name
                    pred_class = self.class_names[pred[0] % len(self.class_names)]
                    
                    # Store prediction results
                    predictions[state] = {
                        'predicted_class': pred_class,
                        'correct': pred_class == state,
                        'probabilities': pred_proba[0] if pred_proba is not None else None
                    }
                    
                    print(f"State: {state}, Predicted: {pred_class}, Correct: {pred_class == state}")
                except Exception as e:
                    print(f"Error making prediction for state {state}: {e}")
            
            # Create visualization of predictions
            if predictions:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                states = list(predictions.keys())
                correct = [1 if predictions[state]['correct'] else 0 for state in states]
                
                # Bar chart
                bars = ax.bar(states, correct, color=['green' if c else 'red' for c in correct])
                
                # Add labels
                for i, state in enumerate(states):
                    pred = predictions[state]['predicted_class']
                    ax.text(i, 0.5, f"Predicted: {pred}", ha='center', rotation=90, color='white')
                
                ax.set_ylim(0, 1.2)
                ax.set_ylabel('Correct Prediction')
                ax.set_title('Model Predictions on Sample Data')
                
                # Add probability bars if available
                proba_available = True
                for p in predictions.values():
                    if p['probabilities'] is None:
                        proba_available = False
                        break
                
                if proba_available:
                    twin_ax = ax.twinx()
                    
                    # Group probabilities by actual class
                    for i, state in enumerate(states):
                        probs = predictions[state]['probabilities']
                        x = np.arange(len(probs)) + (i - 1) * 0.25
                        twin_ax.bar(x, probs, width=0.2, alpha=0.5, 
                                   color=['green', 'blue', 'red'][:len(probs)])
                    
                    twin_ax.set_ylim(0, 1.2)
                    twin_ax.set_ylabel('Probability')
                
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, 'model_predictions.png'), dpi=300)
                plt.close()
                
                # Calculate accuracy
                accuracy = sum(1 for state in states if predictions[state]['correct']) / len(states)
                print(f"Accuracy on sample data: {accuracy * 100:.2f}%")
            
            return True
        except Exception as e:
            print(f"Error analyzing model predictions: {e}")
            return False

def main():
    # Create visualizer
    visualizer = EEGPatternVisualizer()
    
    # Load model
    model_path = input("Enter path to model file (default: high_acc_gb_model.pkl): ").strip() or 'high_acc_gb_model.pkl'
    visualizer.load_model(model_path)
    
    # Choose data file
    data_file = input("Enter path to EEG data file (EDF or CSV format): ").strip()
    
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        # Try to find similar files in current directory
        similar_files = [f for f in os.listdir() if f.endswith(('.edf', '.csv'))]
        if similar_files:
            print("Found similar files in current directory:")
            for i, f in enumerate(similar_files):
                print(f"{i+1}. {f}")
            choice = input("Enter number to select file (or press Enter to skip): ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(similar_files):
                data_file = similar_files[int(choice)-1]
            else:
                print("No valid file selected.")
                return
        else:
            print("No EDF or CSV files found in current directory.")
            return
    
    # Extract samples
    print(f"Extracting samples from {data_file}...")
    if visualizer.extract_samples_from_raw_file(data_file):
        print("Successfully extracted samples.")
        
        # Extract features
        visualizer.extract_features_from_samples()
        
        # Visualize patterns
        output_dir = input("Enter output directory for visualizations (default: eeg_analysis): ").strip() or 'eeg_analysis'
        visualizer.visualize_patterns(save_dir=output_dir)
        
        # Analyze model predictions
        if visualizer.model is not None:
            visualizer.analyze_model_predictions()
    else:
        print("Failed to extract samples from data file.")

if __name__ == "__main__":
    main()