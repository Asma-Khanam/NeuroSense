import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Function to load the saved models
def load_models():
    models = {}
    try:
        models['SVM'] = joblib.load('high_acc_svm_model.pkl')
        print("Loaded SVM model")
    except Exception as e:
        print(f"Error loading SVM model: {e}")
    
    try:
        models['LDA'] = joblib.load('high_acc_lda_model.pkl')
        print("Loaded LDA model")
    except Exception as e:
        print(f"Error loading LDA model: {e}")
    
    try:
        models['GB'] = joblib.load('high_acc_gb_model.pkl')
        print("Loaded GB model")
    except Exception as e:
        print(f"Error loading GB model: {e}")
    
    try:
        models['Stack'] = joblib.load('high_acc_stack_model.pkl')
        print("Loaded Stacked model")
    except Exception as e:
        print(f"Error loading Stacked model: {e}")
    
    return models

# Define the same feature extraction function as used in training
def extract_advanced_features(epochs):
    """Extract a comprehensive set of features for motor imagery classification with NaN handling"""
    n_epochs, n_channels, n_times = epochs.get_data().shape
    sfreq = epochs.info['sfreq']
    
    # Define motor-related frequency bands
    freq_bands = [
        (4, 8),    # Theta
        (8, 10),   # Lower Alpha
        (10, 13),  # Upper Alpha
        (13, 20),  # Lower Beta
        (20, 30),  # Upper Beta
    ]
    
    # Prepare feature matrix - spectral + temporal features
    n_features = n_channels * (len(freq_bands) + 3) + 20  # Additional connectivity features
    X_features = np.zeros((n_epochs, n_features))
    
    # 1. Extract spectral and temporal features
    for epoch_idx, epoch_data in enumerate(epochs.get_data()):
        feature_idx = 0
        for ch_idx in range(n_channels):
            # Handle invalid data in channels
            if np.isnan(epoch_data[ch_idx]).any() or np.isinf(epoch_data[ch_idx]).any():
                # Replace NaN/Inf with zeros for this calculation
                clean_data = np.nan_to_num(epoch_data[ch_idx], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                clean_data = epoch_data[ch_idx]
                
            # Spectral features - calculate band power for each frequency band
            try:
                freqs, psd = signal.welch(clean_data, fs=sfreq, nperseg=min(512, n_times))
                
                for low_freq, high_freq in freq_bands:
                    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                    if np.any(idx_band):
                        band_power = np.mean(psd[idx_band])
                        # Ensure we don't have invalid values
                        if np.isnan(band_power) or np.isinf(band_power):
                            band_power = 0
                    else:
                        band_power = 0
                    X_features[epoch_idx, feature_idx] = band_power
                    feature_idx += 1
            except Exception as e:
                print(f"Error in spectral feature calculation: {e}")
                # Fill with zeros if there's an error
                for _ in range(len(freq_bands)):
                    X_features[epoch_idx, feature_idx] = 0
                    feature_idx += 1
            
            # Temporal features - variance, skewness, kurtosis
            try:
                # Variance
                var_val = np.var(clean_data)
                X_features[epoch_idx, feature_idx] = 0 if np.isnan(var_val) else var_val
                feature_idx += 1
                
                # Skewness - handle case when standard deviation is zero
                if np.std(clean_data) > 1e-10:
                    skew_val = stats.skew(clean_data)
                    X_features[epoch_idx, feature_idx] = 0 if np.isnan(skew_val) else skew_val
                else:
                    X_features[epoch_idx, feature_idx] = 0
                feature_idx += 1
                
                # Kurtosis - handle case when standard deviation is zero
                if np.std(clean_data) > 1e-10:
                    kurt_val = stats.kurtosis(clean_data)
                    X_features[epoch_idx, feature_idx] = 0 if np.isnan(kurt_val) else kurt_val
                else:
                    X_features[epoch_idx, feature_idx] = 0
                feature_idx += 1
            except Exception as e:
                print(f"Error in temporal feature calculation: {e}")
                # Fill with zeros if there's an error
                for _ in range(3):  # 3 temporal features
                    X_features[epoch_idx, feature_idx] = 0
                    feature_idx += 1
    
    # 2. Extract connectivity features for key electrode pairs
    # Calculate correlations between important electrode pairs (C3-C4, etc.)
    ch_names = epochs.ch_names
    
    # Get important channel indices (motor cortex related)
    motor_ch_indices = []
    for name in ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'FCz', 'CP3', 'CP4', 'CPz']:
        indices = [i for i, ch in enumerate(ch_names) if name in ch]
        if indices:
            motor_ch_indices.extend(indices[:1])  # Take first matching index
    
    # Calculate connectivity between channels
    if len(motor_ch_indices) >= 2:
        for i, ch1 in enumerate(motor_ch_indices):
            for j, ch2 in enumerate(motor_ch_indices):
                if i < j and feature_idx < n_features:  # Only use half of the matrix (symmetry)
                    for epoch_idx in range(n_epochs):
                        try:
                            # Get clean data for correlation
                            data1 = epochs.get_data()[epoch_idx, ch1]
                            data2 = epochs.get_data()[epoch_idx, ch2]
                            
                            # Check for invalid values
                            if (np.isnan(data1).any() or np.isinf(data1).any() or 
                                np.isnan(data2).any() or np.isinf(data2).any()):
                                # Fill with zero correlation if invalid data
                                corr = 0
                            else:
                                # Check if both channels have variation
                                if np.std(data1) > 1e-10 and np.std(data2) > 1e-10:
                                    corr = np.corrcoef(data1, data2)[0, 1]
                                    # Check for NaN in correlation result
                                    if np.isnan(corr) or np.isinf(corr):
                                        corr = 0
                                else:
                                    corr = 0
                            X_features[epoch_idx, feature_idx] = corr
                        except Exception as e:
                            print(f"Error in connectivity calculation: {e}")
                            X_features[epoch_idx, feature_idx] = 0
                    feature_idx += 1
    
    # Log transform the spectral features (band powers)
    n_spectral = n_channels * len(freq_bands)
    # Add small epsilon and handle negative values before log transform
    X_features[:, :n_spectral] = np.where(X_features[:, :n_spectral] > 0, 
                                         np.log(X_features[:, :n_spectral] + 1e-10),
                                         0)
    
    # Final check for any NaN or infinite values
    if np.isnan(X_features).any() or np.isinf(X_features).any():
        print("Warning: NaN or Inf values found in features after extraction")
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
    return X_features

# Function to preprocess data for a test subject
def process_test_data(subject_id, subject_num, download_if_missing=True):
    """Process data for a test subject"""
    # Define runs for task 4 and task 5
    task_runs = {
        'Task4': [4],  # Left/right hand imagery
        'Task5': [8]   # Hands/feet imagery
    }
    
    raw_list = []
    print(f"Processing test subject: {subject_id}")
    
    for task, runs in task_runs.items():
        for run in runs:
            # PhysioNet file path
            file_path = os.path.join('files', f'MNE-eegbci-data', f'files', f'eegmmidb', f'1.0.0',
                                    f'S{str(subject_num).zfill(3)}', 
                                    f'S{str(subject_num).zfill(3)}R{str(run).zfill(2)}.edf')
            
            try:
                print(f"Attempting to load: {file_path}")
                raw = mne.io.read_raw_edf(file_path, preload=True)
                raw._data = raw._data.astype(np.float64)
                print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                raw_list.append(raw)
            except FileNotFoundError as e:
                print(f"File not found: {file_path}, error: {e}")
                if download_if_missing:
                    try:
                        print(f"Attempting to download run {run} for subject {subject_num}")
                        eegbci.load_data(subject_num, runs=[run], path='files/')
                        raw = mne.io.read_raw_edf(file_path, preload=True)
                        raw._data = raw._data.astype(np.float64)
                        print(f"Successfully downloaded and loaded data")
                        raw_list.append(raw)
                    except Exception as e2:
                        print(f"Download failed: {e2}")
                        continue
                else:
                    continue
    
    if not raw_list:
        print(f"No data loaded for subject {subject_id}, skipping...")
        return None, None, None
    
    # Concatenate runs
    raw_concat = mne.concatenate_raws(raw_list)
    print(f"Concatenated raw data for {subject_id} with {raw_concat.n_times} total samples")
    
    # Advanced preprocessing pipeline - same as training
    raw_concat.filter(l_freq=1.0, h_freq=None)
    raw_concat.notch_filter(freqs=[50, 60])
    raw_concat.set_eeg_reference('average', projection=False)
    raw_concat.filter(l_freq=4, h_freq=45)
    
    # Select motor cortex channels
    motor_channels = ['C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 
                     'FC3.', 'FC1.', 'FCz.', 'FC2.', 'FC4.', 
                     'CP3.', 'CP1.', 'CPz.', 'CP2.', 'CP4.',
                     'P3..', 'Pz..', 'P4..']
    
    # Get available motor channels
    available_motor = [ch for ch in motor_channels if ch in raw_concat.ch_names]
    
    if available_motor and len(available_motor) >= 5:
        print(f"Selecting {len(available_motor)} motor-related channels")
        raw_concat.pick_channels(available_motor)
    
    # Extract events
    events, event_id = mne.events_from_annotations(raw_concat)
    print(f"Found events: {event_id}")
    
    # Define events of interest for tasks 4 and 5
    event_id_selected = {
        'rest': event_id.get('T0', event_id.get('rest', None)),
        'left': event_id.get('T1', event_id.get('left', None)),
        'right': event_id.get('T2', event_id.get('right', None)),
        'hands': event_id.get('T3', event_id.get('hands', None)),
        'feet': event_id.get('T4', event_id.get('feet', None))
    }
    event_id_selected = {k: v for k, v in event_id_selected.items() if v is not None}
    
    if not event_id_selected:
        print(f"No relevant events found for subject {subject_id}, skipping...")
        return None, None, None
    
    # Create epochs
    epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                       tmin=0.5, tmax=3.5, baseline=(0.5, 1.0), preload=True)
    
    # Handle artifacts with ICA
    try:
        ica = ICA(n_components=15, random_state=42, method='fastica')
        ica.fit(epochs)
        eog_indices, eog_scores = ica.find_bads_eog(epochs)
        if eog_indices:
            ica.exclude = eog_indices[:2]
            print(f"ICA removing {len(ica.exclude)} artifacts")
            ica.apply(epochs)
    except Exception as e:
        print(f"ICA failed with error: {e}, continuing with threshold rejection")
        epochs.drop_bad(reject={'eeg': 200e-6})
    
    # Extract features
    X_features = extract_advanced_features(epochs)
    y = epochs.events[:, 2]
    
    # Get class names for reporting
    class_names = []
    for name, code in sorted(event_id_selected.items(), key=lambda x: x[1]):
        class_names.append(name)
    
    return X_features, y, class_names

# Function to evaluate models
def evaluate_models(models, X, y, class_names, subject_id='test'):
    """Evaluate all models and display results"""
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    
    results = {}
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\nEvaluating {name} model...")
        
        # Make predictions
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        results[name] = accuracy
        
        # Print classification report
        print(f"{name} Test accuracy: {accuracy:.4f}")
        print(classification_report(y, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.subplot(2, 2, i+1)
        cm = confusion_matrix(y, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{name} Model (Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f'{subject_id}_model_comparison.png')
    print(f"Saved confusion matrices to {subject_id}_model_comparison.png")
    
    # Determine best model
    best_model_name = max(results, key=results.get)
    print(f"\nBest model on test data is {best_model_name} with accuracy: {results[best_model_name]:.4f}")
    
    return results

# Test ensemble prediction function if available
def test_ensemble(X, y, class_names, subject_id='test'):
    """Test the saved ensemble prediction function"""
    try:
        ensemble_predict = joblib.load('ensemble_predict_function.pkl')
        print("\nTesting ensemble prediction function...")
        
        y_pred = ensemble_predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Ensemble prediction accuracy: {accuracy:.4f}")
        print(classification_report(y, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Ensemble Prediction (Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'{subject_id}_ensemble_prediction.png')
        print(f"Saved ensemble confusion matrix to {subject_id}_ensemble_prediction.png")
        
        return accuracy
    except Exception as e:
        print(f"Could not test ensemble prediction function: {e}")
        return None

# Main function
def main():
    # Choose a test subject - typically one not used in training
    # If you trained on subjects 1-10, you might want to test on the same ones
    # to measure training accuracy, or on new subjects to measure generalization
    test_subject_id = 'S001'  # Change this to test different subjects
    test_subject_num = 1      # Change this to match the ID
    
    # Load trained models
    print("Loading trained models...")
    models = load_models()
    
    if not models:
        print("No models could be loaded. Please check the model files.")
        return
    
    # Process test data
    print(f"\nProcessing test data for subject {test_subject_id}...")
    X_test, y_test, class_names = process_test_data(test_subject_id, test_subject_num)
    
    if X_test is None or y_test is None:
        print("Could not process test data. Please check the subject data.")
        return
    
    print(f"\nTest data shape: {X_test.shape}, with classes: {class_names}")
    
    # Evaluate models
    model_results = evaluate_models(models, X_test, y_test, class_names, test_subject_id)
    
    # Test ensemble prediction function
    ensemble_accuracy = test_ensemble(X_test, y_test, class_names, test_subject_id)
    
    print("\nModel Accuracy Summary:")
    for name, accuracy in model_results.items():
        print(f"{name} Model: {accuracy:.4f}")
    if ensemble_accuracy:
        print(f"Ensemble Prediction: {ensemble_accuracy:.4f}")

if __name__ == "__main__":
    main()