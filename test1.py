import mne
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def evaluate_model(model_path='test1.pkl', test_subjects=None):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model pickle file
    test_subjects : list of str or int
        List of subject IDs to test on. If None, use all available subjects
    """
    # Base directory
    base_dir = Path(os.path.dirname(os.path.abspath(model_path)) if os.path.dirname(model_path) else '.')
    
    # Load the trained model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # If no test subjects provided, use a default list
    if test_subjects is None:
        test_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Use subject indices
    
    # Define runs for test
    test_runs = {
        'Task2': [4],  # Imagine opening/closing left or right fist (R04)
        'Task3': [5]   # Open/close both fists or both feet (R05)
    }
    
    # Store results for each subject
    all_results = {}
    all_y_true = []
    all_y_pred = []
    
    # Process each test subject
    for subject in test_subjects:
        # Convert to proper subject ID format if needed
        if isinstance(subject, int):
            subject_idx = subject
            subject_id = f'S{str(subject).zfill(3)}'
        else:
            subject_id = subject
            # Extract subject index from string (e.g., 'S001' -> 1)
            subject_idx = int(subject.lstrip('S').lstrip('0') or '0')
        
        print(f"\n{'='*50}")
        print(f"Testing on subject: {subject_id} (index: {subject_idx})")
        print(f"{'='*50}")
        
        # Store raw data for all runs
        raw_list = []
        
        # Try to load the test data for each run
        for task, runs in test_runs.items():
            for run in runs:
                try:
                    # Try to get the file path using MNE's built-in function
                    file_paths = mne.datasets.eegbci.load_data(subjects=[subject_idx], runs=[run], update_path=True)
                    
                    if file_paths:
                        file_path = file_paths[0]
                        print(f"Attempting to load: {file_path}")
                        raw = mne.io.read_raw_edf(file_path, preload=True)
                        print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                        raw_list.append(raw)
                        continue  # Skip to next run if successful
                except Exception as e:
                    print(f"Error loading via MNE API: {e}")
                
                # Fall back to direct path if MNE API fails
                file_path = base_dir / f"files/{subject_id}/{subject_id}R0{run}.edf"
                alt_path = base_dir / f"files{subject_id}/{subject_id}R0{run}.edf"
                
                try:
                    print(f"Attempting to load: {file_path}")
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                    raw_list.append(raw)
                except FileNotFoundError:
                    try:
                        print(f"Attempting to load: {alt_path}")
                        raw = mne.io.read_raw_edf(alt_path, preload=True)
                        print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                        raw_list.append(raw)
                    except FileNotFoundError as e:
                        print(f"File not found: {alt_path}, error: {e}")
                        continue
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        if not raw_list:
            print(f"No data could be loaded for subject {subject_id}, skipping...")
            continue
        
        # Concatenate runs for the subject
        raw_concat = mne.concatenate_raws(raw_list)
        print(f"Concatenated raw data for {subject_id} with {raw_concat.n_times} total samples")
        
        # Apply fine-tuned preprocessing
        raw_concat.filter(l_freq=7, h_freq=30, l_trans_bandwidth=2, h_trans_bandwidth=5)
        raw_concat.notch_filter(freqs=60)  # 60 Hz powerline noise in US
        print("Filtering applied (7-30 Hz with 60 Hz notch)")
        
        # Extract events from annotations
        try:
            events, event_id = mne.events_from_annotations(raw_concat)
            print(f"Found events: {event_id}")
        except Exception as e:
            print(f"Error extracting events: {e}")
            continue
        
        # Map available event IDs to standardized categories
        possible_event_keys = {
            'Idle': ['T0', '768', 'rest', 'Rest'],
            'Task2_left': ['T1', '769', 'left fist', 'Left Fist'], 
            'Task2_right': ['T2', '770', 'right fist', 'Right Fist'],
            'Task3_fists': ['T3', '771', 'both fists', 'Both Fists'],
            'Task3_feet': ['T4', '772', 'both feet', 'Both Feet']
        }
        
        event_id_selected = {}
        for task_name, possible_keys in possible_event_keys.items():
            for key in possible_keys:
                if key in event_id:
                    event_id_selected[task_name] = event_id[key]
                    break
        
        if not event_id_selected:
            print(f"No relevant events found for subject {subject_id}")
            print(f"Available events: {event_id}")
            continue
        
        print(f"Selected events: {event_id_selected}")
        
        # Create epochs with fine-tuned parameters
        try:
            epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                               tmin=0, tmax=4.0, baseline=(0, 0.5), preload=True)
            print(f"Created epochs with shape: {epochs.get_data().shape}")
        except Exception as e:
            print(f"Error creating epochs: {e}")
            continue
        
        # Apply artifact rejection with a higher threshold for test data
        try:
            n_epochs_before = len(epochs)
            epochs.drop_bad(reject={'eeg': 300e-6})  # Increased to 300 µV for test data
            n_epochs_after = len(epochs)
            print(f"Artifact rejection removed {n_epochs_before - n_epochs_after} epochs")
            print(f"After artifact rejection, epochs shape: {epochs.get_data().shape}")
        except Exception as e:
            print(f"Error during artifact rejection: {e}")
            # Continue with all epochs if rejection fails
        
        if len(epochs) == 0:
            print("All epochs were dropped after artifact rejection.")
            try:
                # Plot drop log for diagnosis
                epochs.plot_drop_log()
                plt.title(f"Drop log for {subject_id}")
                plt.savefig(f"{subject_id}_drop_log.png")
                plt.close()
                print(f"Drop log saved to {subject_id}_drop_log.png")
            except Exception as e:
                print(f"Error plotting drop log: {e}")
            continue
        
        # Extract test data and labels
        X_test = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        y_test = epochs.events[:, 2]  # Event codes
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            print("Predictions completed.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
        
        # Calculate metrics
        try:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            print(f"Accuracy for {subject_id}: {accuracy:.2f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            print("Confusion Matrix:")
            print(conf_matrix)
            
            # Store results
            all_results[subject_id] = {
                'accuracy': accuracy,
                'report': report,
                'conf_matrix': conf_matrix,
                'n_samples': len(y_test)
            }
            
            # Collect all predictions for overall metrics
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
            # Map numeric labels to task names for better interpretation
            label_map = {v: k for k, v in event_id_selected.items()}
            y_test_names = [label_map.get(label, f'Unknown_{label}') for label in y_test]
            y_pred_names = [label_map.get(label, f'Unknown_{label}') for label in y_pred]
            
            print("\nMapped Labels - True vs Predicted (sample of max 10):")
            for true, pred in zip(y_test_names[:10], y_pred_names[:10]):
                print(f"True: {true}, Predicted: {pred}")
            
            # Visualize confusion matrix
            plt.figure(figsize=(10, 8))
            class_labels = [label_map.get(label, f'Unknown_{label}') for label in np.unique(y_test)]
            plot_confusion_matrix(conf_matrix, class_labels, title=f"Confusion Matrix - {subject_id}")
            plt.savefig(f"{subject_id}_confusion_matrix.png")
            plt.close()
            print(f"Confusion matrix saved to {subject_id}_confusion_matrix.png")
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            continue
    
    # Calculate overall metrics
    if all_y_true and all_y_pred:
        print("\n" + "="*50)
        print("OVERALL RESULTS ACROSS ALL TEST SUBJECTS")
        print("="*50)
        
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        overall_report = classification_report(all_y_true, all_y_pred, zero_division=0)
        overall_confusion = confusion_matrix(all_y_true, all_y_pred)
        
        print(f"Overall Accuracy: {overall_accuracy:.2f}")
        print("Overall Classification Report:")
        print(overall_report)
        print("Overall Confusion Matrix:")
        print(overall_confusion)
        
        # Print per-subject summary
        print("\nPer-Subject Summary:")
        print(f"{'Subject':^10} | {'Accuracy':^10} | {'Samples':^10}")
        print("-" * 34)
        for subject_id, result in all_results.items():
            print(f"{subject_id:^10} | {result['accuracy']:^10.2f} | {result['n_samples']:^10}")
    else:
        print("\nNo test results collected. Check data paths and event mappings.")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot confusion matrix with labels.
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    return plt

if __name__ == "__main__":
    # Path to the saved model
    model_path = 'test1.pkl'
    
    # Optionally specify test subjects
    # test_subjects = ['S005', 'S006']  # Test specific subjects
    test_subjects = None  # Test all available subjects
    
    # Run evaluation
    evaluate_model(model_path, test_subjects)