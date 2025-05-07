import mne
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os
from pathlib import Path

# Define the base directory
# Update this path to where your data actually is
base_dir = Path("/Users/asmakhanam/Desktop/proj_neuro")

# First, let's download the PhysioNet data if it's not already present
# This will download the data to the default MNE data path and return the path
def download_physionet_data():
    print("Checking for PhysioNet EEG Motor Movement/Imagery Dataset...")
    eeg_path = mne.datasets.eegbci.load_data(subjects=[1], runs=[4, 5], update_path=True)
    # Get the base directory from the returned path
    base_path = Path(eeg_path[0]).parent.parent
    print(f"Data is available at: {base_path}")
    return base_path

# Define subject IDs (e.g., S001 to S010)
subject_ids = [f'S{str(i).zfill(3)}' for i in range(1, 11)]  # ['S001', 'S002', ..., 'S010']

# Define runs for Task 2 and Task 3 based on PhysioNet dataset
task_runs = {
    'Task2': ['R04'],  # Imagine opening/closing left or right fist
    'Task3': ['R05']   # Open/close both fists or both feet
}

# Check if we need to download the data
data_path = download_physionet_data()
print(f"Using data path: {data_path}")

# Lists to store data across all subjects
all_X = []
all_y = []

# Process each subject
for subject_idx, subject_id in enumerate(subject_ids, 1):
    raw_list = []
    print(f"Processing subject: {subject_id} (index {subject_idx})")
    
    for task, runs in task_runs.items():
        for run in runs:
            # Check standard MNE path using subject index instead of ID string
            run_num = int(run[1:])  # Extract numeric part of run (e.g., '04' from 'R04')
            file_path = data_path / f"S{str(subject_idx).zfill(3)}" / f"S{str(subject_idx).zfill(3)}{run}.edf"
            
            # Alternative path using mne function to get file path
            # This is a more reliable way to get the file paths
            try:
                alt_paths = mne.datasets.eegbci.load_data(subjects=[subject_idx], runs=[run_num], update_path=True)
                if alt_paths:
                    file_path = alt_paths[0]  # Use the first path returned
            except Exception as e:
                print(f"Error getting alternative path: {e}")
            
            try:
                print(f"Attempting to load: {file_path}")
                raw = mne.io.read_raw_edf(file_path, preload=True)
                print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                raw_list.append(raw)
            except FileNotFoundError as e:
                print(f"File not found: {file_path}, error: {e}")
                try:
                    # Try the alternative approach using mne.datasets directly
                    print("Attempting to download directly using MNE API...")
                    alt_paths = mne.datasets.eegbci.load_data(subjects=[subject_idx], runs=[run_num], update_path=True)
                    if alt_paths:
                        file_path = alt_paths[0]
                        print(f"Direct download successful, loading: {file_path}")
                        raw = mne.io.read_raw_edf(file_path, preload=True)
                        print(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} samples")
                        raw_list.append(raw)
                except Exception as inner_e:
                    print(f"Direct download/load failed: {inner_e}")
                    continue
            except Exception as e:
                print(f"Other error loading {file_path}: {e}")
                continue
    
    if not raw_list:
        print(f"No data loaded for subject {subject_id}, skipping...")
        continue
    
    # Concatenate runs for the subject
    raw_concat = mne.concatenate_raws(raw_list)
    print(f"Concatenated raw data for {subject_id} with {raw_concat.n_times} total samples")
    
    # Fine-tuned preprocessing: bandpass filter and notch filter
    raw_concat.filter(l_freq=7, h_freq=30, l_trans_bandwidth=2, h_trans_bandwidth=5)  # Narrower band for mu/beta
    raw_concat.notch_filter(freqs=60)  # 60 Hz powerline noise in US
    print("Filtering applied (7-30 Hz with 60 Hz notch)")
    
    # Extract events from annotations
    try:
        events, event_id = mne.events_from_annotations(raw_concat)
        print(f"Found events: {event_id}")
    except Exception as e:
        print(f"Error extracting events: {e}")
        continue
    
    # Define events of interest with numeric mappings from PhysioNet
    # Map the actual event IDs found in this dataset
    # PhysioNet Motor Imagery uses T0=rest, T1=left fist, T2=right fist, etc.
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
    
    # Fine-tuned epoching with baseline correction
    try:
        epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                           tmin=0, tmax=4.0, baseline=(0, 0.5), preload=True)  # Extended window, baseline adjustment
        print(f"Created epochs for {subject_id} with shape: {epochs.get_data().shape}")
    except Exception as e:
        print(f"Error creating epochs: {e}")
        continue
    
    # Artifact rejection
    try:
        epochs.drop_bad(reject={'eeg': 100e-6})  # Reject epochs with EEG > 100 µV
        print(f"After artifact rejection, epochs shape: {epochs.get_data().shape}")
    except Exception as e:
        print(f"Error during artifact rejection: {e}")
        # Continue with all epochs if rejection fails
    
    # Extract data and labels
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]  # Event codes
    
    if X.size > 0 and y.size > 0:
        all_X.append(X)
        all_y.append(y)
        print(f"Added {X.shape[0]} epochs for {subject_id} with labels: {np.unique(y)}")
    else:
        print(f"No valid epochs for {subject_id}")

# Concatenate data from all subjects
if all_X:
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique labels: {np.unique(y)}")
else:
    print("No data was collected. Check your PhysioNet data installation.")
    print("You can manually download the dataset from https://physionet.org/content/eegmmidb/1.0.0/")
    print("Try running: mne.datasets.eegbci.load_data(subjects=[1], runs=[4, 5], update_path=True)")
    raise ValueError("No data or labels found. Check files and annotations.")

# Verify we have enough data to train
if len(np.unique(y)) < 2:
    raise ValueError(f"Not enough classes found in data. Only found labels: {np.unique(y)}")

# Fine-tuned CSP and classifier pipeline with corrected regularization
csp = CSP(n_components=6, reg='shrinkage', log=True, norm_trace=True)
clf = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
pipe = Pipeline([('CSP', csp), ('SVC', clf)])

# Train the model
print(f"Training model on {X.shape[0]} samples with {len(np.unique(y))} classes...")
pipe.fit(X, y)

# Save the model
model_path = base_dir / 'test1.pkl'
joblib.dump(pipe, model_path)
print(f"Model saved as '{model_path}'")