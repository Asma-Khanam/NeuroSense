import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import numpy as np
import os
import joblib
from scipy import signal, stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Define subject IDs
subject_ids = [f'S{str(i).zfill(3)}' for i in range(1, 11)]
subject_nums = list(range(1, 11))

# Define runs for task 4 and task 5 only
task_runs = {
    'Task4': [4],  # Left/right hand imagery
    'Task5': [8]   # Hands/feet imagery
}

# Modified advanced feature extraction function with NaN handling
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

# Data processing pipeline
def process_subject_data(subject_id, subject_num, download_if_missing=True):
    """Process data for a single subject"""
    raw_list = []
    print(f"Processing subject: {subject_id}")
    
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
        return None, None
    
    # Concatenate runs
    raw_concat = mne.concatenate_raws(raw_list)
    print(f"Concatenated raw data for {subject_id} with {raw_concat.n_times} total samples")
    
    # Advanced preprocessing pipeline
    
    # 1. Apply high-pass filter to remove slow drifts
    raw_concat.filter(l_freq=1.0, h_freq=None)
    
    # 2. Apply notch filter for line noise
    raw_concat.notch_filter(freqs=[50, 60])
    
    # 3. Apply Common Average Reference (CAR) for better spatial resolution
    raw_concat.set_eeg_reference('average', projection=False)
    
    # 4. Apply band-pass filter for motor imagery frequencies
    raw_concat.filter(l_freq=4, h_freq=45)
    
    # 5. Select motor cortex channels
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
        return None, None
    
    print(f"Selected events: {event_id_selected}")
    
    # Create epochs with optimized timing
    epochs = mne.Epochs(raw_concat, events, event_id=event_id_selected, 
                       tmin=0.5, tmax=3.5,  # Focus on active motor imagery period
                       baseline=(0.5, 1.0),  # Baseline correction at start
                       preload=True)
    
    # Ensure data is double precision
    epochs._data = epochs._data.astype(np.float64)
    print(f"Created epochs with shape: {epochs.get_data().shape}")
    
    # Handle artifacts with ICA
    try:
        # Try ICA for better artifact removal
        ica = ICA(n_components=15, random_state=42, method='fastica')
        ica.fit(epochs)
        
        # Find components related to eye blinks and movements
        eog_indices, eog_scores = ica.find_bads_eog(epochs)
        if eog_indices:
            ica.exclude = eog_indices[:2]  # Exclude up to 2 EOG components
            print(f"ICA removing {len(ica.exclude)} artifacts")
            ica.apply(epochs)
    except Exception as e:
        print(f"ICA failed with error: {e}, continuing with threshold rejection")
        # Fall back to threshold-based artifact rejection
        epochs.drop_bad(reject={'eeg': 200e-6})
        
    print(f"After artifact handling, epochs shape: {epochs.get_data().shape}")
    
    if len(epochs) == 0:
        print(f"All epochs were dropped for subject {subject_id}, skipping...")
        return None, None
    
    # Extract features
    X_features = extract_advanced_features(epochs)
    y = epochs.events[:, 2]
    
    return X_features, y

# Collect data from all subjects
all_X = []
all_y = []
all_subjects = []  # For tracking which subject each sample came from

for subject_id, subject_num in zip(subject_ids, subject_nums):
    X_features, y = process_subject_data(subject_id, subject_num)
    
    if X_features is not None and y is not None:
        all_X.append(X_features)
        all_y.append(y)
        all_subjects.append(np.full(len(y), subject_num))
        print(f"Added {len(y)} trials for subject {subject_id} with labels: {np.unique(y)}")

# Concatenate data from all subjects
if all_X:
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)
    
    # Check for NaN values before proceeding
    nan_count = np.isnan(X).sum()
    print(f"\nChecking for NaN values in feature matrix...")
    print(f"Found {nan_count} NaN values out of {X.size} total values")

    if nan_count > 0:
        print("Replacing NaN values with zeros")
        X = np.nan_to_num(X, nan=0.0)
    
    # Check class balance
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\nClass distribution in training data:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Balance classes through SMOTE-like approach
    min_count = np.min(counts)
    balanced_X = []
    balanced_y = []
    
    for label in unique_labels:
        mask = (y == label)
        X_class = X[mask]
        y_class = y[mask]
        subj_class = subjects[mask]
        
        # If class is much smaller than others, augment it
        if len(X_class) < 0.7 * np.max(counts):
            n_augment = min(int(np.max(counts) * 0.8), len(X_class) * 2) - len(X_class)
            
            # Generate synthetic samples with small variations
            indices = np.random.choice(len(X_class), n_augment, replace=True)
            augmented = X_class[indices].copy()
            
            # Add random noise scaled to feature variance
            feature_std = np.std(X_class, axis=0) * 0.05  # 5% of std
            # Replace any NaN in std with small value
            feature_std = np.nan_to_num(feature_std, nan=0.01)
            noise = np.random.normal(0, feature_std, (n_augment, X_class.shape[1]))
            augmented += noise
            
            # Add augmented samples
            balanced_X.append(X_class)
            balanced_X.append(augmented)
            balanced_y.append(y_class)
            balanced_y.append(np.full(n_augment, label))
            
            print(f"Augmented class {label} with {n_augment} synthetic samples")
        else:
            # If class is large, use original data
            balanced_X.append(X_class)
            balanced_y.append(y_class)
    
    X = np.vstack(balanced_X)
    y = np.concatenate(balanced_y)
    
    # Final NaN check after augmentation
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values after augmentation, replacing with zeros")
        X = np.nan_to_num(X, nan=0.0)
    
    print("\nAfter augmentation, class distribution:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({count/len(y)*100:.1f}%)")
else:
    raise ValueError("No data or labels found. Check files and annotations.")

# Create multiple classifier pipelines specialized for EEG/BCI with NaN handling

# 1. SVM pipeline with feature selection and imputation
svm_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Add imputer for NaN handling
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=min(150, X.shape[1]))),
    ('classifier', SVC(probability=True, class_weight='balanced'))
])

# 2. LDA pipeline with shrinkage (best for BCI) and imputation
lda_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Add imputer for NaN handling
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=min(100, X.shape[1]))),
    ('classifier', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))
])

# 3. GradientBoosting for complex patterns with imputation
gb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Add imputer for NaN handling
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=min(200, X.shape[1]))),
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                             max_depth=5, subsample=0.8))
])

# Evaluate each pipeline with cross-validation
print("\nEvaluating classifiers with cross-validation...")

# Use stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter grids
svm_param_grid = {
    'classifier__C': [0.1, 1.0, 10.0, 100.0],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
    'classifier__kernel': ['rbf', 'linear'],
    'feature_selection__k': [50, 100, 150]
}

lda_param_grid = {
    'feature_selection__k': [30, 50, 100]
}

gb_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7],
    'feature_selection__k': [50, 100, 200]
}

# Optimize SVM
print("\nOptimizing SVM classifier...")
from sklearn.model_selection import GridSearchCV
svm_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
svm_search.fit(X, y)
print(f"Best SVM parameters: {svm_search.best_params_}")
print(f"Best SVM CV score: {svm_search.best_score_:.4f}")
best_svm = svm_search.best_estimator_

# Optimize LDA
print("\nOptimizing LDA classifier...")
lda_search = GridSearchCV(lda_pipeline, lda_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
lda_search.fit(X, y)
print(f"Best LDA parameters: {lda_search.best_params_}")
print(f"Best LDA CV score: {lda_search.best_score_:.4f}")
best_lda = lda_search.best_estimator_

# Optimize GradientBoosting
print("\nOptimizing GradientBoosting classifier...")
gb_search = GridSearchCV(gb_pipeline, gb_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
gb_search.fit(X, y)
print(f"Best GB parameters: {gb_search.best_params_}")
print(f"Best GB CV score: {gb_search.best_score_:.4f}")
best_gb = gb_search.best_estimator_

# Evaluate best models
cv_scores = {
    'SVM': cross_val_score(best_svm, X, y, cv=cv, scoring='accuracy'),
    'LDA': cross_val_score(best_lda, X, y, cv=cv, scoring='accuracy'),
    'GB': cross_val_score(best_gb, X, y, cv=cv, scoring='accuracy')
}

for name, scores in cv_scores.items():
    print(f"{name} CV accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Create stacked ensemble model
from sklearn.ensemble import StackingClassifier

# Define base estimators
estimators = [
    ('svm', best_svm),
    ('lda', best_lda),
    ('gb', best_gb)
]

# Create stacking classifier with LDA as final estimator
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LinearDiscriminantAnalysis(),
    cv=5
)

# Train stacked model
print("\nTraining stacked ensemble model...")
stack.fit(X, y)

# Evaluate stacked model
stack_scores = cross_val_score(stack, X, y, cv=cv, scoring='accuracy')
print(f"Stacked ensemble CV accuracy: {np.mean(stack_scores):.4f} ± {np.std(stack_scores):.4f}")

# Train final models on all data
print("\nTraining final models on all data...")
best_svm.fit(X, y)
best_lda.fit(X, y)
best_gb.fit(X, y)
stack.fit(X, y)

# Save all models
joblib.dump(best_svm, 'high_acc_svm_model.pkl')
joblib.dump(best_lda, 'high_acc_lda_model.pkl')
joblib.dump(best_gb, 'high_acc_gb_model.pkl')
joblib.dump(stack, 'high_acc_stack_model.pkl')

print("\nAll models trained and saved!")

# Determine best model
model_scores = {
    'SVM': np.mean(cv_scores['SVM']),
    'LDA': np.mean(cv_scores['LDA']),
    'GB': np.mean(cv_scores['GB']),
    'Stack': np.mean(stack_scores)
}

best_model_name = max(model_scores, key=model_scores.get)
print(f"\nBest model is {best_model_name} with CV accuracy: {model_scores[best_model_name]:.4f}")

# Create special ensemble prediction function for external use
def ensemble_predict(X, weights=None):
    """Make ensemble prediction using all trained models"""
    if weights is None:
        # Default weights based on performance
        weights = {
            'SVM': model_scores['SVM'] / sum(model_scores.values()),
            'LDA': model_scores['LDA'] / sum(model_scores.values()),
            'GB': model_scores['GB'] / sum(model_scores.values()),
            'Stack': model_scores['Stack'] / sum(model_scores.values())
        }
    
    # Get probabilities from each model
    svm_probs = best_svm.predict_proba(X)
    lda_probs = best_lda.predict_proba(X)
    gb_probs = best_gb.predict_proba(X)
    stack_probs = stack.predict_proba(X)
    
    # Weighted average of probabilities
    ensemble_probs = (
        svm_probs * weights['SVM'] +
        lda_probs * weights['LDA'] +
        gb_probs * weights['GB'] +
        stack_probs * weights['Stack']
    )
    
    # Return class with highest probability
    return np.argmax(ensemble_probs, axis=1)

# Save ensemble prediction function
joblib.dump(ensemble_predict, 'ensemble_predict_function.pkl')

print("Training complete! Use 'high_acc_stack_model.pkl' for best performance.")