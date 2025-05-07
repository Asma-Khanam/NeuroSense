import pickle as pkl
import pandas as pd
import sys
import joblib

def convert_pickle_to_csv(pickle_file, csv_file):
    print(f"Attempting to convert {pickle_file} to {csv_file}")
    
    # Try multiple methods to load the pickle file
    try:
        # Method 1: Standard pickle
        with open(pickle_file, "rb") as f:
            data = pkl.load(f)
        print("Successfully loaded with standard pickle")
    except Exception as e:
        print(f"Standard pickle failed: {e}")
        try:
            # Method 2: Using joblib (good for sklearn models)
            data = joblib.load(pickle_file)
            print("Successfully loaded with joblib")
        except Exception as e:
            print(f"Joblib failed: {e}")
            return False
    
    # Check what type of data we have
    print(f"Loaded data type: {type(data)}")
    
    # Handle different types of pickle objects
    if isinstance(data, pd.DataFrame):
        # If it's already a DataFrame
        df = data
    elif hasattr(data, 'feature_importances_') or hasattr(data, 'coef_'):
        # It's likely a scikit-learn model
        print("Detected a machine learning model")
        
        # For a GradientBoostingClassifier or similar model
        if hasattr(data, 'feature_importances_'):
            features = getattr(data, 'feature_names_in_', None)
            if features is None:
                features = [f'feature_{i}' for i in range(len(data.feature_importances_))]
            
            df = pd.DataFrame({
                'feature': features,
                'importance': data.feature_importances_
            })
        elif hasattr(data, 'coef_'):
            features = getattr(data, 'feature_names_in_', None)
            if features is None:
                features = [f'feature_{i}' for i in range(data.coef_.shape[1] if len(data.coef_.shape) > 1 else len(data.coef_))]
            
            df = pd.DataFrame({
                'feature': features,
                'coefficient': data.coef_[0] if len(data.coef_.shape) > 1 else data.coef_
            })
    else:
        # Try to convert other types to DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"Could not convert to DataFrame: {e}")
            # Last resort: save useful information about the model
            df = pd.DataFrame({'pickle_attributes': [attr for attr in dir(data) if not attr.startswith('__')]})
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    print(f"Successfully saved to {csv_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 2:
        pickle_file = sys.argv[1]
        csv_file = sys.argv[2]
    else:
        pickle_file = "high_acc_gb_model.pkl"
        csv_file = "model_data.csv"
    
    success = convert_pickle_to_csv(pickle_file, csv_file)
    
    if not success:
        print("Failed to convert the pickle file to CSV.")