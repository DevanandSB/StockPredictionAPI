# inspect_model.py
import pickle
import torch


def inspect_model_files():
    print("Inspecting model files...")

    try:
        # Check features.pkl
        with open('app/data/features.pkl', 'rb') as f:
            features = pickle.load(f)
            print(f"Features type: {type(features)}")
            if hasattr(features, 'n_features_in_'):
                print(f"Scaler with {features.n_features_in_} features")
            else:
                print(f"Features: {features}")
                print(f"Number of features: {len(features)}")

    except Exception as e:
        print(f"Error reading features.pkl: {e}")

    try:
        # Check metadata.pkl
        with open('app/data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"Error reading metadata.pkl: {e}")

    try:
        # Check model architecture
        model_weights = torch.load('app/data/model_weight.pt', map_location='cpu')
        print(f"Model keys: {list(model_weights.keys())[:10]}")  # Show first 10 keys
    except Exception as e:
        print(f"Error reading model weights: {e}")


if __name__ == "__main__":
    inspect_model_files()