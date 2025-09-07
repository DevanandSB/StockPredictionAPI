# inspect_metadata.py
import os
import pickle

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metadata_path = os.path.join(base_dir, "data", "metadata.pkl")

try:
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print("Current metadata contents:")
    print(metadata)
    print("\nKeys in metadata:")
    print(list(metadata.keys()))
except Exception as e:
    print(f"Error reading metadata: {e}")