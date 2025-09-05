import torch
import pickle
import os

# Check the actual model architecture
base_dir = os.path.dirname(os.path.abspath(__file__))  # app/models/
app_dir = os.path.dirname(base_dir)  # app/
model_path = os.path.join(app_dir, "data", "model_weight.pt")

print(f"Loading model from: {model_path}")

# Load state dict to see the architecture
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
print("\nAll keys in state dict:")
for key in state_dict.keys():
    print(f"  {key}")

print(f"\nTotal parameters: {len(state_dict)}")
print("\nModel architecture suggests:")
if "pos_encoder.pe" in state_dict:
    print("  - Has positional encoding")
if "input_proj.weight" in state_dict:
    print("  - Input projection layer name: input_proj")
if "output.0.weight" in state_dict:
    print("  - Output layer is sequential with multiple layers")

# Also check metadata
metadata_path = os.path.join(app_dir, "data", "metadata.pkl")
if os.path.exists(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"\nMetadata: {metadata}")