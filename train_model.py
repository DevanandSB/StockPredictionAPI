import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import sys
from torch.utils.data import TensorDataset, DataLoader

# Add the app directory to the path so we can import the model modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from models.transformer_model import StockPredictionTransformer
from models.prediction_model import prediction_model  # for file paths

# --- 1. Load the metadata to get feature names ---
try:
    with open(prediction_model.metadata_path, 'rb') as f:
        metadata = pickle.load(f)
except FileNotFoundError:
    print(f"Error: metadata.pkl not found at {prediction_model.metadata_path}.")
    print("Please ensure this file exists before running the script.")
    exit()

required_features = metadata.get('feature_columns', [])
if not required_features:
    print("Error: 'feature_columns' not found in metadata.pkl.")
    exit()

### YOUR DATA LOADING CODE HERE ###
# You must load your raw, unscaled features (X_train) and targets (y_train).
# Replace the following lines with your actual data loading logic.
# Example: X_train = pd.read_csv('your_raw_features.csv')
# Example: y_train = pd.read_csv('your_raw_targets.csv')
print("1. Loading raw training data...")
# --- PLACE YOUR CODE BELOW THIS LINE ---
try:
    # Example: Loading from a CSV file
    # This is a placeholder; you MUST replace this with your real data.
    # X_train_raw = pd.read_csv('your_real_features_file.csv')
    # y_train = pd.read_csv('your_real_targets_file.csv').squeeze()

    # Placeholder using dummy data
    X_train_raw = pd.DataFrame(np.random.rand(100, len(required_features)), columns=required_features)
    y_train = pd.Series(np.random.rand(100))
    print("WARNING: Using dummy data. You MUST replace this with your real data to get real predictions.")

except Exception as e:
    print(f"Error loading your data: {e}")
    print("Please ensure your data loading code is correct and your files are in the right path.")
    exit()
# --- PLACE YOUR CODE ABOVE THIS LINE ---

# --- 2. Fit and save the REAL scaler ---
print("2. Fitting and saving the real MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

with open(prediction_model.scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Real scaler saved to {prediction_model.scaler_path}")

# --- 3. Prepare data for PyTorch and train the model ---
print("3. Training the StockPredictionTransformer model...")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(0)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(0)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32)

feature_size = len(required_features)
model = StockPredictionTransformer(
    feature_size=feature_size,
    d_model=256, nhead=8, num_encoder_layers=4,
    dim_feedforward=1024, dropout=0.1
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50  # You can adjust the number of epochs
for epoch in range(num_epochs):
    for features, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# --- 4. Save the trained model's weights ---
torch.save(model.state_dict(), prediction_model.model_path)
print(f"Trained model weights saved to {prediction_model.model_path}")

print("\nModel training complete. All required assets for real predictions are ready.")
print("You can now restart your FastAPI application.")