import os

locations = [
    os.path.join(os.path.dirname(__file__), "app", "data")
]

files_to_check = ["model_weight.pt", "metadata.pkl", "scaler.pkl"]

for location in locations:
    print(f"\nChecking location: {location}")
    if os.path.exists(location):
        for file in files_to_check:
            file_path = os.path.join(location, file)
            exists = os.path.exists(file_path)
            print(f"  {file}: {'✅' if exists else '❌'} {'Exists' if exists else 'Missing'}")
            if exists:
                print(f"    Size: {os.path.getsize(file_path)} bytes")
    else:
        print(f"  Directory does not exist")