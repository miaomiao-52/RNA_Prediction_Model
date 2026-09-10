import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


print("Loading RNA-FM model...")
import fm
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


input_file = "mapped_scores_vivo_pds.pkl" 
print(f"Loading data from {input_file}...")
with open(input_file, "rb") as f:
    data = pickle.load(f)

import os


os.makedirs("temp_features", exist_ok=True)

print("\n=== Collecting training data (Memory Efficient) ===")
total_samples = 0

for sample_id, content in data.items():
    seq = content["sequence"]
    scores = content["scores"]
    
    
    if np.isnan(scores).all():
        continue
        
    print(f"Extracting training points for {sample_id}...")

    
    max_len = 1000
    all_features = []
    for i in range(0, len(seq), max_len):
        chunk_seq = seq[i : i + max_len]
        batch_data = [(sample_id, chunk_seq)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            chunk_features = results["representations"][12][0, 1 : len(chunk_seq) + 1].cpu().numpy()
            all_features.append(chunk_features)
    
    features = np.concatenate(all_features, axis=0)

    valid_indices = np.where(~np.isnan(scores))[0]
    if len(valid_indices) > 0:
        valid_features = features[valid_indices] 
        valid_targets = np.array(scores)[valid_indices] 
        
        
        np.save(f"temp_features/{sample_id}_X.npy", valid_features.astype(np.float32))
        np.save(f"temp_features/{sample_id}_y.npy", valid_targets.astype(np.float32))
        total_samples += len(valid_indices)

print(f"Total training samples identified: {total_samples}")
class DiskDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.files = [f.replace("_X.npy", "") for f in os.listdir(folder) if f.endswith("_X.npy")]
        self.folder = folder
        
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        x = np.load(f"{self.folder}/{name}_X.npy")
        y = np.load(f"{self.folder}/{name}_y.npy")
        return torch.from_numpy(x), torch.from_numpy(y).view(-1, 1)



train_dataset = DiskDataset("temp_features")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) 


print("\n=== Training Global MLP ===")


global_mlp = nn.Sequential(
    nn.Linear(640, 256), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(global_mlp.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    global_mlp.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        
        batch_x = batch_x.squeeze(0).to(device)
        batch_y = batch_y.squeeze(0).to(device)
        
        optimizer.zero_grad()
        outputs = global_mlp(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.5f}")


print("\n=== Global Imputation ===")
global_mlp.eval()
final_results = {}

for sample_id, content in data.items():
    seq = content["sequence"]
    scores = content["scores"]
    
    
    if np.isnan(scores).sum() == 0:
        final_results[sample_id] = scores
        continue

    print(f"Imputing missing values for {sample_id}...")
    
    
    max_len = 1000
    all_features = []
    for i in range(0, len(seq), max_len):
        chunk_seq = seq[i : i + max_len]
        batch_data = [(sample_id, chunk_seq)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            chunk_features = results["representations"][12][0, 1 : len(chunk_seq) + 1].cpu().numpy()
            all_features.append(chunk_features)
    
    features = np.concatenate(all_features, axis=0)
    
    
    nan_indices = np.where(np.isnan(scores))[0]
    nan_features = features[nan_indices]
    
    nan_features_tensor = torch.FloatTensor(nan_features).to(device)
    with torch.no_grad():
        preds = global_mlp(nan_features_tensor).cpu().numpy().flatten()
        
    
    imputed_scores = np.array(scores).copy()
    imputed_scores[nan_indices] = preds
    
    final_results[sample_id] = imputed_scores


output_file = "processed_mapped_scores_vivo_pds.pkl"
print(f"\nSaving results to {output_file}...")
with open(output_file, "wb") as f:
    pickle.dump(final_results, f)
import shutil
shutil.rmtree("temp_features")
print("All transcripts imputed with the global model.")
