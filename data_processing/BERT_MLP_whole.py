import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


input_pkl = "processed_mapped_scores_vivo_pds.pkl"  
input_fasta = "HeLa.fa"  
output_dir = "whole_predicted_scores_dir"  


print("Loading RNA-FM model...")
import fm

model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Loading training data from {input_pkl}...")
with open(input_pkl, "rb") as f:
    training_data = pickle.load(f)

os.makedirs("temp_features", exist_ok=True)


print("\n=== Collecting training data ===")
total_samples = 0
for sample_id, content in training_data.items():
    seq = content["sequence"]
    scores = content["scores"]
    if np.isnan(scores).all():
        continue

    max_len = 1000
    all_features = []
    for i in range(0, len(seq), max_len):
        chunk_seq = seq[i : i + max_len]
        batch_data = [(sample_id, chunk_seq)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            chunk_features = (
                results["representations"][12][0, 1 : len(chunk_seq) + 1]
                .cpu()
                .numpy()
            )
            all_features.append(chunk_features)

    features = np.concatenate(all_features, axis=0)
    valid_indices = np.where(~np.isnan(scores))[0]
    if len(valid_indices) > 0:
        valid_features = features[valid_indices]
        valid_targets = np.array(scores)[valid_indices]
        np.save(
            f"temp_features/{sample_id}_X.npy", valid_features.astype(np.float32)
        )
        np.save(
            f"temp_features/{sample_id}_y.npy", valid_targets.astype(np.float32)
        )
        total_samples += len(valid_indices)

print(f"Total training samples identified: {total_samples}")



class DiskDataset(torch.utils.data.Dataset):

    def __init__(self, folder):
        self.files = [
            f.replace("_X.npy", "")
            for f in os.listdir(folder)
            if f.endswith("_X.npy")
        ]
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
    nn.Linear(640, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
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
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.5f}")

shutil.rmtree("temp_features")



print(
    "\n=== Whole Transcriptome Prediction (Streaming & Resume) ==="
)
os.makedirs(output_dir, exist_ok=True)


def stream_fasta(fasta_path):
    current_id = None
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    yield current_id, "".join(current_seq)

                current_id = line[1:].split()[0].replace("/", "_")
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_id:
            yield current_id, "".join(current_seq)


global_mlp.eval()
count = 0

print(f"Streaming and predicting from FASTA: {input_fasta}...")
for transcript_id, seq in stream_fasta(input_fasta):
    count += 1

    
    save_path = os.path.join(output_dir, f"{transcript_id}.npy")

    
    if os.path.exists(save_path):
        if count % 2000 == 0:
            print(f"Already predicted {count} sequences, skipping...")
        continue

    if count % 200 == 0:
        print(
            f"[{count}] Processing {transcript_id} | Length: {len(seq)}..."
        )

    if len(seq) == 0:
        continue

    
    max_len = 1000
    all_features = []
    for i in range(0, len(seq), max_len):
        chunk_seq = seq[i : i + max_len]
        batch_data = [(transcript_id, chunk_seq)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            chunk_features = (
                results["representations"][12][0, 1 : len(chunk_seq) + 1]
                .cpu()
                .numpy()
            )
            all_features.append(chunk_features)

    features = np.concatenate(all_features, axis=0)

    
    features_tensor = torch.FloatTensor(features).to(device)
    with torch.no_grad():
        preds = global_mlp(features_tensor).cpu().numpy().flatten()

    
    np.save(save_path, preds.astype(np.float32))

print(
    f"All predicted transcripts saved individually in '{output_dir}'"
)