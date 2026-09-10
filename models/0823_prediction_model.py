import math
import os
import random
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CONFIG = {
    "raw_input_csv": "./RNA_whole_label_HeLa_1.csv",
    "balanced_output_csv": "./RNA_training_HeLa_1_filtered.csv",
    "fasta_path": "./HeLa.fa",
    "predicted_scores_dir": "./whole_predicted_scores_dir",
    "model_save_path": "./best_model_0823.pt",
    
    "loss_curve_path": "./loss_curve_0823.png",
    "soft_confusion_matrix_path": "./soft_confusion_matrix_0823.png",
    "predicted_probs_csv": "./test_predicted_probabilities_0823.csv",
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 15,
    "num_classes": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "num_workers": 8,          
    "persistent_workers": True,
    "prefetch_factor": 4,
    "pin_memory": True,
    "use_amp": True,           
    "num_attn_heads": 8,       
    "rg4_window": 12,          
}

CLASS_NAMES = ["G-tetrad", "Stem-loop", "Linear-loop", "Unstructured"]

FEAT_DIM_RNA_FM = 128
FEAT_DIM_REACTIVITY = 64
FEAT_DIM_STRUCT_PRIOR = 8
FEAT_DIM_TOTAL = FEAT_DIM_RNA_FM + FEAT_DIM_REACTIVITY + FEAT_DIM_STRUCT_PRIOR  


def plot_loss_curve(train_losses, val_losses, save_path):
    
    plt.figure(figsize=(9, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4",
              linewidth=2, marker="o", markersize=3)
    plt.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e",
              linewidth=2, marker="s", markersize=3)

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"loss curve: {save_path}")


def plot_soft_confusion_matrix(targets, probs, class_names, save_path):
    
    num_classes = len(class_names)
    soft_cm = np.zeros((num_classes, num_classes), dtype=np.float32)

    targets_np = np.array(targets)
    probs_np = np.array(probs)  

    for c in range(num_classes):
        idx = np.where(targets_np == c)[0]
        if len(idx) > 0:
            soft_cm[c, :] = np.mean(probs_np[idx], axis=0)
        else:
            soft_cm[c, :] = 0.0

    soft_cm_pct = soft_cm * 100.0  

    plt.figure(figsize=(8.5, 7))
    sns.heatmap(
        soft_cm_pct,
        annot=True,
        fmt=".1f",  
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        vmin=0.0,
        vmax=100.0,
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"format": "%.0f%%"},
    )

    ax = plt.gca()
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")

    plt.xlabel("Predicted Class Probability (Sum = 100%)", fontsize=12, labelpad=10)
    plt.ylabel("True Class Label", fontsize=12, labelpad=10)
    plt.title(
        "Soft Confusion Matrix (Average Predicted Probability % per True Class)",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"confusion matrix: {save_path}")



def build_multi_relational_edges(seq, ss_dotbracket, rg4_window=12):
    
    N = len(seq)
    rel_edges = {"seq": [], "stem": [], "rg4": []}

    # 1. 序列相邻边 + 自环
    for i in range(N):
        rel_edges["seq"].append((i, i))
    for i in range(N - 1):
        rel_edges["seq"].append((i, i + 1))
        rel_edges["seq"].append((i + 1, i))

    
    for i in range(N):
        rel_edges["stem"].append((i, i))
    stack = []
    for i, char in enumerate(ss_dotbracket[:N]):
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            j = stack.pop()
            rel_edges["stem"].append((i, j))
            rel_edges["stem"].append((j, i))

    
    for i in range(N):
        rel_edges["rg4"].append((i, i))
    rg4_weight_extra = {}

    
    pos_to_tract = {}
    tract_idx = 0
    curr_i = 0
    seq_upper = seq.upper()
    while curr_i < N:
        if seq_upper[curr_i] == "G":
            start = curr_i
            while curr_i < N and seq_upper[curr_i] == "G":
                curr_i += 1
            if (curr_i - start) >= 2:
                for pos in range(start, curr_i):
                    pos_to_tract[pos] = tract_idx
                tract_idx += 1
        else:
            curr_i += 1

    
    g_candidates = list(pos_to_tract.keys())
    for i in g_candidates:
        t_i = pos_to_tract[i]
        for j in range(max(0, i - rg4_window), min(N, i + rg4_window + 1)):
            if i != j and j in pos_to_tract:
                t_j = pos_to_tract[j]
                if abs(t_i - t_j) <= 4:  
                    rel_edges["rg4"].append((i, j))
                    rg4_weight_extra[(i, j)] = 1.5

    edge_index = {}
    edge_weight = {}
    for rel, edges in rel_edges.items():
        if len(edges) == 0:
            edge_index[rel] = np.zeros((2, 0), dtype=np.int64)
            edge_weight[rel] = np.zeros((0,), dtype=np.float32)
            continue
        ei = np.array(edges, dtype=np.int64).T  # [2, E]
        w = np.ones(ei.shape[1], dtype=np.float32)
        if rel == "rg4":
            for k in range(ei.shape[1]):
                pair = (int(ei[0, k]), int(ei[1, k]))
                if pair in rg4_weight_extra:
                    w[k] = rg4_weight_extra[pair]

        
        deg = np.zeros(N, dtype=np.float32)
        np.add.at(deg, ei[0], w)
        deg_inv_sqrt = np.zeros_like(deg)
        nz = deg > 0
        deg_inv_sqrt[nz] = np.power(deg[nz], -0.5)
        w_norm = w * deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]

        edge_index[rel] = ei
        edge_weight[rel] = w_norm.astype(np.float32)

    return edge_index, edge_weight


def load_fasta_dict(fasta_path):
    
    fasta_dict = {}
    if not os.path.exists(fasta_path):
        print(f"fasta not find: {fasta_path}")
        return fasta_dict

    try:
        from Bio import SeqIO

        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_str = str(record.seq)
            fasta_dict[record.id] = seq_str
            clean_id = record.id.split(".")[0].split()[0]
            fasta_dict[clean_id] = seq_str
    except ImportError:
        with open(fasta_path, "r") as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        full_seq = "".join(current_seq)
                        fasta_dict[current_id] = full_seq
                        fasta_dict[current_id.split(".")[0].split()[0]] = full_seq
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                full_seq = "".join(current_seq)
                fasta_dict[current_id] = full_seq
                fasta_dict[current_id.split(".")[0].split()[0]] = full_seq

    print(f"finished, count: {len(fasta_dict)}")
    return fasta_dict



def prepare_valid_balanced_dataset(input_file, output_file, feature_dir):

    feature_dir = os.path.abspath(feature_dir)
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"not find: {feature_dir}")

    
    file_map = {}
    corrupted_count = 0

    for filename in os.listdir(feature_dir):
        if filename.endswith(".npy"):
            full_path = os.path.join(feature_dir, filename)
            if os.path.getsize(full_path) == 0:
                corrupted_count += 1
                continue
            base_name = filename[:-4]
            clean_id = base_name.split(".")[0]
            file_map[base_name] = filename
            file_map[clean_id] = filename

    print(f"finished: {len(file_map)} , empty number: {corrupted_count} ")

    df = pd.read_csv(input_file, dtype={"sample_id": str})
    print(f"raw csv count: {len(df)}")

    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="sequencing"):
        sample_id = str(row["sample_id"]).strip()
        tx_id = sample_id.split("_")[0]
        clean_tx_id = tx_id.split(".")[0]
        if clean_tx_id in file_map or tx_id in file_map:
            valid_indices.append(idx)

    valid_df = df.loc[valid_indices].reset_index(drop=True)
    print(f"sequence finish: {len(valid_df)} / {len(df)}")

    if len(valid_df) == 0:
        raise RuntimeError("valid number: none")

    print("\n distribution")
    for cls_idx, count in valid_df["label"].value_counts().items():
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
        print(f"  {cls_idx} ({cls_name}): {count} ")

    min_count = valid_df["label"].value_counts().min()
    balanced_df = (
        valid_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(output_file, index=False)

    total_balanced = len(balanced_df)
    n_train = int(0.70 * total_balanced)
    n_val = int(0.15 * total_balanced)

    return balanced_df, n_train, n_val


class RNAGraphDataset(Dataset):
    def __init__(self, df, fasta_dict, feature_dir, rg4_window=12, precompute=True):
        self.df = df.reset_index(drop=True)
        self.fasta_dict = fasta_dict
        self.feature_dir = os.path.abspath(feature_dir)
        self.rg4_window = rg4_window

        self.file_map = {}
        if os.path.exists(self.feature_dir):
            for filename in os.listdir(self.feature_dir):
                if filename.endswith(".npy"):
                    full_p = os.path.join(self.feature_dir, filename)
                    if os.path.getsize(full_p) == 0:
                        continue
                    base_name = filename[:-4]
                    clean_id = base_name.split(".")[0]
                    self.file_map[base_name] = filename
                    self.file_map[clean_id] = filename

        self.cached_graphs = [None] * len(self.df)
        if precompute:
            self._precompute_all_graphs()

    def _resolve_sample(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"]).strip()
        label_val = int(row["label"])
        tx_id = sample_id.split("_")[0]
        clean_tx_id = tx_id.split(".")[0]

        parts = sample_id.split("_")
        start_pos, end_pos = 0, None
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            start_pos, end_pos = int(parts[1]), int(parts[2])

        actual_filename = self.file_map.get(tx_id) or self.file_map.get(clean_tx_id)
        if not actual_filename:
            raise FileNotFoundError(f"file not find")

        feat_file = os.path.join(self.feature_dir, actual_filename)
        raw_feat = np.load(feat_file)

        if raw_feat.ndim == 1:
            raw_feat = np.expand_dims(raw_feat, axis=-1)
        elif raw_feat.ndim > 2:
            raw_feat = np.squeeze(raw_feat)
            if raw_feat.ndim == 1:
                raw_feat = np.expand_dims(raw_feat, axis=-1)

        if end_pos:
            e_i = raw_feat[start_pos:min(end_pos, raw_feat.shape[0])]
        else:
            e_i = raw_feat[start_pos:]
        if e_i.shape[0] == 0:
            e_i = raw_feat

        L_feat = e_i.shape[0]

        raw_seq = self.fasta_dict.get(sample_id) or self.fasta_dict.get(tx_id) or self.fasta_dict.get(clean_tx_id)
        if raw_seq:
            seq = raw_seq[start_pos:end_pos] if (end_pos and len(raw_seq) >= end_pos) else raw_seq[start_pos:]
        else:
            seq = "N" * L_feat

        ss_dotbracket = str(row.get("ss_dotbracket", ""))
        if not ss_dotbracket or ss_dotbracket.lower() == "nan":
            ss_dotbracket = "." * len(seq)

        L = min(L_feat, len(seq))
        e_i = e_i[:L]
        seq = seq[:L]
        ss_dotbracket = ss_dotbracket[:L]

        if e_i.shape[1] < FEAT_DIM_TOTAL:
            pad_dim = FEAT_DIM_TOTAL - e_i.shape[1]
            features = np.hstack([e_i, np.zeros((L, pad_dim), dtype=np.float32)])
        else:
            features = e_i[:, :FEAT_DIM_TOTAL]

        mask = np.zeros(L, dtype=np.float32)
        if L >= 50:
            mask[20:min(50, L)] = 1.0
        elif L > 20:
            mask[20:L] = 1.0
        else:
            mask[:L] = 1.0

        return sample_id, features.astype(np.float32), seq, ss_dotbracket, mask, label_val

    def _precompute_all_graphs(self):
        for idx in tqdm(range(len(self.df)), desc="pre calculation"):
            sample_id, features, seq, ss_dotbracket, mask, label_val = self._resolve_sample(idx)
            edge_index, edge_weight = build_multi_relational_edges(seq, ss_dotbracket, self.rg4_window)
            self.cached_graphs[idx] = {
                "sample_id": sample_id,
                "features": features,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "mask": mask,
                "label": label_val,
                "L": features.shape[0],
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cached_graphs[idx] is None:
            sample_id, features, seq, ss_dotbracket, mask, label_val = self._resolve_sample(idx)
            edge_index, edge_weight = build_multi_relational_edges(seq, ss_dotbracket, self.rg4_window)
            self.cached_graphs[idx] = {
                "sample_id": sample_id,
                "features": features,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "mask": mask,
                "label": label_val,
                "L": features.shape[0],
            }
        return self.cached_graphs[idx]


RELATIONS = ("seq", "stem", "rg4")


def rna_graph_collate_fn(batch):
    batch_size = len(batch)
    lengths = [item["L"] for item in batch]
    total_nodes = sum(lengths)

    features = np.concatenate([item["features"] for item in batch], axis=0)  
    mask = np.concatenate([item["mask"] for item in batch], axis=0)          
    labels = np.array([item["label"] for item in batch], dtype=np.int64)    
    sample_ids = [item["sample_id"] for item in batch]

    node_batch_idx = np.repeat(np.arange(batch_size), lengths)  

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    edge_index_all = {rel: [] for rel in RELATIONS}
    edge_weight_all = {rel: [] for rel in RELATIONS}
    for i, item in enumerate(batch):
        off = offsets[i]
        for rel in RELATIONS:
            ei = item["edge_index"][rel]
            if ei.shape[1] > 0:
                edge_index_all[rel].append(ei + off)
                edge_weight_all[rel].append(item["edge_weight"][rel])

    edge_index_t = {}
    edge_weight_t = {}
    for rel in RELATIONS:
        if len(edge_index_all[rel]) > 0:
            ei_cat = np.concatenate(edge_index_all[rel], axis=1)
            ew_cat = np.concatenate(edge_weight_all[rel], axis=0)
        else:
            ei_cat = np.zeros((2, 0), dtype=np.int64)
            ew_cat = np.zeros((0,), dtype=np.float32)
        edge_index_t[rel] = torch.from_numpy(ei_cat).long()
        edge_weight_t[rel] = torch.from_numpy(ew_cat).float()

    return {
        "features": torch.from_numpy(features).float(),        
        "edge_index": edge_index_t,                             
        "edge_weight": edge_weight_t,                            
        "mask": torch.from_numpy(mask).float(),                  
        "node_batch_idx": torch.from_numpy(node_batch_idx).long(),  
        "labels": torch.from_numpy(labels).long(),               
        "sample_ids": sample_ids,
        "batch_size": batch_size,
        "total_nodes": total_nodes,
    }



def scatter_sum(src, index, num_nodes):
    """src: [E, D], index: [E] -> out: [num_nodes, D]"""
    out = src.new_zeros((num_nodes, src.shape[1]))
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_softmax(logits, index, num_nodes):
    """logits: [E] (或 [E, M])"""
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
        squeeze_back = True
    else:
        squeeze_back = False

    out = torch.zeros_like(logits)
    for m in range(logits.shape[1]):
        col = logits[:, m]
        col_max = torch.full((num_nodes,), float("-inf"), device=col.device, dtype=col.dtype)
        col_max = col_max.scatter_reduce(0, index, col, reduce="amax", include_self=True)
        col_max = torch.nan_to_num(col_max, neginf=0.0)
        shifted = col - col_max[index]
        exp = shifted.exp()
        denom = torch.zeros(num_nodes, device=col.device, dtype=col.dtype)
        denom.scatter_add_(0, index, exp)
        out[:, m] = exp / denom[index].clamp(min=1e-12)
    return out.squeeze(-1) if squeeze_back else out


class RNAMultiRelationalRGATModel(nn.Module):
    def __init__(self, in_dim=FEAT_DIM_TOTAL, hidden_dim=128, num_classes=4,
                 num_heads=8, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rgat1 = SparseRGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.rgat2 = SparseRGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, features, edge_index, edge_weight, node_batch_idx, batch_size, total_nodes):
        h = self.in_proj(features)
        h = self.rgat1(h, edge_index, edge_weight, total_nodes)
        h = self.rgat2(h, edge_index, edge_weight, total_nodes)

        lengths = torch.bincount(node_batch_idx, minlength=batch_size)
        L_max = int(lengths.max().item())
        H = h.shape[1]
        h_padded = h.new_zeros(batch_size, L_max, H)

        offsets = torch.cumsum(lengths, dim=0) - lengths
        for b in range(batch_size):
            l = int(lengths[b].item())
            start = int(offsets[b].item())
            h_padded[b, :l] = h[start:start + l]

        h_lstm, _ = self.lstm(h_padded)
        h_out = self.dropout(h_lstm)
        logits = self.classifier(h_out)
        return logits, lengths


def _pad_mask_and_targets(mask_flat, node_batch_idx, batch_size, L_max, device):
    lengths = torch.bincount(node_batch_idx, minlength=batch_size)
    offsets = torch.cumsum(lengths, dim=0) - lengths
    mask_padded = torch.zeros(batch_size, L_max, device=device)
    for b in range(batch_size):
        l = int(lengths[b].item())
        start = int(offsets[b].item())
        mask_padded[b, :l] = mask_flat[start:start + l]
    return mask_padded


def train_epoch(model, dataloader, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        features = batch["features"].to(device, non_blocking=True)
        edge_index = {r: v.to(device, non_blocking=True) for r, v in batch["edge_index"].items()}
        edge_weight = {r: v.to(device, non_blocking=True) for r, v in batch["edge_weight"].items()}
        mask = batch["mask"].to(device, non_blocking=True)
        node_batch_idx = batch["node_batch_idx"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        batch_size = batch["batch_size"]
        total_nodes = batch["total_nodes"]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
            logits, lengths = model(features, edge_index, edge_weight, node_batch_idx, batch_size, total_nodes)
            L_max = logits.shape[1]
            mask_padded = _pad_mask_and_targets(mask, node_batch_idx, batch_size, L_max, device)

            mask_exp = mask_padded.unsqueeze(-1)
            sample_logits = (logits * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)
            loss = F.cross_entropy(sample_logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs, all_sample_ids = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device, non_blocking=True)
            edge_index = {r: v.to(device, non_blocking=True) for r, v in batch["edge_index"].items()}
            edge_weight = {r: v.to(device, non_blocking=True) for r, v in batch["edge_weight"].items()}
            mask = batch["mask"].to(device, non_blocking=True)
            node_batch_idx = batch["node_batch_idx"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_size = batch["batch_size"]
            total_nodes = batch["total_nodes"]

            with torch.autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
                logits, lengths = model(features, edge_index, edge_weight, node_batch_idx, batch_size, total_nodes)
                L_max = logits.shape[1]
                mask_padded = _pad_mask_and_targets(mask, node_batch_idx, batch_size, L_max, device)

                mask_exp = mask_padded.unsqueeze(-1)
                sample_logits = (logits * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)
                loss = F.cross_entropy(sample_logits, labels)

            total_loss += loss.item()
            probs = F.softmax(sample_logits.float(), dim=-1)
            preds = sample_logits.argmax(dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_sample_ids.extend(batch["sample_ids"])

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1, all_targets, all_preds, np.array(all_probs), all_sample_ids


def main():
    print("launch (Sparse R-GAT)...")
    print(f"equipment {CONFIG['device']}")
    use_amp = CONFIG["use_amp"] and CONFIG["device"] == "cuda"
    print(f"AMP: {use_amp}")

    fasta_dict = load_fasta_dict(CONFIG["fasta_path"])

    balanced_df, n_train, n_val = prepare_valid_balanced_dataset(
        CONFIG["raw_input_csv"], CONFIG["balanced_output_csv"], CONFIG["predicted_scores_dir"],
    )

    train_df = balanced_df.iloc[:n_train].reset_index(drop=True)
    val_df = balanced_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = balanced_df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"\n training: {len(train_df)} , validation: {len(val_df)} , test: {len(test_df)}")

    train_dataset = RNAGraphDataset(train_df, fasta_dict, CONFIG["predicted_scores_dir"], CONFIG["rg4_window"])
    val_dataset = RNAGraphDataset(val_df, fasta_dict, CONFIG["predicted_scores_dir"], CONFIG["rg4_window"])
    test_dataset = RNAGraphDataset(test_df, fasta_dict, CONFIG["predicted_scores_dir"], CONFIG["rg4_window"])

    loader_kwargs = dict(
        collate_fn=rna_graph_collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )
    if CONFIG["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = CONFIG["persistent_workers"]
        loader_kwargs["prefetch_factor"] = CONFIG["prefetch_factor"]

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, **loader_kwargs)

    model = RNAMultiRelationalRGATModel(
        in_dim=FEAT_DIM_TOTAL, hidden_dim=128, num_classes=CONFIG["num_classes"],
        num_heads=CONFIG["num_attn_heads"], dropout=0.3,
    ).to(CONFIG["device"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []

    print("\n begin training model")
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, CONFIG["device"], scaler, use_amp)
        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, CONFIG["device"], use_amp)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  save best model to {CONFIG['model_save_path']} (Val Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n early stopped, validation F1: {CONFIG['patience']} Epochs no increase")
                break

    plot_loss_curve(train_losses, val_losses, CONFIG["loss_curve_path"])

    print("\n" + "=" * 50)
    print("evaluation and results")
    print("=" * 50)

    if os.path.exists(CONFIG["model_save_path"]):
        model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))
        

    test_loss, test_acc, test_f1, targets, preds, probs, sample_ids = evaluate(
        model, test_loader, CONFIG["device"], use_amp
    )

    print(f"\n Overall Accuracy: {test_acc:.4f}")
    print(f" Macro F1 Score:  {test_f1:.4f}\n")
    print(" (Classification Report):\n")
    print(classification_report(targets, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    prob_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": targets,
        "true_class": [CLASS_NAMES[t] for t in targets],
        "pred_label": preds,
        "pred_class": [CLASS_NAMES[p] for p in preds],
        "prob_G-tetrad": probs[:, 0],
        "prob_Stem-loop": probs[:, 1],
        "prob_Linear-loop": probs[:, 2],
        "prob_Unstructured": probs[:, 3],
    })
    prob_df.to_csv(CONFIG["predicted_probs_csv"], index=False)
    print(f"save to: {CONFIG['predicted_probs_csv']}")

    plot_soft_confusion_matrix(targets, probs, CLASS_NAMES, CONFIG["soft_confusion_matrix_path"])


if __name__ == "__main__":
    main()