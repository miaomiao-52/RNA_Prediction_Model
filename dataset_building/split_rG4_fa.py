import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_data(csv_path, npy_path, fasta_path):
    
    
    df_pos = pd.read_csv(csv_path)
    
    
    transcript_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        clean_id = record.id.split('.')[0]
        transcript_dict[clean_id] = str(record.seq).upper()
        
    
    reactivity_data = np.load(npy_path, allow_pickle=True)
    
    reactivity_dict = {}
    
    
    if isinstance(reactivity_data, np.ndarray) and reactivity_data.ndim == 0:
        reactivity_dict = reactivity_data.item()
        
        
    
    elif isinstance(reactivity_data, np.ndarray) and reactivity_data.ndim == 1 and len(reactivity_data) > 0 and isinstance(reactivity_data[0], dict):
        reactivity_dict = reactivity_data[0]
        
        
    
    else:
        shape_info = getattr(reactivity_data, 'shape', len(reactivity_data))
        
        
        fasta_ids = list(transcript_dict.keys())
        
        for i, tx_id in enumerate(fasta_ids):
            if i >= len(reactivity_data):
                break
            actual_len = len(transcript_dict[tx_id])
            
            reactivity_dict[tx_id] = reactivity_data[i][:actual_len]
            
    return df_pos, transcript_dict, reactivity_dict

def extract_features(seq, reactivity_sub):
    
    length = len(seq)
    if length == 0:
        return [0] * 10
    
    
    g_count = seq.count('G')
    c_count = seq.count('C')
    g_content = g_count / length
    gc_content = (g_count + c_count) / length
    
    
    g2_runs = seq.count('GG')
    g3_runs = seq.count('GGG')
    
    
    if len(reactivity_sub) > 0:
        
        reactivity_sub = np.nan_to_num(reactivity_sub, nan=0.0)
        r_mean = np.mean(reactivity_sub)
        r_std = np.std(reactivity_sub)
        r_max = np.max(reactivity_sub)
        r_min = np.min(reactivity_sub)
        
        r_diff = np.mean(np.abs(np.diff(reactivity_sub))) if len(reactivity_sub) > 1 else 0
    else:
        r_mean, r_std, r_max, r_min, r_diff = 0, 0, 0, 0, 0
        
    features = [
        g_content, gc_content, g2_runs, g3_runs, length,
        r_mean, r_std, r_max, r_min, r_diff
    ]
    return features

def build_dataset(df_pos, transcript_dict, reactivity_dict, window_size=35):
    
    X = []
    y = []
    positive_intervals = {}
    
    
    for idx, row in df_pos.iterrows():
        tx_id = str(row['mRNA_id']).split('.')[0]
        start, end = int(row['start']), int(row['end'])
        
        if tx_id not in transcript_dict:
            continue
            
        full_seq = transcript_dict[tx_id]
        full_react = reactivity_dict.get(tx_id, np.zeros(len(full_seq)))
        
        sub_seq = full_seq[start:end]
        sub_react = full_react[start:end]
        
        if len(sub_seq) < 10:  
            continue
            
        feats = extract_features(sub_seq, sub_react)
        X.append(feats)
        y.append(1)
        
        if tx_id not in positive_intervals:
            positive_intervals[tx_id] = []
        positive_intervals[tx_id].append((start, end))

    
    target_neg_count = len(X)
    neg_count = 0
    
    for tx_id, full_seq in transcript_dict.items():
        if neg_count >= target_neg_count:
            break
        
        full_react = reactivity_dict.get(tx_id, np.zeros(len(full_seq)))
        intervals = positive_intervals.get(tx_id, [])
        
        for i in range(0, len(full_seq) - window_size, window_size * 2):
            w_start = i
            w_end = i + window_size
            
            
            overlap = False
            for p_start, p_end in intervals:
                if not (w_end <= p_start or w_start >= p_end):
                    overlap = True
                    break
            
            if not overlap:
                sub_seq = full_seq[w_start:w_end]
                sub_react = full_react[w_start:w_end]
                
                if sub_seq.count('N') > window_size * 0.2:  
                    continue
                    
                feats = extract_features(sub_seq, sub_react)
                X.append(feats)
                y.append(0)
                neg_count += 1
                if neg_count >= target_neg_count:
                    break

    return np.array(X), np.array(y)

def pipeline():
    csv_path = 'G4Atlas_positive.csv'
    npy_path = 'reactivity_matrix_HeLa.npy'
    fasta_path = 'HeLa.fa'
    
    if not (os.path.exists(csv_path) and os.path.exists(npy_path) and os.path.exists(fasta_path)):
        print("warning")
        return
        
    df_pos, transcript_dict, reactivity_dict = load_data(csv_path, npy_path, fasta_path)
    

    X, y = build_dataset(df_pos, transcript_dict, reactivity_dict)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    
    
    
    window_size = 35
    step_size = 15
    threshold = 0.50  
    
    pos_potential_records = []
    no_potential_records = []
    
    for tx_id, full_seq in transcript_dict.items():
        seq_len = len(full_seq)
        if seq_len < window_size:
            continue
            
        full_react = reactivity_dict.get(tx_id, np.zeros(seq_len))
        
        
        is_g4 = np.zeros(seq_len, dtype=bool)
        
        
        tx_pos_df = df_pos[df_pos['mRNA_id'].str.split('.').str[0] == tx_id]
        for _, row in tx_pos_df.iterrows():
            start, end = int(row['start']), int(row['end'])
            is_g4[start:end] = True
            
        
        windows_feats = []
        windows_coords = []
        for start in range(0, seq_len - window_size + 1, step_size):
            end = start + window_size
            sub_seq = full_seq[start:end]
            sub_react = full_react[start:end]
            windows_feats.append(extract_features(sub_seq, sub_react))
            windows_coords.append((start, end))
            
        if len(windows_feats) > 0:
            prob_set = model.predict_proba(windows_feats)[:, 1]
            for idx, prob in enumerate(prob_set):
                if prob >= threshold:
                    w_start, w_end = windows_coords[idx]
                    is_g4[w_start:w_end] = True
                    
        
        def save_segments(mask, value, record_list, prefix):
            state = False
            start_idx = 0
            for i in range(len(mask)):
                if mask[i] == value and not state:
                    start_idx = i
                    state = True
                elif mask[i] != value and state:
                    seg_seq = full_seq[start_idx:i]
                    if len(seg_seq) >= 15: 
                        rec = SeqRecord(Seq(seg_seq), id=f"{tx_id}_{prefix}_{start_idx}_{i}", description=f"coords={start_idx}-{i}")
                        record_list.append(rec)
                    state = False
            if state:
                seg_seq = full_seq[start_idx:len(mask)]
                if len(seg_seq) >= 15:
                    rec = SeqRecord(Seq(seg_seq), id=f"{tx_id}_{prefix}_{start_idx}_{len(mask)}", description=f"coords={start_idx}-{len(mask)}")
                    record_list.append(rec)

        save_segments(is_g4, True, pos_potential_records, "rG4_positive_potential")
        save_segments(is_g4, False, no_potential_records, "rG4_negative")

    SeqIO.write(pos_potential_records, "rG4_positive_and_potential.fa", "fasta")
    SeqIO.write(no_potential_records, "rG4_negative.fa", "fasta")
    
    print(f"\n finished: ")
    print(f" 1. -> rG4_positive_and_potential_HeLa.fa")
    print(f" 2. -> rG4_negative_HeLa.fa")

if __name__ == "__main__":
    pipeline()