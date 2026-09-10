import sys
import os
import re

try:
    import RNA
except ImportError:
    sys.exit(1)

import pandas as pd
from Bio import SeqIO

CONFIG = {
    "fasta_path": "rG4_negative_HeLa.fa", 
    "output_csv": "rG4-_SL+_label_HeLa_1.csv",
    "pqs_regex": r'G{3,}.*?G{3,}.*?G{3,}.*?G{3,}', 
    "sl_mfe_threshold": -18.0,               
    "max_sl_count": 1000000,
    "min_len": 60,
    "max_len": 120
}

def generate_dynamic_sl():
    sl_data = []
    found = 0
    pqs_pattern = re.compile(CONFIG['pqs_regex'], re.IGNORECASE)

    for record in SeqIO.parse(CONFIG['fasta_path'], "fasta"):
        if found >= CONFIG['max_sl_count']: break
        tx_id = record.id.split('.')[0]
        full_seq = str(record.seq).upper().replace('T', 'U')
        L = len(full_seq)
        limit = 0 
        
        for match in pqs_pattern.finditer(full_seq):
            pqs_start, pqs_end = match.start(), match.end()

            
            fold_start = max(0, pqs_start - 10)
            fold_end = min(L, pqs_end + 10)
            candidate_region = full_seq[fold_start:fold_end]
            struct, mfe = RNA.fold(candidate_region)

            if mfe <= CONFIG['sl_mfe_threshold']:
                
                s_len = fold_end - fold_start
                if s_len < CONFIG['min_len']:
                    pad = (CONFIG['min_len'] - s_len) // 2
                    fold_start = max(0, fold_start - pad)
                    fold_end = min(L, fold_end + pad)
                elif s_len > CONFIG['max_len']:
                    fold_end = fold_start + CONFIG['max_len']

                sample_id = f"{tx_id}_{fold_start}_{fold_end}_sl"
                sl_data.append({'sample_id': sample_id, 'label': 1}) 
                found += 1
                limit += 1
                
                if found % 500 == 0: print(f"find {found} SL")
                if found >= CONFIG['max_sl_count'] or limit >= 2: break

    df = pd.DataFrame(sl_data)[['sample_id', 'label']]
    df.to_csv(CONFIG['output_csv'], index=False)
    print(f"finish, {len(df)}")

if __name__ == "__main__":
    generate_dynamic_sl()