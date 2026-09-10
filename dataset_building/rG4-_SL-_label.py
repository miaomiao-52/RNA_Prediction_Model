import pandas as pd
from Bio import SeqIO


positive_csv = "rG4-_SL+_label_HeLa_1.csv"
fasta_file = "rG4_negative_HeLa.fa"
output_csv = "rG4-_SL-_label_HeLa_1.csv"


MIN_LEN = 60
MAX_LEN = 120


pos_df = pd.read_csv(positive_csv)

positive_ids = set(pos_df["sample_id"].tolist())

print(f"count: {len(positive_ids)}")


negative_samples = []


for record in SeqIO.parse(fasta_file, "fasta"):

    transcript_id = record.id.split(".")[0]

    seq = str(record.seq)
    seq_len = len(seq)

    
    occupied = []

    
    for window_size in range(MAX_LEN, MIN_LEN - 1, -1):

        start = 0

        while start + window_size <= seq_len:

            end = start + window_size

            sample_id = f"{transcript_id}_{start}_{end}_sl"

            
            if sample_id in positive_ids:
                start += 1
                continue

            
            overlap = False

            for s, e in occupied:

                if not (end <= s or start >= e):
                    overlap = True
                    break

            
            if not overlap:

                negative_samples.append([sample_id, 0])

                occupied.append((start, end))

                start = end

            else:
                start += 1


neg_df = pd.DataFrame(
    negative_samples,
    columns=["sample_id", "label"]
)

neg_df.to_csv(output_csv, index=False)

print(f"saved: {output_csv}")
print(f"count: {len(neg_df)}")