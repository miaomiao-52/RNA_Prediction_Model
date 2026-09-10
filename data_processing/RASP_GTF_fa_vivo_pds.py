from collections import defaultdict
from Bio import SeqIO
import numpy as np
import pandas as pd
import pyranges as pr


print("Loading FASTA...")
fasta = {}
for record in SeqIO.parse("Homo_sapiens.GRCh38.cdna.all.fa", "fasta"):
    
    tx_id = record.id.split(".")[0]
    fasta[tx_id] = record


print("Loading GTF...")
gtf = pr.read_gtf("Homo_sapiens.GRCh38.115.gtf")
exons = gtf[gtf.Feature == "exon"].df


exons["transcript_id"] = exons["transcript_id"].apply(
    lambda x: str(x).split(".")[0]
)

exons = exons.sort_values(["transcript_id", "Start"])
exons["length"] = exons["End"] - exons["Start"]


total_lens = exons.groupby("transcript_id")["length"].sum().to_dict()


exons["offset"] = (
    exons.groupby("transcript_id")["length"]
    .shift(1)
    .fillna(0)
    .astype(int)
    .groupby(exons["transcript_id"])
    .cumsum()
)


exons_to_join = exons[
    [
        "Chromosome",
        "Start",
        "End",
        "transcript_id",
        "Strand",
        "offset",
        "length",
    ]
]


print("Loading RASP data...")
rasp = pd.read_csv("RASP_vivo_PDS.csv", sep="\t")

rasp["chr"] = rasp["chr"].astype(str)
rasp = rasp.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})

exons["Chromosome"] = exons["Chromosome"].astype(str).str.replace("chr", "")
rasp["Chromosome"] = rasp["Chromosome"].astype(str).str.replace("chr", "")

print(f"RASP unique chromosomes: {rasp['Chromosome'].unique()[:5]}...")
print(f"GTF unique chromosomes: {exons['Chromosome'].unique()[:5]}...")


rasp_pr = pr.PyRanges(rasp)

exons_to_join = exons[["Chromosome", "Start", "End", "transcript_id", "Strand", "offset"]].copy()
exons_pr = pr.PyRanges(exons_to_join)


print("Running PyRanges join...")
overlap = rasp_pr.join(exons_pr)
overlap_df = overlap.df

if overlap_df.empty:
    print("!!! WARNING: overlap_df is EMPTY !!!")
    print("Please check if RASP and GTF use the same Genome Build (e.g., GRCh38).")
    
    print("RASP head:\n", rasp[['Chromosome', 'Start', 'End']].head())
    print("GTF head:\n", exons[['Chromosome', 'Start', 'End']].head())
    exit() 


tid_col = "transcript_id_b" if "transcript_id_b" in overlap_df.columns else "transcript_id"
offset_col = "offset_b" if "offset_b" in overlap_df.columns else "offset"

exon_start_col = "Start_b" 
strand_col = "Strand_b"


print("Mapping scores to transcripts...")
final_data = {}


tid_col = "transcript_id_b" if "transcript_id_b" in overlap_df.columns else "transcript_id"
offset_col = "offset_b" if "offset_b" in overlap_df.columns else "offset"
exon_start_col = "Start_b" if "Start_b" in overlap_df.columns else "Start" 
strand_col = "Strand_b" if "Strand_b" in overlap_df.columns else "Strand"

for tid, group in overlap_df.groupby(tid_col):
    if tid not in fasta:
        continue

    seq = str(fasta[tid].seq)
    scores = np.full(len(seq), np.nan)

    rasp_starts = group["Start"].values     
    exon_starts = group[exon_start_col].values 
    offsets = group[offset_col].values 
    scs = group["score"].values 
    strand = group[strand_col].iloc[0]

    
    t_positions = offsets + (rasp_starts - exon_starts)

    if strand == "-":
        total_len = total_lens[tid]
        t_positions = total_len - t_positions - 1

    t_positions = t_positions.astype(int)
    valid_mask = (t_positions >= 0) & (t_positions < len(seq))
    scores[t_positions[valid_mask]] = scs[valid_mask]

    final_data[tid] = {"sequence": seq, "scores": scores}

print(f"Successfully mapped {len(final_data)} transcripts!")


print("Filtering all-NaN transcripts...")
filtered_data = {}

for tid, content in final_data.items():
    scores = content["scores"]
    
    if not np.all(np.isnan(scores)):
        filtered_data[tid] = content

print(
    f"Filtered: {len(final_data)} -> {len(filtered_data)} transcripts retained."
)

import pickle

output_file = "mapped_scores_vivo_pds.pkl"
print(f"Saving results to {output_file}...")
with open(output_file, "wb") as f:
    pickle.dump(filtered_data, f)
print("Done!")