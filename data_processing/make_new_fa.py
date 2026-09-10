import re
import pandas as pd
from Bio import SeqIO


tsv_path = "HeLa.tsv"
fasta_in = "Homo_sapiens.GRCh38.cdna.all.fa"   
fasta_out = "HeLa.fa"    
target_cell_line = "nTPM"                     

tpm_threshold = 1.0


df = pd.read_csv(tsv_path, sep="\t")

expressed_genes_set = set()

if any(str(x).startswith("ENSG") for x in df.index[:10]):
    
    expressed_genes = df[df[target_cell_line] >= tpm_threshold].index.astype(str)
    expressed_genes_set = set(gene.split('.')[0] for gene in expressed_genes)

else:
    ensg_col = None
    for col in df.columns:
        if any(str(x).startswith("ENSG") for x in df[col].dropna().head(10)):
            ensg_col = col
            break
            
    if ensg_col:
        
        expressed_genes = df[df[target_cell_line] >= tpm_threshold][ensg_col].astype(str)
        expressed_genes_set = set(gene.split('.')[0] for gene in expressed_genes)
    else:
        
        print(df.head(2))
        exit()



count_total = 0
count_kept = 0
filtered_records = []


for record in SeqIO.parse(fasta_in, "fasta"):
    count_total += 1
    header = record.description
    
    
    tx_id = record.id.split('.')[0]
    
    
    gene_id_match = None
    if "gene:" in header:
        match = re.search(r'gene:(ENSG\d+)', header)
        if match:
            gene_id_match = match.group(1)
    elif "|" in header:
        parts = header.split('|')
        for p in parts:
            if p.startswith("ENSG"):
                gene_id_match = p.split('.')[0]
                break

    
    if gene_id_match and (gene_id_match in expressed_genes_set):
        record.id = tx_id
        record.description = f"gene:{gene_id_match}|HeLa_expressed"
        filtered_records.append(record)
        count_kept += 1


SeqIO.write(filtered_records, fasta_out, "fasta")