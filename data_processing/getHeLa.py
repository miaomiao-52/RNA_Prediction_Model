import pandas as pd

df_all = pd.read_csv("rna_cellline.tsv", sep="\t")

df_hela = df_all[df_all['Cell line'] == 'HeLa'].copy()

df_expressed = df_hela[df_hela['nTPM'] >= 1.0]

df_expressed.to_csv("HeLa_real_expressed.tsv", sep="\t", index=False)
print("Saved as: HeLa_real_expressed.tsv")