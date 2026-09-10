import pandas as pd


g4 = pd.read_csv("G4Atlas_positive.csv")
sl = pd.read_csv("rG4+_SL+_label_HeLa_1.csv")


parsed = sl["sample_id"].str.extract(
    r'^(ENST\d+)_[A-Za-z0-9_]+?_(\d+)_(\d+)_\d+_\d+_sl$'
)

parsed.columns = ["mRNA_id", "sl_start", "sl_end"]

parsed["sl_start"] = parsed["sl_start"].astype(int)
parsed["sl_end"] = parsed["sl_end"].astype(int)


g4["row_id"] = range(len(g4))


merged = g4.merge(parsed, on="mRNA_id", how="left")


overlap = (
    (merged["start"] <= merged["sl_end"]) &
    (merged["end"] >= merged["sl_start"])
)


positive_ids = set(
    merged.loc[overlap, "row_id"]
)


sl_plus_final = g4[
    g4["row_id"].isin(positive_ids)
].copy()

sl_minus_final = g4[
    ~g4["row_id"].isin(positive_ids)
].copy()


for df in [sl_plus_final, sl_minus_final]:

    df["sample_id"] = (
        df["mRNA_id"].astype(str)
        + "_"
        + df["start"].astype(str)
        + "_"
        + df["end"].astype(str)
        + "_sl"
    )


sl_plus_final = sl_plus_final[["sample_id"]]
sl_minus_final = sl_minus_final[["sample_id"]]

sl_plus_final["label"] = 1
sl_minus_final["label"] = 2


sl_plus_final.to_csv(
    "rG4+_SL+_label_HeLa_1clean.csv",
    index=False
)

sl_minus_final.to_csv(
    "rG4+_SL-_label_Hela_1.csv",
    index=False
)

print("G4Atlas:", len(g4))
print("SL+:", len(sl_plus_final))
print("SL-:", len(sl_minus_final))
print("SUM:", len(sl_plus_final) + len(sl_minus_final))