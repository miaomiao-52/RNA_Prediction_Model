import os
import pandas as pd

file_list = [
    'rG4+_SL+_label_HeLa_final.csv', 
    'rG4+_SL-_label_HeLa.csv', 
    'rG4-_SL+_label_HeLa.csv', 
    'rG4-_SL-_label_HeLa.csv'
]

output_file = 'RNA_whole_label_HeLa.csv'

dfs = []
for file_name in file_list:
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        dfs.append(df)
        print(f"succeed: {file_name}，count: {len(df)}")
    else:
        print(f"warning {file_name}, please check")


if dfs:
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    
    combined_df.to_csv(output_file, index=False)
    print(f"\n finish")
    print(f"total: {len(combined_df)}")
    print(f"file name: {output_file}")
else:
    print("\n defeat")