import os
import pandas as pd
import numpy as np

os.chdir('mimic-iv-2.2')

# Load your data
mimic = pd.read_csv("modelingv4.csv")
d_items = pd.read_csv("icu/d_items.csv")

print(mimic.columns)

# Build a mapping from itemid (as string) to label
itemid_to_label = d_items.set_index('itemid')['label'].to_dict()

# Rename columns in mimic where applicable
# mimic might have some non-itemid columns like "Unnamed: 0", "gender", etc., so we only rename numeric ones
new_columns = {
    col: itemid_to_label[int(col)]
    for col in mimic.columns
    if col.isdigit() and int(col) in itemid_to_label
}

# Apply renaming
mimic_renamed = mimic.rename(columns=new_columns)

print(mimic_renamed)

mimic_renamed.to_csv("mimic_iv_processed.csv")