import os
import pandas as pd
import numpy as np

os.chdir('mimic-iv-2.2')


admissions = pd.read_csv('hosp/admissions.csv')
chartevents = pd.read_csv('icu/chartevents.csv')
d_labitems = pd.read_csv('hosp/d_labitems.csv') 
icustays = pd.read_csv('icu/icustays.csv')
d_items = pd.read_csv('icu/d_items.csv')
labevents = pd.read_csv('hosp/labevents.csv')
patients = pd.read_csv('hosp/patients.csv')

# Save as Parquet
admissions.to_parquet('raw/admissions.parquet', engine="pyarrow", index=False)
chartevents.to_parquet('raw/chartevents.parquet', engine="pyarrow", index=False)
d_labitems.to_parquet('raw/d_labitems.parquet', engine="pyarrow", index=False) 
d_items.to_parquet('raw/d_items.parquet', engine="pyarrow", index=False) 
icustays.to_parquet('raw/icustays.parquet', engine="pyarrow", index=False)
labevents.to_parquet('raw/labevents.parquet', engine="pyarrow", index=False)
patients.to_parquet('raw/patients.parquet', engine="pyarrow", index=False)

