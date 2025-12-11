import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import opensmile

DATA_ROOT = "/Volumes/T7 exFAT/IRMAS-TrainingData"

# 1) konfiguracja openSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_opensmile_for_file(filepath: str):
    df_feat = smile.process_file(filepath)  # 1 wiersz, kolumny = cechy
    feature_names = df_feat.columns.tolist()
    feature_vector = df_feat.iloc[0].to_numpy(dtype=float)
    return feature_names, feature_vector

# get instrument label from the subfolder name where the file was (eg. "cel")
def get_label_from_path(filepath):
    return os.path.basename(os.path.dirname(filepath))

# recursive search for all .wav files in the DATA_ROOT
wav_files = glob(os.path.join(DATA_ROOT, "**", "*.wav"), recursive=True)
print("Files found:", len(wav_files))

all_rows = []
feature_names_global = None

for path in tqdm(wav_files):
    try:
        feat_names, feat_vec = extract_opensmile_for_file(path)

        # zapamiętaj listę cech z pierwszego pliku
        if feature_names_global is None:
            feature_names_global = feat_names
            print("Number of openSMILE features:", len(feature_names_global))
        else:
            # kontrolnie: czy nazwy cech są identyczne
            if feat_names != feature_names_global:
                print("Warning: feature names mismatch for file:", path)
                # tu możesz np. znormalizować, dopasować lub pominąć plik
                continue

        label = get_label_from_path(path)

        row = {
            "filepath": path,
            "label": label,
        }

        # cechy do słownika
        for name, val in zip(feature_names_global, feat_vec):
            row[name] = float(val)

        all_rows.append(row)

    except Exception as e:
        print("Error at file:", path, "->", e)

# zrób DataFrame
df = pd.DataFrame(all_rows)

print("Final openSMILE feature array size:", df.shape)

df.to_parquet("irmas_opensmile_egemaps.parquet", compression="zstd")

