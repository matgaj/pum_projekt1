import librosa
import numpy as np

DATA_ROOT = "/Volumes/T7 exFAT/IRMAS-TrainingData"

def extract_features_for_file(
    filepath,
    n_mfcc=30,
    n_fft=2048,
    hop_length=512
):
    y, sr = librosa.load(filepath, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return features

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

# get instrument label from the subfolder name where the file was (eg. "cel")
def get_label_from_path(filepath):
    return os.path.basename(os.path.dirname(filepath))

# recursive search for all .wav files in the DATA_ROOT
wav_files = glob(os.path.join(DATA_ROOT, "**", "*.wav"), recursive=True)
print("Files found:", len(wav_files))

all_rows = []
feature_names_global = None
feature_shape = None  # (n_feature_rows, n_frames)

for path in tqdm(wav_files):
    try:
        features = extract_features_for_file(path)
        
        # find out the shape of features
        if feature_shape is None:
            feature_shape = features.shape  # np. (39, T)
            n_flat = features.size          # 39 * T
            feature_names_global = [f"f_{i}" for i in range(n_flat)]
            print("Single object shape:", feature_shape)
            print("Flattened object shape:", n_flat)
        else:
            # check if file gave the same num of features
            if features.shape != feature_shape:
                print("Warning! File gave unexpected number of features:",
                      path, "->", features.shape,
                      "expected:", feature_shape)
                # skip file
                continue
        
        flat = features.reshape(-1)

        label = get_label_from_path(path)
        
        row = {
            "filepath": path,
            "label": label
        }
        for name, val in zip(feature_names_global, flat): # type: ignore
            row[name] = val
        
        all_rows.append(row)
    except Exception as e:
        print("Error at file:", path, "->", e)

# zrób DataFrame
df = pd.DataFrame(all_rows)

print("Final feature array size:", df.shape)

df.to_parquet("irmas_mfcc_features.parquet",compression="zstd")
