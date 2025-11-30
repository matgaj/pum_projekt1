import librosa
import numpy as np

DATA_ROOT = "E:\\IRMAS-TrainingData"

def extract_features_for_file(
    filepath,
    n_mfcc=13,
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

def get_label_from_path(filepath):
    return os.path.basename(os.path.dirname(filepath))

# zbierz wszystkie wav
wav_files = glob(os.path.join(DATA_ROOT, "**", "*.wav"), recursive=True)
print("Znalazłem plików:", len(wav_files))

all_rows = []
feature_names_global = None
feature_shape = None  # (n_feature_rows, n_frames)

for path in tqdm(wav_files):
    try:
        features = extract_features_for_file(path)  # (F, T)
        
        # za pierwszym razem ustalamy rozmiar i nazwy cech
        if feature_shape is None:
            feature_shape = features.shape  # np. (39, T)
            n_flat = features.size          # 39 * T
            feature_names_global = [f"f_{i}" for i in range(n_flat)]
            print("Kształt pojedynczego obiektu cech:", feature_shape)
            print("Liczba cech po spłaszczeniu:", n_flat)
        else:
            # kontrolnie sprawdzamy, czy każdy plik daje ten sam rozmiar
            if features.shape != feature_shape:
                print("Uwaga: inny rozmiar cech dla pliku:",
                      path, "->", features.shape,
                      "oczekiwano:", feature_shape)
                # możesz tu np. pominąć plik:
                continue
        
        # spłaszczenie macierzy (F, T) -> (F*T,)
        flat = features.reshape(-1)  # domyślnie 'C' (wierszami)

        label = get_label_from_path(path)
        
        row = {
            "filepath": path,
            "label": label
        }
        # cechy do słownika
        for name, val in zip(feature_names_global, flat):
            row[name] = val
        
        all_rows.append(row)
    except Exception as e:
        print("Błąd dla pliku:", path, "->", e)

# zrób DataFrame
df = pd.DataFrame(all_rows)

print("Rozmiar macierzy cech:", df.shape)

df.to_parquet("irmas_mfcc_features.parquet")
