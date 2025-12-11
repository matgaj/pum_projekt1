# Podstawy Uczenia Maszynowego <br> Projekt 1 - Klasyfikacja danych audio 

## Struktura plików
W katalogu głównym znajdują się dwa foldery `mfcc` i `openSMILE`, które zawierają kod pracujący na cechach wskazanych w nazwie folderu. 

Pliki `*_feature_exctraction.py` zawierają kod wykorzystany do ekstrakcji cech i zapisu ich do plików `*.parquet`. W celu ich uruchomienia należy pobrać plik `IRMAS-TrainingData.zip` z [tej strony](https://zenodo.org/records/1290750#.WzCwSRyxXMU), wypakować go do folderu i podać jego ścieżkę jako wartość stałej `DATA_ROOT`.

Pliki `classification_*.ipynb` zawierają kod wykorzystany do całego procesu klasyfikacji, od podziału danych na podzbiory, po analizę macierzy pomyłek.

## Projekt
Klasyfikacja instrumentów muzycznych z bazy danych [IRMAS](https://www.upf.edu/web/mtg/irmas) [1] algorytmami klasyfikującymi z biblioteki scikit-learn.

Do projektu wykorzystano zbiór IRMAS-TrainingData, który podzielono na podzbiory treningowy i testowy, ponieważ dane z oryginalnych zbiorów treningowego i testowego pozyskane zostały na inne sposoby a zbiór TrainingData jest wystarczająco liczny. Projekt obejmuje klasyfikację dwoma algorytmami, przy dwóch sposobach reprezentacji danych oraz optymalizację hipermarametrów przy użyciu biblioteki Optuna.

Zbiór składa się z trzysekundowych fragmentów utworów muzycznych w kodowaniu PCM z częstotliwością próbkowania 44,1 kHz i głębii 16 bit. W każdym pliku wyszczególniono instrument "dominujący" w danym fragmencie.

Na zbiorze przeprowadzono ekstrakcję cech, aby nie pracować na surowych danych PCM. Przyjęte metody reprezentacji danych:
1. 30 MFCC + 1. i 2. pochodna,
2. openSMILE eGeMAPSv02 [2].

Oba zbiory cech podzielono na podzbiory treningowy i testowy w proporcjach 1:4. Wszystkie podzbiory ustandaryzowano, a na MFCC dodatkowo zastosowano PCA z zachowaniem 95% wariancji.

Zastosowano dwa algorytmy klasyfikacji:
1. RandomForestClassifier
    - szybkie działanie przy dużej liczbie elementów i cech
2. SVC
    - dobra klasyfikacja trudnych danych

Dla obu algorytmów optymalizowano hiperparametry (tylko dla danych openSMILE)
 - RandomForestClassifier z użyciem optuny
 - SVC z użyciem GridSearchSV

## Bibliografia
 [1] Juan J. Bosch, Ferdinand Fuhrmann, & Perfecto Herrera. (2014). IRMAS: a dataset for instrument recognition in musical audio signals (1.0) [Data set]. 13th International Society for Music Information Retrieval Conference (ISMIR 2012), Porto, Portugal. Zenodo. https://doi.org/10.5281/zenodo.1290750

 [2] F. Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing," in IEEE Transactions on Affective Computing, vol. 7, no. 2, pp. 190-202, 1 April-June 2016. https://doi.org/10.1109/TAFFC.2015.2457417
