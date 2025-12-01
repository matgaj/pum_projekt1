# Podstawy Uczenia Maszynowego - Projekt 1
Klasyfikacja instrumentów muzycznych z bazy danych [IRMAS](https://www.upf.edu/web/mtg/irmas) algorytmami klasyfikującymi z biblioteki scikit-learn.

Do projektu wykorzystano zbiór IRMAS-TrainingData, który podzielono na podzbiory treningowy i testowy, ponieważ dane z oryginalnych zbiorów treningowego i testowego pozyskane zostały na inne sposoby a zbiór TrainingData jest wystarczająco liczny. Projekt obejmuje klasyfikację dwoma algorytmami, przy dwóch sposobach reprezentacji danych oraz optymalizację hipermarametrów przy użyciu biblioteki Optuna.

Zbiór składa się z trzysekundowych fragmentów utworów muzycznych w kodowaniu PCM z częstotliwością próbkowania 44,1 kHz i głębii 16 bit. W każdym pliku wyszczególniono instrument "dominujący" w danym fragmencie.

Na zbiorze przeprowadzono ekstrakcję cech, aby nie pracować na surowych danych PCM. Przyjęte metody reprezentacji danych:
1. 30 MFCC + 1. i 2. pochodna,
2. openSMILE eGeMAPSv02.
