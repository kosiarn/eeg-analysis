# Eksploracja danych

Transformacja falką `cgau4`

1. preprocessing (i labelowanie stanów)
	1. odczytanie fali EEG z pliku EDF (ozn *oryginalny sygnał*) F
	2. ==przeprowadzenie wavelet transform falką `cgau4` (ozn. *transformowany sygnał*)== K
2. zbudowanie modeli
	1. na oryginalny sygnał
		- LSTM F
		- Naive bayes? jak będzie mało treści
		- FC sieć neuronowa F
	2. na transformowany sygnał
		- ==lstm + warstwy głębokie== K
		- ==FC sieć neuronowa== K
		- konwolucyjna sieć neuronowa? F
3. trening + testy
4. wnioski wyniki

## Trening: LSTM na CWT (cgau4)
- Skrypt: `src/train_lstm_wavelet.py`
- Wejście: cechy z CWT spłaszczone do wektora `channels×scales` na każdym kroku czasu, następnie LSTM + head MLP do klasyfikacji.
- Etykiety: pomijane jest `T0`, a `T1/T2/T3` są mapowane do klas `0/1/2`.

Przykład uruchomienia (z katalogu repo):

```bash
python src/train_lstm_wavelet.py --data-dir ./data/physionet.org/files/eegmmidb/1.0.0 --experiment 5
```

Najważniejsze parametry:
- `--seq-len` (domyślnie 640 = 4 s przy 160 Hz)
- `--n-scales`, `--min-scale`, `--max-scale` (domyślnie `logspace(1..64, N=32)`)
- `--batch-size`, `--lr`, `--epochs`
