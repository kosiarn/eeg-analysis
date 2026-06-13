## Summary
- Cel: dodać pipeline do uczenia modelu LSTM na cechach z ciągłej transformaty falkowej (CWT, `pywt.cwt`, fala `cgau4`) pod klasyfikację stanu `codes`.
- Wyjście: skrypt treningowy + nowy Dataset sekwencyjny dla danych po CWT + model LSTM (wejście spłaszczone `channels×scales`) + podstawowa ewaluacja i checkpointy.

## Current State Analysis
- Odczyt EDF + normalizacja istnieją w [preprocessing_DataReader.py](file:///home/walkowiczf/Repos/eeg-analysis/src/preprocessing_DataReader.py).
- CWT istnieje w [preprocessing_Transform.py](file:///home/walkowiczf/Repos/eeg-analysis/src/preprocessing_Transform.py) i zwraca per-kanał macierz `(n_scales, n_samples)` oraz `codes`.
- Loader dla CWT per-timestamp istnieje w [dataloader.py](file:///home/walkowiczf/Repos/eeg-analysis/src/dataloader.py) jako `WaveletTransformDataLoader`, ale nie ma wersji sekwencyjnej pod LSTM.
- Dataset sekwencyjny pod LSTM istnieje tylko dla surowych danych (DataFrame) w [model_DataLoader.py](file:///home/walkowiczf/Repos/eeg-analysis/src/model_DataLoader.py).
- Brak implementacji modelu LSTM i treningu dla danych po CWT; README sugeruje “lstm + warstwy głębokie” dla sygnału transformowanego.
- Etykiety z EDF mogą zawierać `T0..T3` (po `replace("T","")` → `0..3`), natomiast w repo zakładane są 3 stany (`NUM_STATES=3` w [data_specs.py](file:///home/walkowiczf/Repos/eeg-analysis/src/utils/data_specs.py)).

## Assumptions & Decisions (locked)
- Predykcja: klasyfikacja `codes` dla okna sekwencji (etykieta z ostatniej próbki okna).
- Wejście do LSTM: spłaszczone cechy z CWT na każdym kroku czasu: wektor `len(CHANNEL_NAMES) * n_scales`.
- `seq_len = 640` (4 sekundy przy 160 Hz).
- Skale CWT: `logspace` o liczbie skal `N=32`: `scales = np.logspace(np.log10(1), np.log10(64), 32)` (wartości w kodzie jako parametry konfiguracyjne, ale z tym domyślnym baseline).
- Etykiety: 3 klasy poprzez odrzucenie `T0` i remapowanie `{1,2,3} -> {0,1,2}` (tj. `label = code - 1` po odfiltrowaniu `code==0`).

## Proposed Changes
### 1) Dataset sekwencyjny dla CWT
- **Nowy plik**: `src/model_WaveletSequenceDataset.py` (lub analogiczna nazwa zgodna ze stylem repo).
  - Wejście: wynik `Transformer.CWTTransform(...)` (dict: kanał → `(n_scales, n_samples)`, `codes` → `(n_samples,)`).
  - Budowa cech:
    - Złożenie tensora cech w układzie czasowym: `X[t] = concat_over_channels( coef[channel][:, t] )` → kształt `(n_samples, n_channels*n_scales)`.
    - Opcjonalnie: parametr `stride` (domyślnie 1) dla gęstości okien.
  - Filtracja i mapowanie etykiet:
    - Zachować tylko indeksy gdzie `codes != 0`.
    - `y = codes - 1` tak, aby klasy były w zakresie `[0, 2]`.
  - Sekwencje:
    - Dla `i in range(0, n_samples_filtered - seq_len, stride)` tworzyć `X[i:i+seq_len]` i label `y[i+seq_len-1]`.
  - Zwracane typy: `torch.FloatTensor` dla `X` o kształcie `(seq_len, input_dim)` oraz `torch.LongTensor` dla `y`.

### 2) Model LSTM pod CWT (klasyfikacja)
- **Nowy plik**: `src/models/lstm_wavelet_classifier.py` (wprowadzić katalog `src/models/` jeśli repo nie ma jeszcze struktury na modele).
  - Architektura (baseline, “deep layers” po LSTM):
    - `nn.LSTM(input_size=input_dim, hidden_size=H, num_layers=L, batch_first=True, dropout=D)` (dropout aktywny dla `L>1`).
    - Head: `Linear(H -> H2) -> ReLU -> Dropout -> Linear(H2 -> 3)`.
    - Użycie stanu z ostatniego kroku czasu (`output[:, -1, :]`) albo `h_n[-1]` jako reprezentacji sekwencji.
  - Funkcja straty: `nn.CrossEntropyLoss()` na indeksach klas (bez one-hot).

### 3) Skrypt treningowy + podział train/val/test po pacjentach
- **Nowy plik**: `src/train_lstm_wavelet.py`
  - Konfiguracja (na górze pliku lub przez `argparse`):
    - `data_dir`, `patients`, `experiment`, `seq_len`, `stride`, `wavelet`, `scales` (domyślnie `np.logspace(np.log10(1), np.log10(64), 32)`), batch size, lr, epoki, seed.
  - Podział pacjentów:
    - Split w stylu istniejącego `fc_nets.py`: losowanie pacjentów i podział np. 70/15/15 z kontrolą seeda, aby uniknąć leakage między train/test.
  - Pipeline danych:
    - Dla każdej grupy pacjentów:
      - `PreprocessingDataReader.load(patients, experiment)` + `normalize()`.
      - `Transformer.CWTTransform(...)` na wszystkich kanałach i skalach.
      - `WaveletSequenceDataset(...)`.
    - Opakowanie w `torch.utils.data.DataLoader` z batchingiem.
  - Trening:
    - Pętla epok: `train()` + `eval()` na walidacji.
    - Metryki: accuracy (opcjonalnie macro-F1), log średniej straty.
    - Checkpointy: zapis `state_dict` modelu + optimizer + config do `./models/<run_title>/...`.

### 4) Spójność stałych i dokumentacja etykiet
- **Modyfikacja**: `src/utils/data_specs.py`
  - Pozostawić `NUM_STATES = 3`, ale dodać jawne użycie tej stałej w nowym kodzie (bez “magicznych liczb”).
- **(Opcjonalnie, jeśli potrzebne w praktyce)**: nowy helper `src/utils/labels.py`
  - Funkcja `map_codes_to_3class(codes: ArrayLike) -> (filtered_indices, mapped_labels)` żeby centralnie kontrolować mapowanie/filtr.
- **Modyfikacja**: `README.md`
  - Doprecyzować: które klasy są uczone (np. T1/T2/T3) i że T0 jest pomijane.
  - Dopisać komendę uruchomienia treningu LSTM CWT.

## Verification Steps (executor-ready)
- Uruchomić minimalny smoke test na małym wycinku:
  - Jedna osoba/pacjent, jeden eksperyment, kilka pierwszych tysięcy próbek (jeśli dane są dostępne lokalnie) i sprawdzić:
    - kształty: `X` ma `(batch, seq_len, input_dim)`; `y` ma `(batch,)` i wartości w `[0,2]`.
    - forward/backward przechodzi bez błędów.
- Uruchomić trening przez kilka epok i potwierdzić:
  - spadek loss na train,
  - sensowne metryki walidacji (nie NaN),
  - zapis checkpointów w `./models/...`.

## Out of Scope (explicit)
- Strojenie hiperparametrów, dobór pasm EEG, augmentacje, i pełny benchmarking modeli (to można dodać po działającym baseline).
