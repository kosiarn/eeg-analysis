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
