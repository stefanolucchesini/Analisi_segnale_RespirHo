# Analisi del segnale - RespirHo'

Questa repo contiene l'algoritmo (scritto in python) per estrarre la frequenza respiratoria dal segnale raw acquisito dalle tre unità.

Consiglio di aprire l'intero progetto (cartella "Analisi_segnale_RespirHo") in un IDE come pyCharm.

Script disponibili:
* Analisi_Ste_Finestra.py contiene l'algoritmo nuovo, basato su finestra mobile.
* ANALISI_FINALE.py contiene l'algoritmo vecchio (non più utilizzabile perchè il formato dei dati acquisiti è cambiato), può essere utile.
* Quaternions_visualizer.py serve a visualizzare velocemente i quaternioni dell'intera acquisizione, è utile per capire subito se ci sono stati problemi con il magnetometro.
