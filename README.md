# Analisi del segnale - RespirHo'

Questa repo contiene l'algoritmo (scritto in python) per estrarre la frequenza respiratoria dal segnale raw acquisito dalle tre unità.

Consiglio di aprire l'intero progetto (cartella "Analisi_segnale_RespirHo") in un IDE come pyCharm.

Script disponibili:
* Analisi_Ste_Finestra_FAST.py contiene l'algoritmo utilizzato, basato su finestra mobile. La differenza con il file Analisi_Ste_Finestra.py è che la trasformazione dei quaternioni nell'intervallo [-1,1] e l'interpolazione vengono fatte sul dataset globale e non campione per campione. Ciò permette di risparmiare una enorme quantità di tempo. L'inconveniente è che viene fatto prima di entrare nel while(1) e presuppone di avere già tutti i dati disponibili, quindi non si può utilizzare per mostrare dati in tempo reale.
* Analisi_Ste_Finestra.py contiene l'algoritmo nuovo, basato su finestra mobile, pensato per mostrare dati in tempo reale, tutte le operazioni sono svolte in un ciclo infinito.
* ANALISI_FINALE.py contiene l'algoritmo vecchio (non più utilizzabile perchè il formato dei dati acquisiti è cambiato), ma può essere utile.
* Quaternions_visualizer.py serve a visualizzare velocemente i quaternioni dell'intera acquisizione e la tensione delle batterie, è utile per capire subito se ci sono stati problemi con il magnetometro ed evitare di perdere tempo a fare un'analisi su dati corrotti.
