# iperspettrale
Funzioni base per l'analisi, gestione, e visualizzazione di dati iperspettrali e multispettrali.

## Setup
Utilizzare venv e `pip install` per i pacchetti inclusi nel file di requirements (attualmente testato su Mac OSX).

Si consiglia di mantenere i file di dati in una cartella esterna, nominata "../datasets". 

Il formato atteso è HDR, quindi ogni dataset è da mantenere in una directory con nome 'capture' contenente informazioni separate per DARKREF, WHITEREF, e i dati raw (ciascuno con i rispettivi file .hdr, .raw e .log). 
