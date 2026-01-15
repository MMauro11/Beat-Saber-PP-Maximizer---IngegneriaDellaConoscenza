# -*- coding: windows-1252 -*-

"""
LOGIC REPRESENTATION

Costruisce una Knowledge Base Prolog (.pl).
Usa Prolog via PySWIP per calcolare nuove features.
Aggiunge al dataset CSV avg_accuracy e avg_score.
"""

import pandas as pd
import os
import sys


try:
    from pyswip import Prolog
except ImportError:
    print("ERRORE: Libreria 'pyswip' non trovata. Installala con 'pip install pyswip'.")
    sys.exit(1)
except Exception as e:
    print(f"ERRORE CRITICO PySWIP: {e}")
    print("Assicurati di aver installato SWI-Prolog e aggiunto la cartella 'bin' al PATH di sistema.")
    print("Su Windows solitamente è: C:\\Program Files\\swipl\\bin")
    sys.exit(1)

INPUT_FILE = 'maps_clustered.csv'
OUTPUT_CSV = 'maps_enhanced.csv'
OUTPUT_PROLOG = 'beat_saber_logic.pl'

def clean_atom(text):
    """Pulisce una stringa per renderla un atomo Prolog valido."""
    return str(text).lower().replace(' ', '_').replace('/', '_').replace('-', '_')

def generate_prolog_file(df, filepath):
    """
    Crea il file Prolog (.pl) con Fatti e Regole.
    """
    print(f"1. Generazione Knowledge Base Prolog in {filepath}...")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("% --- BEAT SABER KNOWLEDGE BASE ---\n")
        f.write(":- dynamic played/4.\n\n")
        
        # Scrittura dei Fatti
        unique_entries = df[['player_id', 'cluster_name', 'base_score', 'accuracy']].dropna()
        
        # Buffer per scrittura veloce
        facts = []
        for _, row in unique_entries.iterrows():
            p_id = f"'{str(row['player_id'])}'"
            c_name = clean_atom(row['cluster_name'])
            score = int(row['base_score'])
            acc = round(float(row['accuracy']), 4) # 4 decimali di precisione
            
            facts.append(f"played({p_id}, {c_name}, {score}, {acc}).")
        
        f.write('\n'.join(facts))
        
        f.write("\n\n% --- REGOLE DI INFERENZA ---\n")
        
        # Score Medio
        f.write("average_score_cluster(Player, Cluster, AvgScore) :-\n")
        f.write("    findall(Score, played(Player, Cluster, Score, _), ScoreList),\n")
        f.write("    length(ScoreList, Count),\n")
        f.write("    Count > 0,\n")
        f.write("    sum_list(ScoreList, Sum),\n")
        f.write("    AvgScore is Sum / Count.\n\n")

        # Accuracy Media
        f.write("average_accuracy_cluster(Player, Cluster, AvgAcc) :-\n")
        f.write("    findall(Acc, played(Player, Cluster, _, Acc), AccList),\n")
        f.write("    length(AccList, Count),\n")
        f.write("    Count > 0,\n")
        f.write("    sum_list(AccList, Sum),\n")
        f.write("    AvgAcc is Sum / Count.\n")
        
    print(f"   Fatti scritti: {len(facts)}")

def infer_features_with_pyswip(df, prolog_file):
    """
    Interroga Prolog per calcolare le medie.
    """
    print("2. Avvio motore inferenziale Prolog (PySWIP)...")
    prolog = Prolog()
    
    # Caricamento del file KB. Nota: consult richiede il path con slash in stile unix o raw string
    # Su Windows i backslash possono dare problemi, li convertiamo.
    prolog_path = os.path.abspath(prolog_file).replace('\\', '/')
    print(f"   Consulting: {prolog_path}")
    prolog.consult(prolog_path)
    
    print("   Esecuzione query per feature extraction (questo potrebbe richiedere tempo)...")
    
    # Invece di fare una query per ogni riga troviamo le coppie uniche (Player, Cluster) e facciamo caching dei risultati.
    unique_pairs = df[['player_id', 'cluster_name']].drop_duplicates().dropna()
    
    # Dizionario cache: (player_id, cluster_name) -> (avg_score, avg_acc)
    inference_cache = {}
    
    total = len(unique_pairs)
    for i, (_, row) in enumerate(unique_pairs.iterrows()):
        if i % 100 == 0:
            print(f"   Progresso: {i}/{total} coppie elaborate...", end='\r')
            
        p_id_raw = str(row['player_id'])
        c_name_raw = row['cluster_name']
        
        # Preparazione atomi per la query
        p_id_atom = f"'{p_id_raw}'"
        c_name_atom = clean_atom(c_name_raw)
        
        # Query 1: Score Medio
        # Sintassi PySWIP: list(prolog.query("predicato(X, Y)")) restituisce una lista di dizionari
        q_score = list(prolog.query(f"average_score_cluster({p_id_atom}, {c_name_atom}, X)"))
        avg_score = q_score[0]['X'] if q_score else 0
        
        # Query 2: Accuracy Media
        q_acc = list(prolog.query(f"average_accuracy_cluster({p_id_atom}, {c_name_atom}, X)"))
        avg_acc = q_acc[0]['X'] if q_acc else 0
        
        inference_cache[(p_id_raw, c_name_raw)] = (avg_score, avg_acc)
        
    print(f"\n   Inferenza completata. Applicazione al DataFrame...")
    
    # Mapping dei risultati nel DataFrame originale
    # Usiamo una funzione lambda per cercare nel dizionario cache
    def get_inferred_values(row):
        key = (str(row['player_id']), row['cluster_name'])
        if key in inference_cache:
            return pd.Series(inference_cache[key])
        return pd.Series([0, 0])

    df[['avg_score_in_cluster', 'avg_acc_in_cluster']] = df.apply(get_inferred_values, axis=1)
    
    return df

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        
        # Genera il file .pl
        generate_prolog_file(df, OUTPUT_PROLOG)

        try:
            df_enhanced = infer_features_with_pyswip(df, OUTPUT_PROLOG)
            
            # Salvataggio
            df_enhanced.to_csv(OUTPUT_CSV, index=False)
            print(f"\nSUCCESSO: Dataset arricchito salvato in: {OUTPUT_CSV}")
            
        except Exception as e:
            print(f"\nERRORE durante l'inferenza Prolog: {e}")
            print("Suggerimento: Verifica l'installazione di SWI-Prolog.")
        
    else:
        print(f"Errore: File {INPUT_FILE} non trovato.")