# -*- coding: windows-1252 -*-

"""
CONSTRAINT SATISFACTION PROBLEM (CSP)
Genera un 'Piano di Allenamento' di 4 mappe ottimizzato per i PP.
Input: player_rank.
"""

import pandas as pd
import numpy as np
import joblib
import os
from constraint import Problem, AllDifferentConstraint, FunctionConstraint


DATASET_FILE = 'maps_enhanced.csv'
MODEL_FILE = 'models/best_regressor_model.pkl'


CLUSTER_MAP = {0: "HighSpeed", 1: "Warm-up", 2: "Balanced", 3: "Extreme"}

def get_estimated_profile(df, target_rank):
    """
    Dato solo il Rank, stima le statistiche medie (avg_score, avg_acc)
    guardando i giocatori nel dataset con rank simile (Vicini).
    """
    # Range di ricerca: rank +/- 10% o minimo +/- 100 posizioni
    range_window = max(100, target_rank * 0.1)
    
    similar_players = df[
        (df['player_rank'] >= target_rank - range_window) & 
        (df['player_rank'] <= target_rank + range_window)
    ]
    
    if similar_players.empty:
        # Fallback sulla media globale se il rank è fuori scala
        print("   Rank fuori dal range noto. Uso medie globali.")
        avg_score = df['avg_score_in_cluster'].mean()
        avg_acc = df['avg_acc_in_cluster'].mean()
    else:
        # Media delle statistiche dei giocatori simili
        avg_score = similar_players['avg_score_in_cluster'].mean()
        avg_acc = similar_players['avg_acc_in_cluster'].mean()
        
    return avg_score, avg_acc

def generate_predictions(df_maps, model, rank, avg_score, avg_acc):
    """
    Genera le predizioni di PP per il rank target su tutte le mappe uniche.
    """
    # Creiamo un dataset temporaneo con una riga per ogni mappa unica, Usiamo le caratteristiche della mappa e le statistiche stimate del giocatore
    unique_maps = df_maps[['bpm', 'duration', 'notes', 'map_cluster', 'songHash', 'cluster_name']].drop_duplicates(subset=['songHash'])
    
    # Preparazione input per il regressore
    
    pred_input = pd.DataFrame({
        'player_rank': rank,
        'map_cluster': unique_maps['map_cluster'],
        'avg_score_in_cluster': avg_score,
        'avg_acc_in_cluster': avg_acc
    })
    
    # Predizione
    print("   Calcolo predizioni ML per tutte le mappe candidate...")
    predicted_pp = model.predict(pred_input)
    
    unique_maps['predicted_pp'] = predicted_pp
    return unique_maps

def solve_csp_training_plan(candidates):
    """
    Configura e risolve il CSP.
    """
    print("\nAvvio Risolutore CSP (Backtracking)...")
    problem = Problem()
    
    #Vincoli
    domain_m1 = candidates[candidates['map_cluster'] == 1].index.tolist()

    domain_m4 = candidates[candidates['map_cluster'] == 3].index.tolist()

    top_general = candidates.nlargest(30, 'predicted_pp').index.tolist()
    
    # Se le liste sono vuote, impossibile risolvere
    if not domain_m1 or not domain_m4:
        print("ERRORE: Non ci sono abbastanza mappe candidate per soddisfare i vincoli di cluster.")
        return None

    # Ordiniamo i domini per PP decrescente (per trovare prima le soluzioni migliori)
    domain_m1 = sorted(domain_m1, key=lambda idx: candidates.loc[idx, 'predicted_pp'], reverse=True)[:10]
    domain_m4 = sorted(domain_m4, key=lambda idx: candidates.loc[idx, 'predicted_pp'], reverse=True)[:10]
    
    problem.addVariable("Mappa_1_Start", domain_m1)
    problem.addVariable("Mappa_2", top_general)
    problem.addVariable("Mappa_3", top_general)
    problem.addVariable("Mappa_4_End", domain_m4)
    
    #VINCOLI
    # Tutte le mappe devono essere diverse
    problem.addConstraint(AllDifferentConstraint())
    
    print("   Ricerca soluzioni...")
    solutions = problem.getSolutions()
    
    if not solutions:
        return None
        
    print(f"   Trovati {len(solutions)} piani validi. Ottimizzazione (Max PP)...")
    

    # Tra le soluzioni valide, cerchiamo quella con la somma di PP più alta
    best_plan = None
    max_pp = -1
    
    for sol in solutions:
        current_pp = 0
        current_pp += candidates.loc[sol["Mappa_1_Start"], 'predicted_pp']
        current_pp += candidates.loc[sol["Mappa_2"], 'predicted_pp']
        current_pp += candidates.loc[sol["Mappa_3"], 'predicted_pp']
        current_pp += candidates.loc[sol["Mappa_4_End"], 'predicted_pp']
        
        if current_pp > max_pp:
            max_pp = current_pp
            best_plan = sol
            
    return best_plan

def print_training_plan(plan, candidates_df, player_rank):
    """Visualizzazione formattata dell'output."""
    if not plan:
        print("\nNessun piano di allenamento trovato che soddisfi i vincoli.")
        return

    print(f"\n===================================================")
    print(f"   PIANO DI ALLENAMENTO PERSONALIZZATO (Rank {player_rank})")
    print(f"===================================================")
    
    order = ["Mappa_1_Start", "Mappa_2", "Mappa_3", "Mappa_4_End"]
    total_pp = 0
    
    for slot in order:
        idx = plan[slot]
        row = candidates_df.loc[idx]
        cluster_label = CLUSTER_MAP.get(row['map_cluster'], "Unknown")
        pp = row['predicted_pp']
        total_pp += pp
        
        print(f"[{slot}]")
        print(f"   Mappa:   {row['songHash'][:15]}... (Hash)")
        print(f"   Tipo:    {cluster_label} (Cluster {int(row['map_cluster'])})")
        print(f"   BPM:     {row['bpm']:.0f} | Note: {row['notes']:.0f}")
        print(f"   Est. PP: {pp:.2f}")
        print("-" * 30)
        
    print(f"TOTALE PP PREVISTI: {total_pp:.2f}")
    print(f"===================================================")

if __name__ == "__main__":

    if not os.path.exists(DATASET_FILE) or not os.path.exists(MODEL_FILE):
        print("Errore: Mancano i file del dataset o del modello.")
    else:
        #Input Utente
        try:
            print("--- Beat Saber PP MAXIMIZER ---")
            rank_input = int(input("Inserisci il tuo Rank attuale (es. 1500): "))
        except ValueError:
            print("Rank non valido. Uso default 1000.")
            rank_input = 1000
            
        # Caricamento
        df = pd.read_csv(DATASET_FILE)
        model = joblib.load(MODEL_FILE)
        
        # Stima feature mancanti di nuovi utenti
        est_score, est_acc = get_estimated_profile(df, rank_input)
        print(f"\nProfilo Stimato per Rank {rank_input}:")
        print(f"   Avg Score atteso: {est_score:.0f}")
        print(f"   Avg Acc attesa:   {est_acc:.2f}")
        
        # Generazione mappe candidate
        candidates_df = generate_predictions(df, model, rank_input, est_score, est_acc)
        
        # Risoluzione CSP
        best_plan = solve_csp_training_plan(candidates_df)
        
        print_training_plan(best_plan, candidates_df, rank_input)
