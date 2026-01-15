# -*- coding: windows-1252 -*-

"""
CLUSTERING DELLE MAPPE
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os


INPUT_FILE = 'map_performance.csv'
OUTPUT_FILE = 'maps_clustered.csv'
DIR_PLOTS = 'evaluations/clustering/'


CLUSTER_NAMES_MAP = {
    0: "HighSpeed",
    1: "Warm-up",
    2: "Balanced",
    3: "Extreme"
}

# Creazione cartella per i grafici di valutazione
if not os.path.exists(DIR_PLOTS):
    os.makedirs(DIR_PLOTS)

PLOT_ELBOW = os.path.join(DIR_PLOTS, 'elbow_method.png')
PLOT_3D_SCALED = os.path.join(DIR_PLOTS, 'features_3d_scaled.png')

FEATURES_TO_CLUSTER = ['bpm', 'duration', 'notes']
MAX_CLUSTERS_TO_TEST = 10

def load_data(filepath):
    print(f"Caricamento dati da {filepath}...")
    try:
        df = pd.read_csv(filepath)
        if not all(col in df.columns for col in FEATURES_TO_CLUSTER):
            raise ValueError(f"Colonne mancanti. Richieste: {FEATURES_TO_CLUSTER}")
        return df
    except Exception as e:
        print(f"ERRORE: {e}")
        return None

def preprocess_data(df):
    """
    Pulisce e standardizza i dati.
    """
    print("Preprocessing e Scaling...")
    df_clean = df.copy()
    
    # Pulizia righe non valide
    df_clean = df_clean.dropna(subset=FEATURES_TO_CLUSTER)
    df_clean = df_clean[df_clean['duration'] > 0]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[FEATURES_TO_CLUSTER])
    
    return scaled_features, df_clean.index, scaler

def plot_3d_clusters(scaled_data, labels, filepath):
    """Genera visualizzazione 3D dei cluster per la documentazione."""
    print(f"Generazione plot 3D in {filepath}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], 
                        c=labels, cmap='viridis', alpha=0.6)
    
    ax.set_title('Visualizzazione Cluster 3D (BPM, Duration, Notes)')
    ax.set_xlabel('BPM (Scaled)')
    ax.set_ylabel('Duration (Scaled)')
    ax.set_zlabel('Notes (Scaled)')
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.savefig(filepath)
    plt.close()
    print("Plot 3D salvato.")

def find_optimal_k(scaled_data):
    """Metodo del gomito per determinare l'iperparametro K."""
    print("Ricerca K ottimale...")
    wcss = []
    k_range = range(2, MAX_CLUSTERS_TO_TEST + 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
    
    kl = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
    optimal_k = kl.elbow if kl.elbow else 4
    
    # Plot Evaluation
    print(f"Generazione plot gomito in {PLOT_ELBOW}...")
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, 'bx-')
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K: {optimal_k}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Metodo del Gomito')
    plt.legend()
    plt.savefig(PLOT_ELBOW)
    plt.close()
    print("Plot gomito salvato.")
    
    print(f"K ottimale identificato: {optimal_k}")
    return optimal_k

def analyze_cluster_profiles(df, cluster_col):
    """
    Calcola i centroidi (medie) e il conteggio elementi per ogni cluster.
    Essenziale per l'assegnazione dei nomi semantici nella documentazione.
    """
    # Calcola medie e conteggi contemporaneamente
    profiles = df.groupby(cluster_col)[FEATURES_TO_CLUSTER].agg(['mean', 'count'])
    

    counts = df.groupby(cluster_col).size().to_frame(name='count')
    means = df.groupby(cluster_col)[FEATURES_TO_CLUSTER].mean()
    
    summary = pd.concat([counts, means], axis=1)
    
    print("\n--- Analisi dei Cluster (Conteggio e Medie) ---")
    print(summary)
    return summary

if __name__ == "__main__":
    df_raw = load_data(INPUT_FILE)
    
    if df_raw is not None:
        scaled_data, valid_idx, scaler = preprocess_data(df_raw)
        
        #Plot Elbow
        k = find_optimal_k(scaled_data)
        
        #Clustering
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        
        #Integrazione risultati
        df_raw['map_cluster'] = pd.NA
        df_raw.loc[valid_idx, 'map_cluster'] = labels
        

        print("Applicazione nomi semantici ai cluster...")
        df_raw['cluster_name'] = df_raw['map_cluster'].map(CLUSTER_NAMES_MAP)
        
        #Plot 3D
        plot_3d_clusters(scaled_data, labels, PLOT_3D_SCALED)
        analyze_cluster_profiles(df_raw.loc[valid_idx], 'map_cluster')
        
        #Salvataggio dataset
        df_raw.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nFase 2 completata con successo.")
        print(f"Dataset salvato: {OUTPUT_FILE}")
        print(f"Grafici di valutazione salvati in: {DIR_PLOTS}")