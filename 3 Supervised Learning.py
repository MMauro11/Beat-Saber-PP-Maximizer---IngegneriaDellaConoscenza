# -*- coding: windows-1252 -*-

"""
SUPERVISED LEARNING
Input: Player Rank + Performance Storica (Avg Score/Acc) + Cluster.
Target: PP.

1. Preprocessing: Scaling (Numerici) + One-Hot (Categorici) + Filtri dominio (PP>0, PP<=700).
2. Valutazione: K-Fold Cross Validation su più modelli (Linear, Tree, Forest, KNN).
3. Output: Grafici comparativi e salvataggio del modello migliore (basato su RMSE).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings

from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

INPUT_FILE = 'maps_enhanced.csv'
MODEL_DIR = 'models/'
PLOT_DIR = 'evaluations/training/'

# Check cartelle
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'best_regressor_model.pkl')
OUTPUT_PLOT_METRICS = os.path.join(PLOT_DIR, 'model_comparison_metrics.png')
OUTPUT_PLOT_LEARNING_CURVES = os.path.join(PLOT_DIR, 'model_learning_curves.png')

# Definizioni Feature
FEATURES_INPUT = ['player_rank', 'map_cluster', 'avg_score_in_cluster', 'avg_acc_in_cluster']
CATEGORICAL_FEATURES = ['map_cluster']
NUMERICAL_FEATURES = ['player_rank', 'avg_score_in_cluster', 'avg_acc_in_cluster']
TARGET = 'pp'

# Parametri Validazione
N_SPLITS_K_FOLD = 10
RANDOM_STATE = 42

def load_and_prepare_data(filepath):
    """
    Carica i dati, applica i filtri di dominio e prepara X e y.
    """
    print(f"1. Caricamento dati da {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"ERRORE: File '{filepath}' non trovato. Esegui prima logic_feature_extraction.py!")
        return None, None

    df = pd.read_csv(filepath)

    # Pulizia NaN
    features_to_check = FEATURES_INPUT + [TARGET]
    initial_len = len(df)
    df.dropna(subset=features_to_check, inplace=True)
    if len(df) < initial_len:
        print(f"   Rimosse {initial_len - len(df)} righe con valori mancanti.")


    # Pulizia esempi PP=0
    df = df[(df[TARGET] > 0)]
    print(f"   Dataset filtrato (PP>0): {len(df)} campioni rimasti.")

    if df.empty:
        print("ERRORE: Il dataset è vuoto dopo i filtri.")
        return None, None

    X = df[FEATURES_INPUT]
    y = df[TARGET]
    
    return X, y

def create_preprocessor():
    """
    Crea il trasformatore per Scaling e One-Hot Encoding.
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop')
    return preprocessor

def evaluate_models(X, y, preprocessor):
    """
    Valuta diversi modelli usando K-Fold CV.
    """
    print(f"2. Avvio valutazione modelli (K-Fold={N_SPLITS_K_FOLD})...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
        "K-Neighbors (KNN)": KNeighborsRegressor()
    }
    
    cv_strategy = KFold(n_splits=N_SPLITS_K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Cross Validation con metriche multiple
        cv_scores = cross_validate(pipeline, X, y, cv=cv_strategy, 
                                   scoring={'r2': 'r2', 
                                            'mae': 'neg_mean_absolute_error',
                                            'rmse': 'neg_root_mean_squared_error'},
                                   n_jobs=-1)
        
        # Salvataggio medie
        r2_mean = np.mean(cv_scores['test_r2'])
        mae_mean = -np.mean(cv_scores['test_mae'])
        rmse_mean = -np.mean(cv_scores['test_rmse'])
        
        results[model_name] = {
            'R2': r2_mean,
            'MAE': mae_mean,
            'RMSE': rmse_mean
        }
        print(f"   {model_name}: R2={r2_mean:.4f}, RMSE={rmse_mean:.4f}")

    return pd.DataFrame(results).T, models

def plot_model_comparison(results_df, filepath):
    """Genera grafico a barre di confronto."""
    print(f"3. Generazione grafico confronto in {filepath}...")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # R2
        results_df['R2'].plot(kind='bar', ax=ax1, color='teal', alpha=0.8)
        ax1.set_title('Confronto R² (Più alto è meglio)')
        ax1.set_ylabel('R² Score')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.axhline(0, color='black', linewidth=0.8)

        # RMSE
        results_df['RMSE'].plot(kind='bar', ax=ax2, color='coral', alpha=0.8)
        ax2.set_title('Confronto Errore RMSE (Più basso è meglio)')
        ax2.set_ylabel('RMSE (PP)')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    except Exception as e:
        print(f"ERRORE Plot Confronto: {e}")

def plot_learning_curves_all_models(X, y, preprocessor, models, filepath):
    """Genera curve di apprendimento per tutti i modelli."""
    print(f"4. Generazione curve di apprendimento in {filepath}...")
    try:
        n_models = len(models)
        fig, axes = plt.subplots(n_models, 1, figsize=(8, n_models * 4))
        if n_models == 1: axes = [axes]
        
        cv_strategy = KFold(n_splits=N_SPLITS_K_FOLD, shuffle=True, random_state=RANDOM_STATE)
        train_sizes = np.linspace(0.1, 1.0, 5)

        for ax, (model_name, model) in zip(axes, models.items()):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_sizes_abs, train_scores, test_scores = learning_curve(
                    pipeline, X, y, cv=cv_strategy, n_jobs=-1,
                    train_sizes=train_sizes, scoring='neg_root_mean_squared_error',
                    random_state=RANDOM_STATE
                )
            
            train_mean = -np.mean(train_scores, axis=1)
            test_mean = -np.mean(test_scores, axis=1)
            
            ax.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training Error")
            ax.plot(train_sizes_abs, test_mean, 'o-', color="g", label="Validation Error")
            ax.set_title(f"Learning Curve: {model_name}")
            ax.set_ylabel("RMSE")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    except Exception as e:
        print(f"ERRORE Plot Learning Curves: {e}")

def train_and_save_best_model(results_df, X, y, preprocessor, models):
    """Identifica, addestra e salva il modello migliore."""
    best_model_name = results_df['RMSE'].idxmin()
    best_r2 = results_df.loc[best_model_name, 'R2']
    
    print(f"\n--- VINCITORE: {best_model_name} (RMSE: {results_df.loc[best_model_name, 'RMSE']:.2f}, R2: {best_r2:.4f}) ---")
    
    best_model = models[best_model_name]
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])
    
    print("   Riadestramento sull'intero dataset...")
    final_pipeline.fit(X, y)
    
    joblib.dump(final_pipeline, OUTPUT_MODEL_FILE)
    print(f"   Modello salvato in '{OUTPUT_MODEL_FILE}'.")

if __name__ == "__main__":
    # 1. Caricamento e Preparazione
    X, y = load_and_prepare_data(INPUT_FILE)
    
    if X is not None and y is not None:
        # 2. Preprocessing
        preprocessor = create_preprocessor()
        
        # 3. Valutazione
        results_df, models = evaluate_models(X, y, preprocessor)
        
        # 4. Grafici
        plot_model_comparison(results_df, OUTPUT_PLOT_METRICS)
        plot_learning_curves_all_models(X, y, preprocessor, models, OUTPUT_PLOT_LEARNING_CURVES)
        
        # 5. Salvataggio
        train_and_save_best_model(results_df, X, y, preprocessor, models)
        
        print("\nFase 3 Completata.")