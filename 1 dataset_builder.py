# -*- coding: windows-1252 -*-

NUM_SAMPLES = 700
TOTAL_PLAYERS = 35000
PLAYERS_PER_SAMPLED_PAGE = 5 # Quanti giocatori prendere da ogni pagina campionata

import requests
import pandas as pd
import time
import math
import logging
import os
from typing import List, Dict, Any, Optional


# Imposta un logging di base per visualizzare l'avanzamento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SCORESABER_API_URL = "https://scoresaber.com/api"
BEATSAVER_API_URL = "https://api.beatsaver.com"


class RateLimiter:
    """
    Una classe semplice per gestire il rate limiting delle richieste API.
    Attende il tempo necessario tra una chiamata e l'altra per rispettare il limite.
    """
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.delay_between_requests = 60.0 / requests_per_minute
        self.last_request_time = 0
        logging.info(f"RateLimiter inizializzato a {requests_per_minute} req/min (delay: {self.delay_between_requests:.2f}s)")

    def wait(self):
        """Attende il tempo necessario prima di permettere la prossima richiesta."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.delay_between_requests:
            sleep_time = self.delay_between_requests - elapsed
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()

# Valori per Rate Limiter
# ScoreSaber: 400 richieste/minuto al massimo
scoresaber_limiter = RateLimiter(400)
# BeatSaver: Non viene specificato
beatsaver_limiter = RateLimiter(120)



def get_sampled_player_ids() -> List[Dict[str, Any]]:
    """
    Campiona i giocatori in modo stratificato lungo tutto il range del rank.
    Interroga le pagine dei giocatori a intervalli regolari e prende N giocatori
    da ciascuna pagina campionata.
    
    Args:
        total_players: Il numero totale approssimativo di giocatori (per calcolare le pagine).
        num_samples: Il numero di giocatori che si desidera campionare.
        
    Returns:
        Una lista di dizionari, ognuno contenente 'id' e 'rank' del giocatore.
    """
    logging.info(f"Inizio campionamento di {NUM_SAMPLES} giocatori su ~{TOTAL_PLAYERS} totali...")
    logging.info(f"Strategia: {PLAYERS_PER_SAMPLED_PAGE} giocatori per pagina campionata.")
    
    sampled_players: List[Dict[str, Any]] = []
    players_per_page = 50
    total_pages = math.ceil(TOTAL_PLAYERS / players_per_page)
    
    # Calcola quante pagine dobbiamo interrogare
    num_pages_to_sample = math.ceil(NUM_SAMPLES / PLAYERS_PER_SAMPLED_PAGE) # Es: 400 / 5 = 80 pagine
    
    # Calcola l'intervallo tra le pagine Es: 700 pagine totali / 80 pagine da campionare = step 8
    page_step = total_pages // num_pages_to_sample
    if page_step == 0:
        page_step = 1

    sampled_pages = range(1, total_pages + 1, page_step) # Es: 1, 9, 17...

    for page in sampled_pages:
        if len(sampled_players) >= NUM_SAMPLES:
            break
            
        url = f"{SCORESABER_API_URL}/players?page={page}"
        logging.info(f"Querying player page: {page}/{total_pages} (Campionati finora: {len(sampled_players)})")
        
        try:
            scoresaber_limiter.wait()
            response = requests.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            
            data = response.json()
            players_on_page = data.get('players', [])
            
            if players_on_page:
                # Prendi N giocatori dalla pagina
                players_taken_from_this_page = 0
                for i in range(PLAYERS_PER_SAMPLED_PAGE):
                    
                    if i >= len(players_on_page):
                        
                        logging.warning(f"Pagina {page} ha solo {len(players_on_page)} giocatori.")
                        break
                    
                    if len(sampled_players) >= NUM_SAMPLES:
                        break

                    player_info = players_on_page[i]
                    player_id = player_info.get('id')
                    player_rank = player_info.get('rank')
                    
                    if player_id and player_rank is not None:
                        sampled_players.append({'id': player_id, 'rank': player_rank})
                        players_taken_from_this_page += 1
                    else:
                        logging.warning(f"Dati 'id' o 'rank' mancanti per il giocatore {i} a pagina {page}")
                
                logging.info(f"Pagina {page}: presi {players_taken_from_this_page} giocatori.")
                
            else:
                logging.warning(f"Nessun giocatore ('players') trovato a pagina {page}.")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Errore durante la richiesta alla pagina {page}: {e}")
        except Exception as e:
            logging.error(f"Errore inatteso durante l'elaborazione della pagina {page}: {e}")

    logging.info(f"Campionamento completato. Ottenuti {len(sampled_players)} giocatori.")
    return sampled_players

def get_player_top_scores(player_id: str, player_rank: int) -> List[Dict[str, Any]]:
    """
    Ottiene la lista dei "top scores" per un singolo giocatore, filtrando
    quelli che hanno modificatori (mods).
    
    Args:
        player_id: L'ID del giocatore da interrogare.
        player_rank: Il rank del giocatore (da passare al dataset finale).
        
    Returns:
        Una lista di dizionari, ognuno contenente 'player_id', 'accuracy' e 'songHash'.
    """
    logging.info(f"Recupero 'top scores' per il giocatore {player_id} (Rank: {player_rank})...")
    scores_data: List[Dict[str, Any]] = []
    page = 1
    
    while True:
        # Usiamo limit=20 per ridurre num di chiamate a API
        url = f"{SCORESABER_API_URL}/player/{player_id}/scores?sort=top&limit=20&page={page}"
        
        try:
            scoresaber_limiter.wait()
            response = requests.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            
            data = response.json()
            player_scores = data.get('playerScores', [])
            
            if not player_scores:
                logging.info(f"Nessun altro score trovato per {player_id} a pagina {page}. Fine.")
                break

            found_scores_with_mods = 0
            for player_score in player_scores:
                
                score_details = player_score.get('score')
                leaderboard_details = player_score.get('leaderboard')

                # Controlla che i dati ci siano
                if not score_details or not leaderboard_details:
                    logging.warning(f"Score entry malformata per player {player_id}, la salto.")
                    continue

                # non prendere score con "modifiers" (stringa vuota)
                modifiers = score_details.get('modifiers', '')
                

                # prendere solo difficulty == 7, equivale a "Expert" nella scala di ScoreSaber
                difficulty_info = leaderboard_details.get('difficulty', {})
                difficulty = difficulty_info.get('difficulty')


                if (not modifiers) and (difficulty == 7):
                    # L'accuracy va calcolata: baseScore (da 'score') / maxScore (da 'leaderboard')
                    base_score = score_details.get('baseScore', 0)
                    max_score = leaderboard_details.get('maxScore', 0)
                    
                    accuracy = 0.0
                    if max_score > 0:
                        accuracy = (base_score / max_score)
                    else:
                        # Se max_score è 0, non possiamo calcolare l'accuracy
                        logging.warning(f"Skipping score for map {leaderboard_details.get('songHash')} (player {player_id}) because maxScore is 0.")
                        continue 


                    song_hash = leaderboard_details.get('songHash')
                    
                    if not song_hash:
                         logging.warning(f"Skipping score for player {player_id} because songHash is missing.")
                         continue


                    pp = score_details.get('pp', 0.0)

                    scores_data.append({
                        'player_id': player_id,
                        'player_rank': player_rank,
                        'accuracy': accuracy,
                        'base_score': base_score,
                        'pp': pp,
                        'songHash': song_hash
                    })
                else:
                    # Score saltato o per 'modifiers' o per 'difficulty'
                    found_scores_with_mods += 1
            
            logging.info(f"Pagina {page} per {player_id}: {len(player_scores)} scores trovati, {len(scores_data)} validi (no-mods, ExpertSolo), {found_scores_with_mods} ignorati (mods o difficulty errata).")

            # Se l'API restituisce meno di 100 score, siamo all'ultima pagina
            if len(player_scores) < 100:
                logging.info(f"Ultima pagina (n.{page}) raggiunta per {player_id}.")
                break
                
            page += 1
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Errore durante il recupero degli score per {player_id} (pagina {page}): {e}")
            break # Interrompi il ciclo per questo giocatore in caso di errore
        except Exception as e:
            logging.error(f"Errore inatteso during l'elaborazione degli score per {player_id}: {e}")
            break

    return scores_data


def get_map_details(song_hash: str) -> Optional[Dict[str, Any]]:
    """
    Ottiene i dettagli di una mappa (BPM, durata e note) da BeatSaver usando il suo hash.
    
    Args:
        song_hash: L'hash della mappa (ottenuto da ScoreSaber).
        
    Returns:
        Un dizionario con 'songHash', 'bpm', 'duration', e 'notes', o None se fallisce.
    """
    url = f"{BEATSAVER_API_URL}/maps/hash/{song_hash}"
    
    try:
        beatsaver_limiter.wait()
        response = requests.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        
        data = response.json()
        

        metadata = data.get('metadata')
        bpm = None
        duration = None
        
        if metadata:
            bpm = metadata.get('bpm')
            duration = metadata.get('duration')
        else:
            logging.warning(f"Oggetto 'metadata' non trovato per la mappa {song_hash}.")



        notes = None
        versions = data.get('versions')
        
        if versions and isinstance(versions, list) and len(versions) > 0:
            latest_version = versions[0]
            diffs = latest_version.get('diffs')
            
            if diffs and isinstance(diffs, list):
                found_expert = False
                for diff in diffs:

                    if diff.get('difficulty') == 'Expert':
                        notes = diff.get('notes')
                        logging.info(f"Trovate 'notes': {notes} per la difficoltà Expert della mappa {song_hash}.")
                        found_expert = True
                        break
                
                if not found_expert:
                    logging.warning(f"Nessuna difficoltà 'Expert' (e quindi 'notes') trovata per la mappa {song_hash}.")
            else:
                logging.warning(f"Array 'diffs' non trovato o non valido per la mappa {song_hash}.")
        else:
            logging.warning(f"Array 'versions' non trovato o non valido per la mappa {song_hash}.")


        if bpm is not None and duration is not None:
            return {
                'songHash': song_hash,
                'bpm': bpm,
                'duration': duration,
                'notes': notes
            }
        else:
            logging.warning(f"Dati 'bpm' o 'duration' essenziali mancanti per la mappa {song_hash}. Impossibile aggiungerla.")
            return None
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Mappa non trovata su BeatSaver (404): {song_hash}")
        else:
            logging.error(f"Errore HTTP durante il recupero della mappa {song_hash}: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Errore di richiesta per la mappa {song_hash}: {e}")
    except Exception as e:
        logging.error(f"Errore inatteso nell'elaborazione della mappa {song_hash}: {e}")
        
    return None


def main():
    logging.info("--- Inizio Creazione Dataset Beat Saber ---")
    logging.info("--- FASE 1: Creazione 'players_scores' ---")
    
    # PARAMETRI DI CAMPIONAMENTO
    # Si può aggiustare 'num_samples' per un dataset più grande o più piccolo.
    sampled_players = get_sampled_player_ids()
    
    if not sampled_players:
        logging.critical("Nessun giocatore campionato. Impossibile procedere.")
        return

    all_scores_list: List[Dict[str, Any]] = []
    for player_info in sampled_players:
        player_id = player_info['id']
        player_rank = player_info['rank']
        all_scores_list.extend(get_player_top_scores(player_id, player_rank))
        
    if not all_scores_list:
        logging.critical("Nessuno score valido (no-mods) trovato per i giocatori campionati. Impossibile procedere.")
        return

    # Crea il DataFrame e salva
    players_scores_df = pd.DataFrame(all_scores_list)
    output_filename_1 = "players_scores.csv"
    players_scores_df.to_csv(output_filename_1, index=False)
    logging.info(f"FASE 1 completata. Dataset '{output_filename_1}' salvato. Shape: {players_scores_df.shape}")
    print(f"\n--- 'players_scores.csv' (head) ---\n{players_scores_df.head()}\n")




    logging.info("--- FASE 2: Creazione 'maps_data' ---")
    
    # Prende tutti gli hash unici dal dataset precedente
    unique_song_hashes = players_scores_df['songHash'].unique()
    logging.info(f"Trovati {len(unique_song_hashes)} songHash unici da interrogare su BeatSaver.")
    
    map_details_list: List[Dict[str, Any]] = []
    for i, song_hash in enumerate(unique_song_hashes):
        logging.info(f"Recupero dettagli mappa {i+1}/{len(unique_song_hashes)} (Hash: {song_hash})")
        details = get_map_details(song_hash)
        if details:
            map_details_list.append(details)

    if not map_details_list:
        logging.critical("Nessun dettaglio mappa trovato su BeatSaver. Impossibile creare 'maps_data'.")
        return

    # Crea il DataFrame e salva
    maps_data_df = pd.DataFrame(map_details_list)
    output_filename_2 = "maps_data.csv"
    maps_data_df.to_csv(output_filename_2, index=False)
    logging.info(f"FASE 2 completata. Dataset '{output_filename_2}' salvato. Shape: {maps_data_df.shape}")
    print(f"\n--- 'maps_data.csv' (head) ---\n{maps_data_df.head()}\n")


    #Creazione "map_performance.csv" (Join)
    logging.info("--- FASE 3: Creazione 'map_performance' (Join) ---")
    
    map_performance_df = pd.merge(players_scores_df, maps_data_df, on='songHash', how='inner')
    
    output_filename_3 = "map_performance.csv"
    map_performance_df.to_csv(output_filename_3, index=False)
    logging.info(f"FASE 3 completata. Dataset finale '{output_filename_3}' salvato. Shape: {map_performance_df.shape}")
    print(f"\n--- 'map_performance.csv' (head) ---\n{map_performance_df.head()}\n")
    
    logging.info("--- Processo di creazione dataset completato con successo! ---")


if __name__ == "__main__":
    main()