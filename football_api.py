import requests
import os
from datetime import datetime
from typing import List, Dict
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

print(f"[DEBUG] API_KEY: {'CONFIGURADA' if API_KEY else 'NÃO CONFIGURADA'}")

_cache = {}
_cache_ts = {}
CACHE_DUR = 300

def _get(endpoint, params=None):
    key = f"{endpoint}:{str(params)}"
    if key in _cache and time.time() - _cache_ts[key] < CACHE_DUR:
        return _cache[key]
    
    headers = {"x-apisports-key": API_KEY}
    print(f"[DEBUG] Requisitando: {endpoint}")
    
    r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=10)
    print(f"[DEBUG] Status: {r.status_code}")
    
    if r.status_code == 200:
        _cache[key] = r.json()
        _cache_ts[key] = time.time()
        return _cache[key]
    else:
        raise Exception(f"API Error: {r.status_code}")

def get_fixtures_by_date(date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    data = _get("fixtures", {"date": date})
    return data.get("response", [])

def get_fixture_statistics(fixture_id):
    data = _get("fixtures/statistics", {"fixture": fixture_id})
    stats = data.get("response", [])
    if len(stats) >= 2:
        return {
            "home": stats[0].get("statistics", []),
            "away": stats[1].get("statistics", [])
        }
    return {}

def get_fixture_odds(fixture_id):
    data = _get("odds", {"fixture": fixture_id})
    odds_data = data.get("response", [])
    if odds_data:
        bookmakers = odds_data[0].get("bookmakers", [])
        if bookmakers:
            bets = bookmakers[0].get("bets", [])
            for bet in bets:
                if bet.get("name") == "Match Winner":
                    values = bet.get("values", [])
                    odds = {}
                    for value in values:
                        if value.get("value") == "Home":
                            odds["home"] = float(value.get("odd", 0))
                        elif value.get("value") == "Draw":
                            odds["draw"] = float(value.get("odd", 0))
                        elif value.get("value") == "Away":
                            odds["away"] = float(value.get("odd", 0))
                    return odds
    return {}

def get_real_fixtures_for_today():
    print("[INFO] get_real_fixtures_for_today() chamado")
    
    if not API_KEY:
        print("[ERRO] API_KEY não configurada!")
        return _get_mock_fixtures()
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"[INFO] Buscando jogos para {today}...")
        
        fixtures = get_fixtures_by_date(today)
        print(f"[DEBUG] Jogos encontrados: {len(fixtures)}")
        
        valid = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short", "") in ["NS", "1H", "2H", "HT"]]
        print(f"[DEBUG] Jogos válidos: {len(valid)}")
        
        if len(valid) == 0:
            print("[AVISO] Nenhum jogo válido hoje")
            return _get_mock_fixtures()
        
        results = []
        for fixture in valid[:8]:
            try:
                fid = fixture.get("fixture", {}).get("id")
                print(f"[DEBUG] Processando jogo {fid}")
                
                stats = get_fixture_statistics(fid)
                odds = get_fixture_odds(fid)
                
                teams = fixture.get("fixture", {}).get("teams", {})
                results.append({
                    "date": fixture.get("fixture", {}).get("date", "")[:10],
                    "home_team": teams.get("home", {}).get("name", ""),
                    "away_team": teams.get("away", {}).get("name", ""),
                    "competition": fixture.get("league", {}).get("name", ""),
                    "matchday": 15,
                    "xg_home": 1.0,
                    "xg_away": 1.0,
                    "shots_home": 10,
                    "shots_away": 10,
                    "shots_on_home": 4,
                    "shots_on_away": 4,
                    "possession_home": 50,
                    "possession_away": 50,
                    "pass_acc_home": 75,
                    "pass_acc_away": 75,
                    "tackles_home": 18,
                    "tackles_away": 18,
                    "interceptions_home": 9,
                    "interceptions_away": 9,
                    "ppda_home": 9.0,
                    "ppda_away": 9.0,
                    "corners_home": 5,
                    "corners_away": 5,
                    "aerial_won_home": 50,
                    "aerial_won_away": 50,
                    "key_passes_home": 6,
                    "key_passes_away": 6,
                    "prog_passes_home": 24,
                    "prog_passes_away": 24,
                    "touches_box_home": 12,
                    "touches_box_away": 12,
                    "rest_days_home": 7,
                    "rest_days_away": 7,
                    "altitude_home_m": 0,
                    "home_league_pos": 10,
                    "away_league_pos": 10,
                    "home_pts": 20,
                    "away_pts": 20,
                    "home_gf_season": 15,
                    "home_ga_season": 15,
                    "away_gf_season": 15,
                    "away_ga_season": 15,
                    "odds_home": odds.get("home", 2.5),
                    "odds_draw": odds.get("draw", 3.2),
                    "odds_away": odds.get("away", 2.8)
                })
                time.sleep(0.3)
            except Exception as e:
                print(f"[ERRO] Erro no jogo {fid}: {e}")
                continue
        
        print(f"[INFO] Jogos processados: {len(results)}")
        return results if results else _get_mock_fixtures()
        
    except Exception as e:
        print(f"[ERRO CRÍTICO] {e}")
        return _get_mock_fixtures()

def _get_mock_fixtures():
    print("[INFO] Retornando jogos simulados")
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {"home_team": "Mushuc Runa", "away_team": "LDU Quito", "competition": "Liga Pro", "home_str": 0.6, "away_str": 0.8, "date": today},
        {"home_team": "Barcelona SC", "away_team": "Emelec", "competition": "Liga Pro", "home_str": 0.85, "away_str": 0.65, "date": today},
        {"home_team": "Ind. del Valle", "away_team": "Aucas", "competition": "Liga Pro", "home_str": 0.9, "away_str": 0.5, "date": today},
    ]