import requests
import os
from datetime import datetime
from typing import List, Dict
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

print(f"[DEBUG] API_KEY configurada: {'SIM' if API_KEY else 'NÃO'}")
print(f"[DEBUG] API_KEY length: {len(API_KEY) if API_KEY else 0}")

_cache = {}
_cache_timestamps = {}
CACHE_DURATION = 300

def _get_with_cache(endpoint: str, params: dict = None) -> dict:
    cache_key = f"{endpoint}:{str(params)}"
    
    if cache_key in _cache:
        if time.time() - _cache_timestamps[cache_key] < CACHE_DURATION:
            return _cache[cache_key]
    
    headers = {"x-apisports-key": API_KEY}
    print(f"[DEBUG] Fazendo requisição para: {endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=10)
        print(f"[DEBUG] Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            _cache[cache_key] = data
            _cache_timestamps[cache_key] = time.time()
            return data
        else:
            print(f"[ERRO] API Error: {response.status_code} - {response.text}")
            raise Exception(f"API Error: {response.status_code}")
    except Exception as e:
        print(f"[ERRO] Exceção na requisição: {str(e)}")
        raise

def get_fixtures_by_date(date: str = None) -> List[Dict]:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    params = {"date": date}
    print(f"[DEBUG] Buscando jogos para data: {date}")
    data = _get_with_cache("fixtures", params)
    response = data.get("response", [])
    print(f"[DEBUG] Jogos encontrados: {len(response)}")
    return response

def get_fixture_statistics(fixture_id: int) -> Dict:
    params = {"fixture": fixture_id}
    data = _get_with_cache("fixtures/statistics", params)
    stats = data.get("response", [])
    if len(stats) >= 2:
        return {
            "home": stats[0].get("statistics", []),
            "away": stats[1].get("statistics", [])
        }
    return {}

def get_fixture_odds(fixture_id: int) -> Dict:
    params = {"fixture": fixture_id}
    data = _get_with_cache("odds", params)
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

def get_real_fixtures_for_today() -> List[Dict]:
    print("[INFO] Iniciando get_real_fixtures_for_today()")
    
    if not API_KEY:
        print("[ERRO] FOOTBALL_API_KEY não configurada!")
        return _get_mock_fixtures()
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"[INFO] Buscando jogos reais para {today}...")
        
        fixtures = get_fixtures_by_date(today)
        
        valid_fixtures = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short", "") in ["NS", "1H", "2H", "HT"]]
        
        print(f"[INFO] Jogos válidos encontrados: {len(valid_fixtures)}")
        
        if len(valid_fixtures) == 0:
            print("[AVISO] Nenhum jogo válido encontrado hoje")
            return _get_mock_fixtures()
        
        results = []
        for fixture in valid_fixtures[:8]:
            try:
                fid = fixture.get("fixture", {}).get("id")
                print(f"[DEBUG] Processando jogo ID: {fid}")
                
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
                print(f"[ERRO] Erro ao processar jogo {fid}: {str(e)}")
                continue
        
        print(f"[INFO] Total de jogos processados com sucesso: {len(results)}")
        return results if results else _get_mock_fixtures()
        
    except Exception as e:
        print(f"[ERRO CRÍTICO] Erro em get_real_fixtures_for_today: {str(e)}")
        return _get_mock_fixtures()

def _get_mock_fixtures() -> List[Dict]:
    print("[INFO] Retornando jogos simulados (fallback)")
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {"home_team": "Mushuc Runa", "away_team": "LDU Quito", "competition": "Liga Pro", "home_str": 0.6, "away_str": 0.8, "date": today},
        {"home_team": "Barcelona SC", "away_team": "Emelec", "competition": "Liga Pro", "home_str": 0.85, "away_str": 0.65, "date": today},
        {"home_team": "Ind. del Valle", "away_team": "Aucas", "competition": "Liga Pro", "home_str": 0.9, "away_str": 0.5, "date": today},
    ]