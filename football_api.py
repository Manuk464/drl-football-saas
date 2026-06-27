import requests
import os
from datetime import datetime
import time

API_KEY = os.environ.get("FOOTBALL_API_KEY", "28ea505ca8e0fac5bb6d064589244bfc")
BASE_URL = "https://v3.football.api-sports.io"

_cache = {}
_cache_ts = {}
CACHE_DUR = 300

def _get(endpoint, params=None):
    key = f"{endpoint}:{str(params)}"
    if key in _cache and time.time() - _cache_ts[key] < CACHE_DUR:
        return _cache[key]
    
    headers = {"x-apisports-key": API_KEY}
    r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=15)
    
    if r.status_code == 200:
        _cache[key] = r.json()
        _cache_ts[key] = time.time()
        return _cache[key]
    elif r.status_code == 429:
        time.sleep(7)
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            _cache[key] = r.json()
            _cache_ts[key] = time.time()
            return _cache[key]
    return {"response": []}

def extract_stat(stats, stat_name):
    if not stats:
        return 0.0
    for s in stats:
        if s.get("type") == stat_name:
            v = s.get("value")
            if v is None or v == "":
                return 0.0
            if isinstance(v, str):
                v = v.replace("%", "").replace(",", ".").strip()
                try:
                    return float(v)
                except:
                    return 0.0
            try:
                return float(v)
            except:
                return 0.0
    return 0.0

def get_real_fixtures_for_today():
    print("[INFO] Buscando jogos reais...")
    today = datetime.now().strftime("%Y-%m-%d")
    data = _get("fixtures", {"date": today})
    fixtures = data.get("response", [])
    
    valid = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short", "") in ["NS", "1H", "2H", "HT", "P"]]
    
    results = []
    for fixture in valid[:3]:
        try:
            fid = fixture.get("fixture", {}).get("id")
            
            # Estatísticas
            stats_data = _get("fixtures/statistics", {"fixture": fid}).get("response", [])
            home_stats = stats_data[0].get("statistics", []) if len(stats_data) > 0 else []
            away_stats = stats_data[1].get("statistics", []) if len(stats_data) > 1 else []
            time.sleep(2)
            
            # Odds
            odds_data = _get("odds", {"fixture": fid}).get("response", [])
            odds = {"home": 2.5, "draw": 3.2, "away": 2.8}
            if odds_data:
                for bm in odds_data[0].get("bookmakers", []):
                    for bet in bm.get("bets", []):
                        if bet.get("name") == "Match Winner":
                            for val in bet.get("values", []):
                                if val.get("value") == "Home":
                                    odds["home"] = float(val.get("odd", 2.5))
                                elif val.get("value") == "Draw":
                                    odds["draw"] = float(val.get("odd", 3.2))
                                elif val.get("value") == "Away":
                                    odds["away"] = float(val.get("odd", 2.8))
            time.sleep(2)
            
            # Nomes REAIS (SEM ESPAÇOS!)
            teams = fixture.get("fixture", {}).get("teams", {})
            league = fixture.get("league", {})
            home_name = teams.get("home", {}).get("name", "Unknown")
            away_name = teams.get("away", {}).get("name", "Unknown")
            league_name = league.get("name", "Unknown")
            
            round_str = league.get("round", "")
            try:
                matchday = int("".join(filter(str.isdigit, round_str)) or 15)
            except:
                matchday = 15
            
            results.append({
                "date": fixture.get("fixture", {}).get("date", "")[:10],
                "home_team": home_name,
                "away_team": away_name,
                "competition": league_name,
                "matchday": matchday,
                "xg_home": extract_stat(home_stats, "Expected Goals") or 1.2,
                "xg_away": extract_stat(away_stats, "Expected Goals") or 1.0,
                "shots_home": int(extract_stat(home_stats, "Total Shots") or 12),
                "shots_away": int(extract_stat(away_stats, "Total Shots") or 10),
                "shots_on_home": int(extract_stat(home_stats, "Shots On Target") or 5),
                "shots_on_away": int(extract_stat(away_stats, "Shots On Target") or 4),
                "possession_home": extract_stat(home_stats, "Ball Possession") or 50,
                "possession_away": extract_stat(away_stats, "Ball Possession") or 50,
                "pass_acc_home": extract_stat(home_stats, "Passes Accuracy") or 78,
                "pass_acc_away": extract_stat(away_stats, "Passes Accuracy") or 75,
                "tackles_home": int(extract_stat(home_stats, "Tackles") or 18),
                "tackles_away": int(extract_stat(away_stats, "Tackles") or 17),
                "interceptions_home": int(extract_stat(home_stats, "Interceptions") or 10),
                "interceptions_away": int(extract_stat(away_stats, "Interceptions") or 10),
                "ppda_home": 9.0,
                "ppda_away": 9.0,
                "corners_home": int(extract_stat(home_stats, "Corner Kicks") or 6),
                "corners_away": int(extract_stat(away_stats, "Corner Kicks") or 5),
                "aerial_won_home": extract_stat(home_stats, "Aerial Duels Won") or 50,
                "aerial_won_away": extract_stat(away_stats, "Aerial Duels Won") or 50,
                "key_passes_home": int(extract_stat(home_stats, "Key Passes") or 8),
                "key_passes_away": int(extract_stat(away_stats, "Key Passes") or 7),
                "prog_passes_home": int((extract_stat(home_stats, "Total Passes") or 400) * 0.06),
                "prog_passes_away": int((extract_stat(away_stats, "Total Passes") or 380) * 0.06),
                "touches_box_home": int(extract_stat(home_stats, "Shots Inside Box") or 12),
                "touches_box_away": int(extract_stat(away_stats, "Shots Inside Box") or 10),
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
            print(f"[OK] {home_name} vs {away_name}")
        except Exception as e:
            print(f"[ERRO] {e}")
            continue
    
    return results

def _get_mock_fixtures():
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {"home_team": "Mushuc Runa", "away_team": "LDU Quito", "competition": "Liga Pro", "home_str": 0.6, "away_str": 0.8, "date": today},
        {"home_team": "Barcelona SC", "away_team": "Emelec", "competition": "Liga Pro", "home_str": 0.85, "away_str": 0.65, "date": today},
        {"home_team": "Ind. del Valle", "away_team": "Aucas", "competition": "Liga Pro", "home_str": 0.9, "away_str": 0.5, "date": today},
    ]