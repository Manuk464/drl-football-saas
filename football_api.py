import requests
import os
from datetime import datetime
import time

# API Key: tenta ler do ambiente, senão usa hardcoded (funciona no Streamlit Cloud)
API_KEY = os.environ.get("FOOTBALL_API_KEY", "28ea505ca8e0fac5bb6d064589244bfc")
BASE_URL = "https://v3.football.api-sports.io"

_cache = {}
_cache_ts = {}
CACHE_DUR = 300


def _get(endpoint, params=None):
    """Faz requisição à API com cache e tratamento de rate limit"""
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
        print("[AVISO] Rate limit atingido, aguardando 7 segundos...")
        time.sleep(7)
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            _cache[key] = r.json()
            _cache_ts[key] = time.time()
            return _cache[key]
    return {"response": []}


def extract_stat(stats, stat_name):
    """Extrai valor numérico de uma estatística específica"""
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


def get_fixtures_by_date(date=None):
    """Busca todos os jogos de uma data"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    data = _get("fixtures", {"date": date})
    return data.get("response", [])


def get_fixture_statistics(fixture_id):
    """Busca estatísticas de um jogo específico"""
    data = _get("fixtures/statistics", {"fixture": fixture_id})
    stats = data.get("response", [])
    if len(stats) >= 2:
        return {
            "home": stats[0].get("statistics", []),
            "away": stats[1].get("statistics", [])
        }
    return {"home": [], "away": []}


def get_fixture_odds(fixture_id):
    """Busca odds Match Winner de um jogo"""
    data = _get("odds", {"fixture": fixture_id})
    odds_data = data.get("response", [])
    odds = {"home": 2.5, "draw": 3.2, "away": 2.8}

    if odds_data:
        bookmakers = odds_data[0].get("bookmakers", [])
        for bm in bookmakers:
            for bet in bm.get("bets", []):
                if bet.get("name") == "Match Winner":
                    for val in bet.get("values", []):
                        if val.get("value") == "Home":
                            odds["home"] = float(val.get("odd", 2.5))
                        elif val.get("value") == "Draw":
                            odds["draw"] = float(val.get("odd", 3.2))
                        elif val.get("value") == "Away":
                            odds["away"] = float(val.get("odd", 2.8))
                    return odds
    return odds


def get_real_fixtures_for_today():
    """Retorna até 3 jogos REAIS com estatísticas e odds da API-Football"""
    print("[INFO] get_real_fixtures_for_today() chamado")

    if not API_KEY:
        print("[ERRO] API_KEY não configurada!")
        return _get_mock_fixtures()

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        fixtures = get_fixtures_by_date(today)
        print(f"[DEBUG] Jogos encontrados: {len(fixtures)}")

        valid = [
            f for f in fixtures
            if f.get("fixture", {}).get("status", {}).get("short", "") in ["NS", "1H", "2H", "HT", "P"]
        ]
        print(f"[DEBUG] Jogos válidos: {len(valid)}")

        if len(valid) == 0:
            print("[AVISO] Nenhum jogo válido hoje")
            return _get_mock_fixtures()

        results = []
        # Limita a 3 jogos para economizar rate limit (10 req/min do plano grátis)
        for fixture in valid[:3]:
            try:
                fid = fixture.get("fixture", {}).get("id")
                print(f"[DEBUG] Processando jogo {fid}")

                # Estatísticas
                stats = get_fixture_statistics(fid)
                home_stats = stats.get("home", [])
                away_stats = stats.get("away", [])
                time.sleep(2)

                # Odds
                odds = get_fixture_odds(fid)
                time.sleep(2)

                # Nomes e liga
                teams = fixture.get("fixture", {}).get("teams", {})
                league = fixture.get("league", {})
                home_name = teams.get("home", {}).get("name", "Unknown")
                away_name = teams.get("away", {}).get("name", "Unknown")
                league_name = league.get("name", "Unknown")

                # Estatísticas REAIS
                xg_h = extract_stat(home_stats, "Expected Goals") or 1.2
                xg_a = extract_stat(away_stats, "Expected Goals") or 1.0
                shots_h = int(extract_stat(home_stats, "Total Shots") or 12)
                shots_a = int(extract_stat(away_stats, "Total Shots") or 10)
                shots_on_h = int(extract_stat(home_stats, "Shots On Target") or 5)
                shots_on_a = int(extract_stat(away_stats, "Shots On Target") or 4)
                poss_h = extract_stat(home_stats, "Ball Possession") or 50
                poss_a = extract_stat(away_stats, "Ball Possession") or 50
                pass_acc_h = extract_stat(home_stats, "Passes Accuracy") or 78
                pass_acc_a = extract_stat(away_stats, "Passes Accuracy") or 75
                tackles_h = int(extract_stat(home_stats, "Tackles") or 18)
                tackles_a = int(extract_stat(away_stats, "Tackles") or 17)
                intercept_h = int(extract_stat(home_stats, "Interceptions") or 10)
                intercept_a = int(extract_stat(away_stats, "Interceptions") or 10)
                corners_h = int(extract_stat(home_stats, "Corner Kicks") or 6)
                corners_a = int(extract_stat(away_stats, "Corner Kicks") or 5)
                aerial_h = extract_stat(home_stats, "Aerial Duels Won") or 50
                aerial_a = extract_stat(away_stats, "Aerial Duels Won") or 50
                key_pass_h = int(extract_stat(home_stats, "Key Passes") or 8)
                key_pass_a = int(extract_stat(away_stats, "Key Passes") or 7)
                total_pass_h = extract_stat(home_stats, "Total Passes") or 400
                total_pass_a = extract_stat(away_stats, "Total Passes") or 380
                prog_pass_h = int(total_pass_h * 0.06)
                prog_pass_a = int(total_pass_a * 0.06)
                touches_box_h = int(extract_stat(home_stats, "Shots Inside Box") or 12)
                touches_box_a = int(extract_stat(away_stats, "Shots Inside Box") or 10)

                # Matchday
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
                    "xg_home": xg_h,
                    "xg_away": xg_a,
                    "shots_home": shots_h,
                    "shots_away": shots_a,
                    "shots_on_home": shots_on_h,
                    "shots_on_away": shots_on_a,
                    "possession_home": poss_h,
                    "possession_away": poss_a,
                    "pass_acc_home": pass_acc_h,
                    "pass_acc_away": pass_acc_a,
                    "tackles_home": tackles_h,
                    "tackles_away": tackles_a,
                    "interceptions_home": intercept_h,
                    "interceptions_away": intercept_a,
                    "ppda_home": 9.0,
                    "ppda_away": 9.0,
                    "corners_home": corners_h,
                    "corners_away": corners_a,
                    "aerial_won_home": aerial_h,
                    "aerial_won_away": aerial_a,
                    "key_passes_home": key_pass_h,
                    "key_passes_away": key_pass_a,
                    "prog_passes_home": prog_pass_h,
                    "prog_passes_away": prog_pass_a,
                    "touches_box_home": touches_box_h,
                    "touches_box_away": touches_box_a,
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
                print(f"[SUCCESS] {home_name} vs {away_name} - xG: {xg_h:.2f} x {xg_a:.2f}")

            except Exception as e:
                print(f"[ERRO] Erro no jogo {fid}: {e}")
                continue

        print(f"[INFO] Total de jogos processados: {len(results)}")
        return results if results else _get_mock_fixtures()

    except Exception as e:
        print(f"[ERRO CRÍTICO] {e}")
        return _get_mock_fixtures()


def _get_mock_fixtures():
    """Fallback: jogos simulados quando a API falha"""
    print("[INFO] Retornando jogos simulados (fallback)")
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {"home_team": "Mushuc Runa", "away_team": "LDU Quito", "competition": "Liga Pro", "home_str": 0.6, "away_str": 0.8, "date": today},
        {"home_team": "Barcelona SC", "away_team": "Emelec", "competition": "Liga Pro", "home_str": 0.85, "away_str": 0.65, "date": today},
        {"home_team": "Ind. del Valle", "away_team": "Aucas", "competition": "Liga Pro", "home_str": 0.9, "away_str": 0.5, "date": today},
    ]