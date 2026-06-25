import requests
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Configurações da API
API_KEY = os.environ.get("FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

# Cache simples para evitar requisições repetidas
_cache = {}
_cache_timestamps = {}
CACHE_DURATION = 300  # 5 minutos

def _get_with_cache(endpoint: str, params: dict = None) -> dict:
    """Faz requisição com cache para otimizar uso da API"""
    cache_key = f"{endpoint}:{str(params)}"
    
    # Verifica se está no cache e não expirou
    if cache_key in _cache:
        if time.time() - _cache_timestamps[cache_key] < CACHE_DURATION:
            return _cache[cache_key]
    
    # Faz a requisição
    headers = {
        "x-apisports-key": API_KEY
    }
    
    response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        _cache[cache_key] = data
        _cache_timestamps[cache_key] = time.time()
        return data
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def get_fixtures_by_date(date: str = None) -> List[Dict]:
    """
    Busca todos os jogos de uma data específica
    Formato da data: YYYY-MM-DD
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    params = {"date": date}
    data = _get_with_cache("fixtures", params)
    
    return data.get("response", [])

def get_fixture_statistics(fixture_id: int) -> Dict:
    """Busca estatísticas detalhadas de um jogo específico"""
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
    """Busca odds de um jogo específico"""
    params = {"fixture": fixture_id}
    data = _get_with_cache("odds", params)
    
    odds_data = data.get("response", [])
    if odds_data:
        # Pega as odds do primeiro bookmaker
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

def get_league_standings(league_id: int, season: int) -> List[Dict]:
    """Busca classificação de uma liga"""
    params = {"league": league_id, "season": season}
    data = _get_with_cache("standings", params)
    
    standings = data.get("response", [])
    if standings:
        return standings[0].get("league", {}).get("standings", [[]])[0]
    return []

def extract_stat_value(statistics: List[Dict], stat_name: str) -> float:
    """Extrai o valor de uma estatística específica"""
    for stat in statistics:
        if stat.get("type") == stat_name:
            value = stat.get("value")
            if value is None:
                return 0.0
            if isinstance(value, str):
                # Remove % e converte
                value = value.replace("%", "").strip()
                try:
                    return float(value)
                except:
                    return 0.0
            return float(value)
    return 0.0

def map_fixture_to_model_format(fixture: Dict, statistics: Dict, odds: Dict, standings: List[Dict] = None) -> Dict:
    """
    Mapeia dados da API-Football para o formato esperado pelo modelo
    """
    teams = fixture.get("fixture", {}).get("teams", {})
    home_team = teams.get("home", {}).get("name", "")
    away_team = teams.get("away", {}).get("name", "")
    
    # Extrai estatísticas
    home_stats = statistics.get("home", [])
    away_stats = statistics.get("away", [])
    
    # Mapeia features
    mapped_data = {
        "date": fixture.get("fixture", {}).get("date", "")[:10],
        "home_team": home_team,
        "away_team": away_team,
        "competition": fixture.get("league", {}).get("name", "Unknown"),
        "matchday": fixture.get("league", {}).get("round", "").replace("Regular Season - ", ""),
        
        # Estatísticas ofensivas
        "xg_home": extract_stat_value(home_stats, "Expected Goals") or 1.0,
        "xg_away": extract_stat_value(away_stats, "Expected Goals") or 1.0,
        "shots_home": int(extract_stat_value(home_stats, "Total Shots") or 10),
        "shots_away": int(extract_stat_value(away_stats, "Total Shots") or 10),
        "shots_on_home": int(extract_stat_value(home_stats, "Shots On Target") or 4),
        "shots_on_away": int(extract_stat_value(away_stats, "Shots On Target") or 4),
        
        # Posse e passes
        "possession_home": extract_stat_value(home_stats, "Ball Possession") or 50,
        "possession_away": extract_stat_value(away_stats, "Ball Possession") or 50,
        "pass_acc_home": extract_stat_value(home_stats, "Passes Accuracy") or 75,
        "pass_acc_away": extract_stat_value(away_stats, "Passes Accuracy") or 75,
        
        # Defensivo
        "tackles_home": int(extract_stat_value(home_stats, "Tackles") or 18),
        "tackles_away": int(extract_stat_value(away_stats, "Tackles") or 18),
        "interceptions_home": int(extract_stat_value(home_stats, "Interceptions") or 9),
        "interceptions_away": int(extract_stat_value(away_stats, "Interceptions") or 9),
        
        # Pressão (PPDA não está disponível, estimamos)
        "ppda_home": 9.0,  # Valor padrão
        "ppda_away": 9.0,
        
        # Escanteios
        "corners_home": int(extract_stat_value(home_stats, "Corner Kicks") or 5),
        "corners_away": int(extract_stat_value(away_stats, "Corner Kicks") or 5),
        
        # Duelos aéreos
        "aerial_won_home": extract_stat_value(home_stats, "Aerial Duels Won") or 50,
        "aerial_won_away": extract_stat_value(away_stats, "Aerial Duels Won") or 50,
        
        # Passes progressivos (estimativa baseada em passes totais)
        "key_passes_home": int(extract_stat_value(home_stats, "Key Passes") or 6),
        "key_passes_away": int(extract_stat_value(away_stats, "Key Passes") or 6),
        "prog_passes_home": int(extract_stat_value(home_stats, "Total Passes") * 0.3) or 24,
        "prog_passes_away": int(extract_stat_value(away_stats, "Total Passes") * 0.3) or 24,
        
        # Toques na área
        "touches_box_home": int(extract_stat_value(home_stats, "Shots Inside Box") or 12),
        "touches_box_away": int(extract_stat_value(away_stats, "Shots Inside Box") or 12),
        
        # Dias de descanso (estimativa padrão)
        "rest_days_home": 7,
        "rest_days_away": 7,
        
        # Altitude (padrão 0, pode ser atualizado depois)
        "altitude_home_m": 0,
        
        # Posição na tabela (se disponível)
        "home_league_pos": 10,  # Padrão
        "away_league_pos": 10,
        "home_pts": 20,
        "away_pts": 20,
        
        # Gols na temporada (estimativa)
        "home_gf_season": 15,
        "home_ga_season": 15,
        "away_gf_season": 15,
        "away_ga_season": 15,
        
        # Odds
        "odds_home": odds.get("home", 2.5),
        "odds_draw": odds.get("draw", 3.2),
        "odds_away": odds.get("away", 2.8),
    }
    
    # Se temos classificação, atualiza posições
    if standings:
        for team in standings:
            team_info = team.get("team", {})
            if team_info.get("name") == home_team:
                mapped_data["home_league_pos"] = team.get("rank", 10)
                mapped_data["home_pts"] = team.get("points", 20)
                all_matches = team.get("all", {})
                mapped_data["home_gf_season"] = all_matches.get("goals", {}).get("for", 15)
                mapped_data["home_ga_season"] = all_matches.get("goals", {}).get("against", 15)
            elif team_info.get("name") == away_team:
                mapped_data["away_league_pos"] = team.get("rank", 10)
                mapped_data["away_pts"] = team.get("points", 20)
                all_matches = team.get("all", {})
                mapped_data["away_gf_season"] = all