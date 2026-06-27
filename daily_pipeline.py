import requests
from datetime import datetime
from football_api import get_real_fixtures_for_today, _get_mock_fixtures
import os
import random

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def get_today_fixtures():
    """Retorna jogos reais da API ou fallback simulado"""
    try:
        real_fixtures = get_real_fixtures_for_today()
        if real_fixtures and len(real_fixtures) > 0:
            print(f"[SUCCESS] {len(real_fixtures)} jogos reais!")
            return real_fixtures
    except Exception as e:
        print(f"[ERRO] Falha ao buscar jogos reais: {e}")
    return _get_mock_fixtures()


def estimate_prematch_features(fixture):
    """Retorna as features para o modelo DRL. Se já vierem completas (dados reais), retorna direto."""
    if "xg_home" in fixture and "odds_home" in fixture:
        return fixture

    # Fallback: estima features baseado na força dos times
    hs = fixture.get("home_str", 0.5)
    as_ = fixture.get("away_str", 0.5)
    rng = random.Random(hash(fixture.get("home_team", "") + fixture.get("away_team", "")))

    xg_h = max(0.4, 1.3 * hs + rng.uniform(-0.3, 0.3))
    xg_a = max(0.3, 1.1 * as_ + rng.uniform(-0.3, 0.3))
    p_h = (xg_h / (xg_h + xg_a + 0.8)) * 0.95
    p_a = (xg_a / (xg_h + xg_a + 0.8)) * 0.95
    p_d = 1.0 - p_h - p_a
    oh = round(1 / p_h + 0.05, 2) if p_h > 0 else 2.5
    od = round(1 / p_d + 0.06, 2) if p_d > 0 else 3.2
    oa = round(1 / p_a + 0.05, 2) if p_a > 0 else 2.8

    return {
        "date": fixture.get("date", datetime.now().strftime("%Y-%m-%d")),
        "home_team": fixture.get("home_team", "Unknown"),
        "away_team": fixture.get("away_team", "Unknown"),
        "competition": fixture.get("competition", "Unknown"),
        "matchday": 15,
        "xg_home": round(xg_h, 2),
        "xg_away": round(xg_a, 2),
        "shots_home": int(12 * hs + rng.randint(-2, 2)),
        "shots_away": int(10 * as_ + rng.randint(-2, 2)),
        "shots_on_home": int(5 * hs + rng.randint(-1, 1)),
        "shots_on_away": int(4 * as_ + rng.randint(-1, 1)),
        "possession_home": round(50 + 12 * (hs - as_) + rng.uniform(-3, 3), 1),
        "possession_away": round(50 - 12 * (hs - as_) + rng.uniform(-3, 3), 1),
        "pass_acc_home": round(75 + 5 * hs + rng.uniform(-2, 2), 1),
        "pass_acc_away": round(73 + 5 * as_ + rng.uniform(-2, 2), 1),
        "tackles_home": int(18 * (1 - as_) + rng.randint(-3, 3)),
        "tackles_away": int(18 * (1 - hs) + rng.randint(-3, 3)),
        "interceptions_home": int(10 * (1 - as_) + rng.randint(-2, 2)),
        "interceptions_away": int(10 * (1 - hs) + rng.randint(-2, 2)),
        "ppda_home": round(10 - 3 * hs + rng.uniform(-1, 1), 1),
        "ppda_away": round(10 - 3 * as_ + rng.uniform(-1, 1), 1),
        "corners_home": int(6 * hs + rng.randint(-1, 2)),
        "corners_away": int(5 * as_ + rng.randint(-1, 2)),
        "aerial_won_home": round(50 + rng.uniform(-5, 5), 1),
        "aerial_won_away": round(50 + rng.uniform(-5, 5), 1),
        "key_passes_home": int(8 * hs + rng.randint(-1, 2)),
        "key_passes_away": int(7 * as_ + rng.randint(-1, 2)),
        "prog_passes_home": int(25 * hs + rng.randint(-3, 3)),
        "prog_passes_away": int(23 * as_ + rng.randint(-3, 3)),
        "touches_box_home": int(14 * hs + rng.randint(-2, 2)),
        "touches_box_away": int(11 * as_ + rng.randint(-2, 2)),
        "rest_days_home": rng.choice([3, 4, 7]),
        "rest_days_away": rng.choice([3, 4, 7]),
        "altitude_home_m": rng.choice([0, 0, 2577, 2812]),
        "home_league_pos": rng.randint(1, 16),
        "away_league_pos": rng.randint(1, 16),
        "home_pts": rng.randint(5, 35),
        "away_pts": rng.randint(5, 35),
        "home_gf_season": rng.randint(10, 40),
        "home_ga_season": rng.randint(10, 40),
        "away_gf_season": rng.randint(10, 40),
        "away_ga_season": rng.randint(10, 40),
        "odds_home": oh,
        "odds_draw": od,
        "odds_away": oa
    }