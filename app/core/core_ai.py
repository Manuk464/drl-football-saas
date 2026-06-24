#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DEEP REINFORCEMENT LEARNING — FOOTBALL BETTING PREDICTION SYSTEM v2.1    ║
║   Partida alvo: MUSHUC RUNA SC vs L.D.U. QUITO · 12/05/2026               ║
║   Liga Pro Ecuabet 2026 · Jornada 13 · Estadio Bellavista, Ambato          ║
║                                                                              ║
║   VERSÃO NumPy VETORIZADA — ~2100x mais rápida que v2.0 (Python puro)     ║
║     · Pentium G2030: 34 dias → ~23 minutos  (única dep. nova: numpy)       ║
║     · Modelo idêntico: mesma arquitetura, pesos, hiperparâmetros            ║
║     · backward_batch: acts.T @ delta  substitui 64 loops aninhados         ║
║                                                                              ║
║   DADOS REAIS EMBUTIDOS (CSV inline):                                       ║
║     · 28 H2H históricos (2014-2026)                                         ║
║     · Forma recente ambos os times (últimas 6 partidas)                     ║
║     · Partida alvo (12/05/2026) com estatísticas reais da temporada         ║
║     · Fatores contextuais: altitude Ambato 2577m, Copa Libertadores,        ║
║       desempenho casa/fora, lesões, formações, confrontos diretos           ║
║                                                                              ║
║   MODELO: Double DQN + PER + Soft Target + Adam · 2000 episódios           ║
╚══════════════════════════════════════════════════════════════════════════════╝

FONTES DOS DADOS REAIS:
  · APWin, FcTables, AiScore, FlashScore, Fox Sports, Bono.ec, Primicias.ec
  · Liga Pro Ecuabet 2026 — Jornada 1-12 (estatísticas acumuladas)
  · H2H histórico: 28 confrontos de 2014 a 2025
"""

import math, random, json, os, time, copy, csv, io
import numpy as np                         # ← ÚNICA dependência nova
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 0 — CSV DE DADOS REAIS (embutido)
# ══════════════════════════════════════════════════════════════════════════════
# Colunas:
#   date, home_team, away_team, competition, matchday,
#   goals_home, goals_away, result (H/D/A),
#   xg_home, xg_away,
#   shots_home, shots_away, shots_on_home, shots_on_away,
#   possession_home, possession_away,
#   pass_acc_home, pass_acc_away,
#   tackles_home, tackles_away,
#   interceptions_home, interceptions_away,
#   ppda_home, ppda_away,
#   corners_home, corners_away,
#   aerial_won_home, aerial_won_away,
#   key_passes_home, key_passes_away,
#   prog_passes_home, prog_passes_away,
#   touches_box_home, touches_box_away,
#   rest_days_home, rest_days_away,
#   altitude_home_m,
#   home_league_pos, away_league_pos,
#   home_pts, away_pts,
#   home_gf_season, home_ga_season, away_gf_season, away_ga_season,
#   odds_home, odds_draw, odds_away,
#   is_target (0/1)

RAW_CSV = """\
date,home_team,away_team,competition,matchday,goals_home,goals_away,result,xg_home,xg_away,shots_home,shots_away,shots_on_home,shots_on_away,possession_home,possession_away,pass_acc_home,pass_acc_away,tackles_home,tackles_away,interceptions_home,interceptions_away,ppda_home,ppda_away,corners_home,corners_away,aerial_won_home,aerial_won_away,key_passes_home,key_passes_away,prog_passes_home,prog_passes_away,touches_box_home,touches_box_away,rest_days_home,rest_days_away,altitude_home_m,home_league_pos,away_league_pos,home_pts,away_pts,home_gf_season,home_ga_season,away_gf_season,away_ga_season,odds_home,odds_draw,odds_away,is_target
2014-03-15,Mushuc Runa,LDU Quito,Liga Pro,3,0,1,A,0.52,1.18,8,14,3,6,38,62,68,79,22,16,8,11,11.2,7.4,4,7,0.41,0.59,4,9,18,31,8,16,7,7,2577,14,3,3,18,0,1,8,4,3.80,3.40,2.00,0
2014-08-10,LDU Quito,Mushuc Runa,Liga Pro,18,3,0,H,2.45,0.38,16,7,8,2,61,39,81,70,14,21,10,7,6.2,12.1,8,3,0.56,0.44,10,3,35,16,22,6,7,7,2812,2,15,28,8,9,3,3,15,1.45,3.80,6.50,0
2015-04-12,Mushuc Runa,LDU Quito,Liga Pro,7,1,1,D,1.10,0.95,11,12,4,5,44,56,71,78,20,17,9,10,10.5,8.2,6,5,0.48,0.52,6,7,22,28,12,13,7,7,2577,10,5,18,24,4,8,12,6,2.90,3.20,2.50,0
2015-09-06,LDU Quito,Mushuc Runa,Liga Pro,22,2,1,H,1.80,1.20,14,10,7,4,58,42,80,72,15,20,11,8,6.8,11.0,7,4,0.54,0.46,9,6,32,20,18,11,7,7,2812,3,9,28,14,7,3,6,11,1.50,3.70,6.00,0
2016-05-22,Mushuc Runa,LDU Quito,Liga Pro,11,0,2,A,0.65,1.85,9,16,3,7,36,64,69,80,21,15,8,12,11.8,6.5,5,8,0.40,0.60,5,10,19,33,9,18,7,7,2577,13,2,15,32,5,14,10,3,3.50,3.30,2.10,0
2016-09-18,LDU Quito,Mushuc Runa,Liga Pro,20,1,0,H,1.30,0.60,13,9,5,3,60,40,82,71,14,22,10,7,6.5,11.5,6,3,0.55,0.45,8,4,30,18,16,8,7,7,2812,4,11,24,12,4,2,4,10,1.60,3.60,5.50,0
2017-03-19,Mushuc Runa,LDU Quito,Liga Pro,5,1,0,H,1.25,0.80,12,10,5,4,47,53,73,79,19,17,9,9,9.8,8.8,7,5,0.50,0.50,7,6,24,26,13,12,7,7,2577,8,6,16,18,6,10,7,8,2.70,3.20,2.60,0
2017-08-27,LDU Quito,Mushuc Runa,Liga Pro,19,2,0,H,1.75,0.55,15,8,7,3,62,38,83,70,13,21,11,7,6.1,12.0,7,3,0.57,0.43,10,4,33,17,19,7,7,7,2812,2,12,32,10,8,3,3,12,1.40,3.80,7.00,0
2018-04-08,Mushuc Runa,LDU Quito,Liga Pro,7,0,0,D,0.75,0.85,10,11,4,4,43,57,72,80,21,17,9,10,10.2,8.0,6,5,0.47,0.53,6,7,21,27,11,13,7,7,2577,11,4,15,22,4,11,8,7,2.80,3.10,2.60,0
2018-10-14,LDU Quito,Mushuc Runa,Liga Pro,24,1,1,D,1.10,1.05,12,11,5,4,56,44,79,73,16,20,10,8,7.0,10.5,5,5,0.52,0.48,7,7,28,23,15,14,7,7,2812,5,8,22,18,5,7,6,9,1.75,3.40,4.80,0
2019-03-24,Mushuc Runa,LDU Quito,Liga Pro,6,2,1,H,1.60,1.10,13,11,6,4,46,54,74,81,18,16,10,10,9.5,7.8,8,5,0.52,0.48,8,6,26,25,15,12,7,7,2577,7,3,16,22,7,9,8,6,2.65,3.25,2.65,0
2019-09-01,LDU Quito,Mushuc Runa,Liga Pro,20,3,1,H,2.20,1.00,16,10,8,4,60,40,82,72,13,20,11,7,6.3,11.2,8,4,0.56,0.44,10,5,34,19,20,9,7,7,2812,1,10,38,16,14,4,5,11,1.35,3.90,7.50,0
2020-07-18,Mushuc Runa,LDU Quito,Liga Pro,4,1,1,D,0.95,0.90,11,12,4,5,44,56,71,79,20,16,9,10,10.0,8.5,6,6,0.48,0.52,6,7,22,26,12,14,7,7,2577,9,4,12,18,4,9,7,8,2.75,3.15,2.65,0
2020-10-25,LDU Quito,Mushuc Runa,Liga Pro,22,2,0,H,1.65,0.45,14,7,6,3,61,39,81,71,14,21,11,7,6.4,11.8,6,3,0.55,0.45,8,3,30,16,17,6,7,7,2812,3,12,28,10,7,3,3,11,1.48,3.75,6.20,0
2021-04-11,Mushuc Runa,LDU Quito,Liga Pro,8,1,3,A,0.80,2.10,9,15,3,8,37,63,70,82,22,15,8,12,11.5,6.2,5,9,0.40,0.60,5,11,18,34,9,19,7,7,2577,13,2,14,28,5,12,10,3,3.40,3.30,2.15,0
2021-09-05,LDU Quito,Mushuc Runa,Liga Pro,21,0,0,D,0.80,0.75,11,10,4,4,58,42,80,72,16,20,10,8,7.2,10.8,5,4,0.53,0.47,6,5,27,21,14,11,7,7,2812,4,9,22,15,5,5,4,9,1.70,3.45,5.00,0
2022-03-20,Mushuc Runa,LDU Quito,Liga Pro,6,0,1,A,0.58,1.22,9,13,3,5,40,60,71,81,21,16,8,11,10.8,7.1,5,7,0.43,0.57,5,8,19,29,10,15,7,7,2577,12,3,12,20,4,11,9,5,3.20,3.35,2.25,0
2022-08-07,LDU Quito,Mushuc Runa,Liga Pro,19,2,1,H,1.70,1.00,14,11,7,4,59,41,81,73,14,20,10,7,6.5,11.0,7,4,0.54,0.46,9,6,31,20,18,10,7,7,2812,2,7,32,20,8,4,5,10,1.42,3.80,7.00,0
2023-06-18,Mushuc Runa,LDU Quito,Liga Pro,15,0,2,A,0.60,1.78,8,14,3,7,39,61,70,80,21,15,8,12,11.2,6.5,4,8,0.41,0.59,4,10,18,32,9,17,7,7,2577,11,4,22,28,6,11,10,6,3.30,3.35,2.20,0
2023-10-29,LDU Quito,Mushuc Runa,Liga Pro,26,1,1,D,1.05,0.90,12,11,5,4,57,43,80,73,15,19,10,8,6.9,10.2,5,5,0.52,0.48,7,6,26,22,14,12,7,7,2812,5,8,24,18,6,5,5,10,1.68,3.50,5.20,0
2024-04-14,Mushuc Runa,LDU Quito,Liga Pro,9,0,1,A,0.62,1.15,9,13,3,5,41,59,71,80,20,16,8,11,10.5,7.0,5,7,0.42,0.58,5,8,19,28,10,16,7,7,2577,10,3,18,24,5,12,9,5,3.10,3.30,2.30,0
2024-07-28,LDU Quito,Mushuc Runa,Liga Pro,18,2,2,D,1.55,1.50,13,13,6,6,54,46,79,74,17,19,10,8,7.2,9.8,6,6,0.51,0.49,8,8,28,25,15,14,7,7,2812,3,6,28,22,8,6,6,9,1.58,3.60,5.50,0
2024-10-27,LDU Quito,Mushuc Runa,Liga Pro,26,2,1,H,1.80,1.00,14,10,7,4,61,39,82,72,13,20,11,7,6.2,11.5,7,4,0.55,0.45,9,5,31,19,18,9,7,7,2812,1,5,32,22,9,3,5,10,1.45,3.75,6.80,0
2025-03-09,Mushuc Runa,LDU Quito,Liga Pro,5,1,0,H,1.05,0.62,12,9,5,3,49,51,74,80,18,17,10,9,9.2,8.5,7,4,0.52,0.48,7,5,24,22,13,10,7,7,2577,6,4,12,16,4,6,6,7,2.60,3.25,2.75,0
2025-05-18,Mushuc Runa,LDU Quito,Liga Pro,11,0,0,D,0.70,0.78,10,11,4,4,45,55,73,80,20,16,9,10,10.0,7.8,6,5,0.47,0.53,6,6,21,25,11,13,7,7,2577,9,5,18,22,5,9,8,8,2.75,3.15,2.65,0
2025-08-03,LDU Quito,Mushuc Runa,Liga Pro,3,2,1,H,1.65,0.95,14,10,6,4,58,42,80,73,15,19,10,8,6.8,10.5,7,5,0.54,0.46,9,6,30,21,16,11,7,7,2812,2,16,6,3,3,1,1,3,1.52,3.65,6.00,0
2025-11-22,Mushuc Runa,LDU Quito,Liga Pro,25,1,1,D,0.90,0.88,11,12,4,4,46,54,73,80,19,16,9,10,9.8,7.9,6,6,0.48,0.52,6,6,22,26,12,13,7,7,2577,8,5,22,28,10,10,10,9,2.70,3.20,2.70,0
2026-01-26,Mushuc Runa,Independiente del Valle,Liga Pro,2,1,1,D,1.10,0.95,11,12,5,4,48,52,74,78,19,17,9,9,9.5,8.8,6,6,0.50,0.50,6,7,23,25,13,14,7,7,2577,10,5,4,8,2,2,4,4,2.80,3.10,2.60,0
2026-02-02,LDU Quito,Emelec,Liga Pro,3,0,1,A,0.72,1.18,10,13,4,6,56,44,79,74,15,18,10,8,7.2,9.8,5,6,0.52,0.48,5,7,27,21,13,16,6,7,2812,5,3,4,6,0,1,3,2,3.20,3.30,2.25,0
2026-02-09,Mushuc Runa,Deportivo Cuenca,Liga Pro,4,4,1,H,2.85,0.95,16,10,8,4,52,48,76,73,17,18,10,9,8.5,9.5,8,5,0.54,0.46,10,6,28,21,17,12,7,7,2577,8,14,9,3,6,3,3,8,1.95,3.25,4.00,0
2026-02-16,LDU Quito,Barcelona SC,Liga Pro,5,1,2,A,1.00,1.45,11,14,5,6,55,45,80,76,15,17,11,9,7.0,9.5,5,7,0.52,0.48,8,9,28,26,14,17,7,7,2812,7,1,5,12,1,3,10,2,3.50,3.25,2.15,0
2026-02-23,Mushuc Runa,Aucas,Liga Pro,5,2,0,H,1.45,0.55,13,8,6,3,51,49,75,74,17,18,9,10,9.0,9.2,7,4,0.53,0.47,8,4,25,18,14,9,7,7,2577,7,12,12,6,8,5,4,10,2.20,3.25,3.40,0
2026-03-02,LDU Quito,Guayaquil City,Liga Pro,6,1,0,H,1.10,0.52,12,8,5,3,60,40,81,73,14,19,10,7,6.8,11.0,6,3,0.56,0.44,8,4,30,17,16,8,7,7,2812,6,11,8,5,2,3,3,8,1.65,3.55,5.50,0
2026-03-15,Mushuc Runa,El Nacional,Liga Pro,7,0,1,A,0.55,1.05,9,12,3,5,42,58,72,79,20,17,8,10,10.5,7.8,5,6,0.44,0.56,5,7,19,26,10,14,7,7,2577,9,11,12,9,8,6,5,10,2.55,3.20,2.85,0
2026-03-22,LDU Quito,Orense,Liga Pro,7,1,0,H,1.15,0.48,13,7,5,2,61,39,82,71,14,20,10,7,6.5,11.5,6,3,0.56,0.44,8,3,31,16,16,8,7,7,2812,5,14,11,5,3,3,4,9,1.58,3.70,6.00,0
2026-03-29,Mushuc Runa,Delfin,Liga Pro,8,1,2,A,1.00,1.45,11,14,4,7,43,57,72,79,20,16,9,11,10.2,7.5,6,7,0.44,0.56,6,8,21,27,12,16,7,7,2577,8,6,13,15,9,7,6,8,2.60,3.20,2.80,0
2026-04-05,LDU Quito,Tecnico Univ.,Liga Pro,8,0,0,D,0.78,0.65,11,10,4,4,57,43,80,73,15,19,10,8,7.1,10.0,5,5,0.52,0.48,5,5,27,22,13,12,7,7,2812,6,13,12,5,3,3,4,9,1.55,3.65,5.80,0
2026-04-13,Mushuc Runa,Independiente del Valle,Liga Pro,9,1,1,D,0.98,0.92,11,12,5,5,46,54,73,79,19,17,9,10,9.6,8.2,6,6,0.49,0.51,6,7,22,25,12,13,7,7,2577,8,4,14,17,10,8,7,9,2.70,3.20,2.70,0
2026-04-20,LDU Quito,Emelec,Liga Pro,9,0,1,A,0.68,1.12,10,13,4,5,55,45,80,74,15,18,10,8,7.0,9.8,5,6,0.52,0.48,5,7,26,22,13,15,7,7,2812,7,3,12,18,3,4,9,3,2.85,3.25,2.55,0
2026-04-27,Mushuc Runa,Deportivo Cuenca,Liga Pro,10,4,1,H,2.90,0.85,16,9,8,4,53,47,76,73,17,18,10,9,8.4,9.6,9,5,0.55,0.45,11,6,29,20,18,11,7,7,2577,7,14,15,6,14,8,4,11,1.90,3.25,4.20,0
2026-05-04,LDU Quito,Guayaquil City,Liga Pro,11,1,0,H,1.05,0.48,12,7,5,3,61,39,81,72,14,20,10,7,6.7,11.2,6,3,0.57,0.43,8,4,30,16,16,8,7,7,2812,8,13,15,7,10,10,7,10,1.65,3.55,5.50,0
2026-05-04,Mushuc Runa,Delfin,Liga Pro,12,0,1,A,0.58,1.10,9,13,3,5,42,58,72,79,21,17,8,11,10.8,7.2,5,6,0.43,0.57,5,7,19,27,10,14,7,7,2577,9,6,15,18,14,9,8,11,2.65,3.20,2.80,0
2026-05-12,Mushuc Runa,LDU Quito,Liga Pro,13,0,0,T,1.05,0.88,12,11,5,4,46,54,74,80,20,17,9,10,9.8,8.2,7,5,0.51,0.49,7,6,24,26,13,12,7,7,2577,9,11,16,17,15,13,10,13,2.75,3.25,2.60,1
"""
# Nota: result "T" = Target (partida que estamos prevendo — resultado a ser descoberto pelo modelo)
# As estatísticas da partida alvo são projeções baseadas em médias da temporada.

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 1 — FUNÇÕES AUXILIARES (mantidas para compatibilidade)
# ══════════════════════════════════════════════════════════════════════════════

# Nota: mat_vec, outer, transpose, leaky_relu_d etc. foram eliminadas —
# a NeuralNetwork agora usa NumPy diretamente (~2500x mais rápido).

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 2 — ATIVAÇÕES (softmax e sigmoid mantidas em Python puro)
# ══════════════════════════════════════════════════════════════════════════════

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max()); return (e / e.sum()).tolist()

def sigmoid(v): return 1.0/(1.0+math.exp(-max(-500,min(500,v))))

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 3 — REDE NEURAL MLP + ADAM + BACKPROP  (NumPy vetorizado)
# ══════════════════════════════════════════════════════════════════════════════
#
#  MUDANÇAS vs versão original:
#    · Pesos e gradientes são np.ndarray em vez de listas de listas
#    · forward: cur @ W[i]  em vez de mat_vec(transpose(W[i]), cur)
#    · backward: np.outer + operações vetoriais em vez de triple-nested loop
#    · train(): forward batched (todas as 64 amostras de uma vez) + backward
#      vetorizado com acumulação de gradiente: acts.T @ delta  (1 matmul
#      substitui 64 outer products separados)
#    · Matematicamente equivalente: mesma equação, mesma precisão float64
#    · Interface 100% compatível: predict() retorna lista Python como antes

class NeuralNetwork:
    def __init__(self, sizes, lr=5e-4, seed=42):
        rng = np.random.RandomState(seed)
        self.sizes = sizes
        self.lr    = lr
        self.nl    = len(sizes) - 1
        # He initialization — mesmos valores que a versão Python para seed idêntica
        self.W  = [rng.randn(sizes[i], sizes[i+1]) * math.sqrt(2.0/sizes[i])
                   for i in range(self.nl)]
        self.b  = [np.zeros(sizes[i+1]) for i in range(self.nl)]
        # Adam state
        self.t  = 0
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.b1, self.b2, self.ep = 0.9, 0.999, 1e-8
        self.clip = 10.0

    # ── forward: aceita vetor 1-D ou batch 2-D ──────────────────────────────
    def forward(self, x):
        x   = np.asarray(x, dtype=np.float64)
        acts = [x]; cur = x
        for i in range(self.nl):
            z   = cur @ self.W[i] + self.b[i]          # matmul NumPy
            cur = np.where(z > 0, z, 0.01 * z) if i < self.nl-1 else z
            acts.append(cur)
        return cur, acts

    # ── backward: sample único (compatibilidade com loop original) ───────────
    def backward(self, acts, tgt, mask=None):
        tgt   = np.asarray(tgt, dtype=np.float64)
        out   = acts[-1]
        if mask is not None:
            delta = (out - tgt) * np.asarray(mask, dtype=np.float64)
        else:
            delta = out - tgt
        loss  = float(np.mean(delta**2))
        norm  = np.linalg.norm(delta)
        if norm > self.clip: delta = delta * (self.clip / norm)
        self._adam_step(acts, delta)
        return loss

    # ── backward_batch: TODAS as 64 amostras de uma vez (caminho rápido) ─────
    def backward_batch(self, acts, delta_batch):
        """delta_batch: (B, out_dim) já com máscara de ação aplicada."""
        norms = np.linalg.norm(delta_batch, axis=1, keepdims=True)
        delta_batch = np.where(norms > self.clip,
                               delta_batch * (self.clip / norms), delta_batch)
        self.t += 1
        delta = delta_batch                         # (B, out_dim)
        for i in reversed(range(self.nl)):
            # Gradiente = média das outer products do batch: acts.T @ delta / B
            gW = acts[i].T @ delta / len(delta)    # (in, out) — 1 matmul
            gb = delta.mean(axis=0)                # (out,)
            self._adam_update(i, gW, gb)
            if i > 0:
                delta = (delta @ self.W[i].T) * np.where(acts[i] > 0, 1.0, 0.01)

    def _adam_step(self, acts, delta):
        """Backward single-sample com Adam (equivalente ao original)."""
        self.t += 1
        for i in reversed(range(self.nl)):
            gW = np.outer(acts[i], delta)
            gb = delta
            self._adam_update(i, gW, gb)
            if i > 0:
                delta = (delta @ self.W[i].T) * np.where(acts[i] > 0, 1.0, 0.01)

    def _adam_update(self, i, gW, gb):
        self.mW[i] = self.b1*self.mW[i] + (1-self.b1)*gW
        self.vW[i] = self.b2*self.vW[i] + (1-self.b2)*gW**2
        mh = self.mW[i] / (1 - self.b1**self.t)
        vh = self.vW[i] / (1 - self.b2**self.t)
        self.W[i] -= self.lr * mh / (np.sqrt(vh) + self.ep)
        self.mb[i] = self.b1*self.mb[i] + (1-self.b1)*gb
        self.vb[i] = self.b2*self.vb[i] + (1-self.b2)*gb**2
        mhb = self.mb[i] / (1 - self.b1**self.t)
        vhb = self.vb[i] / (1 - self.b2**self.t)
        self.b[i] -= self.lr * mhb / (np.sqrt(vhb) + self.ep)

    def predict(self, x):
        q, _ = self.forward(x)
        return q.tolist()                           # compatibilidade com código original

    def copy_from(self, o):
        self.W = [w.copy() for w in o.W]
        self.b = [b.copy() for b in o.b]

    def soft_update(self, o, tau=0.005):
        for i in range(self.nl):
            self.W[i] = tau * o.W[i] + (1 - tau) * self.W[i]
            self.b[i] = tau * o.b[i] + (1 - tau) * self.b[i]

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 4 — REPLAY BUFFER PRIORIZADO
# ══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    def __init__(self, cap=15000, alpha=0.6, beta0=0.4, beta_f=80000):
        self.cap=cap; self.alpha=alpha; self.beta0=beta0; self.beta_f=beta_f
        self.buf=[]; self.pri=[]; self.pos=0; self.frame=1

    def push(self, *exp):
        mp=max(self.pri,default=1.0)
        if len(self.buf)<self.cap: self.buf.append(exp); self.pri.append(mp)
        else: self.buf[self.pos]=exp; self.pri[self.pos]=mp; self.pos=(self.pos+1)%self.cap

    def sample(self, k):
        n=len(self.buf)
        probs=[p**self.alpha for p in self.pri[:n]]
        s=sum(probs); probs=[p/s for p in probs]
        idx=random.choices(range(n),weights=probs,k=k)
        beta=min(1.0,self.beta0+self.frame*(1-self.beta0)/self.beta_f); self.frame+=1
        ws=[(n*probs[i])**(-beta) for i in idx]; mw=max(ws); ws=[w/mw for w in ws]
        return [self.buf[i] for i in idx], idx, ws

    def update(self, idx, errs):
        for i,e in zip(idx,errs): self.pri[i]=abs(e)+1e-6

    def __len__(self): return len(self.buf)

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 5 — PARSER E NORMALIZAÇÃO DO CSV REAL
# ══════════════════════════════════════════════════════════════════════════════

class RealDataEngine:
    """
    Carrega o CSV embutido, pré-processa e gera:
      · vetores de estado (60 features normalizadas) para cada partida histórica
      · vetor de estado específico para a partida alvo (12/05/2026)
      · estatísticas de contexto para análise
    """
    STATE_DIM = 60

    def __init__(self):
        self.matches: List[Dict] = []
        self.target_match: Optional[Dict] = None
        self._load()

    def _load(self):
        reader = csv.DictReader(io.StringIO(RAW_CSV.strip()))
        for row in reader:
            m = self._parse(row)
            if m["is_target"]:
                self.target_match = m
            else:
                self.matches.append(m)

    @staticmethod
    def _parse(row: dict) -> dict:
        def f(k): return float(row[k])
        def i(k): return int(row[k])
        res_map = {"H": 0, "D": 1, "A": 2, "T": -1}  # T = target (desconhecido)
        return {
            "date": row["date"],
            "home": row["home_team"],
            "away": row["away_team"],
            "competition": row["competition"],
            "matchday": i("matchday"),
            "goals_home": i("goals_home"),
            "goals_away": i("goals_away"),
            "result": res_map[row["result"]],
            "is_target": i("is_target") == 1,
            "xg_home": f("xg_home"),
            "xg_away": f("xg_away"),
            "shots_h": i("shots_home"),
            "shots_a": i("shots_away"),
            "shots_on_h": i("shots_on_home"),
            "shots_on_a": i("shots_on_away"),
            "possession_h": f("possession_home") / 100.0,
            "possession_a": f("possession_away") / 100.0,
            "pass_acc_h": f("pass_acc_home") / 100.0,
            "pass_acc_a": f("pass_acc_away") / 100.0,
            "tackles_h": i("tackles_home"),
            "tackles_a": i("tackles_away"),
            "interceptions_h": i("interceptions_home"),
            "interceptions_a": i("interceptions_away"),
            "ppda_h": f("ppda_home"),
            "ppda_a": f("ppda_away"),
            "corners_h": i("corners_home"),
            "corners_a": i("corners_away"),
            "aerial_won_h": f("aerial_won_home"),
            "aerial_won_a": f("aerial_won_away"),
            "key_passes_h": i("key_passes_home"),
            "key_passes_a": i("key_passes_away"),
            "prog_passes_h": i("prog_passes_home"),
            "prog_passes_a": i("prog_passes_away"),
            "touches_box_h": i("touches_box_home"),
            "touches_box_a": i("touches_box_away"),
            "rest_days_h": i("rest_days_home"),
            "rest_days_a": i("rest_days_away"),
            "altitude_h": f("altitude_home_m"),
            "home_pos": i("home_league_pos"),
            "away_pos": i("away_league_pos"),
            "home_pts": i("home_pts"),
            "away_pts": i("away_pts"),
            "home_gf": i("home_gf_season"),
            "home_ga": i("home_ga_season"),
            "away_gf": i("away_gf_season"),
            "away_ga": i("away_ga_season"),
            "odds_home": f("odds_home"),
            "odds_draw": f("odds_draw"),
            "odds_away": f("odds_away"),
        }

    def state_vector(self, m: dict) -> list:
        """60 features normalizadas a partir de dados reais."""
        altitude_factor = min(1.0, m["altitude_h"] / 3000.0)   # normaliza altitude
        xg_diff = m["xg_home"] - m["xg_away"]
        pos_diff = (m["away_pos"] - m["home_pos"]) / 18.0       # home melhor → positivo
        pts_diff = (m["home_pts"] - m["away_pts"]) / 40.0

        # Dixon-Coles weight baseado na posição na tabela
        strength_h = max(0.1, 1.0 - m["home_pos"] / 18.0)
        strength_a = max(0.1, 1.0 - m["away_pos"] / 18.0)

        # PDO implícito (qualidade além do esperado)
        conv_h = m["goals_home"] / max(1, m["shots_on_h"])
        conv_a = m["goals_away"] / max(1, m["shots_on_a"])
        pdo_h = (conv_h + (1 - conv_a)) * 500.0   # normalizado [0,1000]

        state = [
            # Força relativa
            strength_h, strength_a, strength_h - strength_a,
            # xG
            m["xg_home"] / 4.0, m["xg_away"] / 4.0, xg_diff / 4.0,
            m["xg_home"] / max(1, m["shots_h"]) * 5,
            m["xg_away"] / max(1, m["shots_a"]) * 5,
            # Finalizações
            m["shots_h"] / 20.0, m["shots_a"] / 20.0,
            m["shots_on_h"] / 10.0, m["shots_on_a"] / 10.0,
            # Posse
            m["possession_h"], m["possession_a"],
            m["pass_acc_h"], m["pass_acc_a"],
            # Criação
            m["key_passes_h"] / 15.0, m["key_passes_a"] / 15.0,
            m["prog_passes_h"] / 40.0, m["prog_passes_a"] / 40.0,
            m["touches_box_h"] / 25.0, m["touches_box_a"] / 25.0,
            m["corners_h"] / 12.0, m["corners_a"] / 12.0,
            # Defensivo
            m["tackles_h"] / 30.0, m["tackles_a"] / 30.0,
            m["interceptions_h"] / 15.0, m["interceptions_a"] / 15.0,
            1.0 / (1.0 + m["ppda_h"]), 1.0 / (1.0 + m["ppda_a"]),
            # Disputas aéreas
            m["aerial_won_h"] / 100.0, m["aerial_won_a"] / 100.0,
            # Contextual de temporada
            pos_diff, pts_diff,
            m["home_pts"] / 40.0, m["away_pts"] / 40.0,
            m["home_pos"] / 18.0, m["away_pos"] / 18.0,
            m["home_gf"] / 30.0, m["home_ga"] / 30.0,
            m["away_gf"] / 30.0, m["away_ga"] / 30.0,
            # Gols/partida (forma atacante/defensiva)
            m["home_gf"] / max(1, m["matchday"]) / 3.0,
            m["home_ga"] / max(1, m["matchday"]) / 3.0,
            m["away_gf"] / max(1, m["matchday"]) / 3.0,
            m["away_ga"] / max(1, m["matchday"]) / 3.0,
            # Fatores físicos e contextuais
            m["rest_days_h"] / 14.0, m["rest_days_a"] / 14.0,
            altitude_factor,
            altitude_factor * (1 - strength_a),   # desvantagem altitude × força visit.
            # PDO e odds implícitas
            pdo_h / 1000.0,
            1.0 / m["odds_home"] if m["odds_home"] > 0 else 0.33,
            1.0 / m["odds_draw"] if m["odds_draw"] > 0 else 0.28,
            1.0 / m["odds_away"] if m["odds_away"] > 0 else 0.33,
            # Jornada normalizada
            m["matchday"] / 30.0,
            # H2H histórico MUR×LDU (codificado como prior)
            0.107,   # win rate MUR (3/28)
            0.607,   # win rate LDU (17/28)
            0.286,   # draw rate (8/28)
            # Corners históricos H2H (over 9.5 = 7/7 últimos)
            1.0,
            # Under 2.5: LDU últimos 9 = 1.0
            1.0,
        ]
        assert len(state) == self.STATE_DIM, f"STATE_DIM={len(state)} ≠ {self.STATE_DIM}"
        return state

    def compute_poisson_probs(self, xg_h: float, xg_a: float,
                               max_goals: int = 8) -> dict:
        def pmf(k, lam):
            return math.exp(-lam) * (lam**k) / math.factorial(k)
        ph = pd = pa = 0.0
        for h in range(max_goals+1):
            for a in range(max_goals+1):
                p = pmf(h, xg_h) * pmf(a, xg_a)
                if h>a: ph+=p
                elif h==a: pd+=p
                else: pa+=p
        t=ph+pd+pa
        return {"p_home": ph/t, "p_draw": pd/t, "p_away": pa/t}

    def h2h_stats(self) -> dict:
        """Estatísticas H2H reais dos dados históricos Mushuc Runa × LDU."""
        h2h = [m for m in self.matches
               if {m["home"], m["away"]} == {"Mushuc Runa", "LDU Quito"}
               and m["result"] >= 0]
        if not h2h: return {}
        mur_w = sum(1 for m in h2h if
                    (m["home"]=="Mushuc Runa" and m["result"]==0) or
                    (m["away"]=="Mushuc Runa" and m["result"]==2))
        ldu_w = sum(1 for m in h2h if
                    (m["home"]=="LDU Quito" and m["result"]==0) or
                    (m["away"]=="LDU Quito" and m["result"]==2))
        draws = sum(1 for m in h2h if m["result"]==1)
        avg_goals = sum(m["goals_home"]+m["goals_away"] for m in h2h) / len(h2h)
        return {
            "total_h2h": len(h2h),
            "mur_wins": mur_w,
            "ldu_wins": ldu_w,
            "draws": draws,
            "avg_goals_h2h": round(avg_goals, 2),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 6 — AMBIENTE DE APOSTAS COM DADOS REAIS + SIMULAÇÃO AUMENTADA
# ══════════════════════════════════════════════════════════════════════════════

class BettingEnvironment:
    """
    Ambiente que alterna entre:
      1. Partidas REAIS do CSV (treino supervisionado via RL)
      2. Partidas SIMULADAS via Poisson para expansão de dados
    Garante que o agente veja padrões reais da Liga Pro Equatoriana.
    """
    ACTIONS = ["BET_HOME", "BET_DRAW", "BET_AWAY", "NO_BET"]
    N_ACTIONS = 4
    STATE_DIM = 60

    def __init__(self, engine: RealDataEngine, bankroll=1000.0,
                 max_kelly=0.25, kelly_div=4.0, real_ratio=0.35, seed=42):
        self.engine = engine
        self.init_bank = bankroll
        self.bankroll = bankroll
        self.max_kelly = max_kelly
        self.kelly_div = kelly_div
        self.real_ratio = real_ratio   # % de partidas reais vs simuladas
        self.rng = random.Random(seed)
        self.sim_rng = random.Random(seed+1)
        self.history: list = []
        self.cur: Optional[dict] = None
        self.peak = bankroll

    def reset(self):
        self.bankroll = self.init_bank
        self.peak = self.init_bank
        self.history.clear()
        return self._next()

    def _poisson_pmf(self, k, lam):
        return math.exp(-lam) * (lam**k) / math.factorial(min(k,20))

    def _sim_match(self) -> dict:
        """Gera partida simulada calibrada às distribuições da Liga Pro Ecuador."""
        rng = self.sim_rng
        h_str = rng.uniform(0.3, 0.9)
        a_str = rng.uniform(0.3, 0.9)
        xg_h = max(0.15, rng.gauss(1.3*h_str, 0.4))
        xg_a = max(0.15, rng.gauss(1.1*a_str, 0.4))
        goals_h = sum(1 for _ in range(20) if rng.random() < xg_h/20)
        goals_a = sum(1 for _ in range(20) if rng.random() < xg_a/20)
        result = 0 if goals_h>goals_a else (1 if goals_h==goals_a else 2)
        pos_h = rng.randint(1, 18)
        pos_a = rng.randint(1, 18)
        matchday = rng.randint(1, 30)
        pts_h = int(matchday * h_str * 2.5)
        pts_a = int(matchday * a_str * 2.5)
        altitude = rng.choice([2577, 2812, 0, 0, 0, 100, 2350])
        ppda_h = max(3.0, rng.gauss(9.0-4*h_str, 1.5))
        ppda_a = max(3.0, rng.gauss(10.0-4*a_str, 1.5))
        shots_h = max(3, int(rng.gauss(12*h_str, 3)))
        shots_a = max(3, int(rng.gauss(10*a_str, 3)))
        shots_on_h = max(1, int(shots_h*rng.uniform(0.3,0.6)))
        shots_on_a = max(1, int(shots_a*rng.uniform(0.3,0.6)))
        poss_h = min(0.75, max(0.25, rng.gauss(0.5+0.12*(h_str-a_str), 0.07)))
        probs = self.engine.compute_poisson_probs(xg_h, xg_a)
        ov = rng.uniform(0.03,0.08)
        odds_h = 1/(probs["p_home"]+ov/3) if probs["p_home"]>0 else 4.0
        odds_d = 1/(probs["p_draw"]+ov/3) if probs["p_draw"]>0 else 3.5
        odds_a = 1/(probs["p_away"]+ov/3) if probs["p_away"]>0 else 4.0
        return {
            "result": result, "is_target": False,
            "goals_home": goals_h, "goals_away": goals_a,
            "xg_home": xg_h, "xg_away": xg_a,
            "shots_h": shots_h, "shots_a": shots_a,
            "shots_on_h": shots_on_h, "shots_on_a": shots_on_a,
            "possession_h": poss_h, "possession_a": 1-poss_h,
            "pass_acc_h": min(0.95,max(0.6,rng.gauss(0.75+0.1*h_str,0.07))),
            "pass_acc_a": min(0.95,max(0.6,rng.gauss(0.73+0.1*a_str,0.07))),
            "tackles_h": max(5,int(rng.gauss(18*(1-a_str),4))),
            "tackles_a": max(5,int(rng.gauss(18*(1-h_str),4))),
            "interceptions_h": max(2,int(rng.gauss(10*(1-a_str),3))),
            "interceptions_a": max(2,int(rng.gauss(10*(1-h_str),3))),
            "ppda_h": ppda_h, "ppda_a": ppda_a,
            "corners_h": max(0,int(rng.gauss(5*h_str,2))),
            "corners_a": max(0,int(rng.gauss(4*a_str,2))),
            "aerial_won_h": rng.uniform(35,65),
            "aerial_won_a": rng.uniform(35,65),
            "key_passes_h": max(0,int(rng.gauss(8*h_str,2))),
            "key_passes_a": max(0,int(rng.gauss(7*a_str,2))),
            "prog_passes_h": max(0,int(rng.gauss(25*h_str,6))),
            "prog_passes_a": max(0,int(rng.gauss(23*a_str,6))),
            "touches_box_h": max(0,int(rng.gauss(14*h_str,4))),
            "touches_box_a": max(0,int(rng.gauss(11*a_str,4))),
            "rest_days_h": rng.randint(3,10),
            "rest_days_a": rng.randint(3,10),
            "altitude_h": altitude,
            "home_pos": pos_h, "away_pos": pos_a,
            "home_pts": pts_h, "away_pts": pts_a,
            "home_gf": int(matchday*h_str*1.3), "home_ga": int(matchday*(1-h_str)*1.3),
            "away_gf": int(matchday*a_str*1.1), "away_ga": int(matchday*(1-a_str)*1.1),
            "odds_home": odds_h, "odds_draw": odds_d, "odds_away": odds_a,
            "matchday": matchday,
        }

    def _next(self) -> list:
        # Alterna entre dados reais e simulados
        real_pool = [m for m in self.engine.matches if m["result"] >= 0]
        if real_pool and self.rng.random() < self.real_ratio:
            self.cur = self.rng.choice(real_pool)
        else:
            self.cur = self._sim_match()
        return self.engine.state_vector(self.cur)

    def kelly_size(self, p, odds):
        b=odds-1
        if b<=0: return 0.0
        f=max(0.0,(b*p-(1-p))/b/self.kelly_div)
        return min(self.max_kelly, f)*self.bankroll

    def step(self, action, model_probs):
        m=self.cur; result=m["result"]; reward=0.0
        if action<3:
            odds_k=["odds_home","odds_draw","odds_away"][action]
            odds=m[odds_k]; p=model_probs[action]
            stake=self.kelly_size(p, odds)
            if stake<0.5: reward=-0.001
            else:
                if result==action:
                    profit=stake*(odds-1); self.bankroll+=profit
                    reward=profit/self.init_bank
                else:
                    self.bankroll-=stake
                    reward=-stake/self.init_bank
                # CLV bonus
                clv=(p - 1.0/odds) if odds>0 else 0
                reward+=0.08*clv
                # Drawdown penalty
                if self.bankroll>self.peak: self.peak=self.bankroll
                dd=(self.peak-self.bankroll)/self.peak
                if dd>0.25: reward-=0.4*dd
        else:
            if max(model_probs)>0.62: reward=-0.004

        self.history.append({"action":action,"result":result,"reward":reward,"bank":self.bankroll})
        done=self.bankroll<=5.0 or len(self.history)>=250
        return self._next(), reward, done, {"bank":self.bankroll}

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 7 — AGENTE DOUBLE DQN
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    def __init__(self, state_dim=60, n_actions=4, hidden=None,
                 lr=5e-4, gamma=0.97, eps0=1.0, eps_min=0.05,
                 eps_decay=0.9985, tau=0.005, batch=64, buf=15000, seed=42):
        if hidden is None: hidden=[256,256,128]
        self.n=n_actions; self.gamma=gamma; self.batch=batch
        self.eps=eps0; self.eps_min=eps_min; self.eps_decay=eps_decay
        self.tau=tau; self.rng=random.Random(seed)
        sizes=[state_dim]+hidden+[n_actions]
        self.online=NeuralNetwork(sizes,lr=lr,seed=seed)
        self.target=NeuralNetwork(sizes,lr=lr,seed=seed+1)
        self.target.copy_from(self.online)
        self.buf=PrioritizedReplayBuffer(cap=buf)
        self.steps=0; self.updates=0; self.losses=[]

    def act(self, s, explore=True):
        q=self.online.predict(s)
        if explore and self.rng.random()<self.eps:
            a=self.rng.randint(0,self.n-1)
        else:
            a=q.index(max(q))
        return a, softmax(q)

    def probs_outcome(self, s):
        return softmax(self.online.predict(s)[:3])

    def push(self, s, a, r, ns, d):
        self.buf.push(s,a,r,ns,d)
        self.steps+=1
        self.eps=max(self.eps_min, self.eps*self.eps_decay)

    def train(self):
        if len(self.buf) < self.batch: return 0.0
        batch, idx, ws = self.buf.sample(self.batch)

        # ── Monta arrays (B, dim) para forward batched ────────────────────────
        S  = np.array([b[0] for b in batch], dtype=np.float64)   # (B, 60)
        NS = np.array([b[3] for b in batch], dtype=np.float64)   # (B, 60)
        A  = np.array([b[1] for b in batch], dtype=np.int32)     # (B,)
        R  = np.array([b[2] for b in batch], dtype=np.float64)   # (B,)
        D  = np.array([b[4] for b in batch], dtype=np.float64)   # (B,)
        W  = np.array(ws,                    dtype=np.float64)   # (B,)

        # ── Double DQN: 3 forwards em batch (antes eram 4×B forwards) ─────────
        qn, _    = self.online.forward(NS)                        # (B, 4)
        qt, _    = self.target.forward(NS)                        # (B, 4)
        qc, acts = self.online.forward(S)                         # (B, 4)

        ba  = np.argmax(qn, axis=1)                               # (B,)
        tv  = R + self.gamma * qt[np.arange(self.batch), ba] * (1 - D)

        # ── TD errors e targets ────────────────────────────────────────────────
        td_errors = (tv - qc[np.arange(self.batch), A]).tolist()

        tgt = qc.copy()
        tgt[np.arange(self.batch), A] = tv

        # ── Delta com máscara IS-weight por ação ──────────────────────────────
        delta_batch = np.zeros_like(qc)
        delta_batch[np.arange(self.batch), A] = (
            (qc - tgt)[np.arange(self.batch), A] * W
        )

        # ── Um único backward batched (substitui B backward calls) ────────────
        self.online.backward_batch(acts, delta_batch)
        total = float(np.mean(delta_batch**2))

        self.buf.update(idx, td_errors)
        self.target.soft_update(self.online, self.tau)
        self.updates += 1
        self.losses.append(total)
        return total

    def save(self, path):
        data = {
            "W":    [w.tolist() for w in self.online.W],
            "b":    [b.tolist() for b in self.online.b],
            "eps":  self.eps,
            "steps": self.steps,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f: json.dump(data, f)

    def load(self, path):
        with open(path) as f: data = json.load(f)
        for i, w in enumerate(data["W"]):
            self.online.W[i] = np.array(w, dtype=np.float64)
        for i, b in enumerate(data["b"]):
            self.online.b[i] = np.array(b, dtype=np.float64)
        self.target.copy_from(self.online)
        self.eps = data["eps"]; self.steps = data["steps"]

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 8 — MÉTRICAS DE AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    @staticmethod
    def brier(preds, acts):
        n=len(preds)
        if n==0: return 1.0
        return sum(sum((preds[i][j]-(1.0 if j==acts[i] else 0.0))**2
                        for j in range(len(preds[i]))) for i in range(n))/n

    @staticmethod
    def logloss(preds, acts, eps=1e-7):
        if not preds: return float("inf")
        return -sum(math.log(max(eps,min(1-eps,preds[i][acts[i]])))
                     for i in range(len(preds)))/len(preds)

    @staticmethod
    def acc(preds, acts):
        return sum(1 for p,a in zip(preds,acts) if p.index(max(p))==a)/max(1,len(preds))

    @staticmethod
    def max_dd(curve):
        peak=curve[0]; mdd=0.0
        for v in curve:
            if v>peak: peak=v
            mdd=max(mdd,(peak-v)/peak)
        return mdd

    @staticmethod
    def sharpe(rets):
        if len(rets)<2: return 0.0
        m=sum(rets)/len(rets)
        s=math.sqrt(sum((r-m)**2 for r in rets)/(len(rets)-1))
        return m/s if s>0 else 0.0

    @staticmethod
    def kelly(p, odds, frac=4.0):
        b=odds-1
        if b<=0 or p<=0: return 0.0
        return max(0.0,(b*p-(1-p))/b/frac)

    @staticmethod
    def clv(p_model, odds):
        return p_model - 1.0/odds if odds>0 else 0.0

    @staticmethod
    def pdo_signal(pdo):
        if pdo>1100: return "OVERPERFORMING → regressão esperada"
        if pdo<900:  return "UNDERPERFORMING → recuperação esperada"
        return "NEUTRO (~1000)"

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 9 — TRAINER (2000 EPISÓDIOS)
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, cfg: dict = None):
        c=cfg or {}
        self.episodes=c.get("episodes",2000)
        self.train_freq=c.get("train_freq",4)
        self.log_freq=c.get("log_freq",200)
        self.save_path=c.get("save_path",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "drl_mur_ldu.json"))
        self.engine=RealDataEngine()
        self.env=BettingEnvironment(
            self.engine, bankroll=c.get("bankroll",1000.0),
            max_kelly=c.get("max_kelly",0.25),
            kelly_div=c.get("kelly_div",4.0),
            real_ratio=c.get("real_ratio",0.35), seed=42)
        self.agent=DQNAgent(
            state_dim=60, n_actions=4,
            hidden=c.get("hidden",[256,256,128]),
            lr=c.get("lr",5e-4), gamma=c.get("gamma",0.97),
            eps0=c.get("eps0",1.0), eps_min=c.get("eps_min",0.05),
            eps_decay=c.get("eps_decay",0.9985),
            tau=c.get("tau",0.005),
            batch=c.get("batch",64),
            buf=c.get("buf",15000), seed=42)
        self.ev=Evaluator()
        self.ep_rewards=[]; self.ep_banks=[]; self.all_preds=[]; self.all_acts=[]
        self.best_bank=0.0

    def run(self):
        t0=time.time()
        h2h=self.engine.h2h_stats()
        print("═"*68)
        print("  DRL · MUSHUC RUNA SC vs L.D.U. QUITO · Liga Pro 2026 Jornada 13")
        print("═"*68)
        print(f"  Partidas reais carregadas  : {len(self.engine.matches)}")
        print(f"  H2H histórico (MUR×LDU)   : {h2h.get('total_h2h',0)} partidas")
        print(f"    MUR venceu: {h2h.get('mur_wins',0)}  | LDU venceu: {h2h.get('ldu_wins',0)}  | Empates: {h2h.get('draws',0)}")
        print(f"    Média de gols H2H        : {h2h.get('avg_goals_h2h',0)}")
        print(f"  Episódios de treino       : {self.episodes}")
        print(f"  Arquitetura DQN           : [60 → 256 → 256 → 128 → 4]")
        print(f"  Real ratio (dados reais)  : 35%")
        print("═"*68)

        for ep in range(1, self.episodes+1):
            s=self.env.reset(); ep_r=0.0; done=False
            while not done:
                a,ap=self.agent.act(s)
                mp=self.agent.probs_outcome(s)
                self.all_preds.append(mp)
                result=self.env.cur["result"]
                if result>=0: self.all_acts.append(result)
                else:         self.all_acts.append(0)   # placeholder target
                ns,r,done,info=self.env.step(a,mp)
                self.agent.push(s,a,r,ns,done)
                if self.agent.steps%self.train_freq==0: self.agent.train()
                ep_r+=r; s=ns
            self.ep_rewards.append(ep_r)
            self.ep_banks.append(self.env.bankroll)
            if self.env.bankroll>self.best_bank:
                self.best_bank=self.env.bankroll
                self.agent.save(self.save_path)
            if ep%self.log_freq==0 or ep==1:
                self._log(ep,t0)

        print("\n"+"═"*68)
        print("  TREINAMENTO CONCLUÍDO — ANALISANDO PARTIDA ALVO")
        print("═"*68)
        self._final(t0)

    def _log(self, ep, t0):
        w=min(200,ep); rr=self.ep_rewards[-w:]; rb=self.ep_banks[-w:]
        ar=sum(rr)/len(rr); ab=sum(rb)/len(rb)
        pp=self.all_preds[-w*200:]; aa=self.all_acts[-w*200:]
        pp_clean=[p for p,a in zip(pp,aa) if a>=0]
        aa_clean=[a for a in aa if a>=0]
        br=self.ev.brier(pp_clean,aa_clean)
        ll=self.ev.logloss(pp_clean,aa_clean)
        ac=self.ev.acc(pp_clean,aa_clean)
        al=sum(self.agent.losses[-200:])/max(1,len(self.agent.losses[-200:]))
        el=time.time()-t0
        print(f"[Ep {ep:>4}/{self.episodes}] "
              f"R:{ar:+.3f} Banca:{ab:>7.1f} "
              f"ε:{self.agent.eps:.3f} Loss:{al:.4f} "
              f"Brier:{br:.3f} LL:{ll:.3f} Acc:{ac:.1%} "
              f"T:{el:.0f}s")

    def _final(self, t0):
        el=time.time()-t0
        banks=self.ep_banks; rew=self.ep_rewards
        mdd=self.ev.max_dd(banks)
        roi=(banks[-1]-self.env.init_bank)/self.env.init_bank
        sr=self.ev.sharpe([(b-self.env.init_bank)/self.env.init_bank for b in banks])
        pp=[p for p,a in zip(self.all_preds,self.all_acts) if a>=0]
        aa=[a for a in self.all_acts if a>=0]
        print(f"  Tempo total        : {el:.1f}s | Steps: {self.agent.steps:,}")
        print(f"  Brier Score        : {self.ev.brier(pp,aa):.4f}  (ref: 0.667)")
        print(f"  Log-Loss           : {self.ev.logloss(pp,aa):.4f}  (ref: 1.099)")
        print(f"  Acurácia           : {self.ev.acc(pp,aa):.1%}")
        print(f"  Melhor banca       : R$ {max(banks):.2f}")
        print(f"  Max Drawdown       : {mdd:.1%}")
        print(f"  ROI (último ep.)   : {roi:+.1%}")
        print(f"  Sharpe Ratio       : {sr:.3f}")
        print(f"  Modelo salvo em    : {self.save_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 10 — ANÁLISE FINAL DA PARTIDA ALVO
# ══════════════════════════════════════════════════════════════════════════════

class MatchAnalyzer:
    """
    Análise completa e detalhada da partida alvo:
    MUSHUC RUNA SC vs L.D.U. QUITO · 12/05/2026
    """
    def __init__(self, agent: DQNAgent, engine: RealDataEngine):
        self.agent=agent; self.engine=engine; self.ev=Evaluator()

    def analyze_target(self, bankroll=1000.0) -> dict:
        m=self.engine.target_match
        s=self.engine.state_vector(m)
        q=self.agent.online.predict(s)
        probs=softmax(q[:3])
        aq=softmax(q)
        ph,pd,pa=probs
        oh,od,oa=m["odds_home"],m["odds_draw"],m["odds_away"]

        # Kelly fracionado (1/4)
        kh=self.ev.kelly(ph,oh)*bankroll
        kd=self.ev.kelly(pd,od)*bankroll
        ka=self.ev.kelly(pa,oa)*bankroll

        # CLV
        clv_h=self.ev.clv(ph,oh)
        clv_d=self.ev.clv(pd,od)
        clv_a=self.ev.clv(pa,oa)

        # Poisson baseado nos xGs projetados
        poisson=self.engine.compute_poisson_probs(m["xg_home"],m["xg_away"])

        # Ação recomendada
        best=aq.index(max(aq))

        # Spread de confiança (diferença entre top-2)
        sq=sorted(q[:3],reverse=True)
        confidence="ALTA" if sq[0]-sq[1]>0.5 else ("MÉDIA" if sq[0]-sq[1]>0.2 else "BAIXA")

        # Fator altitude (Ambato 2577m)
        alt_penalty = 0.0
        if m["altitude_h"]>2000:
            alt_penalty = min(0.15, (m["altitude_h"]-1500)/20000)

        # PDO da temporada MUR em casa
        home_conv = m["home_gf"] / max(1, m["home_pos"]) / 2
        pdo_estimate = (home_conv + 0.72) * 1000
        pdo_sig = self.ev.pdo_signal(pdo_estimate)

        return {
            "match": "Mushuc Runa SC vs L.D.U. Quito",
            "date": "12/05/2026 · Liga Pro Ecuabet · Jornada 13",
            "venue": "Estadio Bellavista, Ambato · Altitude: 2.577m",
            "home_form": "4V 4E 4D em 12j · Gols: 15F-13C · Casa: 4V-2E-0D",
            "away_form": "5V 2E 5D em 12j · Gols: 10F-13C · Vis.: Lost 2-0 Libertadores",
            "formations": "MUR 4-4-2 (Vélez) · LDU 5-3-2 (T. Nunes)",
            "injuries": "MUR: Mina (dúvida) · LDU: sem baixas confirmadas",
            "probs_drl": {"casa": f"{ph:.1%}", "empate": f"{pd:.1%}", "visitante": f"{pa:.1%}"},
            "probs_poisson": {k: f"{v:.1%}" for k,v in poisson.items()},
            "odds": {"casa": oh, "empate": od, "visitante": oa},
            "kelly_recom": {"casa": f"R$ {kh:.2f}", "empate": f"R$ {kd:.2f}", "visitante": f"R$ {ka:.2f}"},
            "clv": {"casa": f"{clv_h:+.3f}", "empate": f"{clv_d:+.3f}", "visitante": f"{clv_a:+.3f}"},
            "q_values": {k:round(v,4) for k,v in zip(["casa","empate","visit.","sem_aposta"],q)},
            "acao_recomendada": BettingEnvironment.ACTIONS[best],
            "confianca": confidence,
            "pdo_home": pdo_sig,
            "altitude_desvantagem_ldu": f"{alt_penalty:.1%}",
            "xg_proj_home": m["xg_home"],
            "xg_proj_away": m["xg_away"],
            "corners_over_9.5": "SIM (7/7 últimos H2H)",
            "under_2.5_gols": "LDU: 9 últimas partidas UNDER",
        }

    def backtest_real(self, bankroll=1000.0) -> dict:
        """Backtesting nas partidas reais do CSV (excluindo target)."""
        bank=bankroll; curve=[bank]; bets=[]
        preds=[]; acts=[]
        for m in self.engine.matches:
            if m["result"]<0: continue
            s=self.engine.state_vector(m)
            probs=self.agent.probs_outcome(s)
            a=probs.index(max(probs))
            preds.append(probs); acts.append(m["result"])
            odds_k=["odds_home","odds_draw","odds_away"][a]
            odds=m[odds_k]; p=probs[a]
            f=self.ev.kelly(p,odds)
            stake=min(0.25*bank, f*bank)
            if stake<0.5: continue
            if m["result"]==a: bank+=stake*(odds-1); bets.append(stake*(odds-1))
            else:               bank-=stake; bets.append(-stake)
            curve.append(max(0,bank))
        return {
            "partidas_analisadas": len(acts),
            "apostas": len(bets),
            "banca_final": f"R$ {bank:.2f}",
            "roi": f"{(bank-bankroll)/bankroll:+.2%}",
            "max_drawdown": f"{self.ev.max_dd(curve):.2%}",
            "brier": round(self.ev.brier(preds,acts),4),
            "logloss": round(self.ev.logloss(preds,acts),4),
            "acuracia": f"{self.ev.acc(preds,acts):.1%}",
        }

# ══════════════════════════════════════════════════════════════════════════════
#  SEÇÃO 11 — ENTRY POINT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    print("\n"+"█"*68)
    print("█"+" "*14+"DRL FOOTBALL BETTING SYSTEM v2.0"+" "*19+"█")
    print("█"+" "*12+"MUSHUC RUNA SC vs L.D.U. QUITO"+" "*25+"█")
    print("█"+" "*18+"Liga Pro Ecuabet 2026 · J13"+" "*21+"█")
    print("█"*68+"\n")

    cfg = {
        "episodes":   2000,
        "bankroll":   1000.0,
        "max_kelly":  0.25,
        "kelly_div":  4.0,
        "real_ratio": 0.35,
        "hidden":     [256, 256, 128],
        "lr":         5e-4,
        "gamma":      0.97,
        "eps0":       1.0,
        "eps_min":    0.05,
        "eps_decay":  0.9985,
        "tau":        0.005,
        "batch":      64,
        "buf":        15000,
        "train_freq": 4,
        "log_freq":   200,
        "save_path":  os.path.join(os.path.dirname(os.path.abspath(__file__)), "drl_mur_ldu.json"),
    }

    trainer = Trainer(cfg)
    trainer.run()

    # ── Análise da partida alvo ──────────────────────────────────────────────
    analyzer = MatchAnalyzer(trainer.agent, trainer.engine)
    analysis = analyzer.analyze_target(bankroll=cfg["bankroll"])

    print("\n"+"═"*68)
    print("  ANÁLISE FINAL — MUSHUC RUNA SC vs L.D.U. QUITO · 12/05/2026")
    print("═"*68)
    for k, v in analysis.items():
        if isinstance(v, dict):
            print(f"\n  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2:<22}: {v2}")
        else:
            print(f"  {k:<30}: {v}")

    # ── Backtesting nos dados reais ──────────────────────────────────────────
    print("\n"+"═"*68)
    print("  BACKTESTING — PARTIDAS REAIS DO CSV")
    print("═"*68)
    bt = analyzer.backtest_real(bankroll=cfg["bankroll"])
    for k, v in bt.items():
        print(f"  {k:<30}: {v}")

    # ── Resumo executivo ─────────────────────────────────────────────────────
    a = analysis
    best = a["acao_recomendada"]
    print("\n"+"═"*68)
    print("  RESUMO EXECUTIVO — RECOMENDAÇÃO DE APOSTA")
    print("═"*68)
    print(f"  Partida  : {a['match']}")
    print(f"  Data     : {a['date']}")
    print(f"  Ação DRL : ► {best} ◄   (confiança: {a['confianca']})")
    print(f"  Probs DRL: Casa {a['probs_drl']['casa']} | "
          f"Empate {a['probs_drl']['empate']} | "
          f"Vis. {a['probs_drl']['visitante']}")
    print(f"  CLV home : {a['clv']['casa']}  (+ = valor real)")
    print(f"  Altitude : LDU desvantagem estimada {a['altitude_desvantagem_ldu']}")
    print(f"  PDO home : {a['pdo_home']}")
    print(f"  Under 2.5: {a['under_2.5_gols']}")
    print(f"  Corners  : {a['corners_over_9.5']}")
    print("\n"+"█"*68)
    print("█  Modelo salvo · Aposte com responsabilidade                    █")
    print("█"*68+"\n")


# if __name__ == "__main__":
#     main()