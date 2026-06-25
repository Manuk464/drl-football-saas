import os
import json
import hashlib
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, JSON, Integer, Boolean, select
from contextlib import asynccontextmanager

# ==============================================================================
# 1. BANCO DE DADOS
# ==============================================================================
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    plan: Mapped[str] = mapped_column(String, default="free")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class MatchPrediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(String, index=True)
    teams: Mapped[str] = mapped_column(String)
    raw_input: Mapped[dict] = mapped_column(JSON)
    state_vector: Mapped[list] = mapped_column(JSON)
    q_values: Mapped[list] = mapped_column(JSON)
    probs: Mapped[dict] = mapped_column(JSON)
    recommendation: Mapped[str] = mapped_column(String)
    confidence: Mapped[str] = mapped_column(String)
    kelly_fractions: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

engine = create_async_engine("sqlite+aiosqlite:///./v44_database.db", echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ==============================================================================
# 2. FUNÇÃO DE HASH DE SENHA
# ==============================================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ==============================================================================
# 3. SCHEMAS
# ==============================================================================
class RegisterInput(BaseModel):
    username: str
    email: str
    password: str

class LoginInput(BaseModel):
    username: str
    password: str

class MatchInput(BaseModel):
    date: str
    home_team: str
    away_team: str
    competition: str = "Liga Pro"
    matchday: int
    xg_home: float
    xg_away: float
    shots_home: int
    shots_away: int
    shots_on_home: int
    shots_on_away: int
    possession_home: float
    possession_away: float
    pass_acc_home: float
    pass_acc_away: float
    tackles_home: int
    tackles_away: int
    interceptions_home: int
    interceptions_away: int
    ppda_home: float
    ppda_away: float
    corners_home: int
    corners_away: int
    aerial_won_home: float
    aerial_won_away: float
    key_passes_home: int
    key_passes_away: int
    prog_passes_home: int
    prog_passes_away: int
    touches_box_home: int
    touches_box_away: int
    rest_days_home: int
    rest_days_away: int
    altitude_home_m: float
    home_league_pos: int
    away_league_pos: int
    home_pts: int
    away_pts: int
    home_gf_season: int
    home_ga_season: int
    away_gf_season: int
    away_ga_season: int
    odds_home: float
    odds_draw: float
    odds_away: float

# ==============================================================================
# 4. MOTOR DE INFERÊNCIA
# ==============================================================================
from app.core.core_ai import RealDataEngine, DQNAgent, softmax, Evaluator

class InferenceEngine:
    def __init__(self, model_path: str):
        self.engine = RealDataEngine()
        self.agent = DQNAgent(state_dim=60, n_actions=4, hidden=[256, 256, 128])
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)
            print(f"Modelo v44 carregado de {self.model_path}")
        else:
            print(f"Modelo nao encontrado em {self.model_path}. Usando pesos aleatorios.")

    def predict(self, data: dict) -> dict:
        parsed_data = self._parse_input(data)
        state = self.engine.state_vector(parsed_data)
        q_values = self.agent.online.predict(state)
        probs = softmax(q_values[:3])
        
        oh = data["odds_home"]
        od = data["odds_draw"]
        oa = data["odds_away"]
        ph, pd, pa = probs
        
        kelly = lambda p, o: max(0, (o-1)*p - (1-p)) / (o-1) / 4.0 if o > 1 else 0
        clv = lambda p, o: p - (1/o) if o > 0 else 0
        
        actions = ["BET_HOME", "BET_DRAW", "BET_AWAY", "NO_BET"]
        best_idx = int(np.argmax(q_values))
        rec = actions[best_idx]
        
        sq = sorted(q_values[:3], reverse=True)
        conf = "ALTA" if sq[0]-sq[1] > 0.5 else ("MEDIA" if sq[0]-sq[1] > 0.2 else "BAIXA")
        
        return {
            "state_vector": state,
            "q_values": q_values,
            "probs": {"home": ph, "draw": pd, "away": pa},
            "recommendation": rec,
            "confidence": conf,
            "kelly": {"home": kelly(ph, oh), "draw": kelly(pd, od), "away": kelly(pa, oa)},
            "clv": {"home": clv(ph, oh), "draw": clv(pd, od), "away": clv(pa, oa)}
        }

    def _parse_input(self, d: dict) -> dict:
        return {
            "date": d["date"], "home": d["home_team"], "away": d["away_team"],
            "competition": d["competition"], "matchday": d["matchday"],
            "goals_home": 0, "goals_away": 0, "result": -1, "is_target": True,
            "xg_home": d["xg_home"], "xg_away": d["xg_away"],
            "shots_h": d["shots_home"], "shots_a": d["shots_away"],
            "shots_on_h": d["shots_on_home"], "shots_on_a": d["shots_on_away"],
            "possession_h": d["possession_home"] / 100.0, "possession_a": d["possession_away"] / 100.0,
            "pass_acc_h": d["pass_acc_home"] / 100.0, "pass_acc_a": d["pass_acc_away"] / 100.0,
            "tackles_h": d["tackles_home"], "tackles_a": d["tackles_away"],
            "interceptions_h": d["interceptions_home"], "interceptions_a": d["interceptions_away"],
            "ppda_h": d["ppda_home"], "ppda_a": d["ppda_away"],
            "corners_h": d["corners_home"], "corners_a": d["corners_away"],
            "aerial_won_h": d["aerial_won_home"], "aerial_won_a": d["aerial_won_away"],
            "key_passes_h": d["key_passes_home"], "key_passes_a": d["key_passes_away"],
            "prog_passes_h": d["prog_passes_home"], "prog_passes_a": d["prog_passes_away"],
            "touches_box_h": d["touches_box_home"], "touches_box_a": d["touches_box_away"],
            "rest_days_h": d["rest_days_home"], "rest_days_a": d["rest_days_away"],
            "altitude_h": d["altitude_home_m"],
            "home_pos": d["home_league_pos"], "away_pos": d["away_league_pos"],
            "home_pts": d["home_pts"], "away_pts": d["away_pts"],
            "home_gf": d["home_gf_season"], "home_ga": d["home_ga_season"],
            "away_gf": d["away_gf_season"], "away_ga": d["away_ga_season"],
            "odds_home": d["odds_home"], "odds_draw": d["odds_draw"], "odds_away": d["odds_away"],
        }

# ==============================================================================
# 5. FASTAPI APP
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.ai = InferenceEngine(model_path="./models/drl_v44_real_weights.json")
    yield

app = FastAPI(title="DRL Football API v44", version="4.4.0", lifespan=lifespan)

# CORS liberado para Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.streamlit\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==============================================================================
# 6. ROTAS DE AUTENTICAÇÃO
# ==============================================================================
@app.post("/register")
async def register(data: RegisterInput):
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(User).where(User.username == data.username)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Usuario ja existe.")
        
        user = User(
            username=data.username,
            email=data.email,
            password_hash=hash_password(data.password),
            plan="free"
        )
        session.add(user)
        await session.commit()
        return {"message": "Conta criada com sucesso!", "plan": "free"}

@app.post("/login")
async def login(data: LoginInput):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.username == data.username)
        )
        user = result.scalar_one_or_none()
        
        if not user or user.password_hash != hash_password(data.password):
            raise HTTPException(status_code=401, detail="Usuario ou senha incorretos.")
        
        return {
            "message": "Login realizado!",
            "username": user.username,
            "plan": user.plan,
            "user_id": user.id
        }

@app.post("/upgrade/{username}")
async def upgrade_to_vip(username: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario nao encontrado.")
        
        user.plan = "vip"
        await session.commit()
        return {"message": f"Usuario {username} atualizado para VIP!", "plan": "vip"}

# ==============================================================================
# 7. ROTAS DE PREDIÇÃO E HISTÓRICO
# ==============================================================================
@app.post("/predict")
async def predict_match(data: MatchInput):
    try:
        ai = app.state.ai
        result = ai.predict(data.model_dump())
        
        async with AsyncSessionLocal() as session:
            pred = MatchPrediction(
                match_id=f"{data.date}_{data.home_team}_{data.away_team}",
                teams=f"{data.home_team} vs {data.away_team}",
                raw_input=data.model_dump(),
                state_vector=result["state_vector"],
                q_values=result["q_values"],
                probs=result["probs"],
                recommendation=result["recommendation"],
                confidence=result["confidence"],
                kelly_fractions=result["kelly"]
            )
            session.add(pred)
            await session.commit()
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(limit: int = 20):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(MatchPrediction).order_by(MatchPrediction.id.desc()).limit(limit)
        )
        predictions = result.scalars().all()
        
        return [
            {
                "id": p.id,
                "match_id": p.match_id,
                "teams": p.teams,
                "recommendation": p.recommendation,
                "confidence": p.confidence,
                "probs": p.probs,
                "kelly": p.kelly_fractions,
                "created_at": p.created_at.isoformat() if p.created_at else None
            }
            for p in predictions
        ]

@app.get("/")
def root():
    return {"message": "API DRL v44 Online!"}

# ==============================================================================
# 8. ROTA DE BACKTEST (Transparência)
# ==============================================================================
@app.get("/backtest")
async def get_backtest_stats():
    try:
        ai: InferenceEngine = app.state.ai
        bank = 1000.0
        init_bank = bank
        curve = [bank]
        preds, acts = [], []
        
        for m in ai.engine.matches:
            if m["result"] < 0:
                continue
            
            s = ai.engine.state_vector(m)
            probs = ai.agent.probs_outcome(s)
            
            a = probs.index(max(probs))
            preds.append(probs)
            acts.append(m["result"])
            
            odds_k = ["odds_home", "odds_draw", "odds_away"][a]
            odds = m[odds_k]
            p = probs[a]
            
            b = odds - 1
            f = max(0, (b * p - (1 - p)) / b / 4.0) if b > 0 else 0
            stake = min(0.25 * bank, f * bank)
            
            if stake < 0.5:
                curve.append(bank)
                continue
                
            if m["result"] == a:
                bank += stake * (odds - 1)
            else:
                bank -= stake
            curve.append(max(0, bank))
            
        ev = Evaluator()
        return {
            "initial_bank": init_bank,
            "final_bank": round(bank, 2),
            "roi": f"{((bank - init_bank) / init_bank) * 100:+.2f}%",
            "total_bets": len([c for c in curve if c != curve[0]]),
            "max_drawdown": f"{ev.max_dd(curve) * 100:.2f}%",
            "brier_score": f"{ev.brier(preds, acts):.4f}",
            "log_loss": f"{ev.logloss(preds, acts):.4f}",
            "accuracy": f"{ev.acc(preds, acts) * 100:.1f}%",
            "sharpe_ratio": f"{ev.sharpe([(c - init_bank) / init_bank for c in curve]):.3f}",
            "equity_curve": curve
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))