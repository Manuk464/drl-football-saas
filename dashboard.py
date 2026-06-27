import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os
import time

st.set_page_config(page_title="DRL Football AI v44", page_icon="🧠", layout="wide")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

def api_request_with_retry(method, url, **kwargs):
    timeout = kwargs.pop('timeout', 30)
    for attempt in range(3):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=timeout, **kwargs)
            else:
                resp = requests.post(url, timeout=timeout, **kwargs)
            return resp
        except requests.exceptions.Timeout:
            if attempt < 2:
                st.warning(f"Servidor acordando... Tentativa {attempt + 1}/3")
                time.sleep(5)
            else:
                raise
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                st.warning(f"Reconectando... Tentativa {attempt + 1}/3")
                time.sleep(5)
            else:
                raise

def login_page():
    st.title("🧠 DRL Football AI")
    st.markdown("### Acesse sua conta para ver os palpites da IA")
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔑 Entrar", "📝 Criar Conta"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Utilizador", key="login_user")
        password = st.text_input("Senha", type="password", key="login_pass")
        if st.button("Entrar", type="primary"):
            try:
                resp = api_request_with_retry("POST", f"{API_URL}/login", json={"username": username, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = data['username']
                    st.session_state['plan'] = data['plan']
                    st.success(f"Bem-vindo, {data['username']}! Plano: {data['plan'].upper()}")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Erro no login."))
            except Exception as e:
                st.error(f"Erro de conexão: {e}")
    
    with tab2:
        st.subheader("Criar Conta Grátis")
        new_user = st.text_input("Utilizador", key="reg_user")
        new_email = st.text_input("Email", key="reg_email")
        new_pass = st.text_input("Senha", type="password", key="reg_pass")
        if st.button("Criar Conta", type="primary"):
            try:
                resp = api_request_with_retry("POST", f"{API_URL}/register", json={"username": new_user, "email": new_email, "password": new_pass})
                if resp.status_code == 200:
                    st.success("Conta criada! Faça login na aba ao lado.")
                else:
                    st.error(resp.json().get("detail", "Erro ao criar conta."))
            except Exception as e:
                st.error(f"Erro de conexão: {e}")

def dashboard():
    username = st.session_state.get('username', 'User')
    plan = st.session_state.get('plan', 'free')
    is_vip = plan == "vip"
    
    st.sidebar.markdown(f"### Olá, **{username}**")
    st.sidebar.markdown(f"Plano: **{'⭐ VIP' if is_vip else '🆓 FREE'}**")
    
    if st.sidebar.button("Sair"):
        st.session_state.clear()
        st.rerun()
    
    if not is_vip:
        st.sidebar.warning("Está no plano FREE. Faça upgrade para ver todos os palpites!")
        if st.sidebar.button("⭐ Fazer Upgrade para VIP"):
            try:
                resp = api_request_with_retry("POST", f"{API_URL}/upgrade/{username}")
                if resp.status_code == 200:
                    st.session_state['plan'] = "vip"
                    st.success("Upgrade realizado! Agora é VIP!")
                    st.rerun()
            except:
                st.error("Erro ao fazer upgrade.")
    
    st.sidebar.markdown("---")
    menu_options = ["🏆 Palpites do Dia", "📊 Transparência & ROI"]
    if is_vip:
        menu_options += ["🔮 Análise Manual", "💰 Simulador de Banca", "📜 Histórico"]
    menu = st.sidebar.selectbox("Menu", menu_options)

    if menu == "🏆 Palpites do Dia":
        st.title("🏆 Palpites do Dia")
        limit = 3 if is_vip else 1
        st.caption(f"Top {limit} palpites. {'(VIP)' if is_vip else '(FREE - Upgrade para ver todos)'}")
        
        if st.button("🔄 Buscar Jogos de Hoje", type="primary"):
            from daily_pipeline import get_today_fixtures, estimate_prematch_features
            with st.status("Processando jogos reais...", expanded=True) as status:
                tips = []
                fixtures = get_today_fixtures()
                st.write(f"📊 Analisando {len(fixtures)} jogos...")
                
                for i, fix in enumerate(fixtures):
                    home = fix.get("home_team", fix.get("home", "Unknown"))
                    away = fix.get("away_team", fix.get("away", "Unknown"))
                    league = fix.get("competition", fix.get("league", ""))
                    st.write(f"⚽ **{home}** vs **{away}** ({league})")
                    
                    features = estimate_prematch_features(fix)
                    try:
                        resp = api_request_with_retry("POST", f"{API_URL}/predict", json=features)
                        if resp.status_code == 200:
                            res = resp.json()
                            tips.append({
                                "match": f"{home} vs {away}",
                                "league": league,
                                "rec": res["recommendation"],
                                "conf": res["confidence"],
                                "probs": res["probs"],
                                "kelly": res["kelly"],
                                "clv": res["clv"],
                                "max_clv": max(res["clv"].values()),
                                "odds": {
                                    "home": features.get("odds_home", 2.5),
                                    "draw": features.get("odds_draw", 3.2),
                                    "away": features.get("odds_away", 2.8)
                                }
                            })
                    except Exception as e:
                        st.error(f"Erro ao processar {home}: {e}")
                
                status.update(label=f"✅ {len(tips)} palpites gerados!", state="complete", expanded=False)
                st.session_state['tips'] = sorted(tips, key=lambda x: x["max_clv"], reverse=True)[:limit]
        
        if 'tips' in st.session_state:
            for i, tip in enumerate(st.session_state['tips']):
                st.markdown(f"### {i+1}. {tip['match']}")
                st.caption(f"🏆 {tip['league']}")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("🎯 Ação", tip['rec'])
                with c2:
                    st.metric("📈 Edge (CLV)", f"{tip['max_clv']:+.2%}")
                with c3:
                    st.metric("🎲 Confiança", tip['conf'])
                with c4:
                    odd_media = (tip['odds']['home'] + tip['odds']['draw'] + tip['odds']['away']) / 3
                    st.metric("💰 Odd Média", f"{odd_media:.2f}")
                
                st.caption(
                    f"📊 **Probabilidades IA:** "
                    f"Casa {tip['probs']['home']:.1%} | "
                    f"Empate {tip['probs']['draw']:.1%} | "
                    f"Fora {tip['probs']['away']:.1%}"
                )
                st.caption(
                    f"💵 **Odds de Mercado:** "
                    f"Casa {tip['odds']['home']:.2f} | "
                    f"Empate {tip['odds']['draw']:.2f} | "
                    f"Fora {tip['odds']['away']:.2f}"
                )
                st.markdown("---")
            
            if not is_vip:
                st.info("💡 Está a ver apenas o Top 1. Faça upgrade VIP para ver Top 3 + Simulador de Banca!")

    elif menu == "📊 Transparência & ROI":
        st.title("📊 Auditoria e Transparência do Modelo DRL")
        st.markdown("**Histórico real** da IA sobre partidas passadas. Apenas matemática e backtesting.")
        
        if st.button("🔄 Rodar Backtesting", type="primary"):
            with st.spinner("Processando histórico..."):
                try:
                    resp = api_request_with_retry("GET", f"{API_URL}/backtest", timeout=60)
                    if resp.status_code == 200:
                        st.session_state['backtest'] = resp.json()
                    else:
                        st.error("Erro ao buscar backtest.")
                except Exception as e:
                    st.error(f"Erro de conexão: {e}")
        
        if 'backtest' in st.session_state:
            bt = st.session_state['backtest']
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("💰 ROI Total", bt['roi'])
                st.metric("🎯 Acurácia", bt['accuracy'])
            with c2:
                st.metric("📉 Max Drawdown", bt['max_drawdown'])
                st.metric("📊 Brier Score", bt['brier_score'])
            with c3:
                st.metric("📈 Sharpe Ratio", bt['sharpe_ratio'])
                st.metric("🔢 Log-Loss", bt['log_loss'])
            with c4:
                st.metric("💵 Banca Final", f"R$ {bt['final_bank']}")
                st.metric("🎲 Total Apostas", bt['total_bets'])
            
            st.markdown("---")
            st.subheader("📈 Curva de Capital (Equity Curve)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=bt['equity_curve'], mode='lines',
                name='Banca (R$)', fill='tozeroy',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.update_layout(
                xaxis_title="Sequência de Partidas",
                yaxis_title="Banca (R$)",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if not is_vip:
                st.success("🔥 **Gostou dos resultados?** Faça upgrade para o plano VIP!")

    elif menu == "🔮 Análise Manual":
        st.title("🔮 Análise Manual")
        if st.button("Carregar MUR vs LDU"):
            st.session_state['match_data'] = {
                "date": "2026-05-12", "home_team": "Mushuc Runa", "away_team": "LDU Quito",
                "competition": "Liga Pro", "matchday": 13,
                "xg_home": 1.05, "xg_away": 0.88, "shots_home": 12, "shots_away": 11,
                "shots_on_home": 5, "shots_on_away": 4, "possession_home": 46, "possession_away": 54,
                "pass_acc_home": 74, "pass_acc_away": 80, "tackles_home": 20, "tackles_away": 17,
                "interceptions_home": 9, "interceptions_away": 10, "ppda_home": 9.8, "ppda_away": 8.2,
                "corners_home": 7, "corners_away": 5, "aerial_won_home": 51, "aerial_won_away": 49,
                "key_passes_home": 7, "key_passes_away": 6, "prog_passes_home": 24, "prog_passes_away": 26,
                "touches_box_home": 13, "touches_box_away": 12, "rest_days_home": 7, "rest_days_away": 7,
                "altitude_home_m": 2577, "home_league_pos": 9, "away_league_pos": 11,
                "home_pts": 16, "away_pts": 17, "home_gf_season": 15, "home_ga_season": 13,
                "away_gf_season": 10, "away_ga_season": 13,
                "odds_home": 2.75, "odds_draw": 3.25, "odds_away": 2.60
            }
        
        if 'match_data' in st.session_state:
            if st.button("Executar DRL", type="primary"):
                try:
                    resp = api_request_with_retry("POST", f"{API_URL}/predict", json=st.session_state['match_data'])
                    if resp.status_code == 200:
                        st.session_state['result'] = resp.json()
                except Exception as e:
                    st.error(f"Erro: {e}")
            
            if 'result' in st.session_state:
                r = st.session_state['result']
                st.metric("🎯 Ação", r['recommendation'])
                st.metric("📈 CLV Casa", f"{r['clv']['home']:+.2%}")

    elif menu == "💰 Simulador de Banca":
        st.title("💰 Simulador de Banca (Kelly 1/4)")
        if 'result' not in st.session_state:
            st.warning("Execute uma análise manual primeiro.")
        else:
            r = st.session_state['result']
            bank = st.slider("Banca (€)", 100, 50000, 1000, 100)
            rec = r['recommendation']
            if rec == "BET_HOME":
                kf = r['kelly'].get('home', 0)
            elif rec == "BET_DRAW":
                kf = r['kelly'].get('draw', 0)
            else:
                kf = r['kelly'].get('away', 0)
            st.metric("💶 Aposta Sugerida", f"€ {bank * kf:.2f}")

    elif menu == "📜 Histórico":
        st.title("📜 Histórico de Previsões")
        if st.button("Atualizar"):
            try:
                resp = api_request_with_retry("GET", f"{API_URL}/history?limit=10")
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        df = pd.DataFrame(data)
                        cols = [c for c in ['teams', 'recommendation', 'confidence', 'created_at'] if c in df.columns]
                        st.dataframe(df[cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("Sem histórico.")
            except Exception as e:
                st.error(f"Erro: {e}")

if not st.session_state.get('logged_in', False):
    login_page()
else:
    dashboard()
