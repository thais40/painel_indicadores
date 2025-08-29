# -*- coding: utf-8 -*-
import os
import math
import json
import time
import pytz
import base64
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from requests.auth import HTTPBasicAuth

import streamlit as st

# ==============
# CONFIG INICIAL
# ==============
st.set_page_config(page_title="Painel de Indicadores — Jira", page_icon="📊", layout="wide")

# ---------------
# Jira / Secrets
# ---------------
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets.get("EMAIL", "")
TOKEN = st.secrets.get("TOKEN", "")

if not EMAIL or not TOKEN:
    st.error("Configure EMAIL e TOKEN em st.secrets para acessar o Jira.")
    st.stop()

auth = HTTPBasicAuth(EMAIL, TOKEN)

# Campos custom usados
CAMPO_SLA_SUP = "customfield_13744"   # SLA resolução (SUP) — (TDS/TINE)
CAMPO_ORIGEM   = "customfield_13628"  # Origem do problema — APP NE
CAMPO_QTD_ENC  = "customfield_13666"  # Quantidade de encomendas — Rotinas Manuais

# Campos que vamos pedir ao Jira
JIRA_FIELDS = [
    "key", "summary", "project", "issuetype", "status",
    "created", "updated", "resolutiondate", "resolved",
    CAMPO_SLA_SUP, CAMPO_ORIGEM, CAMPO_QTD_ENC
]

# Metas SLA por projeto
META_SLA = {
    "TDS": 98.00,
    "INT": 96.00,
    "TINE": 96.00,
    "INTEL": 98.00
}
# Limites SLA (millis) — ajuste se necessário
SLA_LIMITES = {
    "TDS": 40 * 60 * 60 * 1000,    # 40h
    "INT": 40 * 60 * 60 * 1000,
    "TINE": 40 * 60 * 60 * 1000,
    "INTEL": 80 * 60 * 60 * 1000
}

# ==========
# Utilitários
# ==========
TZ_BRT = pytz.timezone("America/Sao_Paulo")

def brt_now():
    return datetime.now(TZ_BRT)

def fmt_brt(dt):
    if dt.tzinfo is None:
        dt = TZ_BRT.localize(dt)
    return dt.astimezone(TZ_BRT).strftime("%d/%m/%Y %H:%M:%S")

def load_logo():
    logo = st.secrets.get("LOGO_PATH")
    if logo and os.path.exists(logo):
        with open(logo, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" style="height:48px;margin:2px 0 10px 0;">',
            unsafe_allow_html=True
        )
    else:
        st.markdown("#### Nuvemshop  \n*Painel interno*")

def _jira_get(jql: str, start_at: int = 0, max_results: int = 100):
    url = f"{JIRA_URL}/rest/api/3/search"
    params = {
        "jql": jql,
        "startAt": start_at,
        "maxResults": max_results,
        "fields": ",".join(JIRA_FIELDS),
    }
    r = requests.get(url, params=params, auth=auth, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Erro Jira {r.status_code}: {r.text[:400]}")
    return r.json()

@st.cache_data(show_spinner=False, ttl=60*30)
def jira_search_all(jql: str, max_pages: int = 400) -> pd.DataFrame:
    rows, start, page = [], 0, 0
    while True:
        page += 1
        data = _jira_get(jql, start_at=start, max_results=100)
        issues = data.get("issues", [])
        if not issues:
            break
        for it in issues:
            fields = it.get("fields", {})
            row = {"key": it.get("key", "")}
            for f in JIRA_FIELDS:
                if f == "key": continue
                row[f] = fields.get(f, None)
            rows.append(row)
        start += len(issues)
        if start >= data.get("total", 0): break
        if page >= max_pages: break
    df = pd.DataFrame(rows)
    # normaliza datetimes
    for c in ("created","updated","resolved","resolutiondate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def add_month_cols(df: pd.DataFrame, date_col: str, prefix: str):
    """Cria ano/mes/mes_str a partir de uma coluna de data."""
    out = df.copy()
    out = out.dropna(subset=[date_col])
    out[f"{prefix}_period"]  = out[date_col].dt.to_period("M")
    out[f"{prefix}_ts"]      = out[f"{prefix}_period"].dt.to_timestamp()
    out[f"{prefix}_ano"]     = out[f"{prefix}_period"].dt.year
    out[f"{prefix}_mes"]     = out[f"{prefix}_period"].dt.month
    out[f"{prefix}_mes_str"] = out[f"{prefix}_ts"].dt.strftime("%b/%Y")
    return out

def parse_int_from_any(x):
    if pd.isna(x): return 0
    s = str(x).strip()
    if not s: return 0
    s = s.replace(".", "").replace(",", ".")
    try:
        return int(float(s))
    except Exception:
        return 0

# ==========================
# Cabeçalho + Atualização UI
# ==========================
c1, c2 = st.columns([1,4])
with c1: load_logo()
with c2: st.title("📊 Painel de Indicadores — Jira")

ca, cb = st.columns([0.2,0.8])
with ca:
    if st.button("🔄 Atualizar dados"):
        st.cache_data.clear()
        st.session_state["last_update_brt"] = brt_now()
        st.experimental_rerun()
with cb:
    last = st.session_state.get("last_update_brt") or brt_now()
    st.session_state["last_update_brt"] = last
    st.caption(f"🕒 Última atualização: {fmt_brt(last)} (BRT)")

st.markdown("---")

# ======================
# Filtros Globais
# ======================
st.header("🔎 Filtros Globais")
c_ano, c_mes = st.columns(2)
with c_ano:
    ano_global = st.selectbox("Ano (global)", ["Todos"] + [str(y) for y in range(2024,2031)], index=1)
with c_mes:
    mes_global = st.selectbox("Mês (global)", ["Todos"] + [f"{m:02d}" for m in range(1,13)], index=0)
st.markdown("---")

# =====================
# Busca de dados Jira
# =====================
# Faça ajustes de JQL conforme suas necessidades (status, issuetype, etc)
JQL_TDS   = "project = TDS ORDER BY resolved DESC, created DESC"
JQL_INT   = "project = INT ORDER BY resolved DESC, created DESC"
JQL_TINE  = "project = TINE ORDER BY resolved DESC, created DESC"
JQL_INTEL = "project = INTEL ORDER BY resolved DESC, created DESC"

with st.spinner("Carregando TDS..."):
    df_tds = jira_search_all(JQL_TDS)
with st.spinner("Carregando INT..."):
    df_int = jira_search_all(JQL_INT)
with st.spinner("Carregando TINE..."):
    df_tine = jira_search_all(JQL_TINE)
with st.spinner("Carregando INTEL..."):
    df_intel = jira_search_all(JQL_INTEL)

# ==========================
# Funções de visualizações
# ==========================
def section_criados_resolvidos(df: pd.DataFrame, projeto: str, ano_glob: str, mes_glob: str):
    st.subheader("📈 Tickets Criados vs Resolvidos")

    if df.empty:
        st.info("Sem dados.")
        return

    # Criados por created
    d_created = df.dropna(subset=["created"]).copy()
    d_created = add_month_cols(d_created, "created", "cr")
    # Resolvidos por resolved
    d_res = df.dropna(subset=["resolved"]).copy()
    d_res = add_month_cols(d_res, "resolved", "rs")

    # Filtro global para ambos
    if ano_glob != "Todos":
        d_created = d_created[d_created["cr_ano"] == int(ano_glob)]
        d_res     = d_res[d_res["rs_ano"] == int(ano_glob)]
        if mes_glob != "Todos":
            m = int(mes_glob)
            d_created = d_created[d_created["cr_mes"] == m]
            d_res     = d_res[d_res["rs_mes"] == m]

    g_created = (
        d_created.groupby(["cr_period","cr_ts","cr_mes_str"], as_index=False)
                 .size().rename(columns={"size":"Criados"})
                 .sort_values("cr_ts")
    )
    g_res = (
        d_res.groupby(["rs_period","rs_ts","rs_mes_str"], as_index=False)
             .size().rename(columns={"size":"Resolvidos"})
             .sort_values("rs_ts")
    )

    # junta pela string do mês para facilitar
    g_created.rename(columns={"cr_mes_str":"mes_str"}, inplace=True)
    g_res.rename(columns={"rs_mes_str":"mes_str"}, inplace=True)
    final = pd.merge(g_created[["mes_str","Criados"]],
                     g_res[["mes_str","Resolvidos"]],
                     on="mes_str", how="outer").fillna(0.0)
    final = final.sort_values("mes_str")

    long = final.melt(id_vars=["mes_str"], value_vars=["Criados","Resolvidos"], var_name="variable", value_name="value")
    long["label"] = long["value"].map(lambda v: f"{int(v):,}".replace(",", "."))

    fig = px.bar(long, x="mes_str", y="value", color="variable", barmode="group", text="label", height=420)
    fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="value")
    st.plotly_chart(fig, use_container_width=True)

def section_sla(df: pd.DataFrame, projeto: str, ano_glob: str, mes_glob: str):
    st.subheader(f"⏱️ SLA — {projeto}")

    if df.empty:
        st.info("Sem dados.")
        return

    limite = SLA_LIMITES.get(projeto, 40*60*60*1000)
    meta   = META_SLA.get(projeto, 98.00)

    base = df.dropna(subset=["resolved"]).copy()
    base = add_month_cols(base, "resolved", "m")

    # Precisamos do campo de SLA em millis
    if CAMPO_SLA_SUP not in base.columns:
        st.info(f"Campo {CAMPO_SLA_SUP} ausente. Seção SLA desativada.")
        return

    # Converte para numérico (millis)
    base["sla_millis"] = base[CAMPO_SLA_SUP].apply(parse_int_from_any)
    base["dentro_sla"] = base["sla_millis"] <= int(limite)

    if ano_glob != "Todos":
        base = base[base["m_ano"] == int(ano_glob)]
        if mes_glob != "Todos":
            base = base[base["m_mes"] == int(mes_glob)]

    if base.empty:
        st.info("Sem dados no período.")
        return

    grp = base.groupby(["m_period","m_ts","m_mes_str"], as_index=False).agg(
        total=("key","count"),
        dentro=("dentro_sla","sum")
    ).sort_values("m_ts")
    grp["fora"] = grp["total"] - grp["dentro"]

    # Percentuais com 2 casas
    grp["% Dentro SLA"] = (grp["dentro"]/grp["total"]*100).round(2)
    grp["% Fora SLA"]   = (grp["fora"]/grp["total"]*100).round(2)

    long = grp.melt(
        id_vars=["m_mes_str"],
        value_vars=["% Dentro SLA","% Fora SLA"],
        var_name="variable", value_name="value"
    )
    long["label"] = long["value"].map(lambda v: f"{v:.2f}%")

    fig = px.bar(long, x="m_mes_str", y="value", color="variable", barmode="group", text="label", height=420,
                 color_discrete_map={"% Dentro SLA":"green","% Fora SLA":"red"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0,100], title_text="%", ticksuffix="%")
    st.caption(f"OKR: {grp['% Dentro SLA'].mean():.2f}% — Meta: {meta:.2f}%")
    st.plotly_chart(fig, use_container_width=True)

def section_app_ne(df: pd.DataFrame, ano_glob: str, mes_glob: str):
    st.markdown("### 📱 APP NE (no final)")
    if df.empty:
        st.info("Sem dados.")
        return

    # Filtra por título (mantivemos a definição combinada)
    base = df[df["summary"].str.contains("Volumetria / Tabela de erro CTE", case=False, na=False)].copy()
    if base.empty:
        st.info("Sem tickets com título 'Volumetria / Tabela de erro CTE'.")
        return

    # Origem do problema
    if CAMPO_ORIGEM not in base.columns:
        st.info(f"Campo {CAMPO_ORIGEM} ausente para origem.")
        return

    base = add_month_cols(base.dropna(subset=["resolved"]), "resolved", "m")

    if ano_glob != "Todos":
        base = base[base["m_ano"] == int(ano_glob)]
        if mes_glob != "Todos":
            base = base[base["m_mes"] == int(mes_glob)]

    if base.empty:
        st.info("Sem dados no período.")
        return

    base["origem_cat"] = base[CAMPO_ORIGEM].fillna("Outros/Não informado").astype(str)
    g = base.groupby(["m_period","m_ts","m_mes_str","origem_cat"], as_index=False).size().rename(columns={"size":"Qtd"}).sort_values(["m_ts","origem_cat"])

    fig = px.bar(g, x="m_mes_str", y="Qtd", color="origem_cat", barmode="group", height=420)
    st.plotly_chart(fig, use_container_width=True)

def section_rotinas_manuais(df: pd.DataFrame, ano_glob: str, mes_glob: str):
    st.markdown("### 🛠️ Rotinas Manuais (Quantidade de encomendas por mês de *resolved*)")

    if df.empty:
        st.info("Sem dados.")
        return

    if CAMPO_QTD_ENC not in df.columns:
        st.info(f"Campo {CAMPO_QTD_ENC} ausente.")
        return

    base = df.copy()
    # usamos apenas tickets com resolved e quantidade > 0
    base = base.dropna(subset=["resolved"])
    base["qtd"] = base[CAMPO_QTD_ENC].apply(parse_int_from_any).astype(int)
    base = base[base["qtd"] > 0]

    # deduplicar por key mantendo última resolução
    if "key" in base.columns:
        base = base.sort_values("resolved").drop_duplicates(subset=["key"], keep="last")

    base = add_month_cols(base, "resolved", "m")

    if ano_glob != "Todos":
        base = base[base["m_ano"] == int(ano_glob)]
        if mes_glob != "Todos":
            base = base[base["m_mes"] == int(mes_glob)]

    if base.empty:
        st.info("Sem dados no período.")
        return

    g = base.groupby(["m_period","m_ts","m_mes_str"], as_index=False).agg(Qtd=("qtd","sum")).sort_values("m_ts")
    g["label"] = g["Qtd"].map(lambda v: f"{v:,.0f}".replace(",", "."))

    # Outlier info (igual você gostou)
    outliers = (g["Qtd"] > 100000).sum()
    if outliers > 0:
        st.caption(f"ℹ️ {outliers} registro(s) descartado(s) por > 100.000.")

    g = g[g["Qtd"] <= 100000] if outliers else g

    fig = px.bar(g, x="m_mes_str", y="Qtd", text="label", height=440)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="Qtd encomendas", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Tickets usados (amostra)"):
        cols = ["key","summary","resolved",CAMPO_QTD_ENC]
        cols = [c for c in cols if c in base.columns]
        st.dataframe(base[cols].sort_values("resolved"), use_container_width=True)

def section_onboarding_int(df: pd.DataFrame, ano_glob: str, mes_glob: str):
    st.markdown("### 🚀 Onboarding (INT)")

    if df.empty:
        st.info("Sem dados INT.")
        return

    # simples: possiveis clientes ~ tickets com status de pendência
    # Ajuste as listas se preferir
    pend = ["Aguardando informações adicionais","Em andamento","Aguardando pendências da Triagem",
            "Aguardando validação do cliente","Aguardando Comercial"]

    base = add_month_cols(df.dropna(subset=["created"]), "created", "c")

    if ano_glob != "Todos":
        base = base[base["c_ano"] == int(ano_glob)]
        if mes_glob != "Todos":
            base = base[base["c_mes"] == int(mes_glob)]

    if base.empty:
        st.info("Sem dados no período.")
        return

    base["pend"] = base["status"].astype(str).isin(pend)
    possiveis_clientes = int(base["pend"].sum())

    st.metric("Possíveis clientes", f"{possiveis_clientes:,}".replace(",", "."))

    colL, colR = st.columns([0.5,0.5])
    with colL:
        clientes_sim = st.number_input("Clientes novos (simulação)", min_value=0, value=int(possiveis_clientes), step=1)
    with colR:
        receita_un = st.slider("Cenário Receita por Cliente (R$)", min_value=0, max_value=100000, value=20000, step=500)

    st.subheader(f"Dinheiro perdido (simulação)")
    st.title(f"R$ {(clientes_sim * receita_un):,}".replace(",", "."))

# =================
# RENDER PROJETOS
# =================
st.header("🧭 Projetos")
tabs = st.tabs(["Tech Support (TDS)","Integrations (INT)","IT Support NE (TINE)","Intelligence (INTEL)"])

# ---- TDS
with tabs[0]:
    st.subheader("Projeto: Tech Support (TDS)")
    section_criados_resolvidos(df_tds, "TDS", ano_global, mes_global)
    section_sla(df_tds, "TDS", ano_global, mes_global)
    st.markdown("---")
    section_app_ne(df_tds, ano_global, mes_global)         # fica no final
    section_rotinas_manuais(df_tds, ano_global, mes_global) # fica no final também

# ---- INT
with tabs[1]:
    st.subheader("Projeto: Integrations (INT)")
    section_criados_resolvidos(df_int, "INT", ano_global, mes_global)
    section_sla(df_int, "INT", ano_global, mes_global)
    st.markdown("---")
    section_onboarding_int(df_int, ano_global, mes_global)  # submenu/área no final da página

# ---- TINE
with tabs[2]:
    st.subheader("Projeto: IT Support NE (TINE)")
    section_criados_resolvidos(df_tine, "TINE", ano_global, mes_global)
    section_sla(df_tine, "TINE", ano_global, mes_global)

# ---- INTEL
with tabs[3]:
    st.subheader("Projeto: Intelligence (INTEL)")
    section_criados_resolvidos(df_intel, "INTEL", ano_global, mes_global)
    section_sla(df_intel, "INTEL", ano_global, mes_global)

st.markdown("---")
st.caption("Feito com 💙 — painel unificado (Criados vs Resolvidos, SLA, APP NE, Rotinas Manuais, Onboarding).")
