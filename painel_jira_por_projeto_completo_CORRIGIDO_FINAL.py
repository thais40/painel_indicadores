# -*- coding: utf-8 -*-
# ============================================================
# Painel de Indicadores — Jira (Nuvemshop)
# FULL — migração p/ /rest/api/3/search/jql + Rotinas Manuais corrigido
# ============================================================

from __future__ import annotations

import base64
import re
import unicodedata
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

# =====================
# Config de página
# =====================
st.set_page_config(page_title="Painel de Indicadores — Jira", page_icon="📊", layout="wide")

# =====================
# Credenciais / Jira
# =====================
JIRA_URL = "https://tiendanube.atlassian.net"  # se preferir, coloque em st.secrets["JIRA_URL"]
EMAIL = st.secrets.get("EMAIL", "")
TOKEN = st.secrets.get("TOKEN", "")

if not EMAIL or not TOKEN:
    st.error("⚠️ Configure EMAIL e TOKEN em st.secrets para acessar o Jira.")
    st.stop()

auth = HTTPBasicAuth(EMAIL, TOKEN)
TZ_BR = ZoneInfo("America/Sao_Paulo")

# =====================
# Constantes / Campos
# =====================
# SLA por projeto (JSM)
SLA_CAMPOS = {
    "TDS": "customfield_13744",  # SLA resolução (SUP)
    "TINE": "customfield_13744", # idem
    "INT": "customfield_13686",  # SLA INT
    "INTEL": "customfield_13686"
}

# Assunto Relacionado por projeto
CAMPOS_ASSUNTO = {
    "TDS": "customfield_13712",
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype",  # para INTEL usamos issuetype no lugar de assunto
}

CAMPO_AREA           = "customfield_13719"  # Área solicitante
CAMPO_N3             = "customfield_13659"  # Encaminhamento N3 (Sim/Não)
CAMPO_ORIGEM         = "customfield_13628"  # Origem do problema (APP NE / APP EN)
CAMPO_QTD_ENCOMENDAS = "customfield_13666"  # Rotinas Manuais — Quantidade de encomendas

META_SLA = {"TDS": 98.00, "INT": 96.00, "TINE": 96.00, "INTEL": 96.00}

ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"
TITULO_ROTINA = "Volumetria / Tabela de erro CTE"  # (mantido caso use também por summary)

PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS  = {"TDS": "Tech Support", "INT": "Integrations", "TINE": "IT Support NE", "INTEL": "Intelligence"}

# Campos comuns que pediremos sempre no /search/jql
JIRA_FIELDS_BASE = [
    "key", "summary", "created", "updated", "resolutiondate", "resolved", "status", "issuetype",
    CAMPO_AREA, CAMPO_N3, CAMPO_ORIGEM, CAMPO_QTD_ENCOMENDAS
]
# Campos de SLA e assunto (incluímos todos e usamos conforme projeto)
FIELDS_SLA_ALL = list(set(SLA_CAMPOS.values()))
FIELDS_ASSUNTO_ALL = list(set([v for v in CAMPOS_ASSUNTO.values() if v != "issuetype"]))

FIELDS_ALL: List[str] = list(dict.fromkeys(JIRA_FIELDS_BASE + FIELDS_SLA_ALL + FIELDS_ASSUNTO_ALL))

# Corte mínimo global (inclusive)
DATA_INICIO = "2024-02-01"

# ============
# Aparência UI
# ============
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
.stButton > button{ border-radius:10px; border:1px solid #e6e8ee; box-shadow:0 1px 2px rgba(16,24,40,.04); }
.update-row{ display:inline-flex; align-items:center; gap:12px; margin-bottom:.5rem; }
.update-caption{ color:#6B7280; font-size:.85rem; }
.section-spacer{ height:10px; }
</style>
""",
    unsafe_allow_html=True,
)

def _render_logo_and_title():
    logo_bytes = None
    b64 = st.secrets.get("LOGO_B64")
    if b64:
        try:
            logo_bytes = base64.b64decode(b64)
        except Exception:
            logo_bytes = None
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:8px 0 20px 0;">', unsafe_allow_html=True)
    if logo_bytes:
        st.image(logo_bytes, width=300)
        st.markdown('<span style="color:#111827;font-weight:600;font-size:15px;">Painel interno</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#111827;font-weight:600;font-size:15px;">Nuvemshop · Painel interno</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

_render_logo_and_title()
st.title("📊 Painel de Indicadores")

def now_br_str():
    return datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S")

if "last_update" not in st.session_state:
    st.session_state["last_update"] = now_br_str()

st.markdown('<div class="update-row">', unsafe_allow_html=True)
if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.session_state["last_update"] = now_br_str()
    st.rerun()
st.markdown(f'<span class="update-caption">🕒 Última atualização: {st.session_state["last_update"]} (BRT)</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =================
# Helpers de dados
# =================
def safe_get_value(x, key="value", fallback="—"):
    if isinstance(x, dict):
        return x.get(key, fallback)
    return x if x is not None else fallback

def dentro_sla_from_raw(sla_raw: dict) -> Optional[bool]:
    """
    Lê estrutura Jira Service Management (completedCycles/elapsedTime/goalDuration/breached).
    True = dentro do SLA, False = fora, None = indeterminado.
    """
    try:
        if not sla_raw or not isinstance(sla_raw, dict):
            return None
        cycles = sla_raw.get("completedCycles") or []
        if cycles:
            last = cycles[-1]
            if "breached" in last:
                return not bool(last["breached"])
            elapsed = (last.get("elapsedTime") or {}).get("millis")
            goal = (last.get("goalDuration") or {}).get("millis")
            if elapsed is not None and goal is not None:
                return elapsed <= goal
        return None
    except Exception:
        return None

def normaliza_origem(s: str) -> str:
    if s is None or str(s).strip() == "" or str(s).lower() in ("nan", "none"):
        return "Outros/Não informado"
    t = str(s).strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())
    if "app" in t and "ne" in t:
        return "APP NE"
    if "app" in t and ("en" in t or "eng" in t):
        return "APP EN"
    return "Outros/Não informado"

def parse_qtd_encomendas(v):
    if isinstance(v, list):
        v = next((x for x in reversed(v) if x not in (None, "")), None)
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        try:
            return int(round(float(v)))
        except Exception:
            return 0
    s = str(v).strip()
    if not s:
        return 0
    s = s.replace(".", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except Exception:
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else 0

# =====================
# Jira Enhanced Search
# =====================
def _jira_search_jql(jql: str, next_page_token: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
    """
    GET /rest/api/3/search/jql  (Enhanced)
      params: jql, fields, maxResults, nextPageToken
      response: issues[], nextPageToken, isLast
    """
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    params = {
        "jql": jql,
        "fields": ",".join(FIELDS_ALL),
        "maxResults": max_results
    }
    if next_page_token:
        params["nextPageToken"] = next_page_token

    try:
        resp = requests.get(url, params=params, auth=auth, timeout=60)
    except Exception as e:
        return {"error": str(e), "issues": [], "isLast": True}

    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:400]}", "issues": [], "isLast": True}

    return resp.json()

@st.cache_data(show_spinner="🔄 Buscando dados do Jira...", ttl=60*30)
def buscar_issues(projeto: str, jql: str, max_pages: int = 500) -> pd.DataFrame:
    todos, page, last_error = [], 0, None
    next_token = None
    while True:
        page += 1
        data = _jira_search_jql(jql, next_page_token=next_token, max_results=100)
        if "error" in data and data["error"]:
            last_error = data["error"]
            break
        issues = data.get("issues", [])
        if not issues:
            break
        for it in issues:
            f = it.get("fields", {}) or {}
            row = {
                "projeto": projeto,
                "key": it.get("key"),
                "summary": f.get("summary"),
                "created": f.get("created"),
                "updated": f.get("updated"),
                "resolutiondate": f.get("resolutiondate"),
                "resolved": f.get("resolved") or f.get("resolutiondate"),
                "status": safe_get_value(f.get("status"), "name"),
                "issuetype": f.get("issuetype"),
                "assunto": f.get(CAMPOS_ASSUNTO[projeto]) if CAMPOS_ASSUNTO[projeto] != "issuetype" else f.get("issuetype"),
                "area": f.get(CAMPO_AREA),
                "n3": f.get(CAMPO_N3),
                "origem": f.get(CAMPO_ORIGEM),
                CAMPO_QTD_ENCOMENDAS: f.get(CAMPO_QTD_ENCOMENDAS),
                "sla_raw": f.get(SLA_CAMPOS[projeto], {}),
            }
            todos.append(row)

        next_token = data.get("nextPageToken")
        is_last = bool(data.get("isLast", not bool(next_token)))
        if is_last or page >= max_pages:
            break

    dfp = pd.DataFrame(todos)

    if last_error and dfp.empty:
        st.warning(f"⚠️ Erro ao buscar Jira ({projeto}): {last_error}")
        return dfp

    if not dfp.empty:
        # Converte UTC -> BRT ANTES de derivar mês
        for c in ("created", "resolved", "resolutiondate", "updated"):
            dfp[c] = pd.to_datetime(dfp[c], errors="coerce", utc=True).dt.tz_convert(TZ_BR).dt.tz_localize(None)
        # Colunas mensais
        dfp["mes_created"]  = dfp["created"].dt.to_period("M").dt.to_timestamp()
        dfp["mes_resolved"] = dfp["resolved"].dt.to_period("M").dt.to_timestamp()
    return dfp

# ===================
# Filtros Globais UI
# ===================
# ===================
# Filtros Globais UI
# ===================
st.markdown("### 🔍 Filtros Globais")

# opções de ano: de 2024 até o ano atual
ano_atual = date.today().year
opcoes_ano = ["Todos"] + [str(y) for y in range(2024, ano_atual + 1)]

# opções de mês: 01..12
opcoes_mes = ["Todos"] + [f"{m:02d}" for m in range(1, 13)]

# inicia em "Todos"
colA, colB = st.columns(2)
with colA:
    ano_global = st.selectbox("Ano (global)", opcoes_ano, index=0, key="ano_global")
with colB:
    mes_global = st.selectbox("Mês (global)", opcoes_mes, index=0, key="mes_global")

# ======================
# JQL (com corte mínimo e mês opcional)
# ======================
def jql_projeto(project_key: str, ano_sel: str, mes_sel: str) -> str:
    base = f'project = "{project_key}" AND created >= "{DATA_INICIO}"'
    if mes_sel != "Todos" and ano_sel != "Todos":
        a = int(ano_sel); m = int(mes_sel)
        if m == 12:
            next_month_first = date(a + 1, 1, 1)
        else:
            next_month_first = date(a, m + 1, 1)
        base += f' AND created < "{next_month_first:%Y-%m-%d}"'
    return base + " ORDER BY created ASC"

JQL_TDS   = jql_projeto("TDS",   ano_global, mes_global)
JQL_INT   = jql_projeto("INT",   ano_global, mes_global)   # INT entre aspas evita palavra reservada
JQL_TINE  = jql_projeto("TINE",  ano_global, mes_global)
JQL_INTEL = jql_projeto("INTEL", ano_global, mes_global)

# ======================
# Carrega todos projetos
# ======================
with st.spinner("Carregando TDS..."):
    df_tds = buscar_issues("TDS", JQL_TDS)
with st.spinner("Carregando INT..."):
    df_int = buscar_issues("INT", JQL_INT)
with st.spinner("Carregando TINE..."):
    df_tine = buscar_issues("TINE", JQL_TINE)
with st.spinner("Carregando INTEL..."):
    df_intel = buscar_issues("INTEL", JQL_INTEL)

if all(d.empty for d in [df_tds, df_int, df_tine, df_intel]):
    st.warning("Sem dados do Jira em nenhum projeto (verifique credenciais e permissões).")
    st.stop()

# =====================
# Pré-agregados mensais (criamos uma vez p/ gráficos)
# =====================
@st.cache_data(show_spinner=False)
def build_monthly_tables(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    base = df_all.copy()
    base["per_created"]  = base["created"].dt.to_period("M")
    base["per_resolved"] = base["resolved"].dt.to_period("M")

    # Criados por mês
    created = (base.groupby(["projeto","per_created"])
                    .size().reset_index(name="Criados")
                    .rename(columns={"per_created":"period"}))

    # Resolvidos + SLA
    res = base[base["resolved"].notna()].copy()
    res["dentro_sla"] = res["sla_raw"].apply(dentro_sla_from_raw).fillna(False)
    resolved = (res.groupby(["projeto","per_resolved"])
                  .agg(Resolvidos=("key","count"), Dentro=("dentro_sla","sum"))
                  .reset_index()
                  .rename(columns={"per_resolved":"period"}))

    monthly = pd.merge(created, resolved, how="outer", on=["projeto","period"]).fillna(0)
    monthly["period"]   = monthly["period"].astype("period[M]")
    monthly["period_ts"]= monthly["period"].dt.to_timestamp()
    monthly["ano"]      = monthly["period"].dt.year.astype(int)
    monthly["mes"]      = monthly["period"].dt.month.astype(int)
    monthly["mes_str"]  = monthly["period_ts"].dt.strftime("%b/%Y")

    monthly["Dentro"]     = monthly["Dentro"].astype(int)
    monthly["Resolvidos"] = monthly["Resolvidos"].astype(int)
    monthly["Fora"]       = (monthly["Resolvidos"] - monthly["Dentro"]).clip(lower=0)

    monthly["pct_dentro"] = monthly.apply(
        lambda r: (r["Dentro"]/r["Resolvidos"]*100) if r["Resolvidos"]>0 else 0.0, axis=1
    ).round(2)
    monthly["pct_fora"] = monthly.apply(
        lambda r: (r["Fora"]/r["Resolvidos"]*100) if r["Resolvidos"]>0 else 0.0, axis=1
    ).round(2)

    return monthly.sort_values(["projeto","period"])

df_monthly_all = pd.concat(
    [build_monthly_tables(d) for d in [df_tds, df_int, df_tine, df_intel] if not d.empty],
    ignore_index=True
) if not all(d.empty for d in [df_tds, df_int, df_tine, df_intel]) else pd.DataFrame()

# ===================
# Renderizadores base
# ===================
def aplicar_filtro_global(df_in: pd.DataFrame, col_dt: str, ano: str, mes: str) -> pd.DataFrame:
    out = df_in.copy()
    if ano != "Todos":
        out = out[out[col_dt].dt.year == int(ano)]
    if mes != "Todos":
        out = out[out[col_dt].dt.month == int(mes)]
    return out

def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if "assunto_nome" not in df_proj.columns:
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_proj["assunto_nome"] = df_proj["issuetype"].apply(lambda x: safe_get_value(x, "name"))
        else:
            df_proj["assunto_nome"] = df_proj["assunto"].apply(lambda x: safe_get_value(x, "value"))
    return df_proj

def render_criados_resolvidos(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### 📈 Tickets Criados vs Resolvidos")

    dfm = df_monthly_all[df_monthly_all["projeto"] == projeto].copy()
    if dfm.empty:
        st.info("Sem dados para esta visão.")
        return
    if ano_global != "Todos":
        dfm = dfm[dfm["ano"] == int(ano_global)]
    if mes_global != "Todos":
        dfm = dfm[dfm["mes"] == int(mes_global)]
    if ano_global != "Todos" and mes_global != "Todos":
        alvo = pd.Period(f"{int(ano_global)}-{int(mes_global):02d}", freq="M")
        dfm = dfm[dfm["period"] == alvo]

    dfm = dfm.sort_values("period_ts")
    show = dfm[["mes_str","period_ts","Criados","Resolvidos"]].copy()
    show["Criados"] = show["Criados"].astype(int)
    show["Resolvidos"] = show["Resolvidos"].astype(int)

    fig = px.bar(show, x="mes_str", y=["Criados","Resolvidos"], barmode="group", text_auto=True, height=440)
    fig.update_traces(textangle=0, textfont_size=14, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    st.plotly_chart(fig, use_container_width=True)

def render_sla(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### ⏱️ SLA")

    dfm = df_monthly_all[df_monthly_all["projeto"] == projeto].copy()
    if dfm.empty:
        st.info("Sem dados de SLA.")
        return
    if ano_global != "Todos":
        dfm = dfm[dfm["ano"] == int(ano_global)]
    if mes_global != "Todos":
        dfm = dfm[dfm["mes"] == int(mes_global)]
    if ano_global != "Todos" and mes_global != "Todos":
        alvo = pd.Period(f"{int(ano_global)}-{int(mes_global):02d}", freq="M")
        dfm = dfm[dfm["period"] == alvo]

    okr = dfm["pct_dentro"].mean() if not dfm.empty else 0.0
    meta = META_SLA.get(projeto, 98.0)
    titulo = f"OKR: {okr:.2f}% — Meta: {meta:.2f}%".replace(".", ",")

    show = dfm[["mes_str","period_ts","pct_dentro","pct_fora"]].sort_values("period_ts")
    show = show.rename(columns={"pct_dentro": "% Dentro SLA", "pct_fora":"% Fora SLA"})

    fig = px.bar(
        show, x="mes_str", y=["% Dentro SLA","% Fora SLA"], barmode="group",
        title=titulo, color_discrete_map={"% Dentro SLA":"green","% Fora SLA":"red"},
        height=440
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", textfont_size=14, cliponaxis=False)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    st.plotly_chart(fig, use_container_width=True)

def render_assunto(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### 🧾 Assunto Relacionado")
    df_ass = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_ass.empty:
        st.info("Sem dados para Assunto Relacionado nos filtros atuais.")
        return

    if CAMPOS_ASSUNTO[projeto] == "issuetype":
        df_ass["assunto_nome"] = df_ass["issuetype"].apply(lambda x: safe_get_value(x, "name"))
    else:
        df_ass["assunto_nome"] = df_ass["assunto"].apply(lambda x: safe_get_value(x, "value"))

    assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
    assunto_count.columns = ["Assunto","Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)

def render_area(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📦 Área Solicitante")
    df_area = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_area.empty:
        st.info("Sem dados para Área Solicitante nos filtros atuais.")
        return

    df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
    area_count = df_area["area_nome"].value_counts().reset_index()
    area_count.columns = ["Área","Qtd"]
    st.dataframe(area_count, use_container_width=True, hide_index=True)

def render_encaminhamentos(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🔄 Encaminhamentos")
    df_enc = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_enc.empty:
        st.info("Sem dados para Encaminhamentos nos filtros atuais.")
        return

    col1,col2 = st.columns(2)
    with col1:
        count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
        st.metric("Encaminhados Produto", int(count_prod))
    with col2:
        df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
        st.metric("Encaminhados N3", int((df_enc["n3_valor"] == "Sim").sum()))

# ===================
# APP NE — TDS
# ===================
def render_app_ne(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📱 APP NE")
    if dfp.empty:
        st.info("Sem dados para APP NE.")
        return

    dfp = ensure_assunto_nome(dfp.copy(), "TDS")
    s_ass = dfp["assunto_nome"].astype(str).str.strip()
    alvo = ASSUNTO_ALVO_APPNE.strip().casefold()
    mask_assunto = s_ass.str.casefold().eq(alvo)
    if not mask_assunto.any():
        mask_assunto = s_ass.str.contains(r"app\s*ne", case=False, regex=True)

    df_app = dfp[mask_assunto].copy()
    if df_app.empty:
        st.info(f"Não há chamados para '{ASSUNTO_ALVO_APPNE}'.")
        return

    df_app["origem_nome"] = df_app["origem"].apply(lambda x: safe_get_value(x, "value"))
    df_app["origem_cat"]  = df_app["origem_nome"].apply(normaliza_origem)

    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()
    df_app = aplicar_filtro_global(df_app, "mes_dt", ano_global, mes_global)
    if df_app.empty:
        st.info("Sem dados para exibir com os filtros selecionados.")
        return

    total_app = int(len(df_app))
    contagem  = df_app["origem_cat"].value_counts()

    m1,m2,m3 = st.columns(3)
    m1.metric("Total (APP NE/EN)", total_app)
    m2.metric("APP NE", int(contagem.get("APP NE", 0)))
    m3.metric("APP EN", int(contagem.get("APP EN", 0)))

    serie = (df_app.groupby(["mes_dt","origem_cat"]).size()
             .reset_index(name="Qtd").sort_values("mes_dt"))
    serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
    cats = serie["mes_str"].dropna().unique().tolist()
    serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

    fig_app = px.bar(
        serie, x="mes_str", y="Qtd", color="origem_cat", barmode="group",
        title="APP NE — Volumes por mês e Origem do problema",
        color_discrete_map={"APP NE":"#2ca02c","APP EN":"#1f77b4","Outros/Não informado":"#9ca3af"},
        text="Qtd", height=460
    )
    fig_app.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
    max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
    if max_qtd > 0:
        fig_app.update_yaxes(range=[0, max_qtd * 1.25])
    fig_app.update_layout(yaxis_title="Qtd", xaxis_title="Mês",
                          uniformtext_minsize=14, uniformtext_mode="show",
                          bargap=0.15, margin=dict(t=70, r=20, b=60, l=50))
    st.plotly_chart(fig_app, use_container_width=True)

# ===========================
# Rotinas Manuais — TDS (OK)
# ===========================
def _canonical(s: str) -> str:
    """normaliza texto: sem acentos, minúsculo, sem pontuação, 1 espaço, strip."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)       # remove pontuação (mantém letras/números/_ e espaços)
    s = re.sub(r"\s+", " ", s).strip()   # espaços consecutivos -> 1 só
    return s

import re
import unicodedata

def _canonical(s: str) -> str:
    """normaliza texto: sem acentos, minúsculo, sem pontuação, 1 espaço, strip."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def render_rotinas_manuais(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    """Rotinas Manuais: soma tickets com título-alvo (variações toleradas) e quantidade > 0."""

    COL_QTD_ROTINAS = globals().get("COL_QTD_ROTINAS", "customfield_13666")

    # títulos aceitos
    TITULOS_ROTINAS = [
        "Volumetria / Tabela de erro CTE",
        "Volumetria Correção de Erro de CTE",
    ]
    # normaliza todos
    alvos = [_canonical(t) for t in TITULOS_ROTINAS]

    st.markdown("### 🛠️ Rotinas Manuais")

    if dfp.empty:
        st.info("Sem tickets para o período.")
        return

    # normaliza summaries
    summ_norm = dfp["summary"].fillna("").map(_canonical)
    mask = summ_norm.isin(alvos) | summ_norm.apply(lambda s: any(a in s for a in alvos))
    df_rot = dfp.loc[mask].copy()

    # quantidade numérica > 0
    df_rot[COL_QTD_ROTINAS] = pd.to_numeric(df_rot[COL_QTD_ROTINAS], errors="coerce").fillna(0)
    df_rot = df_rot[df_rot[COL_QTD_ROTINAS] > 0]

    # resolved válido
    df_rot["resolved"] = pd.to_datetime(df_rot["resolved"], errors="coerce")
    df_rot = df_rot.dropna(subset=["resolved"])

    # filtros globais
    if ano_global and str(ano_global).lower() != "todos":
        df_rot = df_rot[df_rot["resolved"].dt.year.astype(str) == str(ano_global)]
    if mes_global and str(mes_global).lower() != "todos":
        m = f"{int(mes_global):02d}"
        df_rot = df_rot[df_rot["resolved"].dt.month.astype(str).str.zfill(2) == m]

    if df_rot.empty:
        st.info("Sem tickets de Rotinas Manuais (títulos alvo) no período.")
        return

    # série mensal
    df_rot["mes_str"] = df_rot["resolved"].dt.to_period("M").dt.strftime("%b/%Y")
    serie = (
        df_rot.groupby("mes_str", as_index=False)[COL_QTD_ROTINAS]
              .sum()
              .rename(columns={COL_QTD_ROTINAS: "qtd_encomendas"})
    )
    serie["mes_ord"] = pd.to_datetime(serie["mes_str"], format="%b/%Y")
    serie = serie.sort_values("mes_ord").drop(columns=["mes_ord"])

    # gráfico
    fig = px.bar(
        serie, x="mes_str", y="qtd_encomendas", text="qtd_encomendas"
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        yaxis_title="Qtd encomendas", xaxis_title="mes_str",
        uniformtext_minsize=8, uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True)

    # amostra
    with st.expander("🔎 Tickets usados (amostra)"):
        st.dataframe(
            df_rot[["key", "summary", "resolved", COL_QTD_ROTINAS]]
            .sort_values("resolved", ascending=True)
            .head(50),
            use_container_width=True,
        )

# ===================
# Onboarding — INT (mantido)
# ===================
def render_onboarding(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🧭 Onboarding")
    if dfp.empty:
        st.info("Sem dados de Onboarding.")
        return

    ASSUNTO_CLIENTE_NOVO = "Nova integração - Cliente novo"
    ASSUNTOS_ERROS = [
        "Erro durante Onboarding - Frete",
        "Erro durante Onboarding - Pedido",
        "Erro durante Onboarding - Rastreio",
        "Erro durante Onboarding - Teste",
    ]
    STATUS_PENDENCIAS = [
        "Aguardando informações adicionais",
        "Em andamento",
        "Aguardando pendências da Triagem",
        "Aguardando validação do cliente",
        "Aguardando Comercial",
    ]

    dfp = ensure_assunto_nome(dfp.copy(), "INT")
    df_onb = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)

    total_clientes_novos = int((df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO).sum())
    df_erros = df_onb[df_onb["assunto_nome"].isin(ASSUNTOS_ERROS)].copy()
    pend_mask = df_onb["status"].isin(STATUS_PENDENCIAS)

    tickets_pendencias = int(pend_mask.sum())
    possiveis_clientes = int(pend_mask.sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tickets clientes novos", total_clientes_novos)
    c2.metric("Erros onboarding", int(len(df_erros)))
    c3.metric("Tickets com pendências", tickets_pendencias)
    c4.metric("Possíveis clientes", possiveis_clientes)

    # gráfico horizontal por erros
    if not df_erros.empty:
        cont_erros = (df_erros["assunto_nome"].value_counts()
                      .reindex(ASSUNTOS_ERROS, fill_value=0).reset_index())
        cont_erros.columns = ["Categoria","Qtd"]

        fig_onb = px.bar(cont_erros, x="Qtd", y="Categoria", orientation="h",
                         text="Qtd", title="Erros Onboarding", height=420)
        fig_onb.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                              textfont_size=16, cliponaxis=False)
        max_q = int(cont_erros["Qtd"].max()) if not cont_erros.empty else 0
        if max_q > 0:
            fig_onb.update_xaxes(range=[0, max_q*1.25])
        fig_onb.update_layout(margin=dict(t=50, r=20, b=30, l=10), bargap=0.25)
        st.plotly_chart(fig_onb, use_container_width=True)

    # simulação de dinheiro perdido
    st.markdown("---")
    st.subheader("💸 Dinheiro perdido (simulação)")
    c_left,c_right = st.columns([1,1])
    with c_left:
        st.number_input("Clientes novos (simulação)", value=possiveis_clientes, disabled=True, key="sim_clientes_onb")
    with c_right:
        receita_cliente = st.slider("Cenário Receita por Cliente (R$)",
                                    min_value=0, max_value=100000, step=500, value=20000,
                                    key="sim_receita_onb")
    dinheiro_perdido = float(possiveis_clientes) * float(receita_cliente)
    st.markdown(f"### **R$ {dinheiro_perdido:,.2f}**",
                help="Cálculo: Clientes novos (simulação) × Cenário Receita por Cliente")

# ======================
# Abas por Projeto/Visão
# ======================
tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")

        if projeto == "TDS":
            dfp = df_tds.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante","APP NE"]
        elif projeto == "INT":
            dfp = df_int.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante","Onboarding"]
        elif projeto == "TINE":
            dfp = df_tine.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante"]
        else:
            dfp = df_intel.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado"]

        if dfp.empty:
            st.info("Sem dados carregados para este projeto.")
            continue

        # garantir colunas auxiliares
        dfp["mes_created"] = pd.to_datetime(dfp["created"], errors="coerce")
        dfp["mes_resolved"] = pd.to_datetime(dfp["resolved"], errors="coerce")

        dfp = ensure_assunto_nome(dfp, projeto)
        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")

        # 1) Criados vs Resolvidos
        if visao == "Criados vs Resolvidos":
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)

        # 2) SLA
        elif visao == "SLA":
            render_sla(dfp, projeto, ano_global, mes_global)

        # 3) Assunto
        elif visao == "Assunto Relacionado":
            render_assunto(dfp, projeto, ano_global, mes_global)

        # 4) Área
        elif visao == "Área Solicitante":
            if projeto == "INTEL":
                st.info("Este projeto não possui Área Solicitante.")
            else:
                render_area(dfp, ano_global, mes_global)

        # 5) Onboarding
        elif visao == "Onboarding":
            if projeto == "INT":
                render_onboarding(dfp, ano_global, mes_global)
            else:
                st.info("Onboarding disponível somente para Integrations.")

        # 6) APP NE
        elif visao == "APP NE":
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
            else:
                st.info("APP NE disponível somente para Tech Support.")

        # Geral — ordem fixa (como você usa)
        else:
            # Criados vs Resolvidos ANTES do SLA (mantido)
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
            render_sla(dfp, projeto, ano_global, mes_global)
            render_assunto(dfp, projeto, ano_global, mes_global)

            if projeto != "INTEL":
                render_area(dfp, ano_global, mes_global)

            if projeto in ("TDS","INT"):
                render_encaminhamentos(dfp, ano_global, mes_global)

            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
                # Rotinas Manuais no final (expander)
                with st.expander("🛠️ Rotinas Manuais", expanded=False):
                    render_rotinas_manuais(dfp, ano_global, mes_global)

            if projeto == "INT":
                with st.expander("🧭 Onboarding", expanded=False):
                    render_onboarding(dfp, ano_global, mes_global)

# Rodapé
st.markdown("---")
st.caption("Feito com 💙 — endpoint Jira /search/jql (enhanced), meses ordenados, SLA com 2 casas, Rotinas Manuais por resolved.")
