# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import date, datetime
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =========================
# CONFIG BÁSICA / SECRETS
# =========================
st.set_page_config(page_title="Painel de Indicadores — Jira", layout="wide")

# ----- Secrets obrigatórios -----
try:
    JIRA_URL = st.secrets["JIRA_URL"].rstrip("/")
    EMAIL    = st.secrets["EMAIL"]
    TOKEN    = st.secrets["TOKEN"]
except Exception:
    st.error("⚠️ Configure `JIRA_URL`, `EMAIL` e `TOKEN` em `st.secrets`.")
    st.stop()

# ----- Metas de SLA por projeto -----
META_SLA = {
    "TDS": 98.00,
    "INT": 96.00,
    "TINE": 96.00,
    "INTEL": 96.00,
}

# ----- Campo(s) usado(s) -----
# Ajuste/adicione conforme necessidade do seu painel
JIRA_FIELDS = [
    "summary",
    "project",
    "created",
    "updated",
    "resolutiondate",
    "customfield_13744",  # SLA resolução (SUP) - numérico em minutos? string? (normalizamos adiante)
    "customfield_13628",  # Origem do problema (APP NE/EN)
    "customfield_13666",  # Quantidade de encomendas (Rotinas Manuais)
]

# Data mínima padrão para recorte
DATA_INICIO = "2024-01-01"

# ===================================
# HELPERS: API NOVA / JQL / FETCH
# ===================================

def _jira_post_search_jql(jql: str, start_at: int = 0, max_results: int = 100) -> Dict[str, Any]:
    """
    Usa a API nova do Jira: POST /rest/api/3/search/jql
    """
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    headers = {"Content-Type": "application/json"}
    payload = {
        "jql": jql,
        "startAt": start_at,
        "maxResults": max_results,
        "fields": JIRA_FIELDS,
    }
    r = requests.post(url, headers=headers, auth=(EMAIL, TOKEN), data=json.dumps(payload))
    if r.status_code >= 400:
        # Exibe erro completo para depuração
        raise RuntimeError(f"Erro Jira {r.status_code}: {r.text}")
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def jira_search_all(jql: str) -> List[Dict[str, Any]]:
    """
    Busca todas as páginas de issues para um JQL.
    """
    issues: List[Dict[str, Any]] = []
    start = 0
    page = _jira_post_search_jql(jql, start_at=start, max_results=100)
    total = page.get("total", 0)
    issues.extend(page.get("issues", []))

    while len(issues) < total:
        start = len(issues)
        page = _jira_post_search_jql(jql, start_at=start, max_results=100)
        issues.extend(page.get("issues", []))
    return issues

def jql_projeto(project_key: str, ano_selecionado: str, mes_selecionado: str) -> str:
    """
    Monta JQL seguro e com recorte:
      - Sempre usa created >= DATA_INICIO
      - Mês != 'Todos': corta em created < 1º dia do MÊS+1
      - Aspas no nome do projeto (ex.: "INT") para evitar conflito (INT = reserved word)
    """
    base = f'project = "{project_key}" AND created >= "{DATA_INICIO}"'
    if mes_selecionado != "Todos" and ano_selecionado != "Todos":
        try:
            a = int(ano_selecionado)
            m = int(mes_selecionado)
            if m == 12:
                next_month_first = date(a + 1, 1, 1)
            else:
                next_month_first = date(a, m + 1, 1)
            base += f' AND created < "{next_month_first:%Y-%m-%d}"'
        except Exception:
            pass
    return base + " ORDER BY created ASC"

# ===================================
# NORMALIZAÇÃO DOS DADOS
# ===================================

def normalize_issues(raw_issues: List[Dict[str, Any]], project_key: str) -> pd.DataFrame:
    if not raw_issues:
        return pd.DataFrame(columns=[
            "key", "project", "summary", "created", "updated", "resolutiondate",
            "sla_sup", "origem_problema", "qtd_encomendas", "mes_str"
        ])

    rows = []
    for it in raw_issues:
        f = (it.get("fields") or {})
        rows.append({
            "key": it.get("key"),
            "project": project_key,
            "summary": f.get("summary"),
            "created": f.get("created"),
            "updated": f.get("updated"),
            "resolutiondate": f.get("resolutiondate"),
            "sla_sup": f.get("customfield_13744"),
            "origem_problema": f.get("customfield_13628"),
            "qtd_encomendas": f.get("customfield_13666"),
        })
    df = pd.DataFrame(rows)

    # Datas
    for col in ["created", "updated", "resolutiondate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Conversão do SLA (se vier como objeto/str)
    # Deixamos como numérico quando possível
    def _to_float(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # Tenta extrair números de strings
        try:
            return float(str(v).replace(",", "."))
        except Exception:
            return None

    df["sla_sup_num"] = df["sla_sup"].map(_to_float)

    # Encomendas numéricas
    df["qtd_encomendas"] = pd.to_numeric(df["qtd_encomendas"], errors="coerce")

    # Label de mês para gráficos
    df["mes_str"] = df["created"].dt.strftime("%b/%Y")
    return df

# ===================================
# UI: HEADER / FILTROS
# ===================================

st.markdown("## 📊 Painel de Indicadores — Jira")

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🔄 Atualizar dados"):
        st.cache_data.clear()
        st.rerun()
with c2:
    agora_brt = pd.Timestamp.utcnow().tz_convert("America/Sao_Paulo").strftime("%d/%m/%Y %H:%M:%S")
    st.caption(f"🕒 Última atualização (BRT): {agora_brt}")

col_ano, col_mes = st.columns(2)
anos_opcoes  = ["Todos"] + [str(y) for y in range(2024, date.today().year + 1)]
meses_opcoes = ["Todos"] + [f"{m:02d}" for m in range(1, 12 + 1)]
with col_ano:
    ano_global = st.selectbox("Ano (global)", anos_opcoes, index=anos_opcoes.index(str(date.today().year)))
with col_mes:
    mes_global = st.selectbox("Mês (global)", meses_opcoes, index=0)  # "Todos"

# ===================================
# BUSCA DOS DADOS (com tratamento de erro visível)
# ===================================

def _fetch_df(project_key: str) -> pd.DataFrame:
    jql = jql_projeto(project_key, ano_global, mes_global)
    try:
        raw = jira_search_all(jql)
        return normalize_issues(raw, project_key)
    except RuntimeError as e:
        st.error(f"⚠️ Erro ao buscar Jira ({project_key}): {e}")
        return pd.DataFrame()

with st.spinner("Carregando TDS..."):
    df_tds = _fetch_df("TDS")
with st.spinner("Carregando INT..."):
    df_int = _fetch_df("INT")
with st.spinner("Carregando TINE..."):
    df_tine = _fetch_df("TINE")
with st.spinner("Carregando INTEL..."):
    df_intel = _fetch_df("INTEL")

# ===================================
# RENDER: Criados vs Resolvidos
# ===================================

def render_criados_resolvidos(df: pd.DataFrame, titulo: str):
    st.markdown(f"### 📈 {titulo}")
    if df.empty:
        st.info("Sem dados para o período.")
        return

    # criados por mês
    criados = df.groupby("mes_str", as_index=False).agg(qty=("key", "count"))
    criados["variable"] = "Criados"

    # resolvidos por mês (considera resolutiondate, não created)
    resolvidos = (
        df[~df["resolutiondate"].isna()]
        .assign(mes_res=lambda x: x["resolutiondate"].dt.strftime("%b/%Y"))
        .groupby("mes_res", as_index=False).agg(qty=("key", "count"))
        .rename(columns={"mes_res": "mes_str"})
    )
    resolvidos["variable"] = "Resolvidos"

    g = pd.concat([criados, resolvidos], ignore_index=True).sort_values("mes_str", key=lambda s: pd.to_datetime(s, format="%b/%Y"))

    fig = px.bar(
        g, x="mes_str", y="qty", color="variable",
        barmode="group", text="qty", height=420
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="mes_str", yaxis_title="value")
    st.plotly_chart(fig, use_container_width=True)

# ===================================
# RENDER: SLA por mês
# ===================================

def render_sla(df: pd.DataFrame, projeto: str):
    st.markdown(f"### ⏱️ SLA — {projeto}")

    if df.empty:
        st.info("Sem dados para o período.")
        return

    meta = META_SLA.get(projeto, 98.0)

    # Consideramos "resolvido no mês" como denominador
    dfr = df[~df["resolutiondate"].isna()].copy()
    if dfr.empty:
        st.info("Não há tickets resolvidos no período.")
        return

    dfr["mes_res"] = dfr["resolutiondate"].dt.strftime("%b/%Y")

    # Dentro/Fora: se sla_sup_num existir e for numérico, definimos um critério simples:
    # - Supondo SLA alvo = 48h (2880 min), ajuste se precisar. Se não houver sla_sup_num, marcamos tudo como dentro.
    SLA_MIN_ALVO = 2880.0  # 48h em minutos (ajuste conforme sua regra). Mantém estável até termos critério oficial.
    dfr["dentro"] = dfr["sla_sup_num"].apply(lambda v: True if pd.isna(v) else (v <= SLA_MIN_ALVO))

    por_mes = dfr.groupby("mes_res", as_index=False).agg(
        resolvidos=("key", "count"),
        dentro_sla=("dentro", "sum")
    )
    por_mes["fora_sla"] = por_mes["resolvidos"] - por_mes["dentro_sla"]
    por_mes["% Dentro SLA"] = (por_mes["dentro_sla"] / por_mes["resolvidos"]) * 100.0
    por_mes["% Fora SLA"]   = 100.0 - por_mes["% Dentro SLA"]

    # Média (OKR) no período
    okr = por_mes["% Dentro SLA"].mean()

    st.caption(f"**OKR: {okr:,.2f}% — Meta: {meta:,.2f}%**".replace(",", "X").replace(".", ",").replace("X", "."))

    # Gráfico barras lado-a-lado
    longp = por_mes.melt(id_vars="mes_res", value_vars=["% Dentro SLA", "% Fora SLA"],
                         var_name="variable", value_name="value")
    longp = longp.sort_values("mes_res", key=lambda s: pd.to_datetime(s, format="%b/%Y"))

    fig = px.bar(longp, x="mes_res", y="value", color="variable",
                 barmode="group", height=420,
                 color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
                 text=longp["value"].map(lambda v: f"{v:.2f}%"))
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="mes_str", yaxis_title="%", uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

# ===================================
# RENDER: Rotinas Manuais (qtd_encomendas por mês)
# ===================================

def render_rotinas_manuais(df: pd.DataFrame):
    st.markdown("### 🧩 Rotinas Manuais")
    if df.empty:
        st.info("Sem dados para o período.")
        return

    dfe = df.copy()
    dfe["qtd_encomendas"] = pd.to_numeric(dfe["qtd_encomendas"], errors="coerce")
    dfe = dfe.dropna(subset=["qtd_encomendas"])
    if dfe.empty:
        st.info("Não há Quantidade de encomendas registrada nas issues para o período.")
        return

    por_mes = dfe.groupby("mes_str", as_index=False).agg(qtd=("qtd_encomendas", "sum"))
    por_mes = por_mes.sort_values("mes_str", key=lambda s: pd.to_datetime(s, format="%b/%Y"))

    # Outlier info (apenas informativo)
    outliers = por_mes[por_mes["qtd"] > 100_000]
    if not outliers.empty:
        st.caption(f"ℹ️ {len(outliers)} registro(s) com quantidade > 100.000 exibido(s) (apenas informativo).")

    fig = px.bar(por_mes, x="mes_str", y="qtd", text=por_mes["qtd"].map(lambda v: f"{int(v):,}".replace(",", ".")),
                 height=430)
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="mes_str", yaxis_title="Qtd encomendas")
    st.plotly_chart(fig, use_container_width=True)

# ===================================
# (Opcional) Placeholders seguros para seções específicas
# ===================================

def render_onboarding_placeholder():
    with st.expander("Onboarding (INT)"):
        st.caption("⚙️ Placeholder — reconecte aqui sua função de Onboarding antiga, se desejar.")

def render_app_ne_placeholder():
    with st.expander("APP NE (TDS)"):
        st.caption("⚙️ Placeholder — reconecte aqui sua função de APP NE antiga, se desejar.")

# ===================================
# LAYOUT POR PROJETO (mínimo necessário)
# ===================================

st.markdown("---")
st.header("Projeto: Tech Support (TDS)")
render_criados_resolvidos(df_tds, "Tickets Criados vs Resolvidos — TDS")
render_sla(df_tds, "TDS")
render_rotinas_manuais(df_tds)
render_app_ne_placeholder()

st.markdown("---")
st.header("Projeto: Integrations (INT)")
render_criados_resolvidos(df_int, "Tickets Criados vs Resolvidos — INT")
render_sla(df_int, "INT")
render_onboarding_placeholder()

st.markdown("---")
st.header("Projeto: IT Support NE (TINE)")
render_criados_resolvidos(df_tine, "Tickets Criados vs Resolvidos — TINE")
render_sla(df_tine, "TINE")

st.markdown("---")
st.header("Projeto: Intelligence (INTEL)")
render_criados_resolvidos(df_intel, "Tickets Criados vs Resolvidos — INTEL")
render_sla(df_intel, "INTEL")
