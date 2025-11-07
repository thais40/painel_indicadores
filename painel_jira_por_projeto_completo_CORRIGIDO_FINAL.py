# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

# ==================== Carrega credenciais (robusto) ====================
from pathlib import Path
def _load_secrets():
    jira_url = st.secrets.get("JIRA_URL", "")
    email    = st.secrets.get("EMAIL", "")
    token    = st.secrets.get("TOKEN", "")

    if not (email and token):
        try:
            try:
                import tomllib  # py3.11+
            except ModuleNotFoundError:
                import tomli as tomllib  # pip install tomli para <3.11
            for cand in (Path(".streamlit/secrets.toml"), Path("secrets.toml"), Path("secrets_example.toml")):
                if cand.exists():
                    with cand.open("rb") as f:
                        data = tomllib.load(f)
                    jira_url = jira_url or data.get("JIRA_URL", "")
                    email    = email    or data.get("EMAIL", "")
                    token    = token    or data.get("TOKEN", "")
                    break
        except Exception:
            pass

    jira_url = jira_url or os.environ.get("JIRA_URL", "")
    email    = email    or os.environ.get("EMAIL", "")
    token    = token    or os.environ.get("TOKEN", "")

    jira_url = jira_url.strip()
    email    = email.strip()
    token    = token.strip()
    return jira_url, email, token

JIRA_URL, EMAIL, TOKEN = _load_secrets()
if not JIRA_URL:
    JIRA_URL = "https://tiendanube.atlassian.net"

if not EMAIL or not TOKEN:
    st.error("⚠️ Configure JIRA_URL, EMAIL e TOKEN (App secrets do Streamlit Cloud ou .streamlit/secrets.toml).")
    st.stop()

auth = HTTPBasicAuth(EMAIL, TOKEN)

# ==================== Config geral ====================
st.set_page_config(page_title="Painel de Indicadores", page_icon="📊", layout="wide")
TZ_BR = ZoneInfo("America/Sao_Paulo")
DATA_INICIO = "2024-06-01"

# Defaults defensivos para filtros globais (evitam NameError)
if "ano_global" not in st.session_state:
    st.session_state["ano_global"] = "Todos"
if "mes_global" not in st.session_state:
    st.session_state["mes_global"] = "Todos"
try:
    ano_global  # noqa
except NameError:
    ano_global = st.session_state.get("ano_global", "Todos")
try:
    mes_global  # noqa
except NameError:
    mes_global = st.session_state.get("mes_global", "Todos")

# ==================== Campos / Constantes Jira ====================
SLA_CAMPOS = {
    "TDS": "customfield_13744",
    "TINE": "customfield_13744",
    "INT": "customfield_13686",
    "INTEL": "customfield_13686",
}

ASSUNTO_TDS_PRIMARY = "customfield_13747"   # Assunto Relacionado (TDS)
ASSUNTO_TDS_FALLBACK = "customfield_13712"  # fallback legacy (TDS)

CAMPOS_ASSUNTO = {
    "TDS": ASSUNTO_TDS_PRIMARY,
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype",
}

CAMPO_AREA = "customfield_13719"           # Área Solicitante
CAMPO_N3 = "customfield_13659"
CAMPO_ORIGEM = "customfield_13628"
CAMPO_QTD_ENCOMENDAS = "customfield_13666"

PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {"TDS": "Tech Support", "INT": "Integrations", "TINE": "IT Support NE", "INTEL": "Intelligence"}
META_SLA = {"TDS": 98.00, "INT": 96.00, "TINE": 96.00, "INTEL": 96.00}
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

JIRA_FIELDS_BASE = [
    "key", "summary", "created", "updated", "resolutiondate", "resolved", "status", "issuetype",
    CAMPO_AREA, CAMPO_N3, CAMPO_ORIGEM, CAMPO_QTD_ENCOMENDAS,
]
FIELDS_SLA_ALL = list(set(SLA_CAMPOS.values()))
FIELDS_ASSUNTO_ALL = list(set([v for v in CAMPOS_ASSUNTO.values() if v != "issuetype"]))
FIELDS_ALL: List[str] = list(dict.fromkeys(JIRA_FIELDS_BASE + FIELDS_SLA_ALL + FIELDS_ASSUNTO_ALL + [ASSUNTO_TDS_FALLBACK]))

# ==================== UI topo ====================
def now_br_str() -> str:
    return datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S")

st.title("📊 Painel de Indicadores")
if "last_update" not in st.session_state:
    st.session_state["last_update"] = now_br_str()

colu1, colu2 = st.columns([0.18, 0.82])
with colu1:
    if st.button("🔄 Atualizar dados"):
        st.cache_data.clear()
        st.session_state["last_update"] = now_br_str()
        st.rerun()
with colu2:
    st.caption(f"🕒 Última atualização: {st.session_state['last_update']} (BRT)")

# ==================== Helpers ====================
def show_plot(fig, nome_bloco: str, projeto: str, ano: str, mes: str):
    st.plotly_chart(fig, use_container_width=True, key=f"plt-{nome_bloco}-{projeto}-{ano}-{mes}-{uuid4()}")

def safe_get_value(x, key: str = "value", fallback: str = "—"):
    if isinstance(x, dict):
        return x.get(key, fallback)
    return x if x is not None else fallback

def dentro_sla_from_raw(sla_raw: dict) -> Optional[bool]:
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

def parse_qtd_encomendas(v) -> int:
    if isinstance(v, list):
        v = next((x for x in reversed(v) if x not in (None, "")), None)
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        try:
            return int(round(float(v)))
        except Exception:
            return 0
    s = str(v).strip().replace(".", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except Exception:
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else 0

def _canonical(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def aplicar_filtro_global(df_in: pd.DataFrame, col_dt: str, ano: str, mes: str) -> pd.DataFrame:
    out = df_in.copy()
    if ano != "Todos":
        out = out[out[col_dt].dt.year == int(ano)]
    if mes != "Todos":
        out = out[out[col_dt].dt.month == int(mes)]
    return out

def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if df_proj is None or df_proj.empty:
        return df_proj
    def _from_field(v):
        if isinstance(v, dict):
            return v.get("value") or v.get("name") or str(v)
        return v
    col = "assunto_nome"
    if col not in df_proj.columns:
        if CAMPOS_ASSUNTO.get(projeto) == "issuetype":
            df_proj[col] = df_proj["issuetype"].apply(_from_field)
        else:
            df_proj[col] = df_proj["assunto"].apply(_from_field)
    if df_proj[col].isna().all() or (df_proj[col].astype(str).str.strip() == "").all():
        if CAMPOS_ASSUNTO.get(projeto) == "issuetype":
            df_proj[col] = df_proj["issuetype"].apply(_from_field)
        else:
            df_proj[col] = df_proj["assunto"].apply(_from_field)
    return df_proj

# ==================== Jira Search ====================
def _jira_search_jql(jql: str, next_page_token: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    params = {"jql": jql, "fields": ",".join(FIELDS_ALL), "maxResults": max_results}
    if next_page_token:
        params["nextPageToken"] = next_page_token
    try:
        resp = requests.get(url, params=params, auth=auth, timeout=60)
    except Exception as e:
        return {"error": str(e), "issues": [], "isLast": True}
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:300]}", "issues": [], "isLast": True}
    return resp.json()

@st.cache_data(show_spinner="🔄 Buscando dados do Jira...", ttl=60*30)
def buscar_issues(projeto: str, jql: str, max_pages: int = 500) -> pd.DataFrame:
    todos, last_error = [], None
    next_token, page = None, 0
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
            if projeto == "TDS":
                assunto_val = f.get(ASSUNTO_TDS_PRIMARY) or f.get(ASSUNTO_TDS_FALLBACK) or f.get("issuetype")
            else:
                assunto_val = f.get(CAMPOS_ASSUNTO[projeto]) if CAMPOS_ASSUNTO[projeto] != "issuetype" else f.get("issuetype")
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
                "assunto": assunto_val,
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
        for c in ("created", "resolved", "resolutiondate", "updated"):
            dfp[c] = pd.to_datetime(dfp[c], errors="coerce", utc=True).dt.tz_convert(TZ_BR).dt.tz_localize(None)
        dfp["mes_created"] = dfp["created"].dt.to_period("M").dt.to_timestamp()
        dfp["mes_resolved"] = dfp["resolved"].dt.to_period("M").dt.to_timestamp()
    return dfp

# ==================== Builders ====================
def build_monthly_tables(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    base = df_all.copy()
    base["per_created"] = base["created"].dt.to_period("M")
    base["per_resolved"] = base["resolved"].dt.to_period("M")

    created = base.groupby(["projeto", "per_created"]).size().reset_index(name="Criados").rename(columns={"per_created":"period"})
    res = base[base["resolved"].notna()].copy()
    res["dentro_sla"] = res["sla_raw"].apply(dentro_sla_from_raw).fillna(False)
    resolved = res.groupby(["projeto", "per_resolved"]).agg(Resolvidos=("key","count"), Dentro=("dentro_sla","sum")).reset_index().rename(columns={"per_resolved":"period"})

    monthly = pd.merge(created, resolved, how="outer", on=["projeto","period"]).fillna(0)
    monthly["period"] = monthly["period"].astype("period[M]")
    monthly["period_ts"] = monthly["period"].dt.to_timestamp()
    monthly["ano"] = monthly["period"].dt.year.astype(int)
    monthly["mes"] = monthly["period"].dt.month.astype(int)
    monthly["mes_str"] = monthly["period_ts"].dt.strftime("%b/%Y")
    monthly["Dentro"] = monthly["Dentro"].astype(int)
    monthly["Resolvidos"] = monthly["Resolvidos"].astype(int)
    monthly["Fora"] = (monthly["Resolvidos"] - monthly["Dentro"]).clip(lower=0)
    monthly["pct_dentro"] = monthly.apply(lambda r: (r["Dentro"]/r["Resolvidos"]*100) if r["Resolvidos"]>0 else 0.0, axis=1).round(2)
    monthly["pct_fora"] = monthly.apply(lambda r: (r["Fora"]/r["Resolvidos"]*100) if r["Resolvidos"]>0 else 0.0, axis=1).round(2)
    return monthly.sort_values(["projeto","period"])

# ==================== Renders genéricos ====================
# Compat: aceita antigo (dfp, projeto, ano, mes) e novo (dfp, ano, mes, projeto=None)
def _render_criados_resolvidos_core(dfp: pd.DataFrame, ano_global: str="Todos", mes_global: str="Todos", projeto: Optional[str]=None):
    df = dfp.copy()
    created  = pd.to_datetime(df.get("created"),  errors="coerce")
    resolved = pd.to_datetime(df.get("resolved"), errors="coerce")

    cdf = (df[created.notna()].assign(created=created[created.notna()]).sort_values(["key","created"]).drop_duplicates("key", keep="first").copy())
    if not cdf.empty:
        cdf["mes_dt"] = cdf["created"].dt.to_period("M").dt.to_timestamp()
    rdf = (df[resolved.notna()].assign(resolved=resolved[resolved.notna()]).sort_values(["key","resolved"]).drop_duplicates("key", keep="last").copy())
    if not rdf.empty:
        rdf["mes_dt"] = rdf["resolved"].dt.to_period("M").dt.to_timestamp()

    if cdf.empty and rdf.empty:
        st.info("Sem dados de criação/resolução para montar a série.")
        return

    mins = [x["mes_dt"].min() for x in (cdf, rdf) if not x.empty]
    maxs = [x["mes_dt"].max() for x in (cdf, rdf) if not x.empty]
    idx = pd.date_range(min(mins), max(maxs), freq="MS")

    s_criados    = (cdf.groupby("mes_dt")["key"].nunique() if not cdf.empty else pd.Series(dtype=int)).reindex(idx, fill_value=0).rename("Criados")
    s_resolvidos = (rdf.groupby("mes_dt")["key"].nunique() if not rdf.empty else pd.Series(dtype=int)).reindex(idx, fill_value=0).rename("Resolvidos")

    monthly = pd.concat([s_criados, s_resolvidos], axis=1).reset_index().rename(columns={"index":"mes_dt"})
    monthly = aplicar_filtro_global(monthly, "mes_dt", ano_global, mes_global)
    if monthly.empty:
        st.info("Sem dados para os filtros selecionados.")
        return

    monthly["mes_str"] = monthly["mes_dt"].dt.strftime("%b/%Y")
    title = f"Tickets Criados vs Resolvidos — {projeto}" if projeto else "Tickets Criados vs Resolvidos"
    fig = px.bar(monthly, x="mes_str", y=["Criados","Resolvidos"], barmode="group", text_auto=True, title=title, height=420)
    fig.update_traces(textangle=0, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())
    show_plot(fig, f"{(projeto or 'all').lower().replace(' ', '_')}_criados_resolvidos_all", (projeto or 'ALL'), ano_global, mes_global)

def render_criados_resolvidos(*args, **kwargs):
    if len(args) == 4 and isinstance(args[1], str):
        dfp, projeto, ano, mes = args
        return _render_criados_resolvidos_core(dfp, ano, mes, projeto=projeto)
    if len(args) >= 3:
        dfp, ano, mes = args[:3]
        projeto = kwargs.get("projeto", None)
        return _render_criados_resolvidos_core(dfp, ano, mes, projeto=projeto)
    dfp = kwargs.get("dfp")
    ano = kwargs.get("ano_global", "Todos")
    mes = kwargs.get("mes_global", "Todos")
    projeto = kwargs.get("projeto", None)
    if dfp is None:
        st.error("Chamada inválida para render_criados_resolvidos.")
        return
    return _render_criados_resolvidos_core(dfp, ano, mes, projeto=projeto)

def render_sla_table(df_monthly_all: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
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
    show = show.rename(columns={"pct_dentro":"% Dentro SLA", "pct_fora":"% Fora SLA"})
    fig = px.bar(show, x="mes_str", y=["% Dentro SLA","% Fora SLA"], barmode="group", title=titulo, height=440)
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    show_plot(fig, "sla", projeto, ano_global, mes_global)

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
    assunto_count.columns = ["Assunto", "Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)

def render_area(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📦 Área Solicitante")
    df_area = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_area.empty:
        st.info("Sem dados para Área Solicitante nos filtros atuais.")
        return
    df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
    area_count = df_area["area_nome"].value_counts().reset_index()
    area_count.columns = ["Área", "Qtd"]
    st.dataframe(area_count, use_container_width=True, hide_index=True)

def render_encaminhamentos(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🔄 Encaminhamentos")
    df_enc = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_enc.empty:
        st.info("Sem dados para Encaminhamentos nos filtros atuais.")
        return
    col1, col2 = st.columns(2)
    with col1:
        count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
        st.metric("Encaminhados Produto", int(count_prod))
    with col2:
        df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
        st.metric("Encaminhados N3", int((df_enc["n3_valor"] == "Sim").sum()))

# ==================== Módulos específicos ====================
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
    df_app["origem_cat"] = df_app["origem_nome"].apply(normaliza_origem)
    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()
    df_app = aplicar_filtro_global(df_app, "mes_dt", ano_global, mes_global)
    if df_app.empty:
        st.info("Sem dados para exibir com os filtros selecionados.")
        return

    serie = df_app.groupby(["mes_dt","origem_cat"]).size().reset_index(name="Qtd").sort_values("mes_dt")
    serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
    cats = serie["mes_str"].dropna().unique().tolist()
    serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

    fig_app = px.bar(
        serie, x="mes_str", y="Qtd", color="origem_cat", barmode="group",
        title="APP NE — Volumes por mês e Origem do problema",
        text="Qtd", height=460
    )
    fig_app.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=16, cliponaxis=False)
    max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
    if max_qtd > 0:
        fig_app.update_yaxes(range=[0, max_qtd*1.25])
    fig_app.update_layout(
        yaxis_title="Qtd", xaxis_title="Mês",
        uniformtext_minsize=14, uniformtext_mode="show",
        bargap=0.15, margin=dict(t=70, r=20, b=60, l=50),
    )
    show_plot(fig_app, "app_ne", "TDS", ano_global, mes_global)

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets clientes novos", total_clientes_novos)
    c2.metric("Erros onboarding", int(len(df_erros)))
    c3.metric("Tickets com pendências", tickets_pendencias)
    c4.metric("Possíveis clientes", possiveis_clientes)

    df_cli_novo = df_onb[df_onb["assunto_nome"].astype(str).str.contains("cliente novo", case=False, na=False)].copy()
    if not df_cli_novo.empty:
        serie = df_cli_novo.groupby(pd.Grouper(key="created", freq="MS")).size().rename("qtd").reset_index()
        if not serie.empty:
            idx = pd.date_range(serie["created"].min(), serie["created"].max(), freq="MS")
            serie = serie.set_index("created").reindex(idx).fillna(0.0).rename_axis("created").reset_index()
            serie["qtd"] = serie["qtd"].astype(int)
            serie["mes_str"] = serie["created"].dt.strftime("%Y %b")
            fig_cli = px.bar(serie, x="mes_str", y="qtd", text="qtd", title="Tickets – Cliente novo", height=420)
            fig_cli.update_traces(textposition="outside", cliponaxis=False)
            fig_cli.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            show_plot(fig_cli, "onb_cli_novo", "INT", ano_global, mes_global)

    def _tipo_from_assunto(s: str) -> str:
        s = (s or "").strip().lower()
        if "cliente novo" in s:
            return "Cliente novo"
        if "alteração" in s and "plataforma" in s:
            return "Alteração de plataforma"
        if "conta filho" in s:
            return "Conta filho"
        return "Outros"

    tipo_counts = df_onb.assign(tipo=df_onb["assunto_nome"].map(_tipo_from_assunto)).groupby("tipo").size().rename("Qtd").reset_index()
    priority = ["Cliente novo", "Outros", "Alteração de plataforma", "Conta filho"]
    tipo_counts["ord"] = tipo_counts["tipo"].apply(lambda x: priority.index(x) if x in priority else len(priority)+1)
    tipo_counts = tipo_counts.sort_values(["ord","Qtd"], ascending=[True, False])
    fig_tipo = px.bar(tipo_counts, x="Qtd", y="tipo", orientation="h", text="Qtd", title="Tipo de Integração", height=420)
    fig_tipo.update_traces(textposition="outside", cliponaxis=False)
    fig_tipo.update_layout(margin=dict(l=10, r=90, t=45, b=10))
    try:
        _max_q = float(tipo_counts["Qtd"].max())
        fig_tipo.update_xaxes(range=[0, _max_q*1.12])
    except Exception:
        pass
    show_plot(fig_tipo, "onb_tipo_int", "INT", ano_global, mes_global)

# ==================== (Coloque aqui sua render_rotinas_manuais atual) ====================
# Mantive seu comportamento acordado: TDS fixo (Ops) para TDS e manuais por Assunto.
# Se você já tem essa função no seu arquivo, pode manter a sua versão.
def render_rotinas_manuais(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    import plotly.express as px
    st.markdown("### 🛠️ Rotinas Manuais — **TDS (Ops)** vs **Manuais por Assunto**")
    if dfp.empty:
        st.info("Sem tickets para o período.")
        return

    try:
        from unidecode import unidecode as _unidecode
    except Exception:
        _unidecode = lambda s: s

    def _canon(s: str) -> str:
        s = str(s or "")
        s = s.replace("–","-").replace("—","-")
        s = _unidecode(s).lower()
        return " ".join(s.split())

    def _parse_dt_col(s):
        x = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        if x.notna().sum() == 0:
            x = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)
        return x

    OPS_AREAS = ["Ops - Conferência","Ops - Cubagem","Ops - Logística","Ops - Coletas","Ops - Expedição","Ops - Divergências"]

    SUBJECT_GROUPS = [
        ("Volumetria - IE / Qliksense",               "Inscrição Estadual"),
        ("Erro no processamento - Inscrição Estadual", "Inscrição Estadual"),
        ("Erro no processamento - CTE",                "CTE"),
        ("Volumetria - Tabela Erro",                   "CTE"),
        ("Volumetria - Tabela Divergência",            "Divergência"),
        ("Volumetria - Cotação/Grafana",               "Cotação"),
        ("Volumetria - Painel sem registro",           "Outros"),
    ]
    GROUP_NEEDLES = [(_canon(src), label) for src, label in SUBJECT_GROUPS]

    df = dfp.copy()
    df = ensure_assunto_nome(df, "TDS")
    df["assunto_nome"] = df["assunto_nome"].astype(str)
    df["area_nome"] = df["area"].apply(lambda x: safe_get_value(x, "value"))

    df["qtd_encomendas"] = df[CAMPO_QTD_ENCOMENDAS].apply(parse_qtd_encomendas)
    df = df[df["qtd_encomendas"] > 0].copy()
    if df.empty:
        st.info("Sem tickets com 'Quantidade de encomendas' > 0.")
        return

    df["resolved"] = _parse_dt_col(df.get("resolved"))
    df["created"]  = _parse_dt_col(df.get("created"))
    df["updated"]  = _parse_dt_col(df.get("updated"))

    df["_best_dt"] = df["resolved"]
    m = df["_best_dt"].isna() & df["created"].notna()
    df.loc[m, "_best_dt"] = df.loc[m, "created"]
    m = df["_best_dt"].isna() & df["updated"].notna()
    df.loc[m, "_best_dt"] = df.loc[m, "updated"]
    df = df[df["_best_dt"].notna()].copy()

    df = (df.sort_values(["key","_best_dt"]).groupby("key", as_index=False).agg({
        "_best_dt":"min","qtd_encomendas":"max","assunto_nome":"first","summary":"first","area_nome":"first"
    }).rename(columns={"_best_dt":"resolved"}).copy())

    df["mes_dt"] = df["resolved"].dt.to_period("M").dt.to_timestamp()
    df["assunto_canon"] = df["assunto_nome"].apply(_canon)

    df_ops = df[df["area_nome"].isin(OPS_AREAS)].copy()

    def _is_manual(canon_text: str) -> bool:
        return any(needle in canon_text for needle, _label in GROUP_NEEDLES)

    df_manual = df[df["assunto_canon"].apply(_is_manual)].copy()

    if not df_ops.empty:
        df_ops = aplicar_filtro_global(df_ops, "mes_dt", ano_global, mes_global)
    if not df_manual.empty:
        df_manual = aplicar_filtro_global(df_manual, "mes_dt", ano_global, mes_global)

    if df_ops.empty and df_manual.empty:
        st.info("Sem dados para exibir com os filtros atuais.")
        return

    monthly_tds = (df_ops.groupby("mes_dt")["qtd_encomendas"].sum().rename("Encomendas TDS") if not df_ops.empty else pd.Series(dtype=float, name="Encomendas TDS"))
    monthly_man = (df_manual.groupby("mes_dt")["qtd_encomendas"].sum().rename("Encomendas manuais") if not df_manual.empty else pd.Series(dtype=float, name="Encomendas manuais"))

    min_m = pd.concat([
        df_ops["mes_dt"] if not df_ops.empty else pd.Series(dtype="datetime64[ns]"),
        df_manual["mes_dt"] if not df_manual.empty else pd.Series(dtype="datetime64[ns]")
    ]).min()
    max_m = pd.concat([
        df_ops["mes_dt"] if not df_ops.empty else pd.Series(dtype="datetime64[ns]"),
        df_manual["mes_dt"] if not df_manual.empty else pd.Series(dtype="datetime64[ns]")
    ]).max()
    idx = pd.date_range(min_m, max_m, freq="MS")

    s_tds    = monthly_tds.reindex(idx, fill_value=0.0)
    s_manual = monthly_man.reindex(idx,  fill_value=0.0)

    monthly = pd.concat([s_manual, s_tds], axis=1).reset_index().rename(columns={"index":"mes_dt"})
    monthly["mes_str"] = monthly["mes_dt"].dt.strftime("%b/%Y")

    fig = px.bar(monthly, x="mes_str", y=["Encomendas manuais","Encomendas TDS"], barmode="group", text_auto=True, title="Encomendas manuais (Assunto) **vs** Encomendas TDS (Ops fixo)", height=420)
    fig.update_traces(textangle=0, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())
    show_plot(fig, "rotinas_manuais_assunto_vs_tds_ops_fixo", "TDS", ano_global, mes_global)

    manual_sum = float(s_manual.sum()); tds_sum = float(s_tds.sum())
    df_donut = pd.DataFrame({"tipo":["Encomendas manuais","Encomendas TDS"], "qtd":[manual_sum, tds_sum]})
    fig_donut = px.pie(df_donut, values="qtd", names="tipo", hole=0.6, title="Totais independentes — pode haver sobreposição")
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    show_plot(fig_donut, "rotinas_manuais_assunto_donut_tds_ops_fixo", "TDS", ano_global, mes_global)

    st.caption("**Obs.:** TDS é fixo (apenas Ops). Pode haver sobreposição entre Manuais e TDS.")

    if not df_manual.empty:
        def _bucket_row(canon_text: str) -> str:
            for needle, label in GROUP_NEEDLES:
                if needle in canon_text:
                    return label
            return "Outros"
        df_manual["bucket_assunto"] = df_manual["assunto_canon"].apply(_bucket_row)
        serie = (df_manual.groupby("bucket_assunto")["qtd_encomendas"].sum().sort_values(ascending=False))
        TOP = 5
        if len(serie) > TOP:
            serie_plot = pd.concat([serie.iloc[:TOP-1], pd.Series({"Outros": serie.iloc[TOP-1:].sum()})])
        else:
            serie_plot = serie
        df_ass = serie_plot.rename_axis("assunto").reset_index(name="qtd")
        fig_ass = px.bar(df_ass, x="qtd", y="assunto", orientation="h", text="qtd", title="Manual | Assunto", height=380)
        fig_ass.update_yaxes(categoryorder="total ascending")
        fig_ass.update_traces(textposition="inside", texttemplate="%{text:,.0f}")
        fig_ass.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        show_plot(fig_ass, "rotinas_manuais_breakdown_assunto", "TDS", ano_global, mes_global)

# ==================== Coleta & Vistas ====================
def jql_projeto(project_key: str, ano_sel: str, mes_sel: str) -> str:
    base = f'project = "{project_key}" AND created >= "{DATA_INICIO}"'
    if mes_sel != "Todos" and ano_sel != "Todos":
        a = int(ano_sel); m = int(mes_sel)
        if m == 12:
            next_month_first = date(a+1, 1, 1)
        else:
            next_month_first = date(a, m+1, 1)
        base += f' AND created < "{next_month_first:%Y-%m-%d}"'
    return base + " ORDER BY created ASC"

JQL_TDS   = jql_projeto("TDS", "Todos", "Todos")
JQL_INT   = jql_projeto("INT", "Todos", "Todos")
JQL_TINE  = jql_projeto("TINE", "Todos", "Todos")
JQL_INTEL = jql_projeto("INTEL", "Todos", "Todos")

with st.spinner("Carregando TDS..."):   df_tds = buscar_issues("TDS", JQL_TDS)
with st.spinner("Carregando INT..."):   df_int = buscar_issues("INT", JQL_INT)
with st.spinner("Carregando TINE..."):  df_tine = buscar_issues("TINE", JQL_TINE)
with st.spinner("Carregando INTEL..."): df_intel = buscar_issues("INTEL", JQL_INTEL)

if all(d.empty for d in [df_tds, df_int, df_tine, df_intel]):
    st.warning("Sem dados do Jira em nenhum projeto (verifique credenciais e permissões).")
    st.stop()

_df_monthly_all = pd.concat(
    [build_monthly_tables(d) for d in [df_tds, df_int, df_tine, df_intel] if not d.empty],
    ignore_index=True,
) if not all(d.empty for d in [df_tds, df_int, df_tine, df_intel]) else pd.DataFrame()

tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")
        if projeto == "TDS":
            dfp = df_tds.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante","APP NE","Rotinas Manuais"]
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

        dfp["mes_created"]  = pd.to_datetime(dfp["created"],  errors="coerce")
        dfp["mes_resolved"] = pd.to_datetime(dfp["resolved"], errors="coerce")
        dfp = ensure_assunto_nome(dfp, projeto)

        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")

        if visao == "Criados vs Resolvidos":
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)  # compat
        elif visao == "SLA":
            render_sla_table(_df_monthly_all, projeto, ano_global, mes_global)
        elif visao == "Assunto Relacionado":
            render_assunto(dfp, projeto, ano_global, mes_global)
        elif visao == "Área Solicitante":
            if projeto == "INTEL":
                st.info("Este projeto não possui Área Solicitante.")
            else:
                render_area(dfp, ano_global, mes_global)
        elif visao == "Onboarding":
            if projeto == "INT":
                render_onboarding(dfp, ano_global, mes_global)
            else:
                st.info("Onboarding disponível apenas para Integrations.")
        elif visao == "APP NE":
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
            else:
                st.info("APP NE disponível apenas para Tech Support.")
        elif visao == "Rotinas Manuais":
            if projeto == "TDS":
                render_rotinas_manuais(dfp, ano_global, mes_global)
            else:
                st.info("Rotinas Manuais disponível apenas para Tech Support.")
        else:
            # Geral
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
            render_sla_table(_df_monthly_all, projeto, ano_global, mes_global)
            render_assunto(dfp, projeto, ano_global, mes_global)
            if projeto != "INTEL":
                render_area(dfp, ano_global, mes_global)
            if projeto in ("TDS", "INT"):
                render_encaminhamentos(dfp, ano_global, mes_global)
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
                with st.expander("🛠️ Rotinas Manuais", expanded=False):
                    render_rotinas_manuais(dfp, ano_global, mes_global)
            if projeto == "INT":
                with st.expander("🧭 Onboarding", expanded=False):
                    render_onboarding(dfp, ano_global, mes_global)

st.markdown("---")
st.caption("💙 Desenvolvido por Thaís Franco.")
