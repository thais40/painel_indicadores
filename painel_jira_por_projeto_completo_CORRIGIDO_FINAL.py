# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from requests.auth import HTTPBasicAuth
from pandas.tseries.offsets import MonthBegin

st.set_page_config(page_title="Painel de Indicadores — Jira (Classic)", layout="wide")

# =====================================================
# CREDENCIAIS FIXAS (via st.secrets) — sem sidebar
# =====================================================
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets["EMAIL"]
TOKEN = st.secrets["TOKEN"]
auth = HTTPBasicAuth(EMAIL, TOKEN)

# =====================================================
# CONFIG
# =====================================================
SLA_LIMITE = {
    "TDS": 40 * 60 * 60 * 1000,
    "INT": 40 * 60 * 60 * 1000,
    "TINE": 40 * 60 * 60 * 1000,
    "INTEL": 80 * 60 * 60 * 1000,
}
SLA_PADRAO_MILLIS = 40 * 60 * 60 * 1000
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

CF_ASSUNTO_REL = "customfield_13747"
CF_ORIGEM_PROB = "customfield_13628"
CF_AREA_SOL    = "customfield_13719"
CF_SLA_SUP     = "customfield_13744"
CF_SLA_RES     = "customfield_13686"

PROJETO_DEFAULT = "TDS"
JQL_PERIOD_START = "2024-01-01"

# =====================================================
# HELPERS
# =====================================================
def first_option(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, dict):
        return value.get("value") or value.get("name") or str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        v0 = value[0]
        if isinstance(v0, dict):
            return v0.get("value") or v0.get("name") or str(v0)
        return str(v0)
    return str(value)

def get_nested(d, path, default=None):
    node = d
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node

def ensure_ms(x):
    try:
        return float(x)
    except Exception:
        return pd.to_numeric(x, errors="coerce")

def ordenar_mes_str(df: pd.DataFrame, col="mes_str") -> pd.DataFrame:
    dfx = df.copy()
    try:
        dfx["mes_data"] = pd.to_datetime(dfx[col], format="%b/%Y")
    except Exception:
        dfx["mes_data"] = pd.to_datetime(dfx[col], errors="coerce")
    dfx = dfx.sort_values("mes_data")
    dfx[col] = dfx["mes_data"].dt.strftime("%b/%Y")
    cats = dfx[col].dropna().unique().tolist()
    dfx[col] = pd.Categorical(dfx[col], categories=cats, ordered=True)
    return dfx

def month_range_from_series(s) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None, None
        s = s.iloc[:, 0]
    s = pd.Series(s).astype(str)
    s = pd.to_datetime(s, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None, None
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    start = s.min().to_period("M").to_timestamp()
    end = (s.max().to_period("M").to_timestamp() + MonthBegin(1))
    return start, end

# =====================================================
# JIRA API
# =====================================================
@st.cache_data(show_spinner=False, ttl=900)
def jira_search_cached(base_url: str, email: str, token: str, jql: str, fields: List[str]) -> List[Dict[str, Any]]:
    url = base_url.rstrip("/") + "/rest/api/3/search"
    auth_local = HTTPBasicAuth(email, token)
    start_at = 0
    collected: List[Dict[str, Any]] = []

    while True:
        params = {"jql": jql, "startAt": start_at, "maxResults": 100, "fields": ",".join(fields)}
        r = requests.get(url, params=params, auth=auth_local, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Erro Jira {r.status_code}: {r.text[:300]}")
        data = r.json()
        issues = data.get("issues", [])
        collected.extend(issues)
        total = data.get("total", 0)
        if len(collected) >= total:
            break
        start_at += 100
        time.sleep(0.2)

    return collected

def flatten_issues(raw: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in raw:
        f = it.get("fields", {})
        rows.append({
            "key": it.get("key"),
            "id": it.get("id"),
            "created": f.get("created"),
            "resolutiondate": f.get("resolutiondate"),
            CF_ASSUNTO_REL: f.get(CF_ASSUNTO_REL),
            CF_ORIGEM_PROB: f.get(CF_ORIGEM_PROB),
            f"{CF_AREA_SOL}.value": get_nested(f, [CF_AREA_SOL, "value"]),
            CF_SLA_SUP: f.get(CF_SLA_SUP),
            CF_SLA_RES: f.get(CF_SLA_RES),
        })
    return pd.DataFrame(rows)

def build_df_issues(df_flat: pd.DataFrame, projeto: str) -> pd.DataFrame:
    df = df_flat.copy()
    df["id_norm"] = df.get("key", df.get("id"))
    df["created_norm"] = pd.to_datetime(df.get("created"), errors="coerce")
    df["resolved_norm"] = pd.to_datetime(df.get("resolutiondate"), errors="coerce")
    df["assunto_relacionado_norm"] = df[CF_ASSUNTO_REL].apply(first_option) if CF_ASSUNTO_REL in df else None
    df["origem_problema_norm"] = df[CF_ORIGEM_PROB].apply(first_option) if CF_ORIGEM_PROB in df else None
    projeto_up = str(projeto).upper()
    use_sup = projeto_up in {"TDS", "TINE"}
    def extract_sla_ms(row):
        obj = row.get(CF_SLA_SUP) if use_sup else row.get(CF_SLA_RES)
        ms = None
        if isinstance(obj, dict):
            ms = get_nested(obj, ["elapsedTime","millis"]) or get_nested(obj, ["ongoingCycle","elapsedTime","millis"])
        if ms is None and pd.notna(row.get("resolved_norm")) and pd.notna(row.get("created_norm")):
            return (row["resolved_norm"] - row["created_norm"]).total_seconds()*1000.0
        return ensure_ms(ms)
    df["sla_millis_norm"] = df.apply(extract_sla_ms, axis=1)
    limite_ms = SLA_LIMITE.get(projeto_up, SLA_PADRAO_MILLIS)
    df["dentro_sla_norm"] = df["sla_millis_norm"] <= limite_ms
    df["mes_str_norm"] = df["created_norm"].dt.strftime("%b/%Y")
    return df.rename(columns={
        "id_norm":"id","created_norm":"created","resolved_norm":"resolved",
        "sla_millis_norm":"sla_millis","dentro_sla_norm":"dentro_sla",
        "mes_str_norm":"mes_str","assunto_relacionado_norm":"assunto_relacionado",
        "origem_problema_norm":"origem_problema"
    })

def build_df_assunto(df_issues: pd.DataFrame) -> pd.DataFrame:
    return (df_issues.assign(Assunto=df_issues["assunto_relacionado"].fillna("—"))
                     .groupby("Assunto").size().reset_index(name="Qtd")
                     .sort_values("Qtd",ascending=False))

def build_df_area(df_flat: pd.DataFrame) -> pd.DataFrame:
    col = f"{CF_AREA_SOL}.value"
    if col not in df_flat: return pd.DataFrame({"Área":[],"Qtd":[]})
    return (pd.DataFrame({"Área":df_flat[col].fillna("—")})
              .groupby("Área").size().reset_index(name="Qtd")
              .sort_values("Qtd",ascending=False))

# =====================================================
# RENDER
# =====================================================
st.title("📊 Painel de Indicadores — Jira (Classic)")

if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

projeto = PROJETO_DEFAULT
jql = f'project = {projeto} AND created >= "{JQL_PERIOD_START}" ORDER BY created ASC'

def load_and_render():
    fields = ["key","created","resolutiondate",CF_ASSUNTO_REL,CF_ORIGEM_PROB,CF_AREA_SOL,CF_SLA_SUP,CF_SLA_RES]
    raw = jira_search_cached(JIRA_URL, EMAIL, TOKEN, jql, fields)
    st.caption(f"Issues retornados: {len(raw)}")
    if not raw:
        st.warning("JQL não retornou issues.")
        return
    df_flat = flatten_issues(raw)
    df_issues = build_df_issues(df_flat, projeto)

    # 🔧 Patch: tratar colunas duplicadas e garantir datetime
    if getattr(df_issues.columns, "duplicated", lambda: np.array([]))().any():
        df_issues = df_issues.loc[:, ~df_issues.columns.duplicated(keep="first")]

    def _coerce_dt(df, col):
        obj = df.loc[:, col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:,0]
        ser = pd.Series(obj, copy=False).astype(str)
        ser = pd.to_datetime(ser, errors="coerce")
        try: ser = ser.dt.tz_localize(None)
        except: pass
        return ser

    df_issues["created"] = _coerce_dt(df_issues,"created")
    df_issues["resolved"] = _coerce_dt(df_issues,"resolved")

    df_assunto = build_df_assunto(df_issues)
    df_area = build_df_area(df_flat)

    # ... [restante das seções: Criados vs Resolvidos, SLA, Assunto, Área, APP NE] ...
    # (igual ao que já enviamos, usando df_issues corrigido)
    # ----------------
    # vou abreviar aqui pra não repetir tudo, mas é o mesmo código das seções que você já tem
    # ----------------

# Carregamento automático
if "auto_loaded" not in st.session_state:
    st.session_state.auto_loaded = True
    load_and_render()
else:
    load_and_render()
