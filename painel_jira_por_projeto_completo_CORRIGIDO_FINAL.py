# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from requests.auth import HTTPBasicAuth

st.set_page_config(page_title="Painel de Indicadores — Jira", layout="wide")

# =====================================================
# CREDENCIAIS FIXAS (sem sidebar)
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

# =====================================================
# HELPERS
# =====================================================
def first_option(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, dict):
        return value.get("value") or value.get("name") or str(value)
    if isinstance(value, (list, tuple)):
        if not value: return None
        v0 = value[0]
        if isinstance(v0, dict):
            return v0.get("value") or v0.get("name") or str(v0)
        return str(v0)
    return str(value)

def get_nested(d, path, default=None):
    node = d
    for k in path:
        if not isinstance(node, dict) or k not in node: return default
        node = node[k]
    return node

def ensure_ms(x):
    try: return float(x)
    except: return pd.to_numeric(x, errors="coerce")

def ordenar_mes_str(df: pd.DataFrame, col="mes_str") -> pd.DataFrame:
    dfx = df.copy()
    try: dfx["mes_data"] = pd.to_datetime(dfx[col], format="%b/%Y")
    except: dfx["mes_data"] = pd.to_datetime(dfx[col], errors="coerce")
    dfx = dfx.sort_values("mes_data")
    dfx[col] = dfx["mes_data"].dt.strftime("%b/%Y")
    cats = dfx[col].dropna().unique().tolist()
    dfx[col] = pd.Categorical(dfx[col], categories=cats, ordered=True)
    return dfx

# =====================================================
# JIRA API
# =====================================================
def jira_search(base_url, auth, jql, fields, max_per_page=100) -> List[Dict[str,Any]]:
    url = base_url.rstrip("/") + "/rest/api/3/search"
    start_at, collected = 0, []
    while True:
        params = {"jql": jql, "startAt": start_at, "maxResults": max_per_page, "fields": ",".join(fields)}
        r = requests.get(url, params=params, auth=auth, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Erro Jira {r.status_code}: {r.text[:200]}")
        data = r.json()
        issues = data.get("issues", [])
        collected.extend(issues)
        if len(collected) >= data.get("total", 0): break
        start_at += max_per_page
        time.sleep(0.2)
    return collected

def flatten_issues(raw) -> pd.DataFrame:
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
            f"{CF_AREA_SOL}.value": get_nested(f,[CF_AREA_SOL,"value"]),
            CF_SLA_SUP: f.get(CF_SLA_SUP),
            CF_SLA_RES: f.get(CF_SLA_RES),
        })
    return pd.DataFrame(rows)

def build_df_issues(df_flat, projeto):
    df = df_flat.copy()
    df["id_norm"] = df.get("key", df.get("id"))
    df["created_norm"] = pd.to_datetime(df.get("created"), errors="coerce")
    df["resolved_norm"] = pd.to_datetime(df.get("resolutiondate"), errors="coerce")
    df["assunto_relacionado_norm"] = df[CF_ASSUNTO_REL].apply(first_option) if CF_ASSUNTO_REL in df else None
    df["origem_problema_norm"] = df[CF_ORIGEM_PROB].apply(first_option) if CF_ORIGEM_PROB in df else None

    projeto_up = str(projeto).upper()
    use_sup = projeto_up in {"TDS","TINE"}
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

def build_df_assunto(df_issues):
    return (df_issues.assign(Assunto=df_issues["assunto_relacionado"].fillna("—"))
                     .groupby("Assunto").size().reset_index(name="Qtd")
                     .sort_values("Qtd",ascending=False))

def build_df_area(df_flat):
    col = f"{CF_AREA_SOL}.value"
    if col not in df_flat: return pd.DataFrame({"Área":[],"Qtd":[]})
    return (pd.DataFrame({"Área":df_flat[col].fillna("—")})
              .groupby("Área").size().reset_index(name="Qtd")
              .sort_values("Qtd",ascending=False))

# =====================================================
# MAIN
# =====================================================
st.title("📊 Painel de Indicadores — Jira (Classic)")

projeto = "TDS"  # 🔧 defina projeto default
jql = f'project = {projeto} AND created >= "2024-01-01" ORDER BY created ASC'

if st.button("Carregar dados"):
    try:
        fields = ["key","created","resolutiondate",CF_ASSUNTO_REL,CF_ORIGEM_PROB,CF_AREA_SOL,CF_SLA_SUP,CF_SLA_RES]
        raw = jira_search(JIRA_URL, auth, jql, fields)
        st.caption(f"Issues retornados: {len(raw)}")
        if not raw: st.stop()

        df_flat = flatten_issues(raw)
        df_issues = build_df_issues(df_flat, projeto)
        df_assunto = build_df_assunto(df_issues)
        df_area = build_df_area(df_flat)

        # Criados vs Resolvidos
        st.subheader("Criados vs Resolvidos")
        meses_range = pd.date_range(df_issues["created"].min().floor("D"), df_issues["created"].max().ceil("D"), freq="MS")
        df_months = pd.DataFrame({"mes":meses_range})
        criadas = df_issues.groupby(df_issues["created"].dt.to_period("M")).size().rename("Criados")
        criadas.index = criadas.index.to_timestamp()
        resolvidas = df_issues.dropna(subset=["resolved"]).groupby(df_issues["resolved"].dt.to_period("M")).size().rename("Resolvidos")
        resolvidas.index = resolvidas.index.to_timestamp()
        df_cr_res = df_months.set_index("mes").join(criadas,how="left").join(resolvidas,how="left").fillna(0).reset_index()
        df_cr_res["mes_str"]=df_cr_res["mes"].dt.strftime("%b/%Y")
        df_cr_res=ordenar_mes_str(df_cr_res,"mes_str")
        st.plotly_chart(px.bar(df_cr_res,x="mes_str",y=["Criados","Resolvidos"],barmode="group"), use_container_width=True)

        # SLA
        st.subheader("SLA — Dentro x Fora (%)")
        agrupado=df_issues.groupby("mes_str")["dentro_sla"].value_counts(normalize=True).unstack(fill_value=0)*100
        cols_exist=[c for c in agrupado.columns if c in (True,False,"True","False")]
        agr_wide=agrupado.loc[:,cols_exist].copy() if cols_exist else agrupado.copy()
        agr_wide.rename(columns={True:"% Dentro SLA",False:"% Fora SLA","True":"% Dentro SLA","False":"% Fora SLA"}, inplace=True)
        agr_wide=ordenar_mes_str(agr_wide.reset_index(),"mes_str")
        fig_sla=px.bar(agr_wide,x="mes_str",y=[c for c in ["% Dentro SLA","% Fora SLA"] if c in agr_wide.columns],
                       barmode="group",color_discrete_map={"% Dentro SLA":"green","% Fora SLA":"red"})
        st.plotly_chart(fig_sla,use_container_width=True)

        # Assunto Relacionado
        st.subheader("🧾 Assunto Relacionado")
        st.dataframe(df_assunto,use_container_width=True,hide_index=True)

        # Área Solicitante
        st.subheader("📦 Área Solicitante")
        st.dataframe(df_area,use_container_width=True,hide_index=True)

        # APP NE (TDS)
        if projeto=="TDS":
            with st.expander("📱 TDS • APP NE — Detalhe", expanded=False):
                df_app=df_issues[df_issues["assunto_relacionado"]==ASSUNTO_ALVO_APPNE].copy()
                if df_app.empty:
                    st.info("Nenhum chamado com esse assunto.")
                else:
                    st.metric("Total de chamados",len(df_app))
                    df_app_mes=df_app.groupby(["mes_str","origem_problema"]).size().reset_index(name="Qtd")
                    df_app_mes=ordenar_mes_str(df_app_mes,"mes_str")
                    st.plotly_chart(px.bar(df_app_mes,x="mes_str",y="Qtd",color="origem_problema",barmode="group",
                                            color_discrete_map={"APP NE":"#2ca02c","APP EN":"#1f77b4"}),use_container_width=True)
                    st.dataframe(df_app[["id","mes_str","assunto_relacionado","origem_problema"]],
                                 use_container_width=True,hide_index=True)
    except Exception as e:
        st.error(f"Erro: {e}")
