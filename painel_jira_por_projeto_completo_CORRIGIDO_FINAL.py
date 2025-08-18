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
EMAIL = st.secrets["EMAIL"]          # defina em .streamlit/secrets.toml
TOKEN = st.secrets["TOKEN"]          # defina em .streamlit/secrets.toml
auth = HTTPBasicAuth(EMAIL, TOKEN)

# =====================================================
# CONFIG
# =====================================================
SLA_LIMITE = {
    "TDS": 40 * 60 * 60 * 1000,  # 40h
    "INT": 40 * 60 * 60 * 1000,
    "TINE": 40 * 60 * 60 * 1000, # 40h
    "INTEL": 80 * 60 * 60 * 1000,# 80h
}
SLA_PADRAO_MILLIS = 40 * 60 * 60 * 1000
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

CF_ASSUNTO_REL = "customfield_13747"   # Assunto Relacionado
CF_ORIGEM_PROB = "customfield_13628"   # Origem do problema
CF_AREA_SOL    = "customfield_13719"   # Área Solicitante (.value)
CF_SLA_SUP     = "customfield_13744"   # SLA (SUP) p/ TDS/TINE
CF_SLA_RES     = "customfield_13686"   # Tempo de resolução (fallback)

PROJETO_DEFAULT = "TDS"           # ajuste se quiser
JQL_PERIOD_START = "2024-01-01"   # ajuste o período “fixo” do classic

# =====================================================
# HELPERS
# =====================================================
def first_option(value):
    """Extrai .value de option Jira (dict/list/str)."""
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

# 🔒 Helper robusto para ranges mensais (evita "duplicate keys" no to_datetime)
def month_range_from_series(s) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Retorna (início_do_mês_mín, início_do_próximo_mês_do_máx) para usar com freq='MS'.
    Robusto para: DataFrame com colunas duplicadas, Series com dicts/objetos, timezone.
    """
    # 1) Se vier DataFrame (ex.: colunas duplicadas "created"), usa a 1ª coluna
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None, None
        s = s.iloc[:, 0]

    # 2) Converte tudo para string antes de to_datetime (evita assemble-from-units)
    s = pd.Series(s)  # garante Series
    s = s.astype(str)

    # 3) Parse para datetime (coerce) e remove NaT
    s = pd.to_datetime(s, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None, None

    # 4) Remove timezone se houver
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass

    # 5) Gera limites mensais
    start = s.min().to_period("M").to_timestamp()                   # 1º dia do mês do mínimo
    end   = (s.max().to_period("M").to_timestamp() + MonthBegin(1)) # 1º dia do mês seguinte ao máximo
    return start, end

# =====================================================
# JIRA API (com cache)
# =====================================================
@st.cache_data(show_spinner=False, ttl=900)  # 15 min de cache
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
            ms = get_nested(obj, ["elapsedTime", "millis"]) or get_nested(obj, ["ongoingCycle", "elapsedTime", "millis"])
        if ms is None and pd.notna(row.get("resolved_norm")) and pd.notna(row.get("created_norm")):
            return (row["resolved_norm"] - row["created_norm"]).total_seconds() * 1000.0
        return ensure_ms(ms)

    df["sla_millis_norm"] = df.apply(extract_sla_ms, axis=1)
    limite_ms = SLA_LIMITE.get(projeto_up, SLA_PADRAO_MILLIS)
    df["dentro_sla_norm"] = df["sla_millis_norm"] <= limite_ms
    df["mes_str_norm"] = df["created_norm"].dt.strftime("%b/%Y")

    return df.rename(columns={
        "id_norm": "id",
        "created_norm": "created",
        "resolved_norm": "resolved",
        "sla_millis_norm": "sla_millis",
        "dentro_sla_norm": "dentro_sla",
        "mes_str_norm": "mes_str",
        "assunto_relacionado_norm": "assunto_relacionado",
        "origem_problema_norm": "origem_problema",
    })

def build_df_assunto(df_issues: pd.DataFrame) -> pd.DataFrame:
    return (
        df_issues.assign(Assunto=df_issues["assunto_relacionado"].fillna("—"))
                 .groupby("Assunto").size().reset_index(name="Qtd")
                 .sort_values("Qtd", ascending=False)
    )

def build_df_area(df_flat: pd.DataFrame) -> pd.DataFrame:
    col = f"{CF_AREA_SOL}.value"
    if col not in df_flat:
        return pd.DataFrame({"Área": [], "Qtd": []})
    return (
        pd.DataFrame({"Área": df_flat[col].fillna("—")})
        .groupby("Área").size().reset_index(name="Qtd")
        .sort_values("Qtd", ascending=False)
    )

# =====================================================
# RENDER
# =====================================================
st.title("📊 Painel de Indicadores — Jira (Classic)")

# 🔄 Botão para atualizar dados do Jira manualmente
if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

# Parâmetros ‘hardcoded’ do classic
projeto = PROJETO_DEFAULT
jql = f'project = {projeto} AND created >= "{JQL_PERIOD_START}" ORDER BY created ASC'

def load_and_render():
    fields = ["key", "created", "resolutiondate",
              CF_ASSUNTO_REL, CF_ORIGEM_PROB, CF_AREA_SOL,
              CF_SLA_SUP, CF_SLA_RES]

    raw = jira_search_cached(JIRA_URL, EMAIL, TOKEN, jql, fields)
    st.caption(f"Issues retornados: {len(raw)}")
    if not raw:
        st.warning("JQL não retornou issues. Ajuste o período ou verifique permissões.")
        return

    df_flat = flatten_issues(raw)
    df_issues = build_df_issues(df_flat, projeto)

    # 🔧 Patch: tratar colunas duplicadas e garantir datetime
    if getattr(df_issues.columns, "duplicated", lambda: np.array([]))().any():
        df_issues = df_issues.loc[:, ~df_issues.columns.duplicated(keep="first")]

    def _coerce_dt(df, col):
        obj = df.loc[:, col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        ser = pd.Series(obj, copy=False).astype(str)
        ser = pd.to_datetime(ser, errors="coerce")
        try:
            ser = ser.dt.tz_localize(None)
        except Exception:
            pass
        return ser

    df_issues["created"] = _coerce_dt(df_issues, "created")
    df_issues["resolved"] = _coerce_dt(df_issues, "resolved")

    df_assunto = build_df_assunto(df_issues)
    df_area = build_df_area(df_flat)

    # =====================================================
    # 1) Criados vs Resolvidos
    # =====================================================
    st.subheader("Criados vs Resolvidos")

    start, end = month_range_from_series(df_issues["created"])
    if start is None or end is None:
        st.warning("Não há datas de criação válidas para montar a série mensal.")
        return

    df_months = pd.DataFrame({"mes": pd.date_range(start=start, end=end, freq="MS")})

    criadas = (
        df_issues
        .groupby(df_issues["created"].dt.to_period("M"))
        .size()
        .rename("Criados")
    )
    criadas.index = criadas.index.to_timestamp()

    if df_issues["resolved"].notna().any():
        resolvidas = (
            df_issues.dropna(subset=["resolved"])
                     .groupby(df_issues["resolved"].dt.to_period("M"))
                     .size()
                     .rename("Resolvidos")
        )
        resolvidas.index = resolvidas.index.to_timestamp()
    else:
        resolvidas = pd.Series(dtype=float, name="Resolvidos")

    df_cr_res = (
        df_months.set_index("mes")
                 .join(criadas, how="left")
                 .join(resolvidas, how="left")
                 .fillna(0)
                 .reset_index()
    )
    df_cr_res["mes_str"] = df_cr_res["mes"].dt.strftime("%b/%Y")
    df_cr_res = ordenar_mes_str(df_cr_res, "mes_str")

    fig_cr = px.bar(
        df_cr_res,
        x="mes_str",
        y=["Criados", "Resolvidos"],
        barmode="group",
        title=f"Criados vs Resolvidos — {projeto}",
    )
    st.plotly_chart(fig_cr, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # 2) SLA — Dentro x Fora (%)
    # =====================================================
    st.subheader("SLA — Dentro x Fora (%)")

    agrupado = (
        df_issues
        .groupby("mes_str")["dentro_sla"]
        .value_counts(normalize=True)
        .unstack(fill_value=0) * 100.0
    )

    cols_exist = [c for c in agrupado.columns if c in (True, False, "True", "False")]
    agr_wide = agrupado.loc[:, cols_exist].copy() if cols_exist else agrupado.copy()
    agr_wide.rename(columns={
        True: "% Dentro SLA",
        False: "% Fora SLA",
        "True": "% Dentro SLA",
        "False": "% Fora SLA",
    }, inplace=True)

    agr_wide = ordenar_mes_str(agr_wide.reset_index(), "mes_str")

    fig_sla = px.bar(
        agr_wide,
        x="mes_str",
        y=[c for c in ["% Dentro SLA", "% Fora SLA"] if c in agr_wide.columns],
        barmode="group",
        color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
        title=f"Percentual dentro/fora do SLA — {projeto}",
    )
    fig_sla.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig_sla.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_sla, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # 3) Assunto Relacionado
    # =====================================================
    st.subheader("🧾 Assunto Relacionado")
    st.dataframe(df_assunto, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================================================
    # 4) Área Solicitante
    # =====================================================
    st.subheader("📦 Área Solicitante")
    st.dataframe(df_area, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================================================
    # 5) TDS • APP NE (final em expander)
    # =====================================================
    if projeto.upper() == "TDS":
        with st.expander("📱 TDS • APP NE — Detalhe", expanded=False):
            df_app = df_issues[df_issues["assunto_relacionado"] == ASSUNTO_ALVO_APPNE].copy()

            if df_app.empty:
                st.info(f"Não há chamados para '{ASSUNTO_ALVO_APPNE}' no período selecionado.")
            else:
                df_app = ordenar_mes_str(df_app, "mes_str")

                total_app = len(df_app)
                por_origem = df_app["origem_problema"].value_counts(dropna=False).to_dict()
                k1, k2, k3 = st.columns(3)
                k1.metric("Total de chamados (APP NE/EN)", total_app)
                k2.metric("APP NE", por_origem.get("APP NE", 0))
                k3.metric("APP EN", por_origem.get("APP EN", 0))

                df_app_mes = (
                    df_app
                    .groupby(["mes_str", "origem_problema"])
                    .size()
                    .reset_index(name="Qtd")
                )
                df_app_mes = ordenar_mes_str(df_app_mes, "mes_str")

                fig_app = px.bar(
                    df_app_mes,
                    x="mes_str",
                    y="Qtd",
                    color="origem_problema",
                    barmode="group",
                    title="APP NE — Volumes por mês e origem",
                    color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4"},
                )
                st.plotly_chart(fig_app, use_container_width=True)

                st.subheader("Chamados — Detalhe")
                st.dataframe(
                    df_app[["id", "mes_str", "assunto_relacionado", "origem_problema"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ============================
# CARREGAMENTO AUTOMÁTICO
# ============================
# Na primeira renderização, carrega automaticamente
if "auto_loaded" not in st.session_state:
    st.session_state.auto_loaded = True
    load_and_render()
else:
    # Em novas renderizações (ex.: interação), também carregamos;
    # Se preferir evitar chamadas repetidas, comente a linha abaixo.
    load_and_render()
