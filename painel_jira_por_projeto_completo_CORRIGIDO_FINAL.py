# -*- coding: utf-8 -*-
import math
import json
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Painel de Indicadores", layout="wide")

# 🔄 Botão para atualizar dados do Jira manualmente
if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

# =====================================================
# CONFIG GERAL
# =====================================================
# Limites de SLA por projeto (ms)
SLA_LIMITE = {
    "TDS": 40 * 60 * 60 * 1000,     # 40h
    "INT": 40 * 60 * 60 * 1000,     # ajuste se quiser
    "TINE": 40 * 60 * 60 * 1000,    # 40h
    "INTEL": 80 * 60 * 60 * 1000,   # 80h
}
SLA_PADRAO_MILLIS = 40 * 60 * 60 * 1000

ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

# Campos custom do seu Jira
CF_ASSUNTO_REL = "customfield_13747"
CF_ORIGEM_PROB = "customfield_13628"
CF_AREA_SOL    = "customfield_13719"
CF_SLA_SUP     = "customfield_13744"  # SLA resolução (SUP)
CF_SLA_RES     = "customfield_13686"  # Tempo de resolução (fallback p/ não TDS/TINE)

# =====================================================
# HELPERS
# =====================================================
def ordenar_mes_str(df: pd.DataFrame, col: str = "mes_str") -> pd.DataFrame:
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

def first_option(value):
    """Extrai .value de opção Jira. Suporta dict, lista de dicts, lista simples, string."""
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

def get_nested(d, path: List[str], default=None):
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

# =====================================================
# JIRA API
# =====================================================
def jira_search(
    base_url: str,
    email: str,
    token: str,
    jql: str,
    fields: List[str],
    max_per_page: int = 100,
    limit_pages: int = 200
) -> List[Dict[str, Any]]:
    """
    Pagina /rest/api/3/search até trazer todos os issues da JQL.
    Retorna lista de issues (dict) crus.
    """
    url = base_url.rstrip("/") + "/rest/api/3/search"
    auth = (email, token)
    start_at = 0
    collected = []
    page = 0

    while True:
        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_per_page,
            "fields": ",".join(fields),
            "expand": "",  # adicione se quiser
        }
        resp = requests.get(url, params=params, auth=auth, timeout=60)
        if resp.status_code == 401:
            raise RuntimeError("Não autorizado: verifique e-mail e API token.")
        if resp.status_code == 403:
            raise RuntimeError("Acesso negado: o usuário não tem permissão para esta busca.")
        if resp.status_code >= 400:
            raise RuntimeError(f"Erro na API Jira ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        issues = data.get("issues", [])
        collected.extend(issues)

        total = data.get("total", 0)
        if len(collected) >= total:
            break

        page += 1
        if page >= limit_pages:
            break

        start_at += max_per_page
        # opcional: respeitar rate limit
        time.sleep(0.2)

    return collected

def flatten_issues(raw_issues: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Achata os campos essenciais numa linha por issue.
    """
    rows = []
    for it in raw_issues:
        fields = it.get("fields", {})
        row = {
            "key": it.get("key"),
            "id": it.get("id"),
            "created": fields.get("created"),
            "resolutiondate": fields.get("resolutiondate"),
            # campos custom podem aparecer como objetos/estruturas
            f"{CF_ASSUNTO_REL}": fields.get(CF_ASSUNTO_REL),
            f"{CF_ORIGEM_PROB}": fields.get(CF_ORIGEM_PROB),
            f"{CF_AREA_SOL}.value": get_nested(fields, [CF_AREA_SOL, "value"]),
            # blocos de SLA: podem ter subestruturas (elapsedTime, etc.)
            f"{CF_SLA_SUP}": fields.get(CF_SLA_SUP),
            f"{CF_SLA_RES}": fields.get(CF_SLA_RES),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# =====================================================
# NORMALIZAÇÃO PARA O PAINEL
# =====================================================
def build_df_issues(df_raw: pd.DataFrame, projeto: str) -> pd.DataFrame:
    df = df_raw.copy()

    # ID / Datas
    df["id_norm"] = df.get("key", df.get("id"))
    df["created_norm"] = pd.to_datetime(df.get("created"), errors="coerce")
    df["resolved_norm"] = pd.to_datetime(df.get("resolutiondate"), errors="coerce")

    # Assunto Relacionado
    # Pode vir como lista de opções/dict; extrair .value
    assunto_series = df[CF_ASSUNTO_REL].apply(first_option) if CF_ASSUNTO_REL in df.columns else None
    df["assunto_relacionado_norm"] = assunto_series if assunto_series is not None else None

    # Origem do problema
    origem_series = df[CF_ORIGEM_PROB].apply(first_option) if CF_ORIGEM_PROB in df.columns else None
    df["origem_problema_norm"] = origem_series if origem_series is not None else None

    # SLA millis:
    # Para TDS/TINE usar CF_SLA_SUP; demais tentar CF_SLA_RES
    projeto_up = str(projeto).upper()
    use_sup = projeto_up in {"TDS", "TINE"}

    def extract_sla_ms(row):
        obj = row.get(CF_SLA_SUP) if use_sup else row.get(CF_SLA_RES)
        # obj pode ser dict com elapsedTime/ongoingCycle/etc.
        candidates = [
            get_nested(obj, ["elapsedTime", "millis"]),
            get_nested(obj, ["ongoingCycle", "elapsedTime", "millis"]),
        ]
        ms = next((c for c in candidates if c is not None), None)
        if ms is None:
            # fallback: diferença entre resolved e created
            c, r = row.get("created_norm"), row.get("resolved_norm")
            if pd.notna(c) and pd.notna(r):
                return (r - c).total_seconds() * 1000.0
            return np.nan
        return ensure_ms(ms)

    df["sla_millis_norm"] = df.apply(extract_sla_ms, axis=1)

    # Dentro do SLA
    limite_ms = SLA_LIMITE.get(projeto_up, SLA_PADRAO_MILLIS)
    df["dentro_sla_norm"] = df["sla_millis_norm"] <= limite_ms

    # mes_str
    df["mes_str_norm"] = df["created_norm"].dt.strftime("%b/%Y")

    cols = {
        "id": "id_norm",
        "created": "created_norm",
        "resolved": "resolved_norm",
        "sla_millis": "sla_millis_norm",
        "dentro_sla": "dentro_sla_norm",
        "mes_str": "mes_str_norm",
        "assunto_relacionado": "assunto_relacionado_norm",
        "origem_problema": "origem_problema_norm",
    }
    out = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    return out

def build_df_assunto(df_issues: pd.DataFrame) -> pd.DataFrame:
    tmp = df_issues.assign(Assunto=df_issues["assunto_relacionado"].fillna("—"))
    return (
        tmp.groupby("Assunto")
           .size()
           .reset_index(name="Qtd")
           .sort_values("Qtd", ascending=False)
    )

def build_df_area(df_flat: pd.DataFrame) -> pd.DataFrame:
    # Área Solicitante = customfield_13719.value (já flatten)
    col = f"{CF_AREA_SOL}.value"
    if col not in df_flat.columns:
        return pd.DataFrame({"Área": [], "Qtd": []})
    tmp = pd.DataFrame({"Área": df_flat[col].fillna("—")})
    return tmp.groupby("Área").size().reset_index(name="Qtd").sort_values("Qtd", ascending=False)

# =====================================================
# SIDEBAR — CREDENCIAIS / PARÂMETROS
# =====================================================
st.sidebar.header("⚙️ Conexão Jira")
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets['EMAIL']
TOKEN = st.secrets['TOKEN']
auth = HTTPBasicAuth(EMAIL, TOKEN)

st.sidebar.header("🎯 Escopo")
projeto = st.sidebar.selectbox("Projeto", options=["TDS", "INT", "TINE", "INTEL"], index=0)
use_jql = st.sidebar.toggle("Usar JQL personalizado", value=False)
if use_jql:
    jql = st.sidebar.text_area(
        "JQL",
        value=f'project = {projeto} ORDER BY created DESC',
        height=80
    )
else:
    data_ini = st.sidebar.date_input("De (created >=)", value=pd.to_datetime("2024-01-01"))
    data_fim = st.sidebar.date_input("Até (created <=)", value=pd.to_datetime("today"))
    jql = f'project = {projeto} AND created >= "{data_ini}" AND created <= "{data_fim}" ORDER BY created ASC'

st.sidebar.caption("Dica: gere seu API token em https://id.atlassian.com/manage/api-tokens")

btn = st.sidebar.button("Carregar dados")

# =====================================================
# MAIN
# =====================================================
st.title("📊 Painel de Indicadores — Jira")

if btn:
    if not base_url or not email or not token:
        st.error("Preencha domínio, e-mail e API token.")
        st.stop()

    st.info("Consultando Jira… isso pode levar alguns segundos conforme o volume de tickets.")
    try:
        fields = [
            "key", "created", "resolutiondate",
            CF_ASSUNTO_REL, CF_ORIGEM_PROB, CF_AREA_SOL,
            CF_SLA_SUP, CF_SLA_RES
        ]
        raw = jira_search(base_url, email, token, jql, fields)
        if not raw:
            st.warning("JQL não retornou issues.")
            st.stop()

        df_flat = flatten_issues(raw)
        # Normaliza para o painel
        df_issues = build_df_issues(df_flat, projeto)
        # Agregados
        df_assunto = build_df_assunto(df_issues)
        df_area = build_df_area(df_flat)

        # =====================================================
        # 1) Criados vs Resolvidos (PRIMEIRO)
        # =====================================================
        st.subheader("Criados vs Resolvidos")

        # Série mensal completa (com base no período presente nos dados)
        meses = pd.date_range(df_issues["created"].min().floor("D"),
                              df_issues["created"].max().ceil("D"),
                              freq="MS")
        df_months = pd.DataFrame({"mes": meses})

        # Contagens
        criadas = df_issues.groupby(df_issues["created"].dt.to_period("M")).size().rename("Criados")
        criadas.index = criadas.index.to_timestamp()
        resolvidas = (
            df_issues.dropna(subset=["resolved"])
                     .groupby(df_issues["resolved"].dt.to_period("M"))
                     .size()
                     .rename("Resolvidos")
        )
        resolvidas.index = resolvidas.index.to_timestamp()

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
        # 2) SLA (lado a lado, meses ordenados)
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

        rename_map = {True: "% Dentro SLA", False: "% Fora SLA", "True": "% Dentro SLA", "False": "% Fora SLA"}
        agr_wide.rename(columns=rename_map, inplace=True)
        agr_wide = agr_wide.reset_index()
        agr_wide = ordenar_mes_str(agr_wide, "mes_str")

        fig_sla = px.bar(
            agr_wide,
            x="mes_str",
            y=[c for c in ["% Dentro SLA", "% Fora SLA"] if c in agr_wide.columns],
            barmode="group",
            title=f"Percentual dentro/fora do SLA — {projeto}",
            color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
        )
        fig_sla.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig_sla.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_sla, use_container_width=True)

        st.markdown("---")

        # =====================================================
        # 3) Assunto Relacionado (filtros + tabela alinhada)
        # =====================================================
        st.subheader("🧾 Assunto Relacionado")
        LEFT_RATIO, RIGHT_RATIO = 62, 38

        colA, colB = st.columns((LEFT_RATIO, RIGHT_RATIO), gap="large")
        with colA:
            anos = ["Todos"] + sorted(df_issues["created"].dropna().dt.year.unique().tolist())
            filtro_ano_assunto = st.selectbox("Ano - Tech Support (Assunto)", anos, index=0)
        with colB:
            meses_str = ["Todos"] + agr_wide["mes_str"].cat.categories.tolist()
            filtro_mes_assunto = st.selectbox("Mês - Tech Support (Assunto)", meses_str, index=0)

        # (filtros demonstrativos — ajuste conforme desejar)
        st.dataframe(
            df_assunto[["Assunto", "Qtd"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Assunto": st.column_config.TextColumn("Assunto", width=900),
                "Qtd": st.column_config.NumberColumn("Qtd", format="%d", width=340),
            },
        )

        st.markdown("---")

        # =====================================================
        # 4) Área Solicitante (filtros + tabela alinhada)
        # =====================================================
        st.subheader("📦 Área Solicitante")

        colC, colD = st.columns((LEFT_RATIO, RIGHT_RATIO), gap="large")
        with colC:
            anos2 = anos
            filtro_ano_area = st.selectbox("Ano - Tech Support (Área)", anos2, index=0)
        with colD:
            filtro_mes_area = st.selectbox("Mês - Tech Support (Área)", meses_str, index=0)

        st.dataframe(
            df_area[["Área", "Qtd"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Área": st.column_config.TextColumn("Área", width=900),
                "Qtd": st.column_config.NumberColumn("Qtd", format="%d", width=340),
            },
        )

        st.markdown("---")

        # =====================================================
        # 5) TDS • APP NE (no final, expander)
        # =====================================================
        if str(projeto).upper() == "TDS":
            with st.expander("📱 TDS • APP NE — Detalhe", expanded=False):
                df_app = df_issues[df_issues["assunto_relacionado"] == ASSUNTO_ALVO_APPNE].copy()

                if df_app.empty:
                    st.info(f"Não há chamados para o assunto '{ASSUNTO_ALVO_APPNE}' no período selecionado.")
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

                    view_cols = {
                        "id": "ID",
                        "mes_str": "Mês",
                        "assunto_relacionado": "Assunto relacionado",
                        "origem_problema": "Origem do problema",
                    }
                    st.subheader("Chamados — Detalhe")
                    st.dataframe(
                        df_app[list(view_cols.keys())].rename(columns=view_cols),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv = (
                        df_app[list(view_cols.keys())]
                        .rename(columns=view_cols)
                        .to_csv(index=False)
                        .encode("utf-8")
                    )
                    st.download_button("Baixar CSV (APP NE)", data=csv, file_name="tds_app_ne.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Falha ao carregar dados: {e}")
        st.stop()
else:
    st.info("Preencha as credenciais e clique em **Carregar dados**.")
