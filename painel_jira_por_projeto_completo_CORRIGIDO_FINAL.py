import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

st.set_page_config(layout='wide')
st.title("📊 Painel de Indicadores")

JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets["jira"]["email"]
TOKEN = st.secrets["jira"]["token"]
auth = HTTPBasicAuth(EMAIL, TOKEN)

PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {
    "TDS": "Tech Support",
    "INT": "Integrations",
    "TINE": "IT Support NE",
    "INTEL": "Intelligence"
}
SLA_METAS = {
    "TDS": 98,
    "INT": 96,
    "TINE": 96,
    "INTEL": 96
}
SLA_HORAS = {
    "TDS": 40,
    "INT": 40,
    "TINE": 40,
    "INTEL": 80
}
CUTOFF_DATE = "2024-01-01"

def extrair_sla_millis(sla_field):
    try:
        if sla_field.get("completedCycles"):
            cycles = sla_field["completedCycles"]
            if isinstance(cycles, list) and cycles:
                return cycles[0].get("elapsedTime", {}).get("millis")
        if sla_field.get("ongoingCycle"):
            return sla_field["ongoingCycle"].get("elapsedTime", {}).get("millis")
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=True)
def buscar_issues():
    headers = {"Accept": "application/json"}
    all_issues = []

    for projeto in PROJETOS:
        start_at = 0
        while True:
            jql = f'project = "{projeto}" AND created >= "{CUTOFF_DATE}" ORDER BY created ASC'
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": 100,
                "fields": (
                    "summary,created,resolutiondate,status,"
                    "customfield_13686,customfield_13744,"
                    "customfield_13719,customfield_13712,customfield_13643,"
                    "customfield_13699,customfield_13659,customfield_10010,"
                    "issuetype"
                )
            }
            res = requests.get(f"{JIRA_URL}/rest/api/3/search", headers=headers, auth=auth, params=params)
            if res.status_code != 200:
                break
            issues = res.json().get("issues", [])
            if not issues:
                break
            for issue in issues:
                f = issue["fields"]
                sla_millis = extrair_sla_millis(
                    f.get("customfield_13744", {}) if projeto in ["TDS", "TINE"]
                    else f.get("customfield_13686", {})
                )
                all_issues.append({
                    "projeto": projeto,
                    "created": f["created"],
                    "resolved": f.get("resolutiondate"),
                    "status": f.get("status", {}).get("name", ""),
                    "sla_millis": sla_millis,
                    "customfield_13719": f.get("customfield_13719"),
                    "customfield_13712": f.get("customfield_13712"),
                    "customfield_13643": f.get("customfield_13643"),
                    "customfield_13699": f.get("customfield_13699"),
                    "customfield_13659": f.get("customfield_13659"),
                    "customfield_10010": f.get("customfield_10010"),
                    "issuetype": f.get("issuetype")
                })
            start_at += 100
    return pd.DataFrame(all_issues)


df = buscar_issues()
df["created"] = pd.to_datetime(df["created"])
df["resolved"] = pd.to_datetime(df["resolved"])
df["mes_str"] = df["created"].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")
df["ano"] = df["created"].dt.year.astype(str)
df["mes"] = df["created"].dt.strftime("%B")
df["escalado_n3_valor"] = df["customfield_13659"].apply(lambda x: x.get("value") if isinstance(x, dict) else None)

abas = st.tabs([TITULOS[projeto] for projeto in PROJETOS])

for projeto, aba in zip(PROJETOS, abas):
    with aba:
        st.header(f"📊 Projeto: {TITULOS[projeto]}")
        df_proj = df[df["projeto"] == projeto].copy()

        # Filtros de ANO e MÊS
        anos_disponiveis = sorted(df_proj["ano"].unique())
        meses_disponiveis = sorted(df_proj["mes"].unique(), key=lambda m: pd.to_datetime(m, format='%B'))

        col1, col2 = st.columns(2)
        with col1:
            filtro_ano = st.selectbox("Filtrar por Ano", ["Todos"] + anos_disponiveis, key=f"ano_{projeto}")
        with col2:
            filtro_mes = st.selectbox("Filtrar por Mês", ["Todos"] + meses_disponiveis, key=f"mes_{projeto}")

        df_filtrado = df_proj.copy()
        if filtro_ano != "Todos":
            df_filtrado = df_filtrado[df_filtrado["ano"] == filtro_ano]
        if filtro_mes != "Todos":
            df_filtrado = df_filtrado[df_filtrado["mes"] == filtro_mes]

        # Meses ordenados para base do gráfico
        meses_filtrados = sorted(df_filtrado["mes_str"].unique(), key=lambda x: pd.to_datetime(x, format="%b/%Y"))
        base_meses = pd.DataFrame({"mes_str": meses_filtrados})

        # 📈 Criados vs Resolvidos
        st.subheader("📈 Tickets Criados | Resolvidos")
        criados = df_filtrado.groupby("mes_str")["created"].count().reset_index(name="Criados")
        resolvidos = df_filtrado[df_filtrado["resolved"].notna()].groupby("mes_str")["resolved"].count().reset_index(name="Resolvidos")
        grafico = base_meses.merge(criados, on="mes_str", how="left").merge(resolvidos, on="mes_str", how="left").fillna(0)

        fig = go.Figure()
        fig.add_bar(x=grafico["mes_str"], y=grafico["Criados"], name="Criados",
                    text=grafico["Criados"], textposition="outside")
        fig.add_bar(x=grafico["mes_str"], y=grafico["Resolvidos"], name="Resolvidos",
                    text=grafico["Resolvidos"], textposition="outside")
        fig.update_layout(barmode="group", xaxis_title="Mês", yaxis_title="Chamados")
        st.plotly_chart(fig, use_container_width=True)

        # ⏱️ SLA por Mês
        st.subheader("⏱️ SLA por Mês")
        df_sla = df_filtrado.dropna(subset=["sla_millis"]).copy()
        df_sla["sla_horas"] = df_sla["sla_millis"] / (1000 * 60 * 60)
        limite_sla = SLA_HORAS[projeto]
        meta_porcentagem = SLA_METAS[projeto]
        df_sla["dentro_sla"] = df_sla["sla_horas"] <= limite_sla
        df_sla["mes_str"] = df_sla["created"].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")

        sla_mensal = df_sla.groupby("mes_str")["dentro_sla"].mean().reset_index()
        sla_mensal["percentual"] = (sla_mensal["dentro_sla"] * 100).round(2)
        sla_mensal["fora"] = (100 - sla_mensal["percentual"]).round(2)
        sla_plot = base_meses.merge(sla_mensal[["mes_str", "percentual", "fora"]], on="mes_str", how="left").fillna(0)

        fig_sla = go.Figure([
            go.Bar(name="Dentro SLA", x=sla_plot["mes_str"], y=sla_plot["percentual"],
                   text=sla_plot["percentual"].map(lambda x: f"{x:.2f}"), textposition="outside", marker_color="green"),
            go.Bar(name="Fora SLA", x=sla_plot["mes_str"], y=sla_plot["fora"],
                   text=sla_plot["fora"].map(lambda x: f"{x:.2f}"), textposition="outside", marker_color="red")
        ])
        fig_sla.update_layout(
            barmode="group",
            yaxis_title="%",
            xaxis_title="Mês",
            title=f"SLA Mensal ({meta_porcentagem}% - {limite_sla}h)"
        )
        st.plotly_chart(fig_sla, use_container_width=True)

        # 🎯 OKR SLA Total
        total_sla_registros = len(df_sla)
        sla_ok = df_sla["dentro_sla"].sum()
        okr_percentual = (sla_ok / total_sla_registros * 100) if total_sla_registros > 0 else 0
        cor = "green" if okr_percentual >= meta_porcentagem else "red"
        st.markdown(f"<h4 style='text-align:right;'>🎯 OKR SLA Total: <span style='color:{cor}'>{okr_percentual:.2f}%</span> (meta {meta_porcentagem}%)</h4>", unsafe_allow_html=True)

        # 📈 Criados vs Resolvidos
        st.subheader("📈 Tickets Criados | Resolvidos")
        criados = df_filtrado.groupby("mes_str")["created"].count().reset_index(name="Criados")
        resolvidos = df_filtrado[df_filtrado["resolved"].notna()].groupby("mes_str")["resolved"].count().reset_index(name="Resolvidos")
        grafico = base_meses.merge(criados, on="mes_str", how="left").merge(resolvidos, on="mes_str", how="left").fillna(0)

        fig = go.Figure()
        fig.add_bar(x=grafico["mes_str"], y=grafico["Criados"], name="Criados",
                    text=grafico["Criados"], textposition="outside")
        fig.add_bar(x=grafico["mes_str"], y=grafico["Resolvidos"], name="Resolvidos",
                    text=grafico["Resolvidos"], textposition="outside")
        fig.update_layout(barmode="group", xaxis_title="Mês", yaxis_title="Chamados")
        st.plotly_chart(fig, use_container_width=True)

        # ⏱️ SLA por Mês
        st.subheader("⏱️ SLA por Mês")
        df_sla = df_filtrado.dropna(subset=["sla_millis"]).copy()
        df_sla["sla_horas"] = df_sla["sla_millis"] / (1000 * 60 * 60)
        limite_sla = SLA_HORAS[projeto]
        meta_porcentagem = SLA_METAS[projeto]
        df_sla["dentro_sla"] = df_sla["sla_horas"] <= limite_sla
        df_sla["mes_str"] = df_sla["created"].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")

        sla_mensal = df_sla.groupby("mes_str")["dentro_sla"].mean().reset_index()
        sla_mensal["percentual"] = (sla_mensal["dentro_sla"] * 100).round(2)
        sla_mensal["fora"] = (100 - sla_mensal["percentual"]).round(2)
        sla_plot = base_meses.merge(sla_mensal[["mes_str", "percentual", "fora"]], on="mes_str", how="left").fillna(0)

        fig_sla = go.Figure([
            go.Bar(name="Dentro SLA", x=sla_plot["mes_str"], y=sla_plot["percentual"],
                   text=sla_plot["percentual"].map(lambda x: f"{x:.2f}"), textposition="outside", marker_color="green"),
            go.Bar(name="Fora SLA", x=sla_plot["mes_str"], y=sla_plot["fora"],
                   text=sla_plot["fora"].map(lambda x: f"{x:.2f}"), textposition="outside", marker_color="red")
        ])
        fig_sla.update_layout(
            barmode="group",
            yaxis_title="%",
            xaxis_title="Mês",
            title=f"SLA Mensal ({meta_porcentagem}% - {limite_sla}h)"
        )
        st.plotly_chart(fig_sla, use_container_width=True)

        # 🎯 OKR SLA Total
        total_sla_registros = len(df_sla)
        sla_ok = df_sla["dentro_sla"].sum()
        okr_percentual = (sla_ok / total_sla_registros * 100) if total_sla_registros > 0 else 0
        cor = "green" if okr_percentual >= meta_porcentagem else "red"
        st.markdown(f"<h4 style='text-align:right;'>🎯 OKR SLA Total: <span style='color:{cor}'>{okr_percentual:.2f}%</span> (meta {meta_porcentagem}%)</h4>", unsafe_allow_html=True)
