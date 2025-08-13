import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Painel de Indicadores")

# 🔄 Botão para atualizar dados do Jira manualmente
if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets['EMAIL']
TOKEN = st.secrets['TOKEN']
auth = HTTPBasicAuth(EMAIL, TOKEN)

PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {
    "TDS": "Tech Support",
    "INT": "Integrations",
    "TINE": "IT Support NE",
    "INTEL": "Intelligence"
}
SLA_CAMPOS = {
    "TDS": "customfield_13744",
    "TINE": "customfield_13744",
    "INT": "customfield_13686",
    "INTEL": "customfield_13686"
}
CAMPOS_ASSUNTO = {
    "TDS": "customfield_13712",
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype"
}
CAMPO_AREA = "customfield_13719"
CAMPO_N3 = "customfield_13659"
SLA_METAS = {"TDS": 98, "INT": 96, "TINE": 96, "INTEL": 96}
SLA_HORAS = {"TDS": 40, "INT": 40, "TINE": 40, "INTEL": 80}


def extrair_sla_millis(sla_field):
    try:
        if sla_field.get("completedCycles"):
            return sla_field["completedCycles"][0].get("elapsedTime", {}).get("millis")
        if sla_field.get("ongoingCycle"):
            return sla_field["ongoingCycle"].get("elapsedTime", {}).get("millis")
    except:
        return None
    return None

@st.cache_data(show_spinner="🔄 Buscando dados do Jira...")
def buscar_issues():
    todos = []
    for projeto in PROJETOS:
        start = 0
        while True:
            jql = f'project = "{projeto}" AND created >= "2024-01-01" ORDER BY created ASC'
            params = {
                "jql": jql,
                "startAt": start,
                "maxResults": 100,
                "fields": "created,resolutiondate,status,issuetype," +
                          f"{SLA_CAMPOS[projeto]},{CAMPOS_ASSUNTO[projeto]},{CAMPO_AREA},{CAMPO_N3}"
            }
            res = requests.get(f"{JIRA_URL}/rest/api/3/search", auth=auth, params=params)
            if res.status_code != 200:
                break
            dados = res.json().get("issues", [])
            if not dados:
                break
            for issue in dados:
                f = issue["fields"]
                sla_raw = f.get(SLA_CAMPOS[projeto], {})
                todos.append({
                    "projeto": projeto,
                    "created": f.get("created"),
                    "resolved": f.get("resolutiondate"),
                    "status": f.get("status", {}).get("name"),
                    "sla_millis": extrair_sla_millis(sla_raw),
                    "area": f.get(CAMPO_AREA),
                    "assunto": f.get(CAMPOS_ASSUNTO[projeto]),
                    "issuetype": f.get("issuetype"),
                    "n3": f.get(CAMPO_N3)
                })
            start += 100
    return pd.DataFrame(todos)

# ===== CARREGAR DADOS =====
df = buscar_issues()
df["created"] = pd.to_datetime(df["created"])
df["resolved"] = pd.to_datetime(df["resolved"])
df["mes_created"] = df["created"].dt.to_period("M").dt.to_timestamp()
df["mes_resolved"] = df["resolved"].dt.to_period("M").dt.to_timestamp()

# ===== VISUALIZAÇÃO DE DADOS =====
tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")
        dfp = df[df["projeto"] == projeto].copy()
        sla_limite = SLA_HORAS[projeto] * 60 * 60 * 1000
        sla_meta = SLA_METAS[projeto]

        # Gráfico: Criados vs Resolvidos
        st.markdown("### 📈 Tickets Criados vs Resolvidos")
        meses_cr = sorted(dfp["mes_created"].dropna().unique())
        anos_cr = sorted(dfp["mes_created"].dt.year.unique())
        meses_cr = sorted(dfp["mes_created"].dt.month.unique())
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            ano_cr = st.selectbox(f"Ano - {TITULOS[projeto]} (Criados vs Resolvidos)", ["Todos"] + [str(a) for a in anos_cr], key=f"ano_cr_{projeto}")
        with col_cr2:
            mes_cr = st.selectbox(f"Mês - {TITULOS[projeto]} (Criados vs Resolvidos)", ["Todos"] + [str(m).zfill(2) for m in meses_cr], key=f"mes_cr_{projeto}")
        df_cr = dfp.copy()
        if ano_cr != "Todos":
            df_cr = df_cr[df_cr["mes_created"].dt.year == int(ano_cr)]
        if mes_cr != "Todos":
            df_cr = df_cr[df_cr["mes_created"].dt.month == int(mes_cr)]
        criados = df_cr.groupby("mes_created").size().reset_index(name="Criados")
        resolvidos = df_cr[df_cr["resolved"].notna()].groupby("mes_resolved").size().reset_index(name="Resolvidos")
        criados.rename(columns={"mes_created": "mes_str"}, inplace=True)
        resolvidos.rename(columns={"mes_resolved": "mes_str"}, inplace=True)
        grafico = pd.merge(criados, resolvidos, how="outer", on="mes_str").fillna(0).sort_values("mes_str")
        grafico["mes_str"] = grafico["mes_str"].dt.strftime("%b/%Y")
        fig = px.bar(grafico, x="mes_str", y=["Criados", "Resolvidos"], barmode="group", text_auto=True, height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"crv_{projeto}")

        # SLA
        st.markdown("### ⏱️ SLA + OKR")
        df_sla = dfp[dfp["resolved"].notna()].copy()
        df_sla["dentro_sla"] = df_sla["sla_millis"] <= sla_limite
        agrupado = df_sla.groupby("mes_resolved")["dentro_sla"].agg([("Dentro do SLA", "sum"), ("Fora do SLA", lambda x: (~x).sum())]).reset_index()
        agrupado["Total"] = agrupado["Dentro do SLA"] + agrupado["Fora do SLA"]
        agrupado["%"] = (agrupado["Dentro do SLA"] / agrupado["Total"] * 100).round(2)
        agrupado["mes"] = agrupado["mes_resolved"].dt.strftime("%b/%Y")
        fig_sla = px.bar(
    agrupado,
    x="mes",
    y=["Dentro do SLA", "Fora do SLA"],
    barmode="group",
    text_auto=True,
    color_discrete_map={
        "Dentro do SLA": "green",
        "Fora do SLA": "red"
    },
    height=400
)
        okr_total = agrupado["Dentro do SLA"].sum() / agrupado["Total"].sum() * 100 if agrupado["Total"].sum() else 0
        fig_sla.add_annotation(text=f"🎯 OKR: {okr_total:.1f}%", x=0.99, y=0.95, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="green" if okr_total >= sla_meta else "red"))
        st.plotly_chart(fig_sla, use_container_width=True, key=f"sla_{projeto}_plot")

        # Assunto Relacionado
        st.markdown("### 🧾 Assunto Relacionado")
        anos_ass = sorted(dfp["mes_created"].dt.year.unique())
        meses_ass = sorted(dfp["mes_created"].dt.month.unique())
        col_ass1, col_ass2 = st.columns(2)
        with col_ass1:
            ano_ass = st.selectbox(f"Ano - {TITULOS[projeto]} (Assunto)", ["Todos"] + [str(a) for a in anos_ass], key=f"ano_ass_{projeto}")
        with col_ass2:
            mes_ass = st.selectbox(f"Mês - {TITULOS[projeto]} (Assunto)", ["Todos"] + [str(m).zfill(2) for m in meses_ass], key=f"mes_ass_{projeto}")
        df_ass = dfp.copy()
        if ano_ass != "Todos":
            df_ass = df_ass[df_ass["mes_created"].dt.year == int(ano_ass)]
        if mes_ass != "Todos":
            df_ass = df_ass[df_ass["mes_created"].dt.month == int(mes_ass)]
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_ass["assunto_nome"] = df_ass["issuetype"].apply(lambda x: x.get("name") if isinstance(x, dict) else str(x))
        else:
            df_ass["assunto_nome"] = df_ass["assunto"].apply(lambda x: x.get("value") if isinstance(x, dict) else str(x))
        assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
        assunto_count.columns = ["Assunto", "Qtd"]
        st.dataframe(assunto_count)

        # Área Solicitante
        if projeto != "INTEL":
            st.markdown("### 📦 Área Solicitante")
            anos_area = sorted(dfp["mes_created"].dt.year.unique())
            meses_area = sorted(dfp["mes_created"].dt.month.unique())
            col_ar1, col_ar2 = st.columns(2)
            with col_ar1:
                ano_area = st.selectbox(f"Ano - {TITULOS[projeto]} (Área)", ["Todos"] + [str(a) for a in anos_area], key=f"ano_area_{projeto}")
            with col_ar2:
                mes_area = st.selectbox(f"Mês - {TITULOS[projeto]} (Área)", ["Todos"] + [str(m).zfill(2) for m in meses_area], key=f"mes_area_{projeto}")
            df_area = dfp.copy()
            if ano_area != "Todos":
                df_area = df_area[df_area["mes_created"].dt.year == int(ano_area)]
            if mes_area != "Todos":
                df_area = df_area[df_area["mes_created"].dt.month == int(mes_area)]
            df_area["area_nome"] = df_area["area"].apply(lambda x: x.get("value") if isinstance(x, dict) else str(x))
            area_count = df_area["area_nome"].value_counts().reset_index()
            area_count.columns = ["Área", "Qtd"]
            st.dataframe(area_count)

        # Encaminhamentos
        st.markdown("### 🔄 Encaminhamentos")
        anos_enc = sorted(dfp["mes_created"].dt.year.unique())
        meses_enc = sorted(dfp["mes_created"].dt.month.unique())
        col_en1, col_en2 = st.columns(2)
        with col_en1:
            ano_enc = st.selectbox(f"Ano - {TITULOS[projeto]} (Encaminhamentos)", ["Todos"] + [str(a) for a in anos_enc], key=f"ano_enc_{projeto}")
        with col_en2:
            mes_enc = st.selectbox(f"Mês - {TITULOS[projeto]} (Encaminhamentos)", ["Todos"] + [str(m).zfill(2) for m in meses_enc], key=f"mes_enc_{projeto}")
        df_enc = dfp.copy()
        if ano_enc != "Todos":
            df_enc = df_enc[df_enc["mes_created"].dt.year == int(ano_enc)]
        if mes_enc != "Todos":
            df_enc = df_enc[df_enc["mes_created"].dt.month == int(mes_enc)]
        col1, col2 = st.columns(2)
        with col1:
            count_prod = df_enc["status"].str.contains("Produto", case=False, na=False).sum()
            st.metric("Encaminhados Produto", count_prod)
        with col2:
            df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: x.get("value") if isinstance(x, dict) else None)
            st.metric("Encaminhados N3", (df_enc["n3_valor"] == "Sim").sum())
