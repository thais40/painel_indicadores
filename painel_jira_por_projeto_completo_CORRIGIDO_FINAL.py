import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

# Limites de SLA por projeto (em milissegundos)
SLA_LIMITE = {
    "TDS": 40 * 60 * 60 * 1000,     # 40 horas
    "INT": 40 * 60 * 60 * 1000,     # 40 horas
    "TINE": 40 * 60 * 60 * 1000,    # 40 horas
    "INTEL": 80 * 60 * 60 * 1000    # 80 horas
}

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
CAMPO_ORIGEM = "customfield_13628"  # Origem do problema (para APP NE)
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

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
                "fields": (
                    "created,resolutiondate,status,issuetype,"
                    f"{SLA_CAMPOS[projeto]},{CAMPOS_ASSUNTO[projeto]},{CAMPO_AREA},{CAMPO_N3},{CAMPO_ORIGEM}"
                )
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
                    "n3": f.get(CAMPO_N3),
                    "origem": f.get(CAMPO_ORIGEM),
                    "key": issue.get("key")
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

        # ======================
        # 1) Criados vs Resolvidos (COM FILTROS)
        # ======================
        st.markdown("### 📈 Tickets Criados vs Resolvidos")
        anos_cr = sorted(dfp["mes_created"].dt.year.dropna().unique())
        meses_cr = sorted(dfp["mes_created"].dt.month.dropna().unique())
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            ano_cr = st.selectbox(f"Ano - {TITULOS[projeto]}", ["Todos"] + [str(a) for a in anos_cr], key=f"ano_cr_{projeto}")
        with col_cr2:
            mes_cr = st.selectbox(f"Mês - {TITULOS[projeto]}", ["Todos"] + [str(m).zfill(2) for m in meses_cr], key=f"mes_cr_{projeto}")

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

        if grafico.empty:
            st.info("Sem dados para exibir.")
        else:
            grafico["mes_str"] = grafico["mes_str"].dt.strftime("%b/%Y")
            fig = px.bar(grafico, x="mes_str", y=["Criados", "Resolvidos"], barmode="group", text_auto=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"crv_{projeto}")

        # ======================
        # 2) SLA (COM FILTROS)
        # ======================
        st.markdown("### ⏱️ SLA")
        anos_sla = sorted(dfp["mes_resolved"].dropna().dt.year.unique())
        meses_sla = sorted(dfp["mes_resolved"].dropna().dt.month.unique())
        col_sla1, col_sla2 = st.columns(2)
        with col_sla1:
            ano_sla = st.selectbox(f"Ano - {TITULOS[projeto]} (SLA)", ["Todos"] + [str(a) for a in anos_sla], key=f"ano_sla_{projeto}")
        with col_sla2:
            mes_sla = st.selectbox(f"Mês - {TITULOS[projeto]} (SLA)", ["Todos"] + [str(m).zfill(2) for m in meses_sla], key=f"mes_sla_{projeto}")

        df_sla = dfp[dfp["sla_millis"].notna()].copy()
        if ano_sla != "Todos":
            df_sla = df_sla[df_sla["mes_resolved"].dt.year == int(ano_sla)]
        if mes_sla != "Todos":
            df_sla = df_sla[df_sla["mes_resolved"].dt.month == int(mes_sla)]

        df_sla["mes_str"] = df_sla["mes_resolved"].dt.strftime("%b/%Y")
        if df_sla.empty:
            st.info("Sem dados de SLA para exibir.")
        else:
            df_sla["dentro_sla"] = df_sla["sla_millis"] <= SLA_LIMITE[projeto]
            agrup = (
                df_sla.groupby("mes_str")["dentro_sla"]
                    .value_counts(normalize=True)
                    .unstack(fill_value=0) * 100
            ).rename(columns={True: "% Dentro SLA", False: "% Fora SLA"})

            agr_wide = agrup.reset_index().copy()
            y_cols = [c for c in ["% Dentro SLA", "% Fora SLA"] if c in agr_wide.columns]
            if not y_cols:
                st.info("Sem colunas de SLA para exibir.")
            else:
                try:
                    agr_wide["mes_data"] = pd.to_datetime(agr_wide["mes_str"], format="%b/%Y")
                except Exception:
                    agr_wide["mes_data"] = pd.to_datetime(agr_wide["mes_str"], errors="coerce")
                agr_wide = agr_wide.sort_values("mes_data")
                agr_wide["mes_str"] = agr_wide["mes_data"].dt.strftime("%b/%Y")
                cats = agr_wide["mes_str"].dropna().unique().tolist()
                agr_wide["mes_str"] = pd.Categorical(agr_wide["mes_str"], categories=cats, ordered=True)
                for c in y_cols:
                    agr_wide[c] = pd.to_numeric(agr_wide[c], errors="coerce").fillna(0)

                fig_sla = px.bar(
                    agr_wide,
                    x="mes_str",
                    y=y_cols,
                    barmode="group",
                    title=f"SLA — {TITULOS[projeto]}",
                    color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"}
                )
                fig_sla.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
                fig_sla.update_yaxes(ticksuffix="%")
                st.plotly_chart(fig_sla, use_container_width=True, key=f"sla_{projeto}")

        # ======================
        # 3) Assunto Relacionado (COM FILTROS)
        # ======================
        st.markdown("### 🧾 Assunto Relacionado")
        anos_ass = sorted(dfp["mes_created"].dt.year.dropna().unique())
        meses_ass = sorted(dfp["mes_created"].dt.month.dropna().unique())
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
        st.dataframe(assunto_count, use_container_width=True)

        # ======================
        # 3.1) Submenu APP NE — só para TDS (gráfico com números grandes)
        # ======================
        if projeto == "TDS":
            with st.expander("📱 APP NE — Origem do problema", expanded=False):
                if "assunto_nome" not in dfp.columns:
                    if CAMPOS_ASSUNTO[projeto] == "issuetype":
                        dfp["assunto_nome"] = dfp["issuetype"].apply(
                            lambda x: x.get("name") if isinstance(x, dict) else (str(x) if x is not None else "—")
                        )
                    else:
                        dfp["assunto_nome"] = dfp["assunto"].apply(
                            lambda x: x.get("value") if isinstance(x, dict) else (str(x) if x is not None else "—")
                        )

                df_app = dfp[dfp["assunto_nome"] == ASSUNTO_ALVO_APPNE].copy()
                if df_app.empty:
                    st.info(f"Não há chamados com Assunto '{ASSUNTO_ALVO_APPNE}'.")
                else:
                    df_app["origem_nome"] = df_app["origem"].apply(
                        lambda x: x.get("value") if isinstance(x, dict) else (str(x) if x is not None else "—")
                    )
                    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()

                    anos_app = sorted(df_app["mes_dt"].dt.year.dropna().unique())
                    meses_app = sorted(df_app["mes_dt"].dt.month.dropna().unique())
                    col_app1, col_app2 = st.columns(2)
                    with col_app1:
                        ano_app = st.selectbox("Ano (APP NE)", ["Todos"] + [str(a) for a in anos_app], key=f"ano_app_{projeto}")
                    with col_app2:
                        mes_app = st.selectbox("Mês (APP NE)", ["Todos"] + [str(m).zfill(2) for m in meses_app], key=f"mes_app_{projeto}")

                    df_app_f = df_app.copy()
                    if ano_app != "Todos":
                        df_app_f = df_app_f[df_app_f["mes_dt"].dt.year == int(ano_app)]
                    if mes_app != "Todos":
                        df_app_f = df_app_f[df_app_f["mes_dt"].dt.month == int(mes_app)]

                    if df_app_f.empty:
                        st.info("Sem dados para exibir com os filtros selecionados.")
                    else:
                        total_app = len(df_app_f)
                        contagem = df_app_f["origem_nome"].value_counts(dropna=False).to_dict()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total (APP NE/EN)", total_app)
                        c2.metric("APP NE", contagem.get("APP NE", 0))
                        c3.metric("APP EN", contagem.get("APP EN", 0))

                        serie = (
                            df_app_f.groupby(["mes_dt", "origem_nome"])
                                    .size()
                                    .reset_index(name="Qtd")
                                    .sort_values("mes_dt")
                        )
                        serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
                        cats = serie["mes_str"].dropna().unique().tolist()
                        serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

                        fig_app = px.bar(
                            serie,
                            x="mes_str",
                            y="Qtd",
                            color="origem_nome",
                            barmode="group",
                            title="APP NE — Volumes por mês e Origem do problema",
                            color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4"},
                            text="Qtd",
                            height=800,
                        )
                        fig_app.update_traces(
                            texttemplate="%{text:.0f}",
                            textposition="outside",
                            textfont_size=18,
                            cliponaxis=False,
                        )
                        max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
                        if max_qtd > 0:
                            fig_app.update_yaxes(range=[0, max_qtd * 1.25])
                        fig_app.update_layout(
                            yaxis_title="Qtd",
                            xaxis_title="Mês",
                            uniformtext_minsize=16,
                            uniformtext_mode="show",
                            bargap=0.15,
                            margin=dict(t=70, r=20, b=60, l=50),
                        )
                        st.plotly_chart(fig_app, use_container_width=True)

                        df_app_f["mes_str"] = df_app_f["mes_dt"].dt.strftime("%b/%Y")
                        cols_show = ["key", "created", "mes_str", "assunto_nome", "origem_nome", "status"]
                        cols_show = [c for c in cols_show if c in df_app_f.columns]
                        st.dataframe(df_app_f[cols_show], use_container_width=True, hide_index=True)

        # ======================
        # 4) Área Solicitante (COM FILTROS, exceto INTEL)
        # ======================
        if projeto != "INTEL":
            st.markdown("### 📦 Área Solicitante")
            anos_area = sorted(dfp["mes_created"].dt.year.dropna().unique())
            meses_area = sorted(dfp["mes_created"].dt.month.dropna().unique())
            col_ar1, col_ar2 = st.columns(2)
            with col_ar1:
                ano_area = st.selectbox(f"Ano - {TITULOS[projeto]} (Área)", ["Todos"] + [
