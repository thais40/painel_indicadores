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

        # ... (seções 1 a 5: Criados vs Resolvidos, SLA, Assunto, APP NE (TDS), Área, Encaminhamentos — já estavam prontas)

        # ======================
        # 6) Onboarding (somente INT)
        # ======================
        if projeto == "INT":
            with st.expander("🧭 Onboarding", expanded=False):

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

                if "assunto_nome" not in dfp.columns:
                    if CAMPOS_ASSUNTO[projeto] == "issuetype":
                        dfp["assunto_nome"] = dfp["issuetype"].apply(
                            lambda x: x.get("name") if isinstance(x, dict) else str(x)
                        )
                    else:
                        dfp["assunto_nome"] = dfp["assunto"].apply(
                            lambda x: x.get("value") if isinstance(x, dict) else str(x)
                        )

                anos_ob = sorted(dfp["mes_created"].dt.year.dropna().unique())
                meses_ob = sorted(dfp["mes_created"].dt.month.dropna().unique())
                col_ob1, col_ob2 = st.columns(2)
                with col_ob1:
                    ano_ob = st.selectbox("Ano (Onboarding)", ["Todos"] + [str(a) for a in anos_ob], key=f"ano_onb_{projeto}")
                with col_ob2:
                    mes_ob = st.selectbox("Mês (Onboarding)", ["Todos"] + [str(m).zfill(2) for m in meses_ob], key=f"mes_onb_{projeto}")

                df_onb = dfp.copy()
                if ano_ob != "Todos":
                    df_onb = df_onb[df_onb["mes_created"].dt.year == int(ano_ob)]
                if mes_ob != "Todos":
                    df_onb = df_onb[df_onb["mes_created"].dt.month == int(mes_ob)]

                total_clientes_novos = (df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO).sum()
                df_erros = df_onb[df_onb["assunto_nome"].isin(ASSUNTOS_ERROS)].copy()
                pend_mask = df_onb["status"].isin(STATUS_PENDENCIAS)
                tickets_pendencias = pend_mask.sum()
                possiveis_clientes = pend_mask.sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tickets clientes novos", total_clientes_novos)
                m2.metric("Erros onboarding", len(df_erros))
                m3.metric("Tickets com pendências", tickets_pendencias)
                m4.metric("Possíveis clientes", possiveis_clientes)

                st.markdown("---")

                # --- Erros Onboarding ---
                if df_erros.empty:
                    st.info("Sem erros de Onboarding no período/filtros selecionados.")
                else:
                    cont_erros = (
                        df_erros["assunto_nome"].value_counts()
                        .reindex(ASSUNTOS_ERROS, fill_value=0)
                        .reset_index()
                    )
                    cont_erros.columns = ["Categoria", "Qtd"]
                    fig_onb = px.bar(
                        cont_erros, x="Qtd", y="Categoria",
                        orientation="h", text="Qtd",
                        title="Erros Onboarding", height=420
                    )
                    fig_onb.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
                    max_q = int(cont_erros["Qtd"].max()) if not cont_erros.empty else 0
                    if max_q > 0:
                        fig_onb.update_xaxes(range=[0, max_q * 1.25])
                    st.plotly_chart(fig_onb, use_container_width=True)

                st.markdown("---")

                # --- Gráfico Clientes Novos (OBG Rotulo) ---
                df_cli = df_onb[df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO].copy()
                if not df_cli.empty:
                    serie_cli = (
                        df_cli.groupby(df_cli["mes_created"].dt.to_period("M"))
                              .size()
                              .reset_index(name="ClientesNovos")
                    )
                    serie_cli["mes_dt"] = serie_cli["mes_created"].dt.to_timestamp()
                    serie_cli = serie_cli.sort_values("mes_dt")
                    serie_cli["mes_str"] = serie_cli["mes_dt"].dt.strftime("%Y %b")
                    serie_cli["OBG_Rotulo"] = (serie_cli["ClientesNovos"] * 1.35).round(0).astype(int)
                    serie_cli["MoM"] = serie_cli["ClientesNovos"].pct_change() * 100

                    fig_cli = px.bar(
                        serie_cli, x="mes_str", y="ClientesNovos",
                        title="Tickets - Cliente novo", text="ClientesNovos", height=360
                    )
                    fig_cli.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=14, cliponaxis=False)
                    y_top = (serie_cli["ClientesNovos"].max() * 1.45) if len(serie_cli) else 10
                    fig_cli.update_yaxes(range=[0, y_top])

                    for _, r in serie_cli.iterrows():
                        x = r["mes_str"]; y_bar = r["ClientesNovos"]
                        fig_cli.add_annotation(x=x, y=y_bar + (y_top * 0.05), text=f"OBG {r['OBG_Rotulo']}",
                                               showarrow=False, font=dict(size=12, color="#6b7280"))
                        if pd.notna(r["MoM"]):
                            up = r["MoM"] >= 0
                            arrow = "▲" if up else "▼"
                            color = "#2563eb" if up else "#dc2626"
                            fig_cli.add_annotation(x=x, y=y_bar + (y_top * 0.12),
                                                   text=f"{arrow} {abs(r['MoM']):.0f}%",
                                                   showarrow=False, font=dict(size=12, color=color))
                    st.plotly_chart(fig_cli, use_container_width=True)

                st.markdown("---")

                # --- Tabela (Receita se existir) ---
                col_receita = None
                for c in df_onb.columns:
                    if "receita" in str(c).lower():
                        col_receita = c
                        break
                df_tabela = df_onb[df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO].copy()
                df_tabela["Assunto"] = df_tabela["assunto_nome"]
                df_tabela["Status"] = df_tabela["status"]
                df_tabela["Chave"] = df_tabela["key"]
                df_tabela["Receita"] = df_tabela[col_receita] if col_receita is not None else pd.NA
                st.write("**Tabela — Receita por ticket (se disponível nos dados)**")
                st.dataframe(df_tabela[["Chave", "Status", "Assunto", "Receita"]], use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("💸 Dinheiro perdido (simulação)")
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    clientes_sim = st.number_input("Cliente novos (simulação)", min_value=0, step=1, value=int(total_clientes_novos), key=f"sim_clientes_{projeto}")
                with c_right:
                    receita_cliente = st.slider("Cenário Receita por Cliente (R$)", min_value=0, max_value=100000, step=500, value=20000, key=f"sim_receita_{projeto}")
                dinheiro_perdido = float(clientes_sim) * float(receita_cliente)
                st.markdown(f"### **R$ {dinheiro_perdido:,.2f}**", help="Cálculo: Cliente novos (simulação) × Cenário Receita por Cliente")
