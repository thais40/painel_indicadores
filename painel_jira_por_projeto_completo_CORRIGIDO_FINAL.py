import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")
st.title("Painel de Indicadores")

# Simulação de DataFrame - Substitua com sua função real
# df = carregar_dados_jira()
# df = pd.read_csv("seus_dados.csv")

# Conversão de datas
df["created"] = pd.to_datetime(df["created"])
df["resolved"] = pd.to_datetime(df["resolved"])

# Extração de ano e mês
df["ano_created"] = df["created"].dt.year
df["mes_created"] = df["created"].dt.month
df["ano_resolved"] = df["resolved"].dt.year
df["mes_resolved"] = df["resolved"].dt.month

# Filtros
anos_disponiveis = sorted(df["ano_created"].dropna().unique())
meses_disponiveis = sorted(df["mes_created"].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    ano_selecionado = st.selectbox("Selecione o Ano", options=["Todos"] + list(map(str, anos_disponiveis)))
with col2:
    mes_selecionado = st.selectbox("Selecione o Mês", options=["Todos"] + list(map(str, meses_disponiveis)))

# Aplicação dos filtros
df_filtrado = df.copy()
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["ano_created"] == int(ano_selecionado)]
if mes_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["mes_created"] == int(mes_selecionado)]

# Criados vs Resolvidos
df_filtrado["mes_str_created"] = df_filtrado["created"].dt.to_period("M").dt.to_timestamp()
df_filtrado["mes_str_resolved"] = df_filtrado["resolved"].dt.to_period("M").dt.to_timestamp()

criados = df_filtrado.groupby("mes_str_created").size().reset_index(name="Criados")
resolvidos = df_filtrado[df_filtrado["resolved"].notna()].groupby("mes_str_resolved").size().reset_index(name="Resolvidos")

criados.rename(columns={"mes_str_created": "mes_str"}, inplace=True)
resolvidos.rename(columns={"mes_str_resolved": "mes_str"}, inplace=True)

grafico_cr = pd.merge(criados, resolvidos, how="outer", on="mes_str").fillna(0).sort_values("mes_str")
grafico_cr["mes_str"] = grafico_cr["mes_str"].dt.strftime("%b %Y")

fig_cr = px.bar(
    grafico_cr,
    x="mes_str",
    y=["Criados", "Resolvidos"],
    barmode="group",
    text_auto=True,
    title="Chamados Criados vs Resolvidos por Mês",
    height=400  # reduzido para evitar corte
)
fig_cr.update_layout(
    xaxis_title="Mês",
    yaxis_title="Chamados",
    legend_title="Tipo",
    bargap=0.15
)
st.plotly_chart(fig_cr, use_container_width=True, key="grafico_criados_resolvidos")

# Gráfico de SLA com OKR consolidado
if "sla_millis" in df.columns:
    SLA_LIMITE = 40 * 60 * 60 * 1000  # 40 horas em milissegundos

    df_filtrado["mes_resolvido"] = df_filtrado["resolved"].dt.to_period("M").dt.to_timestamp()
    df_filtrado["dentro_sla"] = df_filtrado["sla_millis"] <= SLA_LIMITE

    sla_mes = df_filtrado[df_filtrado["resolved"].notna()].groupby("mes_resolvido")["dentro_sla"].agg([
        ("Dentro do SLA", lambda x: (x == True).sum()),
        ("Fora do SLA", lambda x: (x == False).sum())
    ]).reset_index()

    sla_mes["Total"] = sla_mes["Dentro do SLA"] + sla_mes["Fora do SLA"]
    sla_mes["percentual"] = (sla_mes["Dentro do SLA"] / sla_mes["Total"]) * 100
    sla_mes["mes"] = sla_mes["mes_resolvido"].dt.strftime("%b %Y")

    fig_sla = px.bar(
        sla_mes,
        x="mes",
        y=["Dentro do SLA", "Fora do SLA"],
        text_auto=True,
        barmode="group",
        title="SLA Mensal (com OKR Consolidado)",
        height=400  # reduzido
    )
    fig_sla.update_layout(
        yaxis_title="Chamados",
        xaxis_title="Mês",
        bargap=0.15
    )

    # 🎯 OKR geral no topo
    percentual_total = sla_mes["Dentro do SLA"].sum() / sla_mes["Total"].sum() * 100
    fig_sla.add_annotation(
        x=0.99, y=0.95, xref="paper", yref="paper",
        text=f"🎯 OKR Geral: {percentual_total:.2f}%",
        showarrow=False,
        font=dict(size=14, color="green"),
        align="right"
    )

    st.plotly_chart(fig_sla, use_container_width=True, key="grafico_sla_ok")

# Área Solicitante (caso o campo esteja presente)
if "customfield_13719" in df_filtrado.columns:
    st.subheader("📦 Área Solicitante")

    df_area = df_filtrado.dropna(subset=["customfield_13719"]).copy()
    df_area["area"] = df_area["customfield_13719"].apply(
        lambda x: x.get("value") if isinstance(x, dict) else str(x)
    )

    dados_area = df_area["area"].value_counts().reset_index()
    dados_area.columns = ["Área", "Qtd. Chamados"]
    st.dataframe(dados_area)

# Assunto Relacionado adaptado por projeto
st.subheader("🧾 Assunto Relacionado")

campos_assunto = {
    "TDS": "customfield_13712",
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype"
}

campo_assunto = None
for c in campos_assunto.values():
    if c in df_filtrado.columns:
        campo_assunto = c
        break

if campo_assunto:
    df_assunto = df_filtrado.dropna(subset=[campo_assunto]).copy()
    if campo_assunto == "issuetype":
        df_assunto["assunto"] = df_assunto["issuetype"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )
    else:
        df_assunto["assunto"] = df_assunto[campo_assunto].apply(
            lambda x: x.get("value") if isinstance(x, dict) else str(x)
        )

    dados_assunto = df_assunto["assunto"].value_counts().reset_index()
    dados_assunto.columns = ["Assunto Relacionado", "Qtd. Chamados"]
    st.dataframe(dados_assunto)

# Encaminhamentos (Produto e N3)
st.subheader("🔄 Encaminhamentos")

col1, col2 = st.columns(2)

with col1:
    count_produto = df_filtrado["status"].str.contains("Produto", case=False, na=False).sum()
    st.metric("Encaminhados Produto", count_produto)

with col2:
    if "escalado_n3_valor" in df_filtrado.columns:
        count_n3 = (df_filtrado["escalado_n3_valor"] == "Sim").sum()
        st.metric("Encaminhados N3", count_n3)
