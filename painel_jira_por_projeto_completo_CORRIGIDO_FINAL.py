# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Painel de Indicadores — Completo", layout="wide")

# =====================================================
# Configurações gerais e helpers
# =====================================================
# Limites de SLA por projeto (exemplo, em ms)
SLA_LIMITE = {
    "TDS": 40 * 60 * 60 * 1000,     # 40 horas
    "INT": 40 * 60 * 60 * 1000,
    "TINE": 40 * 60 * 60 * 1000,
    "INTEL": 80 * 60 * 60 * 1000,   # 80 horas
}
SLA_PADRAO_MILLIS = 40 * 60 * 60 * 1000

def ordenar_mes_str(df: pd.DataFrame, col: str = "mes_str") -> pd.DataFrame:
    """Cria coluna mes_data, ordena e fixa categoria ordenada para mes_str (%b/%Y)."""
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

def garantir_projeto():
    """Garante variável 'projeto' (sidebar → fallback)."""
    try:
        _ = projeto  # noqa
        return projeto
    except NameError:
        pass
    projetos = list(SLA_LIMITE.keys()) or ["TDS"]
    return st.sidebar.selectbox("Projeto", projetos, index=0)

# =====================================================
# MOCKS DE DADOS (substitua pelos seus dataframes reais)
# =====================================================
np.random.seed(42)
# período de 20 meses
meses = pd.date_range("2024-01-01", periods=20, freq="MS")
mes_str = [d.strftime("%b/%Y") for d in meses]

# Issues "criadas" vs "resolvidas"
df_issues = pd.DataFrame({
    "id": range(1, 1201),
    "created": np.random.choice(meses, 1200),
})
# resolvidas em até 90 dias em média
df_issues["resolved"] = df_issues["created"] + pd.to_timedelta(np.random.randint(0, 90, size=len(df_issues)), unit="D")

# SLA por issue
projeto = garantir_projeto()
limite_ms = SLA_LIMITE.get(projeto, SLA_PADRAO_MILLIS)
# gera um "tempo de atendimento" aleatório (em horas → ms)
atendimento_horas = np.random.randint(1, 120, size=len(df_issues))
df_issues["sla_millis"] = atendimento_horas * 60 * 60 * 1000
df_issues["dentro_sla"] = df_issues["sla_millis"] <= limite_ms
df_issues["mes_str"] = pd.to_datetime(df_issues["created"]).dt.strftime("%b/%Y")

# =======================
# MOCK dos campos Jira (Assunto/Origem)
# (Remova ao conectar com seus dados reais)
# =======================
ASSUNTO_ALVO = "Problemas no App NE - App EN"
prob_assunto_alvo = 0.35  # ~35% dos chamados entram no tema alvo
df_issues["assunto_relacionado"] = np.where(
    np.random.rand(len(df_issues)) < prob_assunto_alvo,
    ASSUNTO_ALVO,
    "Outros"
)
mask_app = df_issues["assunto_relacionado"] == ASSUNTO_ALVO
df_issues.loc[mask_app, "origem_problema"] = np.random.choice(["APP NE", "APP EN"], size=mask_app.sum())

# Tabelas "Assunto Relacionado" e "Área Solicitante" (mocks)
assuntos = [
    "Problemas no App NE - App EN",
    "Extravio - Transportadora Ponto de Postagem NE",
    "Erro no processamento - CTE",
    "PLP transportadora - Não coletado",
    "Alteração de status - Cancelada",
    "Erro no processamento - Cotação/CEP no custo",
    "Cancelar reembolso",
    "Erro no processamento - Outros",
    "Alteração de status - Devolução",
    "Erro no processamento - Inscrição Estadual",
]
qtd_assuntos = np.random.randint(600, 6000, size=len(assuntos))
df_assunto = pd.DataFrame({"Assunto": assuntos, "Qtd": qtd_assuntos}).sort_values("Qtd", ascending=False)

areas = [
    "None","Customer Success","Ops - Expedição","Backoffice","Ops - Cubagem","Ops - Divergências",
    "Suporte - Infra","Relacionamento","Ops - Coletas","Ops - Logística"
]
qtd_areas = np.random.randint(700, 6000, size=len(areas))
df_area = pd.DataFrame({"Área": areas, "Qtd": qtd_areas}).sort_values("Qtd", ascending=False)

# =====================================================
# Sidebar — Submenu quando projeto = TDS
# =====================================================
sub_menu = None
if projeto == "TDS":
    sub_menu = st.sidebar.radio("TDS • Submenu", ["Geral", "APP NE"], index=0)

# =====================================================
# SEÇÃO 1 — Criados vs Resolvidos (PRIMEIRO)
# =====================================================
st.title("📊 Painel de Indicadores — Completo")
st.markdown("Gráfico de **Criados vs Resolvidos** aparece primeiro, como solicitado.")

df_months = pd.DataFrame({"mes": meses})
criadas = df_issues.groupby(pd.to_datetime(df_issues["created"]).dt.to_period("M")).size().rename("Criados")
criadas.index = criadas.index.to_timestamp()
resolvidas = df_issues.dropna(subset=["resolved"]).groupby(pd.to_datetime(df_issues["resolved"]).dt.to_period("M")).size().rename("Resolvidos")
resolvidas.index = resolvidas.index.to_timestamp()

df_cr_res = df_months.set_index("mes").join(criadas, how="left").join(resolvidas, how="left").fillna(0).reset_index()
df_cr_res["mes_str"] = df_cr_res["mes"].dt.strftime("%b/%Y")
df_cr_res = ordenar_mes_str(df_cr_res, "mes_str")

fig_cr = px.bar(
    df_cr_res,
    x="mes_str",
    y=["Criados", "Resolvidos"],
    barmode="group",
    title="Criados vs Resolvidos",
)
st.plotly_chart(fig_cr, use_container_width=True)

st.markdown("---")

# =====================================================
# SEÇÃO 2 — SLA (barras lado a lado, meses ordenados)
# =====================================================
okr_label = "🎯 OKR (mock) — Percentual dentro/fora do SLA por mês"
agrupado = (
    df_issues
    .groupby("mes_str")["dentro_sla"]
    .value_counts(normalize=True)
    .unstack(fill_value=0) * 100.0
)

# Garante apenas colunas True/False e renomeia para legenda desejada
cols_bool = []
if True in agrupado.columns: cols_bool.append(True)
if False in agrupado.columns: cols_bool.append(False)
agr_wide = agrupado[cols_bool].copy() if cols_bool else agrupado.copy()
agr_wide.rename(columns={True: "% Dentro SLA", False: "% Fora SLA"}, inplace=True)
agr_wide = agr_wide.reset_index()
agr_wide = ordenar_mes_str(agr_wide, "mes_str")

fig_sla = px.bar(
    agr_wide,
    x="mes_str",
    y=[c for c in ["% Dentro SLA", "% Fora SLA"] if c in agr_wide.columns],
    barmode="group",  # lado a lado
    title=okr_label,
    color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
)
fig_sla.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
fig_sla.update_yaxes(ticksuffix="%")
st.plotly_chart(fig_sla, use_container_width=True)

# =====================================================
# SEÇÃO — TDS • APP NE (submenu)
# =====================================================
if projeto == "TDS" and sub_menu == "APP NE":
    st.markdown("---")
    st.header("📱 TDS • APP NE")

    # Filtra somente chamados do assunto alvo
    df_app = df_issues[df_issues["assunto_relacionado"] == ASSUNTO_ALVO].copy()

    if df_app.empty:
        st.info("Não há chamados para o assunto 'Problemas no App NE - App EN' no período selecionado.")
    else:
        # Ordena meses
        df_app = ordenar_mes_str(df_app, "mes_str")

        # KPIs
        total_app = len(df_app)
        por_origem = df_app["origem_problema"].value_counts(dropna=False).to_dict()
        colk1, colk2, colk3 = st.columns(3)
        colk1.metric("Total de chamados (APP NE/EN)", total_app)
        colk2.metric("APP NE", por_origem.get("APP NE", 0))
        colk3.metric("APP EN", por_origem.get("APP EN", 0))

        # Gráfico: chamados por mês, lado a lado por origem
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
            barmode="group",  # lado a lado
            title="APP NE — Volumes por mês e origem",
            color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4"},  # verde e azul
        )
        st.plotly_chart(fig_app, use_container_width=True)

        # Tabela com campos solicitados
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

        # Exportar CSV (opcional)
        csv = df_app[list(view_cols.keys())].rename(columns=view_cols).to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV (APP NE)", data=csv, file_name="tds_app_ne.csv", mime="text/csv")

st.markdown("---")

# =====================================================
# SEÇÃO 3 — Assunto Relacionado (filtros + tabela alinhada)
# =====================================================
st.header("🧾 Assunto Relacionado")
LEFT_RATIO, RIGHT_RATIO = 62, 38  # mantenha igual na tabela

colA, colB = st.columns((LEFT_RATIO, RIGHT_RATIO), gap="large")
with colA:
    filtro_ano_assunto = st.selectbox("Ano - Tech Support (Assunto)", ["Todos", "2024", "2025"], index=0)
with colB:
    filtro_mes_assunto = st.selectbox("Mês - Tech Support (Assunto)", ["Todos"] + [d.strftime("%b/%Y") for d in meses], index=0)

ASSUNTO_W = 900
QTD_W = 340
st.dataframe(
    df_assunto[["Assunto", "Qtd"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Assunto": st.column_config.TextColumn("Assunto", width=ASSUNTO_W),
        "Qtd": st.column_config.NumberColumn("Qtd", format="%d", width=QTD_W),
    },
)

st.markdown("---")

# =====================================================
# SEÇÃO 4 — Área Solicitante (filtros + tabela alinhada)
# =====================================================
st.header("📦 Área Solicitante")

colC, colD = st.columns((LEFT_RATIO, RIGHT_RATIO), gap="large")
with colC:
    filtro_ano_area = st.selectbox("Ano - Tech Support (Área)", ["Todos", "2024", "2025"], index=0)
with colD:
    filtro_mes_area = st.selectbox("Mês - Tech Support (Área)", ["Todos"] + [d.strftime("%b/%Y") for d in meses], index=0)

AREA_W = 900
QTD2_W = 340
st.dataframe(
    df_area[["Área", "Qtd"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Área": st.column_config.TextColumn("Área", width=AREA_W),
        "Qtd": st.column_config.NumberColumn("Qtd", format="%d", width=QTD2_W),
    },
)

st.caption("Substitua os MOCKs pelos seus DataFrames reais. Ordem: Criados vs Resolvidos → SLA → (se TDS e APP NE) submenu → Assunto → Área.")
