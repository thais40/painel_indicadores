import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.auth import HTTPBasicAuth

# ======================
# Configuração geral
# ======================
st.set_page_config(layout="wide")
st.title("📊 Painel de Indicadores — Jira")

# 🔄 Botão para atualizar dados do Jira manualmente
if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

# ======================
# Conexão Jira (secrets)
# ======================
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets["EMAIL"]
TOKEN = st.secrets["TOKEN"]
auth = HTTPBasicAuth(EMAIL, TOKEN)

# ======================
# Parâmetros por projeto
# ======================
PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {
    "TDS": "Tech Support",
    "INT": "Integrations",
    "TINE": "IT Support NE",
    "INTEL": "Intelligence",
}
# Campo de SLA por projeto
SLA_CAMPOS = {
    "TDS": "customfield_13744",
    "TINE": "customfield_13744",
    "INT": "customfield_13686",
    "INTEL": "customfield_13686",
}
# Campo de "Assunto" por projeto
CAMPOS_ASSUNTO = {
    "TDS": "customfield_13712",
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype",  # INTEL usa issuetype como assunto
}
CAMPO_AREA = "customfield_13719"
CAMPO_N3 = "customfield_13659"
CAMPO_ORIGEM = "customfield_13628"  # TDS APP NE — Origem do problema
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

# Limites de SLA (ms)
SLA_LIMITE = {
    "TDS": 40 * 60 * 60 * 1000,   # 40h
    "INT": 40 * 60 * 60 * 1000,
    "TINE": 40 * 60 * 60 * 1000,
    "INTEL": 80 * 60 * 60 * 1000, # 80h
}

# ======================
# Funções auxiliares
# ======================
def extrair_sla_millis(sla_field: dict):
    """Extrai o elapsedTime.millis do SLA (ciclo completo ou em andamento)."""
    try:
        if sla_field and isinstance(sla_field, dict):
            if sla_field.get("completedCycles"):
                return sla_field["completedCycles"][0].get("elapsedTime", {}).get("millis")
            if sla_field.get("ongoingCycle"):
                return sla_field["ongoingCycle"].get("elapsedTime", {}).get("millis")
    except Exception:
        return None
    return None

def safe_get_value(x, key="value", fallback="—"):
    if isinstance(x, dict):
        return x.get(key, fallback)
    return x if x is not None else fallback

def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    """Garante a coluna assunto_nome em df_proj, conforme o projeto."""
    if "assunto_nome" not in df_proj.columns:
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_proj["assunto_nome"] = df_proj["issuetype"].apply(
                lambda x: safe_get_value(x, "name")
            )
        else:
            df_proj["assunto_nome"] = df_proj["assunto"].apply(
                lambda x: safe_get_value(x, "value")
            )
    return df_proj

@st.cache_data(show_spinner="🔄 Buscando dados do Jira...")
def buscar_issues() -> pd.DataFrame:
    """Coleta issues dos projetos com os campos necessários desde 2024-01-01."""
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
                ),
            }
            resp = requests.get(f"{JIRA_URL}/rest/api/3/search", auth=auth, params=params)
            if resp.status_code != 200:
                break
            issues = resp.json().get("issues", [])
            if not issues:
                break
            for it in issues:
                f = it.get("fields", {})
                sla_raw = f.get(SLA_CAMPOS[projeto], {})
                todos.append(
                    {
                        "projeto": projeto,
                        "key": it.get("key"),
                        "created": f.get("created"),
                        "resolutiondate": f.get("resolutiondate"),
                        "status": safe_get_value(f.get("status"), "name"),
                        "sla_millis": extrair_sla_millis(sla_raw),
                        "issuetype": f.get("issuetype"),
                        "assunto": f.get(CAMPOS_ASSUNTO[projeto]),
                        "area": f.get(CAMPO_AREA),
                        "n3": f.get(CAMPO_N3),
                        "origem": f.get(CAMPO_ORIGEM),
                    }
                )
            start += 100
    df_all = pd.DataFrame(todos)
    if df_all.empty:
        return df_all
    # Datas e colunas mensais
    df_all["created"] = pd.to_datetime(df_all["created"], errors="coerce")
    df_all["resolved"] = pd.to_datetime(df_all["resolutiondate"], errors="coerce")
    df_all["mes_created"] = df_all["created"].dt.to_period("M").dt.to_timestamp()
    df_all["mes_resolved"] = df_all["resolved"].dt.to_period("M").dt.to_timestamp()
    return df_all

# ======================
# Carregar dados
# ======================
df = buscar_issues()

# ======================
# UI por projeto (abas)
# ======================
tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")
        dfp = df[df["projeto"] == projeto].copy()
        if dfp.empty:
            st.info("Sem dados carregados para este projeto.")
            continue

        # Garante coluna assunto_nome
        dfp = ensure_assunto_nome(dfp, projeto)

        # ======================
        # 1) Criados vs Resolvidos (ANTES do SLA)
        # ======================
        st.markdown("### 📈 Tickets Criados vs Resolvidos")
        anos_cr = sorted(dfp["mes_created"].dt.year.dropna().unique())
        meses_cr = sorted(dfp["mes_created"].dt.month.dropna().unique())
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            ano_cr = st.selectbox(
                f"Ano - {TITULOS[projeto]}",
                ["Todos"] + [str(a) for a in anos_cr],
                key=f"ano_cr_{projeto}",
            )
        with col_cr2:
            mes_cr = st.selectbox(
                f"Mês - {TITULOS[projeto]}",
                ["Todos"] + [str(m).zfill(2) for m in meses_cr],
                key=f"mes_cr_{projeto}",
            )

        df_cr = dfp.copy()
        if ano_cr != "Todos":
            df_cr = df_cr[df_cr["mes_created"].dt.year == int(ano_cr)]
        if mes_cr != "Todos":
            df_cr = df_cr[df_cr["mes_created"].dt.month == int(mes_cr)]

        criados = df_cr.groupby("mes_created").size().reset_index(name="Criados")
        resolvidos = (
            df_cr[df_cr["resolved"].notna()]
            .groupby("mes_resolved")
            .size()
            .reset_index(name="Resolvidos")
        )
        criados.rename(columns={"mes_created": "mes_str"}, inplace=True)
        resolvidos.rename(columns={"mes_resolved": "mes_str"}, inplace=True)
        grafico = pd.merge(criados, resolvidos, how="outer", on="mes_str").fillna(0)
        grafico = grafico.sort_values("mes_str")
        if grafico.empty:
            st.info("Sem dados para exibir.")
        else:
            grafico["mes_str"] = grafico["mes_str"].dt.strftime("%b/%Y")
            fig = px.bar(
                grafico,
                x="mes_str",
                y=["Criados", "Resolvidos"],
                barmode="group",
                text_auto=True,
                height=440,
            )
            # 🔧 valores dos rótulos na horizontal
            fig.update_traces(textangle=0, textfont_size=14, cliponaxis=False)
            fig.update_layout(margin=dict(t=40, r=20, b=50, l=40))
            st.plotly_chart(fig, use_container_width=True, key=f"crv_{projeto}")

        # ======================
        # 2) SLA
        # ======================
        st.markdown("### ⏱️ SLA")
        anos_sla = sorted(dfp["mes_resolved"].dropna().dt.year.unique())
        meses_sla = sorted(dfp["mes_resolved"].dropna().dt.month.unique())
        col_sla1, col_sla2 = st.columns(2)
        with col_sla1:
            ano_sla = st.selectbox(
                f"Ano - {TITULOS[projeto]} (SLA)",
                ["Todos"] + [str(a) for a in anos_sla],
                key=f"ano_sla_{projeto}",
            )
        with col_sla2:
            mes_sla = st.selectbox(
                f"Mês - {TITULOS[projeto]} (SLA)",
                ["Todos"] + [str(m).zfill(2) for m in meses_sla],
                key=f"mes_sla_{projeto}",
            )

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
                .unstack(fill_value=0)
                * 100
            ).rename(columns={True: "% Dentro SLA", False: "% Fora SLA"})

            agr_wide = agrup.reset_index().copy()
            # ordenar meses
            try:
                agr_wide["mes_data"] = pd.to_datetime(agr_wide["mes_str"], format="%b/%Y")
            except Exception:
                agr_wide["mes_data"] = pd.to_datetime(agr_wide["mes_str"], errors="coerce")
            agr_wide = agr_wide.sort_values("mes_data")
            agr_wide["mes_str"] = agr_wide["mes_data"].dt.strftime("%b/%Y")
            cats = agr_wide["mes_str"].dropna().unique().tolist()
            agr_wide["mes_str"] = pd.Categorical(agr_wide["mes_str"], categories=cats, ordered=True)

            # safe numeric
            for c in ["% Dentro SLA", "% Fora SLA"]:
                if c in agr_wide.columns:
                    agr_wide[c] = pd.to_numeric(agr_wide[c], errors="coerce").fillna(0)

            y_cols = [c for c in ["% Dentro SLA", "% Fora SLA"] if c in agr_wide.columns]
            fig_sla = px.bar(
                agr_wide,
                x="mes_str",
                y=y_cols,
                barmode="group",
                title=f"SLA — {TITULOS[projeto]}",
                color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
                height=440,
            )
            fig_sla.update_traces(texttemplate="%{y:.1f}%", textposition="outside", textfont_size=14, cliponaxis=False)
            fig_sla.update_yaxes(ticksuffix="%")
            fig_sla.update_layout(margin=dict(t=60, r=20, b=50, l=40))
            st.plotly_chart(fig_sla, use_container_width=True, key=f"sla_{projeto}")

        # ======================
        # 3) Assunto Relacionado
        # ======================
        st.markdown("### 🧾 Assunto Relacionado")
        anos_ass = sorted(dfp["mes_created"].dt.year.dropna().unique())
        meses_ass = sorted(dfp["mes_created"].dt.month.dropna().unique())
        col_ass1, col_ass2 = st.columns(2)
        with col_ass1:
            ano_ass = st.selectbox(
                f"Ano - {TITULOS[projeto]} (Assunto)",
                ["Todos"] + [str(a) for a in anos_ass],
                key=f"ano_ass_{projeto}",
            )
        with col_ass2:
            mes_ass = st.selectbox(
                f"Mês - {TITULOS[projeto]} (Assunto)",
                ["Todos"] + [str(m).zfill(2) for m in meses_ass],
                key=f"mes_ass_{projeto}",
            )

        df_ass = dfp.copy()
        if ano_ass != "Todos":
            df_ass = df_ass[df_ass["mes_created"].dt.year == int(ano_ass)]
        if mes_ass != "Todos":
            df_ass = df_ass[df_ass["mes_created"].dt.month == int(mes_ass)]

        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_ass["assunto_nome"] = df_ass["issuetype"].apply(lambda x: safe_get_value(x, "name"))
        else:
            df_ass["assunto_nome"] = df_ass["assunto"].apply(lambda x: safe_get_value(x, "value"))

        assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
        assunto_count.columns = ["Assunto", "Qtd"]
        st.dataframe(assunto_count, use_container_width=True, hide_index=True)

        # ======================
        # 4) Área Solicitante (exceto INTEL)
        # ======================
        if projeto != "INTEL":
            st.markdown("### 📦 Área Solicitante")
            anos_area = sorted(dfp["mes_created"].dt.year.dropna().unique())
            meses_area = sorted(dfp["mes_created"].dt.month.dropna().unique())
            col_ar1, col_ar2 = st.columns(2)
            with col_ar1:
                ano_area = st.selectbox(
                    f"Ano - {TITULOS[projeto]} (Área)",
                    ["Todos"] + [str(a) for a in anos_area],
                    key=f"ano_area_{projeto}",
                )
            with col_ar2:
                mes_area = st.selectbox(
                    f"Mês - {TITULOS[projeto]} (Área)",
                    ["Todos"] + [str(m).zfill(2) for m in meses_area],
                    key=f"mes_area_{projeto}",
                )

            df_area = dfp.copy()
            if ano_area != "Todos":
                df_area = df_area[df_area["mes_created"].dt.year == int(ano_area)]
            if mes_area != "Todos":
                df_area = df_area[df_area["mes_created"].dt.month == int(mes_area)]

            df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
            area_count = df_area["area_nome"].value_counts().reset_index()
            area_count.columns = ["Área", "Qtd"]
            st.dataframe(area_count, use_container_width=True, hide_index=True)

        # ======================
        # 5) Encaminhamentos (TDS e INT)
        # ======================
        if projeto in ("TDS", "INT"):
            st.markdown("### 🔄 Encaminhamentos")
            anos_enc = sorted(dfp["mes_created"].dt.year.dropna().unique())
            meses_enc = sorted(dfp["mes_created"].dt.month.dropna().unique())
            col_en1, col_en2 = st.columns(2)
            with col_en1:
                ano_enc = st.selectbox(
                    f"Ano - {TITULOS[projeto]} (Encaminhamentos)",
                    ["Todos"] + [str(a) for a in anos_enc],
                    key=f"ano_enc_{projeto}",
                )
            with col_en2:
                mes_enc = st.selectbox(
                    f"Mês - {TITULOS[projeto]} (Encaminhamentos)",
                    ["Todos"] + [str(m).zfill(2) for m in meses_enc],
                    key=f"mes_enc_{projeto}",
                )

            df_enc = dfp.copy()
            if ano_enc != "Todos":
                df_enc = df_enc[df_enc["mes_created"].dt.year == int(ano_enc)]
            if mes_enc != "Todos":
                df_enc = df_enc[df_enc["mes_created"].dt.month == int(mes_enc)]

            col1, col2 = st.columns(2)
            with col1:
                count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
                st.metric("Encaminhados Produto", count_prod)
            with col2:
                df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
                st.metric("Encaminhados N3", (df_enc["n3_valor"] == "Sim").sum())

        # ======================
        # 6) Onboarding (somente INT) — no final
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

                # Filtros (Ano/Mês) do submenu
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

                # Métricas
                total_clientes_novos = (df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO).sum()
                df_erros = df_onb[df_onb["assunto_nome"].isin(ASSUNTOS_ERROS)].copy()
                pend_mask = df_onb["status"].isin(STATUS_PENDENCIAS)
                tickets_pendencias = pend_mask.sum()
                possiveis_clientes = pend_mask.sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tickets clientes novos", total_clientes_novos)
                c2.metric("Erros onboarding", len(df_erros))
                c3.metric("Tickets com pendências", tickets_pendencias)
                c4.metric("Possíveis clientes", possiveis_clientes)

                st.markdown("---")

                # Erros Onboarding — gráfico horizontal
                if df_erros.empty:
                    st.info("Sem erros de Onboarding no período/filtros selecionados.")
                else:
                    cont_erros = (
                        df_erros["assunto_nome"]
                        .value_counts()
                        .reindex(ASSUNTOS_ERROS, fill_value=0)
                        .reset_index()
                    )
                    cont_erros.columns = ["Categoria", "Qtd"]
                    fig_onb = px.bar(
                        cont_erros,
                        x="Qtd",
                        y="Categoria",
                        orientation="h",
                        text="Qtd",
                        title="Erros Onboarding",
                        height=420,
                    )
                    fig_onb.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
                    max_q = int(cont_erros["Qtd"].max()) if not cont_erros.empty else 0
                    if max_q > 0:
                        fig_onb.update_xaxes(range=[0, max_q * 1.25])
                    fig_onb.update_layout(margin=dict(t=50, r=20, b=30, l=10), bargap=0.25)
                    st.plotly_chart(fig_onb, use_container_width=True)

                st.markdown("---")

                # Gráfico mensal — Clientes Novos com OBG e variação MoM (sem sobrepor)
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
                        serie_cli,
                        x="mes_str",
                        y="ClientesNovos",
                        title="Tickets - Cliente novo",
                        text="ClientesNovos",
                        height=380,
                    )
                    fig_cli.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=14, cliponaxis=False)

                    # 👉 Mais folga no topo (aumentado) para separar OBG e MoM de vez
                    y_top = (serie_cli["ClientesNovos"].max() * 2.2) if len(serie_cli) else 10
                    fig_cli.update_yaxes(range=[0, y_top])

                    # OBG logo acima da barra e MoM lá no topo (sem sobreposição)
                    for _, r in serie_cli.iterrows():
                        x = r["mes_str"]
                        yb = float(r["ClientesNovos"])

                        # OBG (próximo da barra)
                        fig_cli.add_annotation(
                            x=x, y=yb + (y_top * 0.05),
                            text=f"OBG {int(r['OBG_Rotulo'])}",
                            showarrow=False,
                            font=dict(size=12, color="#6b7280"),
                            align="center",
                        )

                        # MoM (bem mais alto)
                        if pd.notna(r["MoM"]):
                            mom_abs = abs(r["MoM"])
                            if mom_abs >= 1:
                                up = r["MoM"] >= 0
                                arrow = "▲" if up else "▼"
                                color = "#2563eb" if up else "#dc2626"
                                fig_cli.add_annotation(
                                    x=x, y=yb + (y_top * 0.20),
                                    text=f"{arrow} {mom_abs:.0f}%",
                                    showarrow=False,
                                    font=dict(size=12, color=color),
                                    align="center",
                                )

                    fig_cli.update_layout(
                        margin=dict(t=50, r=20, b=35, l=40),
                        bargap=0.18,
                        xaxis_title=None, yaxis_title="ClientesNovos",
                        uniformtext_mode="show", uniformtext_minsize=12,
                    )
                    st.plotly_chart(fig_cli, use_container_width=True)

                st.markdown("---")

                # Tabela (Receita se existir)
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

                # --- Dinheiro perdido (simulação) — clientes = Possíveis clientes
                st.markdown("---")
                st.subheader("💸 Dinheiro perdido (simulação)")

                c_left, c_right = st.columns([1, 1])
                with c_left:
                    clientes_sim = int(possiveis_clientes)  # igual ao "Possíveis clientes"
                    st.number_input(
                        "Cliente novos (simulação)",
                        value=clientes_sim,
                        disabled=True,          # bloqueia edição
                        key=f"sim_clientes_{projeto}"
                    )
                with c_right:
                    receita_cliente = st.slider(
                        "Cenário Receita por Cliente (R$)",
                        min_value=0, max_value=100000, step=500, value=20000,
                        key=f"sim_receita_{projeto}"
                    )

                dinheiro_perdido = float(clientes_sim) * float(receita_cliente)
                st.markdown(
                    f"### **R$ {dinheiro_perdido:,.2f}**",
                    help="Cálculo: Cliente novos (simulação) × Cenário Receita por Cliente",
                )

        # ======================
        # 7) APP NE — só TDS (no final)
        # ======================
        if projeto == "TDS":
            with st.expander("📱 APP NE — Origem do problema", expanded=False):
                # filtro robusto do assunto alvo
                s_ass = dfp["assunto_nome"].astype(str).str.strip()
                alvo = ASSUNTO_ALVO_APPNE.strip().casefold()
                mask_assunto = s_ass.str.casefold().eq(alvo)
                if not mask_assunto.any():
                    # fallback: contém "app ne"
                    mask_assunto = s_ass.str.contains(r"app\s*ne", case=False, regex=True)

                df_app = dfp[mask_assunto].copy()
                if df_app.empty:
                    st.info(f"Não há chamados para '{ASSUNTO_ALVO_APPNE}'.")
                else:
                    df_app["origem_nome"] = df_app["origem"].apply(lambda x: safe_get_value(x, "value"))
                    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()

                    anos_app = sorted(df_app["mes_dt"].dt.year.dropna().unique())
                    meses_app = sorted(df_app["mes_dt"].dt.month.dropna().unique())
                    c_a1, c_a2 = st.columns(2)
                    with c_a1:
                        ano_app = st.selectbox("Ano (APP NE)", ["Todos"] + [str(a) for a in anos_app], key=f"ano_app_{projeto}")
                    with c_a2:
                        mes_app = st.selectbox("Mês (APP NE)", ["Todos"] + [str(m).zfill(2) for m in meses_app], key=f"mes_app_{projeto}")

                    df_app_f = df_app.copy()
                    if ano_app != "Todos":
                        df_app_f = df_app_f[df_app_f["mes_dt"].dt.year == int(ano_app)]
                    if mes_app != "Todos":
                        df_app_f = df_app_f[df_app_f["mes_dt"].dt.month == int(mes_app)]

                    if df_app_f.empty:
                        st.info("Sem dados para exibir com os filtros selecionados.")
                    else:
                        # Métricas
                        total_app = len(df_app_f)
                        contagem = df_app_f["origem_nome"].value_counts(dropna=False).to_dict()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total (APP NE/EN)", total_app)
                        m2.metric("APP NE", contagem.get("APP NE", 0))
                        m3.metric("APP EN", contagem.get("APP EN", 0))

                        # Série mensal
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
                            height=460,
                        )
                        fig_app.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
                        max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
                        if max_qtd > 0:
                            fig_app.update_yaxes(range=[0, max_qtd * 1.25])
                        fig_app.update_layout(
                            yaxis_title="Qtd",
                            xaxis_title="Mês",
                            uniformtext_minsize=14,
                            uniformtext_mode="show",
                            bargap=0.15,
                            margin=dict(t=70, r=20, b=60, l=50),
                        )
                        st.plotly_chart(fig_app, use_container_width=True)

                        # Tabela
                        df_app_f["mes_str"] = df_app_f["mes_dt"].dt.strftime("%b/%Y")
                        cols_show = ["key", "created", "mes_str", "assunto_nome", "origem_nome", "status"]
                        cols_show = [c for c in cols_show if c in df_app_f.columns]
                        st.dataframe(df_app_f[cols_show], use_container_width=True, hide_index=True)
