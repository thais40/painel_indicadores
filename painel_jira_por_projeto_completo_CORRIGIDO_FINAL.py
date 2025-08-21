import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from zoneinfo import ZoneInfo  # timezone Brasil

# ======================
# 0) Configuração geral
# ======================
st.set_page_config(layout="wide")

# ===== Estilo Nuvemshop (leve) =====
st.markdown("""
<style>
html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji" !important;
}
:root {
  --ns-primary: #2E7DFF;
  --ns-muted:   #6B7280;
}
h1 { letter-spacing: .2px; }
.stButton > button {
  border-radius: 10px;
  border: 1px solid #e6e8ee;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.update-row { display: inline-flex; align-items: center; gap: 12px; }
.update-caption { color: var(--ns-muted); font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# --- Brand bar com logo (opcional via st.secrets["LOGO_URL"]) ---
# --- Brand bar com logo robusto (prioriza base64) ---
import base64

def _render_logo_and_title():
    logo_bytes = None

    # Opção A: base64 nos secrets (mais estável)
    b64 = st.secrets.get("LOGO_B64")
    if b64:
        try:
            logo_bytes = base64.b64decode(b64)
        except Exception:
            logo_bytes = None

    # Render
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:8px 0 20px 0;">',
        unsafe_allow_html=True,
    )
    if logo_bytes:
        st.image(logo_bytes, width=220)  # você pode ajustar o width aqui
        st.markdown(
            '<span style="color:#111827;font-weight:600;font-size:15px;">Painel interno</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#111827;font-weight:600;font-size:15px;">Nuvemshop · Painel interno</span>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

_render_logo_and_title()

st.title("📊 Painel de Indicadores")

# função de horário em Brasília
TZ_BR = ZoneInfo("America/Sao_Paulo")
def now_br_str():
    return datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S")

# inicializa timestamp
if "last_update" not in st.session_state:
    st.session_state["last_update"] = now_br_str()

# botão + legenda juntos
st.markdown('<div class="update-row">', unsafe_allow_html=True)
clicked = st.button("🔄 Atualizar dados")
st.markdown(
    f'<span class="update-caption">🕒 Última atualização: {st.session_state["last_update"]} (BRT)</span>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

if clicked:
    st.cache_data.clear()
    st.session_state["last_update"] = now_br_str()
    st.rerun()

# ======================
# 1) Conexão Jira
# ======================
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets["EMAIL"]
TOKEN = st.secrets["TOKEN"]
auth = HTTPBasicAuth(EMAIL, TOKEN)

# ======================
# 2) Parâmetros
# ======================
PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {"TDS": "Tech Support", "INT": "Integrations", "TINE": "IT Support NE", "INTEL": "Intelligence"}
SLA_CAMPOS = {"TDS": "customfield_13744", "TINE": "customfield_13744", "INT": "customfield_13686", "INTEL": "customfield_13686"}
CAMPOS_ASSUNTO = {"TDS": "customfield_13712", "INT": "customfield_13643", "TINE": "customfield_13699", "INTEL": "issuetype"}
CAMPO_AREA = "customfield_13719"
CAMPO_N3 = "customfield_13659"
CAMPO_ORIGEM = "customfield_13628"
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

SLA_LIMITE = {"TDS": 40*60*60*1000, "INT": 40*60*60*1000, "TINE": 40*60*60*1000, "INTEL": 80*60*60*1000}
META_SLA = {"TDS": 98.0, "INT": 98.0, "TINE": 98.0, "INTEL": 95.0}

# ======================
# 3) Funções
# ======================
def extrair_sla_millis(sla_field: dict):
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
    if isinstance(x, dict): return x.get(key, fallback)
    return x if x is not None else fallback

def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if "assunto_nome" not in df_proj.columns:
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_proj["assunto_nome"] = df_proj["issuetype"].apply(lambda x: safe_get_value(x, "name"))
        else:
            df_proj["assunto_nome"] = df_proj["assunto"].apply(lambda x: safe_get_value(x, "value"))
    return df_proj

@st.cache_data(show_spinner="🔄 Buscando dados do Jira...")
def buscar_issues() -> pd.DataFrame:
    todos = []
    for projeto in PROJETOS:
        start = 0
        while True:
            jql = f'project="{projeto}" AND created >= "2024-01-01" ORDER BY created ASC'
            params = {"jql": jql, "startAt": start, "maxResults": 100,
                      "fields": f"created,resolutiondate,status,issuetype,{SLA_CAMPOS[projeto]},{CAMPOS_ASSUNTO[projeto]},{CAMPO_AREA},{CAMPO_N3},{CAMPO_ORIGEM}"}
            resp = requests.get(f"{JIRA_URL}/rest/api/3/search", auth=auth, params=params)
            if resp.status_code != 200:
                break
            issues = resp.json().get("issues", [])
            if not issues:
                break
            for it in issues:
                f = it.get("fields", {})
                sla_raw = f.get(SLA_CAMPOS[projeto], {})
                todos.append({
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
                })
            start += 100
    df_all = pd.DataFrame(todos)
    if df_all.empty:
        return df_all
    df_all["created"] = pd.to_datetime(df_all["created"], errors="coerce")
    df_all["resolved"] = pd.to_datetime(df_all["resolutiondate"], errors="coerce")
    df_all["mes_created"] = df_all["created"].dt.to_period("M").dt.to_timestamp()
    df_all["mes_resolved"] = df_all["resolved"].dt.to_period("M").dt.to_timestamp()
    return df_all

def filtros_ano_mes(prefixo: str, serie_dt: pd.Series):
    anos = sorted(pd.Series(serie_dt.dt.year.dropna().unique()).astype(int).tolist())
    meses = sorted(pd.Series(serie_dt.dt.month.dropna().unique()).astype(int).tolist())
    col1, col2 = st.columns(2)
    with col1:
        ano = st.selectbox(f"Ano - {prefixo}", ["Todos"] + [str(a) for a in anos], key=f"ano_{prefixo}")
    with col2:
        mes = st.selectbox(f"Mês - {prefixo}", ["Todos"] + [str(m).zfill(2) for m in meses], key=f"mes_{prefixo}")
    return ano, mes

def aplicar_filtro(df_in: pd.DataFrame, col_dt: str, ano: str, mes: str) -> pd.DataFrame:
    out = df_in.copy()
    if ano != "Todos":
        out = out[out[col_dt].dt.year == int(ano)]
    if mes != "Todos":
        out = out[out[col_dt].dt.month == int(mes)]
    return out

# ======================
# 4) Carregar
# ======================
df = buscar_issues()

# ======================
# 5) UI por projeto
# ======================
tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")
        dfp = df[df["projeto"] == projeto].copy()
        if dfp.empty:
            st.info("Sem dados carregados.")
            continue
        dfp = ensure_assunto_nome(dfp, projeto)

        # --- Criados vs Resolvidos ---
        st.markdown("### 📈 Tickets Criados vs Resolvidos")
        ano_cv, mes_cv = filtros_ano_mes(f"{TITULOS[projeto]}", dfp["mes_created"])
        df_cv = aplicar_filtro(dfp, "mes_created", ano_cv, mes_cv)

        criados = df_cv.groupby("mes_created").size().reset_index(name="Criados")
        resolvidos = df_cv[df_cv["resolved"].notna()].groupby("mes_resolved").size().reset_index(name="Resolvidos")

        criados.rename(columns={"mes_created":"mes_str"}, inplace=True)
        resolvidos.rename(columns={"mes_resolved":"mes_str"}, inplace=True)
        grafico = pd.merge(criados, resolvidos, how="outer", on="mes_str").fillna(0).sort_values("mes_str")
        if not grafico.empty:
            grafico["mes_str"] = grafico["mes_str"].dt.strftime("%b/%Y")
            fig = px.bar(grafico, x="mes_str", y=["Criados","Resolvidos"], barmode="group", text_auto=True, height=440)
            fig.update_traces(textangle=0, textfont_size=14, cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados para os filtros selecionados.")

        # --- SLA ---
        st.markdown("### ⏱️ SLA")
        ano_sla, mes_sla = filtros_ano_mes(f"{TITULOS[projeto]} (SLA)", dfp["mes_resolved"])
        df_sla = aplicar_filtro(dfp[dfp["sla_millis"].notna()], "mes_resolved", ano_sla, mes_sla).copy()

        if not df_sla.empty:
            df_sla["mes_str"] = df_sla["mes_resolved"].dt.strftime("%b/%Y")
            df_sla["dentro_sla"] = df_sla["sla_millis"] <= SLA_LIMITE[projeto]

            agrup = df_sla.groupby("mes_str")["dentro_sla"].value_counts(normalize=True).unstack(fill_value=0) * 100.0
            if True not in agrup.columns:
                agrup[True] = 0.0
            if False not in agrup.columns:
                agrup[False] = 0.0
            agrup = agrup.rename(columns={True:"% Dentro SLA", False:"% Fora SLA"}).reset_index()

            try:
                agrup["mes_data"] = pd.to_datetime(agrup["mes_str"], format="%b/%Y")
            except Exception:
                agrup["mes_data"] = pd.to_datetime(agrup["mes_str"], errors="coerce")
            agrup = agrup.sort_values("mes_data")
            agrup["mes_str"] = agrup["mes_data"].dt.strftime("%b/%Y")

            for col in ["% Dentro SLA", "% Fora SLA"]:
                agrup[col] = pd.to_numeric(agrup[col], errors="coerce").fillna(0.0)

            okr = agrup["% Dentro SLA"].mean() if not agrup.empty else 0.0
            meta = META_SLA.get(projeto, 98.0)
            titulo_sla = f"🎯 OKR: {okr:.1f}% — Meta: {meta:.1f}%"

            if not agrup.empty:
                fig_sla = px.bar(
                    agrup, x="mes_str",
                    y=["% Dentro SLA","% Fora SLA"],
                    barmode="group",
                    title=titulo_sla,
                    color_discrete_map={"% Dentro SLA":"green","% Fora SLA":"red"},
                    height=440
                )
                fig_sla.update_traces(texttemplate="%{y:.1f}%", textposition="outside", textfont_size=14, cliponaxis=False)
                fig_sla.update_yaxes(ticksuffix="%")
                st.plotly_chart(fig_sla, use_container_width=True)
            else:
                st.info("Sem dados de SLA para os filtros selecionados.")
        else:
            st.info("Sem dados de SLA para os filtros selecionados.")

        # --- Assunto Relacionado ---
        st.markdown("### 🧾 Assunto Relacionado")
        df_ass = dfp.copy()
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_ass["assunto_nome"] = df_ass["issuetype"].apply(lambda x: safe_get_value(x, "name"))
        else:
            df_ass["assunto_nome"] = df_ass["assunto"].apply(lambda x: safe_get_value(x, "value"))
        assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
        assunto_count.columns = ["Assunto", "Qtd"]
        st.dataframe(assunto_count, use_container_width=True, hide_index=True)

        # --- Área Solicitante (exceto INTEL) ---
        if projeto != "INTEL":
            st.markdown("### 📦 Área Solicitante")
            df_area = dfp.copy()
            df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
            area_count = df_area["area_nome"].value_counts().reset_index()
            area_count.columns = ["Área", "Qtd"]
            st.dataframe(area_count, use_container_width=True, hide_index=True)

        # --- Encaminhamentos (TDS e INT) ---
        if projeto in ("TDS", "INT"):
            st.markdown("### 🔄 Encaminhamentos")
            df_enc = dfp.copy()
            col1, col2 = st.columns(2)
            with col1:
                count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
                st.metric("Encaminhados Produto", count_prod)
            with col2:
                df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
                st.metric("Encaminhados N3", (df_enc["n3_valor"] == "Sim").sum())

        # --- Onboarding (somente INT) ---
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

                df_onb = dfp.copy()

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

                if df_erros.empty:
                    st.info("Sem erros de Onboarding no período.")
                else:
                    cont_erros = (
                        df_erros["assunto_nome"]
                        .value_counts()
                        .reindex(ASSUNTOS_ERROS, fill_value=0)
                        .reset_index()
                    )
                    cont_erros.columns = ["Categoria", "Qtd"]
                    fig_onb = px.bar(
                        cont_erros, x="Qtd", y="Categoria", orientation="h",
                        text="Qtd", title="Erros Onboarding", height=420,
                    )
                    fig_onb.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
                    max_q = int(cont_erros["Qtd"].max()) if not cont_erros.empty else 0
                    if max_q > 0:
                        fig_onb.update_xaxes(range=[0, max_q * 1.25])
                    fig_onb.update_layout(margin=dict(t=50, r=20, b=30, l=10), bargap=0.25)
                    st.plotly_chart(fig_onb, use_container_width=True)

                st.markdown("---")

                df_cli = df_onb[df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO].copy()
                if not df_cli.empty:
                    serie_cli = (
                        df_cli.groupby(df_cli["mes_created"].dt.to_period("M")).size().reset_index(name="ClientesNovos")
                    )
                    serie_cli["mes_dt"] = serie_cli["mes_created"].dt.to_timestamp()
                    serie_cli = serie_cli.sort_values("mes_dt")
                    serie_cli["mes_str"] = serie_cli["mes_dt"].dt.strftime("%Y %b")
                    serie_cli["MoM"] = serie_cli["ClientesNovos"].pct_change() * 100

                    fig_cli = px.bar(
                        serie_cli, x="mes_str", y="ClientesNovos",
                        title="Tickets - Cliente novo", text="ClientesNovos", height=380,
                    )
                    fig_cli.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=14, cliponaxis=False)

                    y_top = (serie_cli["ClientesNovos"].max() * 2.2) if len(serie_cli) else 10
                    fig_cli.update_yaxes(range=[0, y_top])

                    for _, r in serie_cli.iterrows():
                        x = r["mes_str"]; yb = float(r["ClientesNovos"])
                        if pd.notna(r["MoM"]):
                            mom_abs = abs(r["MoM"])
                            if mom_abs >= 1:
                                up = r["MoM"] >= 0
                                arrow = "▲" if up else "▼"
                                color = "#2563eb" if up else "#dc2626"
                                fig_cli.add_annotation(
                                    x=x, y=yb + (y_top * 0.20),
                                    text=f"{arrow} {mom_abs:.0f}%",
                                    showarrow=False, font=dict(size=12, color=color), align="center",
                                )
                    fig_cli.update_layout(margin=dict(t=50, r=20, b=35, l=40), bargap=0.18, xaxis_title=None, yaxis_title="ClientesNovos")
                    st.plotly_chart(fig_cli, use_container_width=True)

                st.markdown("---")

                # Tabela Receita (se existir)
                col_receita = None
                for c in df_onb.columns:
                    if "receita" in str(c).lower():
                        col_receita = c; break
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
                    clientes_sim = int(possiveis_clientes)
                    st.number_input("Cliente novos (simulação)", value=clientes_sim, disabled=True, key=f"sim_clientes_{projeto}")
                with c_right:
                    receita_cliente = st.slider("Cenário Receita por Cliente (R$)", min_value=0, max_value=100000, step=500, value=20000, key=f"sim_receita_{projeto}")
                dinheiro_perdido = float(clientes_sim) * float(receita_cliente)
                st.markdown(f"### **R$ {dinheiro_perdido:,.2f}**", help="Cálculo: Cliente novos (simulação) × Cenário Receita por Cliente")

        # --- APP NE (somente TDS) ---
        if projeto == "TDS":
            with st.expander("📱 APP NE", expanded=False):
                s_ass = dfp["assunto_nome"].astype(str).str.strip()
                alvo = ASSUNTO_ALVO_APPNE.strip().casefold()
                mask_assunto = s_ass.str.casefold().eq(alvo)
                if not mask_assunto.any():
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
                        total_app = len(df_app_f)
                        contagem = df_app_f["origem_nome"].value_counts(dropna=False).to_dict()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total (APP NE/EN)", total_app)
                        m2.metric("APP NE", contagem.get("APP NE", 0))
                        m3.metric("APP EN", contagem.get("APP EN", 0))

                        serie = (
                            df_app_f.groupby(["mes_dt", "origem_nome"]).size().reset_index(name="Qtd").sort_values("mes_dt")
                        )
                        serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
                        cats = serie["mes_str"].dropna().unique().tolist()
                        serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

                        fig_app = px.bar(
                            serie, x="mes_str", y="Qtd", color="origem_nome",
                            barmode="group", title="APP NE — Volumes por mês e Origem do problema",
                            color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4"},
                            text="Qtd", height=460,
                        )
                        fig_app.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
                        max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
                        if max_qtd > 0:
                            fig_app.update_yaxes(range=[0, max_qtd * 1.25])
                        fig_app.update_layout(yaxis_title="Qtd", xaxis_title="Mês", uniformtext_minsize=14, uniformtext_mode="show",
                                              bargap=0.15, margin=dict(t=70, r=20, b=60, l=50))
                        st.plotly_chart(fig_app, use_container_width=True)

                        df_app_f["mes_str"] = df_app_f["mes_dt"].dt.strftime("%b/%Y")
                        cols_show = ["key", "created", "mes_str", "assunto_nome", "origem_nome", "status"]
                        cols_show = [c for c in cols_show if c in df_app_f.columns]
                        st.dataframe(df_app_f[cols_show], use_container_width=True, hide_index=True)
