import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from zoneinfo import ZoneInfo
import base64

# ======================
# 0) Configurações e estilo
# ======================
st.set_page_config(layout="wide")

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
.section-spacer { height: 10px; }
</style>
""", unsafe_allow_html=True)

def _render_logo_and_title():
    logo_bytes = None
    b64 = st.secrets.get("LOGO_B64")
    if b64:
        try:
            logo_bytes = base64.b64decode(b64)
        except Exception:
            logo_bytes = None

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:8px 0 20px 0;">',
        unsafe_allow_html=True,
    )
    if logo_bytes:
        st.image(logo_bytes, width=300)
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

st.title("📊 Painel de Indicadores — Jira")

# ======================
# 1) Conexão / horário / atualizar
# ======================
TZ_BR = ZoneInfo("America/Sao_Paulo")
def now_br_str():
    return datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S")

if "last_update" not in st.session_state:
    st.session_state["last_update"] = now_br_str()

st.markdown('<div class="update-row">', unsafe_allow_html=True)
if st.button("🔄 Atualizar dados"):
    st.cache_data.clear()
    st.session_state["last_update"] = now_br_str()
    st.rerun()
st.markdown(
    f'<span class="update-caption">🕒 Última atualização: {st.session_state["last_update"]} (BRT)</span>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

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
# 3) Helpers
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
            params = {
                "jql": jql, "startAt": start, "maxResults": 100,
                "fields": f"created,resolutiondate,status,issuetype,{SLA_CAMPOS[projeto]},{CAMPOS_ASSUNTO[projeto]},{CAMPO_AREA},{CAMPO_N3},{CAMPO_ORIGEM}"
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

def aplicar_filtro_global(df_in: pd.DataFrame, col_dt: str, ano: str, mes: str) -> pd.DataFrame:
    out = df_in.copy()
    if ano != "Todos":
        out = out[out[col_dt].dt.year == int(ano)]
    if mes != "Todos":
        out = out[out[col_dt].dt.month == int(mes)]
    return out

def normaliza_origem(s: str) -> str:
    if s is None or str(s).strip() == "" or str(s).lower() in ("nan", "none"):
        return "Outros/Não informado"
    t = str(s).strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())
    if "app" in t and "ne" in t:
        return "APP NE"
    if "app" in t and ("en" in t or "eng" in t):
        return "APP EN"
    return "Outros/Não informado"

# ======================
# 4) Carregar e filtros globais
# ======================
df = buscar_issues()
st.markdown("### 🔍 Filtros Globais")
if df.empty:
    st.warning("Sem dados retornados do Jira.")
    st.stop()

anos_glob = sorted(pd.Series(df["mes_created"].dt.year.dropna().unique()).astype(int).tolist())
meses_glob = sorted(pd.Series(df["mes_created"].dt.month.dropna().unique()).astype(int).tolist())
c1, c2 = st.columns(2)
with c1:
    ano_global = st.selectbox("Ano (global)", ["Todos"] + [str(a) for a in anos_glob], key="ano_global")
with c2:
    mes_global = st.selectbox("Mês (global)", ["Todos"] + [str(m).zfill(2) for m in meses_glob], key="mes_global")
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

# ======================
# 5) Renderizadores de seções (blocos)
# ======================
def render_criados_resolvidos(dfp, ano_global, mes_global):
    st.markdown("### 📈 Tickets Criados vs Resolvidos")
    df_cv = aplicar_filtro_global(dfp, "mes_created", ano_global, mes_global)
    criados = df_cv.groupby("mes_created").size().reset_index(name="Criados")
    resolvidos = df_cv[df_cv["resolved"].notna()].groupby("mes_resolved").size().reset_index(name="Resolvidos")
    criados.rename(columns={"mes_created": "mes_ts"}, inplace=True)
    resolvidos.rename(columns={"mes_resolved": "mes_ts"}, inplace=True)
    grafico = pd.merge(criados, resolvidos, how="outer", on="mes_ts").fillna(0).sort_values("mes_ts")
    if grafico.empty:
        st.info("Sem dados para os filtros selecionados.")
        return
    grafico["mes_str"] = grafico["mes_ts"].dt.strftime("%b/%Y")
    fig = px.bar(grafico, x="mes_str", y=["Criados", "Resolvidos"], barmode="group", text_auto=True, height=440)
    fig.update_traces(textangle=0, textfont_size=14, cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

def render_sla(dfp, projeto, ano_global, mes_global):
    st.markdown("### ⏱️ SLA")
    df_sla = dfp[dfp["sla_millis"].notna()].copy()
    df_sla = aplicar_filtro_global(df_sla, "mes_resolved", ano_global, mes_global)
    if df_sla.empty:
        st.info("Sem dados de SLA para os filtros selecionados.")
        return
    df_sla["dentro_sla"] = df_sla["sla_millis"] <= SLA_LIMITE[projeto]
    df_sla["mes_str"] = df_sla["mes_resolved"].dt.strftime("%b/%Y")

    agr = df_sla.groupby("mes_str").agg(
        total=("dentro_sla", "count"),
        dentro=("dentro_sla", "sum")
    ).reset_index()
    agr["fora"] = agr["total"] - agr["dentro"]
    agr["% Dentro SLA"] = (agr["dentro"] / agr["total"]) * 100.0
    agr["% Fora SLA"] = (agr["fora"] / agr["total"]) * 100.0

    try:
        agr["mes_data"] = pd.to_datetime(agr["mes_str"], format="%b/%Y")
    except Exception:
        agr["mes_data"] = pd.to_datetime(agr["mes_str"], errors="coerce")
    agr = agr.sort_values("mes_data")
    agr["mes_str"] = agr["mes_data"].dt.strftime("%b/%Y")

    okr = agr["% Dentro SLA"].mean() if not agr.empty else 0.0
    meta = META_SLA.get(projeto, 98.0)
    titulo_sla = f"Meta: {meta:.2f}%".replace(".", ",")

    fig_sla = px.bar(
        agr, x="mes_str", y=["% Dentro SLA", "% Fora SLA"],
        barmode="group", title=titulo_sla,
        color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
        height=440
    )
    fig_sla.update_traces(texttemplate="%{y:.2f}%", textposition="outside", textfont_size=14, cliponaxis=False)
    fig_sla.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_sla, use_container_width=True)
    st.caption(f"**Média (OKR)** no período filtrado: {okr:.2f}%".replace(".", ","))

def render_assunto(dfp, projeto, ano_global, mes_global):
    st.markdown("### 🧾 Assunto Relacionado")
    df_ass = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if CAMPOS_ASSUNTO[projeto] == "issuetype":
        df_ass["assunto_nome"] = df_ass["issuetype"].apply(lambda x: safe_get_value(x, "name"))
    else:
        df_ass["assunto_nome"] = df_ass["assunto"].apply(lambda x: safe_get_value(x, "value"))
    assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
    assunto_count.columns = ["Assunto", "Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)

def render_area(dfp, ano_global, mes_global):
    st.markdown("### 📦 Área Solicitante")
    df_area = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
    area_count = df_area["area_nome"].value_counts().reset_index()
    area_count.columns = ["Área", "Qtd"]
    st.dataframe(area_count, use_container_width=True, hide_index=True)

def render_encaminhamentos(dfp, ano_global, mes_global):
    st.markdown("### 🔄 Encaminhamentos")
    df_enc = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    col1, col2 = st.columns(2)
    with col1:
        count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
        st.metric("Encaminhados Produto", int(count_prod))
    with col2:
        df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
        st.metric("Encaminhados N3", int((df_enc["n3_valor"] == "Sim").sum()))

def render_onboarding(dfp, ano_global, mes_global):
    st.markdown("### 🧭 Onboarding")
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

    df_onb = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)

    total_clientes_novos = (df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO).sum()
    df_erros = df_onb[df_onb["assunto_nome"].isin(ASSUNTOS_ERROS)].copy()
    pend_mask = df_onb["status"].isin(STATUS_PENDENCIAS)
    tickets_pendencias = int(pend_mask.sum())
    possiveis_clientes = int(pend_mask.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets clientes novos", int(total_clientes_novos))
    c2.metric("Erros onboarding", int(len(df_erros)))
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

def render_app_ne(dfp, ano_global, mes_global):
    st.markdown("### 📱 APP NE — Origem do problema")
    s_ass = dfp["assunto_nome"].astype(str).str.strip()
    alvo = ASSUNTO_ALVO_APPNE.strip().casefold()
    mask_assunto = s_ass.str.casefold().eq(alvo)
    if not mask_assunto.any():
        mask_assunto = s_ass.str.contains(r"app\s*ne", case=False, regex=True)

    df_app = dfp[mask_assunto].copy()
    if df_app.empty:
        st.info(f"Não há chamados para '{ASSUNTO_ALVO_APPNE}'.")
        return

    df_app["origem_nome"] = df_app["origem"].apply(lambda x: safe_get_value(x, "value"))
    df_app["origem_cat"] = df_app["origem_nome"].apply(normaliza_origem)
    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()

    df_app = aplicar_filtro_global(df_app, "mes_dt", ano_global, mes_global)
    if df_app.empty:
        st.info("Sem dados para exibir com os filtros selecionados.")
        return

    total_app = int(len(df_app))
    contagem = df_app["origem_cat"].value_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total (APP NE/EN)", total_app)
    m2.metric("APP NE", int(contagem.get("APP NE", 0)))
    m3.metric("APP EN", int(contagem.get("APP EN", 0)))

    serie = (
        df_app.groupby(["mes_dt", "origem_cat"])
        .size().reset_index(name="Qtd").sort_values("mes_dt")
    )
    serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
    cats = serie["mes_str"].dropna().unique().tolist()
    serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

    fig_app = px.bar(
        serie, x="mes_str", y="Qtd", color="origem_cat",
        barmode="group", title="APP NE — Volumes por mês e Origem do problema",
        color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4", "Outros/Não informado": "#9ca3af"},
        text="Qtd", height=460,
    )
    fig_app.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
    max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
    if max_qtd > 0:
        fig_app.update_yaxes(range=[0, max_qtd * 1.25])
    fig_app.update_layout(yaxis_title="Qtd", xaxis_title="Mês",
                          uniformtext_minsize=14, uniformtext_mode="show",
                          bargap=0.15, margin=dict(t=70, r=20, b=60, l=50))
    st.plotly_chart(fig_app, use_container_width=True)

    df_app["mes_str"] = df_app["mes_dt"].dt.strftime("%b/%Y")
    cols_show = ["key", "created", "mes_str", "assunto_nome", "origem_cat", "status"]
    st.dataframe(df_app[cols_show], use_container_width=True, hide_index=True)

# ======================
# 6) Abas por projeto + seletor de “Visão”
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

        # Opções por projeto
        if projeto == "TDS":
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante", "APP NE"]
        elif projeto == "INT":
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante", "Onboarding"]
        elif projeto == "TINE":
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante"]
        else:  # INTEL
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado"]

        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")

        # Navegação por visão: renderiza somente o bloco escolhido
        if visao == "Criados vs Resolvidos":
            render_criados_resolvidos(dfp, ano_global, mes_global)

        elif visao == "SLA":
            render_sla(dfp, projeto, ano_global, mes_global)

        elif visao == "Assunto Relacionado":
            render_assunto(dfp, projeto, ano_global, mes_global)

        elif visao == "Área Solicitante":
            if projeto == "INTEL":
                st.info("Este projeto não possui Área Solicitante.")
            else:
                render_area(dfp, ano_global, mes_global)

        elif visao == "Onboarding":
            if projeto == "INT":
                render_onboarding(dfp, ano_global, mes_global)
            else:
                st.info("Onboarding disponível somente para Integrations.")

        elif visao == "APP NE":
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
            else:
                st.info("APP NE disponível somente para Tech Support.")

        else:
            # ---------- Visão GERAL ----------
            render_criados_resolvidos(dfp, ano_global, mes_global)
            render_sla(dfp, projeto, ano_global, mes_global)
            render_assunto(dfp, projeto, ano_global, mes_global)
            if projeto != "INTEL":
                render_area(dfp, ano_global, mes_global)
            if projeto in ("TDS", "INT"):
                render_encaminhamentos(dfp, ano_global, mes_global)
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
            if projeto == "INT":
                render_onboarding(dfp, ano_global, mes_global)
