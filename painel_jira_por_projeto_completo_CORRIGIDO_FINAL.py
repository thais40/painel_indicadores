# -*- coding: utf-8 -*-
import base64
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth


# ============================
# CONFIGURAÇÃO / ESTILO
# ============================
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
:root { --ns-muted:#6B7280; }
h1 { letter-spacing:.2px; }
.stButton > button{ border-radius:10px; border:1px solid #e6e8ee; box-shadow:0 1px 2px rgba(16,24,40,.04); }
.update-row{ display:inline-flex; align-items:center; gap:12px; }
.update-caption{ color:var(--ns-muted); font-size:.85rem; }
.section-spacer{ height:10px; }
</style>
""",
    unsafe_allow_html=True,
)


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
    st.markdown("</div>", unsafe_allow_html=True)


_render_logo_and_title()
st.title("📊 Painel de Indicadores — Jira")


# ============================
# ATUALIZAÇÃO / HORÁRIO (BRT)
# ============================
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
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================
# PARÂMETROS / CAMPOS
# ============================
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets["EMAIL"]
TOKEN = st.secrets["TOKEN"]
auth = HTTPBasicAuth(EMAIL, TOKEN)

PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {
    "TDS": "Tech Support",
    "INT": "Integrations",
    "TINE": "IT Support NE",
    "INTEL": "Intelligence",
}

# Campo SLA por projeto (JSM)
SLA_CAMPOS = {
    "TDS": "customfield_13744",  # SLA resolução (SUP)
    "TINE": "customfield_13744",
    "INT": "customfield_13686",
    "INTEL": "customfield_13686",
}

# Assunto por projeto
CAMPOS_ASSUNTO = {
    "TDS": "customfield_13712",
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype",  # usa issuetype.name
}

CAMPO_AREA = "customfield_13719"
CAMPO_N3 = "customfield_13659"
CAMPO_ORIGEM = "customfield_13628"  # Origem do problema (TDS/TINE)

# App NE alvo
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

# Metas por projeto (exibição)
META_SLA = {"TDS": 98.0, "INT": 96.0, "TINE": 96.0, "INTEL": 95.0}


# ============================
# HELPERS
# ============================
def safe_get_value(x, key="value", fallback="—"):
    if isinstance(x, dict):
        return x.get(key, fallback)
    return x if x is not None else fallback


def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if "assunto_nome" not in df_proj.columns:
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_proj["assunto_nome"] = df_proj["issuetype"].apply(lambda x: safe_get_value(x, "name"))
        else:
            df_proj["assunto_nome"] = df_proj["assunto"].apply(lambda x: safe_get_value(x, "value"))
    return df_proj


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


def dentro_sla_from_raw(sla_raw: dict) -> bool | None:
    try:
        if not sla_raw or not isinstance(sla_raw, dict):
            return None
        cycles = sla_raw.get("completedCycles") or []
        if cycles:
            last = cycles[-1]
            if "breached" in last:
                return not bool(last["breached"])
            elapsed = (last.get("elapsedTime") or {}).get("millis")
            goal = (last.get("goalDuration") or {}).get("millis")
            if elapsed is not None and goal is not None:
                return elapsed <= goal
        return None
    except Exception:
        return None


# ============================
# BUSCA DE DADOS
# ============================
@st.cache_data(show_spinner="🔄 Buscando dados do Jira...")
def buscar_issues() -> pd.DataFrame:
    todos = []
    for projeto in PROJETOS:
        start = 0
        while True:
            jql = f'project="{projeto}" AND created >= "2024-01-01" ORDER BY created ASC'
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
                sla_field = f.get(SLA_CAMPOS[projeto], {})
                todos.append(
                    {
                        "projeto": projeto,
                        "key": it.get("key"),
                        "created": f.get("created"),
                        "resolutiondate": f.get("resolutiondate"),
                        "status": safe_get_value(f.get("status"), "name"),
                        "sla_raw": sla_field,
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


# ============================
# DADOS + FILTRO GLOBAL
# ============================
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


# ============================
# RENDERIZADORES
# ============================
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
    df_sla = dfp[dfp["resolved"].notna()].copy()
    df_sla["mes_resolved"] = df_sla["resolved"].dt.to_period("M").dt.to_timestamp()
    df_sla = aplicar_filtro_global(df_sla, "mes_resolved", ano_global, mes_global)
    if df_sla.empty:
        st.info("Sem dados de SLA para os filtros selecionados.")
        return
    df_sla["dentro_sla"] = df_sla["sla_raw"].apply(dentro_sla_from_raw)
    df_sla["dentro_sla"] = df_sla["dentro_sla"].fillna(False)
    agr = (
        df_sla.groupby(df_sla["mes_resolved"].dt.strftime("%b/%Y"))
        .agg(total=("dentro_sla", "size"), dentro=("dentro_sla", "sum"))
        .reset_index()
        .rename(columns={"mes_resolved": "mes_str"})
    )
    agr["fora"] = (agr["total"] - agr["dentro"]).astype(int)
    agr["% Dentro SLA"] = (agr["dentro"].astype(float) / agr["total"].astype(float)) * 100.0
    agr["% Fora SLA"] = (agr["fora"].astype(float) / agr["total"].astype(float)) * 100.0
    agr["mes_data"] = pd.to_datetime(agr["mes_str"], format="%b/%Y")
    agr = agr.sort_values("mes_data")
    agr["mes_str"] = agr["mes_data"].dt.strftime("%b/%Y")
    okr = agr["% Dentro SLA"].mean() if not agr.empty else 0.0
    meta = META_SLA.get(projeto, 98.0)
    titulo = f"OKR: {okr:.2f}% — Meta: {meta:.2f}%".replace(".", ",")
    fig_sla = px.bar(
        agr, x="mes_str", y=["% Dentro SLA", "% Fora SLA"], barmode="group", title=titulo,
        color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"}, height=440,
    )
    fig_sla.update_traces(texttemplate="%{y:.2f}%", textposition="outside", textfont_size=14, cliponaxis=False)
    fig_sla.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_sla, use_container_width=True)


# ... (mantém render_assunto, render_area, render_encaminhamentos, render_onboarding, render_app_ne iguais ao último arquivo) ...


# ============================
# ABAS POR PROJETO
# ============================
tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")
        dfp = df[df["projeto"] == projeto].copy()
        if dfp.empty:
            st.info("Sem dados carregados.")
            continue
        dfp = ensure_assunto_nome(dfp, projeto)
        if projeto == "TDS":
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante", "APP NE"]
        elif projeto == "INT":
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante", "Onboarding"]
        elif projeto == "TINE":
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante"]
        else:
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado"]
        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")
        if visao == "Onboarding" and projeto == "INT":
            render_onboarding(dfp, ano_global, mes_global)
        elif visao == "APP NE" and projeto == "TDS":
            render_app_ne(dfp, ano_global, mes_global)
        elif visao == "Geral":
            # ... renderizações gerais ...
            if projeto == "INT":
                # agora o Onboarding em sub-menu
                with st.expander("🧭 Onboarding"):
                    render_onboarding(dfp, ano_global, mes_global)
