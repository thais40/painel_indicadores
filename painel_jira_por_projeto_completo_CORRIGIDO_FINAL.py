# -*- coding: utf-8 -*-
"""
Painel de Indicadores — Jira (Nuvemshop)
Arquivo COMPLETO sem dependência de planilhas do BI.
Classificação de Rotinas Manuais (TDS) 100% por campos do Jira:
  - Área Solicitante (customfield_13719) ∈ {Ops - Conferência, Ops - Cubagem, Ops - Logística, Ops - Coletas, Ops - Expedição, Ops - Divergências}
  - Quantidade de encomendas (customfield_13666) > 0
  - Data base: mês de resolved
  - Manual x Encomendas TDS por "Assunto Relacionado" (customfield_13747; fallback 13712).

Mantém cache de dados (st.cache_data) e botão de Atualizar — não refaz fetch a cada filtro.
Necessário EMAIL e TOKEN em st.secrets.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, date
from typing import Dict, Any, Optional, List
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

# ================= Config da página =======================
st.set_page_config(page_title="Painel de Indicadores", page_icon="📊", layout="wide")

# ================= Credenciais Jira ========================
JIRA_URL = "https://tiendanube.atlassian.net"
EMAIL = st.secrets.get("EMAIL", "")
TOKEN = st.secrets.get("TOKEN", "")
if not EMAIL or not TOKEN:
    st.error("⚠️ Configure EMAIL e TOKEN em st.secrets para acessar o Jira.")
    st.stop()

auth = HTTPBasicAuth(EMAIL, TOKEN)
TZ_BR = ZoneInfo("America/Sao_Paulo")
DATA_INICIO = "2024-06-01"

# ================= Campos / Constantes =====================
SLA_CAMPOS = {
    "TDS": "customfield_13744",
    "TINE": "customfield_13744",
    "INT": "customfield_13686",
    "INTEL": "customfield_13686",
}

ASSUNTO_TDS_PRIMARY = "customfield_13747"   # Assunto Relacionado
ASSUNTO_TDS_FALLBACK = "customfield_13712"  # fallback

CAMPOS_ASSUNTO = {
    "TDS": ASSUNTO_TDS_PRIMARY,
    "INT": "customfield_13643",
    "TINE": "customfield_13699",
    "INTEL": "issuetype",
}

CAMPO_AREA = "customfield_13719"
CAMPO_N3 = "customfield_13659"
CAMPO_ORIGEM = "customfield_13628"
CAMPO_QTD_ENCOMENDAS = "customfield_13666"

PROJETOS = ["TDS", "INT", "TINE", "INTEL"]
TITULOS = {"TDS": "Tech Support", "INT": "Integrations", "TINE": "IT Support NE", "INTEL": "Intelligence"}
META_SLA = {"TDS": 98.00, "INT": 96.00, "TINE": 96.00, "INTEL": 96.00}
ASSUNTO_ALVO_APPNE = "Problemas no App NE - App EN"

JIRA_FIELDS_BASE = [
    "key", "summary", "created", "updated", "resolutiondate", "resolved", "status", "issuetype",
    CAMPO_AREA, CAMPO_N3, CAMPO_ORIGEM, CAMPO_QTD_ENCOMENDAS,
]
FIELDS_SLA_ALL = list(set(SLA_CAMPOS.values()))
FIELDS_ASSUNTO_ALL = list(set([v for v in CAMPOS_ASSUNTO.values() if v != "issuetype"]))
FIELDS_ALL: List[str] = list(
    dict.fromkeys(JIRA_FIELDS_BASE + FIELDS_SLA_ALL + FIELDS_ASSUNTO_ALL + [ASSUNTO_TDS_FALLBACK])
)

# ================= UI: Cabeçalho ===========================

def _render_head():
    st.markdown(
        """
        <style>
        html, body, [class*=\"css\"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
        .update-row{ display:flex; align-items:center; gap:12px; margin:8px 0 18px 0; }
        .update-caption{ color:#6B7280; font-size:.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("📊 Painel de Indicadores")


def now_br_str() -> str:
    return datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S")


_render_head()
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

# ================= Helpers ================================

def show_plot(fig, nome_bloco: str, projeto: str, ano: str, mes: str):
    st.plotly_chart(fig, use_container_width=True, key=f"plt-{nome_bloco}-{projeto}-{ano}-{mes}-{uuid4()}")


def safe_get_value(x, key: str = "value", fallback: str = "—"):
    if isinstance(x, dict):
        return x.get(key, fallback)
    return x if x is not None else fallback


def dentro_sla_from_raw(sla_raw: dict) -> Optional[bool]:
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


def parse_qtd_encomendas(v) -> int:
    if isinstance(v, list):
        v = next((x for x in reversed(v) if x not in (None, "")), None)
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        try:
            return int(round(float(v)))
        except Exception:
            return 0
    s = str(v).strip().replace(".", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except Exception:
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else 0


def _canonical(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def aplicar_filtro_global(df_in: pd.DataFrame, col_dt: str, ano: str, mes: str) -> pd.DataFrame:
    out = df_in.copy()
    if ano != "Todos":
        out = out[out[col_dt].dt.year == int(ano)]
    if mes != "Todos":
        out = out[out[col_dt].dt.month == int(mes)]
    return out


# 👉 Garante 'assunto_nome' antes do uso

def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if df_proj is None or df_proj.empty:
        return df_proj

    def _from_field(v):
        if isinstance(v, dict):
            return v.get("value") or v.get("name") or str(v)
        return v

    col = "assunto_nome"
    if col not in df_proj.columns:
        if CAMPOS_ASSUNTO.get(projeto) == "issuetype":
            df_proj[col] = df_proj["issuetype"].apply(_from_field)
        else:
            df_proj[col] = df_proj["assunto"].apply(_from_field)

    if df_proj[col].isna().all() or (df_proj[col].astype(str).str.strip() == "").all():
        if CAMPOS_ASSUNTO.get(projeto) == "issuetype":
            df_proj[col] = df_proj["issuetype"].apply(_from_field)
        else:
            df_proj[col] = df_proj["assunto"].apply(_from_field)

    return df_proj


# ================= Jira fetch =============================

def _jira_search_jql(jql: str, next_page_token: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    params = {"jql": jql, "fields": ",".join(FIELDS_ALL), "maxResults": max_results}
    if next_page_token:
        params["nextPageToken"] = next_page_token
    try:
        resp = requests.get(url, params=params, auth=auth, timeout=60)
    except Exception as e:
        return {"error": str(e), "issues": [], "isLast": True}
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:300]}", "issues": [], "isLast": True}
    return resp.json()


@st.cache_data(show_spinner="🔄 Buscando dados do Jira...", ttl=60 * 30)
def buscar_issues(projeto: str, jql: str, max_pages: int = 500) -> pd.DataFrame:
    todos, last_error = [], None
    next_token, page = None, 0
    while True:
        page += 1
        data = _jira_search_jql(jql, next_page_token=next_token, max_results=100)
        if "error" in data and data["error"]:
            last_error = data["error"]
            break
        issues = data.get("issues", [])
        if not issues:
            break
        for it in issues:
            f = it.get("fields", {}) or {}
            if projeto == "TDS":
                assunto_val = f.get(ASSUNTO_TDS_PRIMARY) or f.get(ASSUNTO_TDS_FALLBACK) or f.get("issuetype")
            else:
                assunto_val = (
                    f.get(CAMPOS_ASSUNTO[projeto]) if CAMPOS_ASSUNTO[projeto] != "issuetype" else f.get("issuetype")
                )

            row = {
                "projeto": projeto,
                "key": it.get("key"),
                "summary": f.get("summary"),
                "created": f.get("created"),
                "updated": f.get("updated"),
                "resolutiondate": f.get("resolutiondate"),
                "resolved": f.get("resolved") or f.get("resolutiondate"),
                "status": safe_get_value(f.get("status"), "name"),
                "issuetype": f.get("issuetype"),
                "assunto": assunto_val,
                "area": f.get(CAMPO_AREA),
                "n3": f.get(CAMPO_N3),
                "origem": f.get(CAMPO_ORIGEM),
                CAMPO_QTD_ENCOMENDAS: f.get(CAMPO_QTD_ENCOMENDAS),
                "sla_raw": f.get(SLA_CAMPOS[projeto], {}),
            }
            todos.append(row)
        next_token = data.get("nextPageToken")
        is_last = bool(data.get("isLast", not bool(next_token)))
        if is_last or page >= max_pages:
            break

    dfp = pd.DataFrame(todos)
    if last_error and dfp.empty:
        st.warning(f"⚠️ Erro ao buscar Jira ({projeto}): {last_error}")
        return dfp

    if not dfp.empty:
        for c in ("created", "resolved", "resolutiondate", "updated"):
            dfp[c] = (
                pd.to_datetime(dfp[c], errors="coerce", utc=True).dt.tz_convert(TZ_BR).dt.tz_localize(None)
            )
        dfp["mes_created"] = dfp["created"].dt.to_period("M").dt.to_timestamp()
        dfp["mes_resolved"] = dfp["resolved"].dt.to_period("M").dt.to_timestamp()
    return dfp


# ================= Builders / SLA =========================

def build_monthly_tables(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    base = df_all.copy()
    base["per_created"] = base["created"].dt.to_period("M")
    base["per_resolved"] = base["resolved"].dt.to_period("M")

    created = (
        base.groupby(["projeto", "per_created"]).size().reset_index(name="Criados").rename(columns={"per_created": "period"})
    )
    res = base[base["resolved"].notna()].copy()
    res["dentro_sla"] = res["sla_raw"].apply(dentro_sla_from_raw).fillna(False)
    resolved = (
        res.groupby(["projeto", "per_resolved"]).agg(Resolvidos=("key", "count"), Dentro=("dentro_sla", "sum")).reset_index().rename(columns={"per_resolved": "period"})
    )

    monthly = pd.merge(created, resolved, how="outer", on=["projeto", "period"]).fillna(0)
    monthly["period"] = monthly["period"].astype("period[M]")
    monthly["period_ts"] = monthly["period"].dt.to_timestamp()
    monthly["ano"] = monthly["period"].dt.year.astype(int)
    monthly["mes"] = monthly["period"].dt.month.astype(int)
    monthly["mes_str"] = monthly["period_ts"].dt.strftime("%b/%Y")
    monthly["Dentro"] = monthly["Dentro"].astype(int)
    monthly["Resolvidos"] = monthly["Resolvidos"].astype(int)
    monthly["Fora"] = (monthly["Resolvidos"] - monthly["Dentro"]).clip(lower=0)
    monthly["pct_dentro"] = monthly.apply(lambda r: (r["Dentro"] / r["Resolvidos"] * 100) if r["Resolvidos"] > 0 else 0.0, axis=1).round(2)
    monthly["pct_fora"] = monthly.apply(lambda r: (r["Fora"] / r["Resolvidos"] * 100) if r["Resolvidos"] > 0 else 0.0, axis=1).round(2)
    return monthly.sort_values(["projeto", "period"])


# ================= Visuais Genéricos ======================

def render_criados_resolvidos(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    """
    Tickets Criados vs Resolvidos — TODAS as áreas (para o projeto atual).
    - Criados: primeiro 'created' por key (mês de criação)
    - Resolvidos: último 'resolved' por key (mês de resolução final)
    - Eixo mensal contínuo (não pula meses).
    """
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    df = dfp.copy()

    # datas robustas
    created  = pd.to_datetime(df.get("created"),  errors="coerce")
    resolved = pd.to_datetime(df.get("resolved"), errors="coerce")

    # --- criados por mês (primeira criação por key) ---
    cdf = (
        df[created.notna()]
        .assign(created=created[created.notna()])
        .sort_values(["key", "created"])
        .drop_duplicates("key", keep="first")
        .copy()
    )
    if not cdf.empty:
        cdf["mes_dt"] = cdf["created"].dt.to_period("M").dt.to_timestamp()

    # --- resolvidos por mês (última resolução por key) ---
    rdf = (
        df[resolved.notna()]
        .assign(resolved=resolved[resolved.notna()])
        .sort_values(["key", "resolved"])
        .drop_duplicates("key", keep="last")
        .copy()
    )
    if not rdf.empty:
        rdf["mes_dt"] = rdf["resolved"].dt.to_period("M").dt.to_timestamp()

    if cdf.empty and rdf.empty:
        st.info("Sem dados de criação/resolução para montar a série.")
        return

    # eixo mensal contínuo
    mins = [x["mes_dt"].min() for x in (cdf, rdf) if not x.empty]
    maxs = [x["mes_dt"].max() for x in (cdf, rdf) if not x.empty]
    idx = pd.date_range(min(mins), max(maxs), freq="MS")

    s_criados    = (cdf.groupby("mes_dt")["key"].nunique() if not cdf.empty else pd.Series(dtype=int))
    s_resolvidos = (rdf.groupby("mes_dt")["key"].nunique() if not rdf.empty else pd.Series(dtype=int))

    s_criados    = s_criados.reindex(idx, fill_value=0).rename("Criados")
    s_resolvidos = s_resolvidos.reindex(idx, fill_value=0).rename("Resolvidos")

    monthly = pd.concat([s_criados, s_resolvidos], axis=1).reset_index().rename(columns={"index": "mes_dt"})

    # aplica seus filtros globais (ano/mês)
    monthly = aplicar_filtro_global(monthly, "mes_dt", ano_global, mes_global)
    if monthly.empty:
        st.info("Sem dados para os filtros selecionados.")
        return

    monthly["mes_str"] = monthly["mes_dt"].dt.strftime("%b/%Y")

    fig = px.bar(
        monthly,
        x="mes_str",
        y=["Criados", "Resolvidos"],
        barmode="group",
        text_auto=True,
        title=f"Tickets Criados vs Resolvidos — {projeto} (todas as áreas)",
        height=420,
    )
    fig.update_traces(textangle=0, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())

    # usa o mesmo helper de sempre para exibir/salvar
    show_plot(fig, f"{projeto.lower().replace(' ', '_')}_criados_resolvidos_all", projeto, ano_global, mes_global)

def render_sla_table(df_monthly_all: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### ⏱️ SLA")
    dfm = df_monthly_all[df_monthly_all["projeto"] == projeto].copy()
    if dfm.empty:
        st.info("Sem dados de SLA.")
        return
    if ano_global != "Todos":
        dfm = dfm[dfm["ano"] == int(ano_global)]
    if mes_global != "Todos":
        dfm = dfm[dfm["mes"] == int(mes_global)]
    if ano_global != "Todos" and mes_global != "Todos":
        alvo = pd.Period(f"{int(ano_global)}-{int(mes_global):02d}", freq="M")
        dfm = dfm[dfm["period"] == alvo]

    okr = dfm["pct_dentro"].mean() if not dfm.empty else 0.0
    meta = META_SLA.get(projeto, 98.0)
    titulo = f"OKR: {okr:.2f}% — Meta: {meta:.2f}%".replace(".", ",")

    show = dfm[["mes_str", "period_ts", "pct_dentro", "pct_fora"]].sort_values("period_ts")
    show = show.rename(columns={"pct_dentro": "% Dentro SLA", "pct_fora": "% Fora SLA"})
    fig = px.bar(
        show,
        x="mes_str",
        y=["% Dentro SLA", "% Fora SLA"],
        barmode="group",
        title=titulo,
        color_discrete_map={"% Dentro SLA": "green", "% Fora SLA": "red"},
        height=440,
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    show_plot(fig, "sla", projeto, ano_global, mes_global)


def render_assunto(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### 🧾 Assunto Relacionado")
    df_ass = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_ass.empty:
        st.info("Sem dados para Assunto Relacionado nos filtros atuais.")
        return
    if CAMPOS_ASSUNTO[projeto] == "issuetype":
        df_ass["assunto_nome"] = df_ass["issuetype"].apply(lambda x: safe_get_value(x, "name"))
    else:
        df_ass["assunto_nome"] = df_ass["assunto"].apply(lambda x: safe_get_value(x, "value"))
    assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
    assunto_count.columns = ["Assunto", "Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)


def render_area(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📦 Área Solicitante")
    df_area = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_area.empty:
        st.info("Sem dados para Área Solicitante nos filtros atuais.")
        return
    df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
    area_count = df_area["area_nome"].value_counts().reset_index()
    area_count.columns = ["Área", "Qtd"]
    st.dataframe(area_count, use_container_width=True, hide_index=True)


def render_encaminhamentos(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🔄 Encaminhamentos")
    df_enc = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_enc.empty:
        st.info("Sem dados para Encaminhamentos nos filtros atuais.")
        return
    col1, col2 = st.columns(2)
    with col1:
        count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
        st.metric("Encaminhados Produto", int(count_prod))
    with col2:
        df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
        st.metric("Encaminhados N3", int((df_enc["n3_valor"] == "Sim").sum()))


# ================= Módulos específicos ====================
# ---- APP NE (TDS)

def render_app_ne(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📱 APP NE")
    if dfp.empty:
        st.info("Sem dados para APP NE.")
        return
    dfp = ensure_assunto_nome(dfp.copy(), "TDS")
    s_ass = dfp["assunto_nome"].astype(str).str.strip()
    alvo = ASSUNTO_ALVO_APPNE.strip().casefold()
    mask_assunto = s_ass.str.casefold().eq(alvo)
    if not mask_assunto.any():
        mask_assunto = s_ass.str.contains(r"app\\s*ne", case=False, regex=True)

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

    serie = df_app.groupby(["mes_dt", "origem_cat"]).size().reset_index(name="Qtd").sort_values("mes_dt")
    serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
    cats = serie["mes_str"].dropna().unique().tolist()
    serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

    fig_app = px.bar(
        serie,
        x="mes_str",
        y="Qtd",
        color="origem_cat",
        barmode="group",
        title="APP NE — Volumes por mês e Origem do problema",
        color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4", "Outros/Não informado": "#9ca3af"},
        text="Qtd",
        height=460,
        category_orders={"origem_cat": ["APP NE", "APP EN", "Outros/Não informado"]},
    )
    fig_app.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=16, cliponaxis=False)
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
    show_plot(fig_app, "app_ne", "TDS", ano_global, mes_global)


# ---- Onboarding (INT)

def render_onboarding(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🧭 Onboarding")
    if dfp.empty:
        st.info("Sem dados de Onboarding.")
        return

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

    dfp = ensure_assunto_nome(dfp.copy(), "INT")
    df_onb = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)

    total_clientes_novos = int((df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO).sum())
    df_erros = df_onb[df_onb["assunto_nome"].isin(ASSUNTOS_ERROS)].copy()
    pend_mask = df_onb["status"].isin(STATUS_PENDENCIAS)
    tickets_pendencias = int(pend_mask.sum())
    possiveis_clientes = int(pend_mask.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets clientes novos", total_clientes_novos)
    c2.metric("Erros onboarding", int(len(df_erros)))
    c3.metric("Tickets com pendências", tickets_pendencias)
    c4.metric("Possíveis clientes", possiveis_clientes)

    df_cli_novo = df_onb[
        df_onb["assunto_nome"].astype(str).str.contains("cliente novo", case=False, na=False)
    ].copy()
    if not df_cli_novo.empty:
        serie = df_cli_novo.groupby(pd.Grouper(key="created", freq="MS")).size().rename("qtd").reset_index()
        if not serie.empty:
            idx = pd.date_range(serie["created"].min(), serie["created"].max(), freq="MS")
            serie = (
                serie.set_index("created").reindex(idx).fillna(0.0).rename_axis("created").reset_index()
            )
            serie["qtd"] = serie["qtd"].astype(int)
            serie["mes_str"] = serie["created"].dt.strftime("%Y %b")
            fig_cli = px.bar(
                serie, x="mes_str", y="qtd", text="qtd", title="Tickets – Cliente novo", height=420
            )
            fig_cli.update_traces(textposition="outside", cliponaxis=False)
            fig_cli.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            show_plot(fig_cli, "onb_cli_novo", "INT", ano_global, mes_global)

    def _tipo_from_assunto(s: str) -> str:
        s = (s or "").strip().lower()
        if "cliente novo" in s:
            return "Cliente novo"
        if "alteração" in s and "plataforma" in s:
            return "Alteração de plataforma"
        if "conta filho" in s:
            return "Conta filho"
        return "Outros"

    tipo_counts = (
        df_onb.assign(tipo=df_onb["assunto_nome"].map(_tipo_from_assunto)).groupby("tipo").size().rename("Qtd").reset_index()
    )
    priority = ["Cliente novo", "Outros", "Alteração de plataforma", "Conta filho"]
    tipo_counts["ord"] = tipo_counts["tipo"].apply(
        lambda x: priority.index(x) if x in priority else len(priority) + 1
    )
    tipo_counts = tipo_counts.sort_values(["ord", "Qtd"], ascending=[True, False])
    fig_tipo = px.bar(
        tipo_counts, x="Qtd", y="tipo", orientation="h", text="Qtd", title="Tipo de Integração", height=420
    )
    fig_tipo.update_traces(textposition="outside", cliponaxis=False)
    fig_tipo.update_layout(margin=dict(l=10, r=90, t=45, b=10))
    try:
        _max_q = float(tipo_counts["Qtd"].max())
        fig_tipo.update_xaxes(range=[0, _max_q * 1.12])
    except Exception:
        pass
    show_plot(fig_tipo, "onb_tipo_int", "INT", ano_global, mes_global)


# ---- Rotinas Manuais (TDS) — 100% por Jira

def render_rotinas_manuais(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    """
    Rotinas Manuais (TDS)
    - Filtra somente as áreas Ops.
    - Encomendas TDS = TOTAL mensal de 'Quantidade de encomendas' > 0
    - Manual = subconjunto se (assunto + título) contiver:
        'divergência', 'IE (Qliksense)', 'CTE', 'IE (tabela)', 'alteração de status'
      (match normalizado, sem acento/caixa)
    - Dedup por key usando o PRIMEIRO instante confiável (resolved → created → updated).
    - Reindex mensal, então todos os meses aparecem (com zero).
    - Expander de DEBUG para investigar picos mensais.
    """
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    try:
        from unidecode import unidecode as _unidecode
    except Exception:
        _unidecode = lambda s: s

    st.markdown("### 🛠️ Rotinas Manuais")

    if dfp.empty:
        st.info("Sem tickets para o período.")
        return

    OPS_AREAS = [
        "Ops - Conferência", "Ops - Cubagem", "Ops - Logística",
        "Ops - Coletas", "Ops - Expedição", "Ops - Divergências",
    ]

    def _canonical_local(txt: str) -> str:
        """Normaliza texto: minúsculas, sem acentos, sem quebras extras."""
        s = (txt or "")
        s = _unidecode(str(s)).lower()
        # normaliza espaços
        return " ".join(s.split())

    def _parse_dt_col(s):
        """Parse de datas robusto (tenta ISO e depois dayfirst)."""
        x = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        if x.notna().sum() == 0:
            x = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)
        return x

    KEYWORDS_MANUAL = [
        "divergencia",
        "ie qliksense",
        "cte",
        "ie tabela",
        "alteracao de status",
        "alteracao status",
        "volumetria",
        "inscrição estadual",
        "inscricao estadual",
    ]

    def _is_manual_by_keywords(text: str) -> bool:
        c = _canonical_local(text)
        # garante casar "alteracao ... status" mesmo com palavras no meio
        if ("alteracao" in c and "status" in c):
            return True
        return any(k in c for k in KEYWORDS_MANUAL)

    # 1) Base + assunto consolidado + área
    df = dfp.copy()
    df = ensure_assunto_nome(df, "TDS")
    df["area_nome"] = df["area"].apply(lambda x: safe_get_value(x, "value"))

    # 2) Somente as áreas Ops
    base = df[df["area_nome"].isin(OPS_AREAS)].copy()
    if base.empty:
        st.info("Sem tickets nas áreas Ops para os filtros atuais.")
        return

    # 3) Quantidade de encomendas > 0 (usa seu parser)
    base["qtd_encomendas"] = base[CAMPO_QTD_ENCOMENDAS].apply(parse_qtd_encomendas)
    base = base[base["qtd_encomendas"] > 0].copy()
    if base.empty:
        st.info("Sem tickets com 'Quantidade de encomendas' > 0 nas áreas Ops.")
        return

    # 4) Datas robustas
    base["resolved"] = _parse_dt_col(base.get("resolved"))
    base["created"]  = _parse_dt_col(base.get("created"))
    base["updated"]  = _parse_dt_col(base.get("updated"))

    # 5) PRIMEIRO instante por ticket (evita concentrar tudo no último mês)
    base["_best_dt"] = base["resolved"]
    m = base["_best_dt"].isna() & base["created"].notna()
    base.loc[m, "_best_dt"] = base.loc[m, "created"]
    m = base["_best_dt"].isna() & base["updated"].notna()
    base.loc[m, "_best_dt"] = base.loc[m, "updated"]
    base = base[base["_best_dt"].notna()].copy()

    agg_cols = {
        "_best_dt": "min",
        "qtd_encomendas": "max",
        "assunto_nome": "first",
        "summary": "first",
        "area_nome": "first",
    }
    base = (
        base.sort_values(["key", "_best_dt"])
            .groupby("key", as_index=False)
            .agg(agg_cols)
            .rename(columns={"_best_dt": "resolved"})
            .copy()
    )

    # 6) Mês + texto_busca
    base["mes_dt"] = base["resolved"].dt.to_period("M").dt.to_timestamp()
    base["assunto_nome"] = base["assunto_nome"].astype(str)
    base["summary"]      = base["summary"].astype(str)
    base["texto_busca"]  = (base["assunto_nome"].fillna("") + " " + base["summary"].fillna("")).astype(str)

    # guarda base (sem filtro global) p/ export
    full_base = base.copy()

    # 7) Filtro global (ano/mês) — só para os gráficos
    base = aplicar_filtro_global(base, "mes_dt", ano_global, mes_global)
    if base.empty:
        st.info("Sem dados para exibir com os filtros atuais.")
        return

    # 8) Série mensal (Manual x TOTAL) com reindex
    manual_mask = base["texto_busca"].apply(_is_manual_by_keywords)

    # ================= DEBUG DE MÊS (abrir mês e entender pico) =================
    with st.expander("🔍 Debug de mês (Manual | TDS)", expanded=False):

        def _manual_reason(text: str) -> str:
            c = _canonical_local(text)
            reasons = []
            if ("alteracao" in c and "status" in c):
                reasons.append("alteração de status")
            for k in KEYWORDS_MANUAL:
                if k in ("alteracao de status", "alteracao status"):
                    continue
                if k in c:
                    reasons.append(k)
            return ", ".join(sorted(set(reasons))) or "—"

        dbg = base.copy()
        dbg["is_manual"] = dbg["texto_busca"].apply(_is_manual_by_keywords)
        dbg["reason"]    = dbg["texto_busca"].apply(_manual_reason)

        mensal_manual = dbg[dbg["is_manual"]].groupby("mes_dt")["qtd_encomendas"].sum()
        mes_default = mensal_manual.idxmax() if not mensal_manual.empty else dbg["mes_dt"].min()

        meses_disponiveis = sorted(dbg["mes_dt"].dropna().unique())
        mes_debug = st.selectbox(
            "Mês para investigar",
            meses_disponiveis,
            index=max(0, meses_disponiveis.index(mes_default) if mes_default in meses_disponiveis else 0),
            format_func=lambda d: pd.to_datetime(d).strftime("%b/%Y"),
        )

        dm = dbg[dbg["mes_dt"] == mes_debug].copy()
        dm_manual = dm[dm["is_manual"]].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Tickets (mês)", int(dm["key"].nunique()))
        c2.metric("Manual — soma", int(dm_manual["qtd_encomendas"].sum()))
        c3.metric("Total — soma", int(dm["qtd_encomendas"].sum()))

        st.markdown("**Por motivo (keyword)**")
        st.dataframe(
            dm_manual.groupby("reason")["qtd_encomendas"]
                .sum().sort_values(ascending=False).reset_index(),
            use_container_width=True, hide_index=True
        )

        st.markdown("**Por assunto**")
        st.dataframe(
            dm_manual.groupby("assunto_nome")["qtd_encomendas"]
                .sum().sort_values(ascending=False).reset_index().head(30),
            use_container_width=True, hide_index=True
        )

        st.markdown("**Por área**")
        st.dataframe(
            dm_manual.groupby("area_nome")["qtd_encomendas"]
                .sum().sort_values(ascending=False).reset_index(),
            use_container_width=True, hide_index=True
        )

        if not dm_manual.empty:
            p99 = dm_manual["qtd_encomendas"].quantile(0.99)
            med = dm_manual["qtd_encomendas"].median()
            lim = max(p99, med * 5)

            top = (dm_manual.assign(outlier=lambda x: x["qtd_encomendas"] >= lim)
                   .sort_values("qtd_encomendas", ascending=False)
                   [["key","resolved","area_nome","assunto_nome","summary","reason","qtd_encomendas","outlier"]]
                   .head(50))

            st.markdown("**Top 50 tickets (Manual) do mês**")
            st.dataframe(top, use_container_width=True, hide_index=True)

            st.download_button(
                "Baixar CSV (Manual do mês)",
                data=dm_manual[["key","resolved","mes_dt","area_nome","assunto_nome","summary","reason","qtd_encomendas"]]
                    .sort_values("qtd_encomendas", ascending=False)
                    .to_csv(index=False).encode("utf-8-sig"),
                file_name=f"debug_manual_{pd.to_datetime(mes_debug).strftime('%Y_%m')}.csv",
                mime="text/csv",
            )
    # ============================================================================

    monthly_total  = base.groupby("mes_dt")["qtd_encomendas"].sum().rename("Encomendas TDS")
    monthly_manual = base[manual_mask].groupby("mes_dt")["qtd_encomendas"].sum().rename("Manual")

    idx = pd.date_range(
        start=base["mes_dt"].min().to_period("M").to_timestamp(),
        end=base["mes_dt"].max().to_period("M").to_timestamp(),
        freq="MS",
    )
    monthly = pd.concat([monthly_manual, monthly_total], axis=1).reindex(idx).fillna(0.0)
    monthly = monthly.rename_axis("mes_dt").reset_index()
    monthly["mes_str"] = monthly["mes_dt"].dt.strftime("%b/%Y")

    # 9) Barras Manual x TDS
    fig = px.bar(
        monthly,
        x="mes_str",
        y=["Manual", "Encomendas TDS"],
        barmode="group",
        text_auto=True,
        title="Encomendas corrigidas — Manual | TDS (mês a mês)",
        height=420,
    )
    fig.update_traces(textangle=0, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())
    show_plot(fig, "rotinas_ops_mensal_manual_tds", "TDS", ano_global, mes_global)

    # 10) Donut de participação
    total_sum  = float(monthly["Encomendas TDS"].sum())
    manual_sum = float(monthly["Manual"].sum())
    restante   = max(total_sum - manual_sum, 0.0)
    df_donut = pd.DataFrame({"tipo": ["Manual", "Encomendas TDS"], "qtd": [manual_sum, restante]})
    fig_donut = px.pie(df_donut, values="qtd", names="tipo", hole=0.6, title="Manual | TDS — Participação")
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    show_plot(fig_donut, "rotinas_ops_donut_manual_tds", "TDS", ano_global, mes_global)

    # 11) Ranking por Assunto (Manual)
    df_manual = base[manual_mask].copy()
    if not df_manual.empty:
        rank = (df_manual.groupby("assunto_nome")["qtd_encomendas"]
                .sum().sort_values(ascending=False).reset_index())
        fig_ass = px.bar(
            rank, x="qtd_encomendas", y="assunto_nome",
            orientation="h", text="qtd_encomendas",
            title="Manual | Assunto", height=420
        )
        fig_ass.update_traces(textposition="outside", cliponaxis=False)
        fig_ass.update_layout(margin=dict(l=10, r=90, t=45, b=10))
        try:
            _max_q = float(rank["qtd_encomendas"].max())
            fig_ass.update_xaxes(range=[0, _max_q * 1.12])
        except Exception:
            pass
        show_plot(fig_ass, "rotinas_ops_manual_assunto", "TDS", ano_global, mes_global)

    # 12) Export/diagnóstico (sem filtro global)
    with st.expander("📤 Exportar / diagnóstico — Ops (qtd > 0)", expanded=False):
        df_export = full_base.copy()
        df_export["texto_busca"] = (df_export["assunto_nome"].fillna("") + " " + df_export["summary"].fillna("")).astype(str)
        df_export["tipo_encomenda"] = df_export["texto_busca"].apply(
            lambda s: "Manual" if _is_manual_by_keywords(s) else "Encomendas TDS"
        )
        df_export = df_export[
            ["key","resolved","mes_dt","area_nome","assunto_nome","summary","qtd_encomendas","tipo_encomenda"]
        ].sort_values("resolved")

        csv = df_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV (todos)", data=csv, file_name="rotinas_manuais_ops.csv", mime="text/csv")
        st.dataframe(df_export.head(5000), use_container_width=True, hide_index=True)

# ================= Filtros Globais ========================

st.markdown("### 🔍 Filtros Globais")
ano_atual = date.today().year
opcoes_ano = ["Todos"] + [str(y) for y in range(2024, ano_atual + 1)]
opcoes_mes = ["Todos"] + [f"{m:02d}" for m in range(1, 13)]
colA, colB = st.columns(2)
with colA:
    ano_global = st.selectbox("Ano (global)", opcoes_ano, index=0, key="ano_global")
with colB:
    mes_global = st.selectbox("Mês (global)", opcoes_mes, index=0, key="mes_global")

# ================= Coleta de dados ========================

def jql_projeto(project_key: str, ano_sel: str, mes_sel: str) -> str:
    base = f'project = "{project_key}" AND created >= "{DATA_INICIO}"'
    if mes_sel != "Todos" and ano_sel != "Todos":
        a = int(ano_sel)
        m = int(mes_sel)
        if m == 12:
            next_month_first = date(a + 1, 1, 1)
        else:
            next_month_first = date(a, m + 1, 1)
        base += f' AND created < "{next_month_first:%Y-%m-%d}"'
    return base + " ORDER BY created ASC"

JQL_TDS = jql_projeto("TDS", "Todos", "Todos")
JQL_INT = jql_projeto("INT", "Todos", "Todos")
JQL_TINE = jql_projeto("TINE", "Todos", "Todos")
JQL_INTEL = jql_projeto("INTEL", "Todos", "Todos")

with st.spinner("Carregando TDS..."):
    df_tds = buscar_issues("TDS", JQL_TDS)
with st.spinner("Carregando INT..."):
    df_int = buscar_issues("INT", JQL_INT)
with st.spinner("Carregando TINE..."):
    df_tine = buscar_issues("TINE", JQL_TINE)
with st.spinner("Carregando INTEL..."):
    df_intel = buscar_issues("INTEL", JQL_INTEL)

if all(d.empty for d in [df_tds, df_int, df_tine, df_intel]):
    st.warning("Sem dados do Jira em nenhum projeto (verifique credenciais e permissões).")
    st.stop()

_df_monthly_all = pd.concat(
    [build_monthly_tables(d) for d in [df_tds, df_int, df_tine, df_intel] if not d.empty],
    ignore_index=True,
) if not all(d.empty for d in [df_tds, df_int, df_tine, df_intel]) else pd.DataFrame()

# ================= Abas / Vistas ==========================

tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")

        if projeto == "TDS":
            dfp = df_tds.copy()
            opcoes = [
                "Geral",
                "Criados vs Resolvidos",
                "SLA",
                "Assunto Relacionado",
                "Área Solicitante",
                "APP NE",
                "Rotinas Manuais",
            ]
        elif projeto == "INT":
            dfp = df_int.copy()
            opcoes = [
                "Geral",
                "Criados vs Resolvidos",
                "SLA",
                "Assunto Relacionado",
                "Área Solicitante",
                "Onboarding",
            ]
        elif projeto == "TINE":
            dfp = df_tine.copy()
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante"]
        else:
            dfp = df_intel.copy()
            opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado"]

        if dfp.empty:
            st.info("Sem dados carregados para este projeto.")
            continue

        dfp["mes_created"] = pd.to_datetime(dfp["created"], errors="coerce")
        dfp["mes_resolved"] = pd.to_datetime(dfp["resolved"], errors="coerce")
        dfp = ensure_assunto_nome(dfp, projeto)

        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")

        if visao == "Criados vs Resolvidos":
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
        elif visao == "SLA":
            render_sla_table(_df_monthly_all, projeto, ano_global, mes_global)
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
        elif visao == "Rotinas Manuais":
            if projeto == "TDS":
                render_rotinas_manuais(dfp, ano_global, mes_global)
            else:
                st.info("Rotinas Manuais disponível somente para Tech Support.")
        else:
            # Geral
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
            render_sla_table(_df_monthly_all, projeto, ano_global, mes_global)
            render_assunto(dfp, projeto, ano_global, mes_global)
            if projeto != "INTEL":
                render_area(dfp, ano_global, mes_global)
            if projeto in ("TDS", "INT"):
                render_encaminhamentos(dfp, ano_global, mes_global)
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
                with st.expander("🛠️ Rotinas Manuais", expanded=False):
                    render_rotinas_manuais(dfp, ano_global, mes_global)
            if projeto == "INT":
                with st.expander("🧭 Onboarding", expanded=False):
                    render_onboarding(dfp, ano_global, mes_global)

st.markdown("---")
st.caption("💙 Desenvolvido por Thaís Franco.")
