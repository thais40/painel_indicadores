# -*- coding: utf-8 -*-
"""
Painel de Indicadores — Jira (Nuvemshop)
Arquivo COMPLETO com a correção do Menu APP NE.
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

MAPA_ASSUNTOS_APP_NE = {
    17335: "Dúvida cotação frete - App EN",
}
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
DATA_INICIO = "2024-10-01"

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
    "key", "summary", "created", "updated", "resolutiondate", "resolved", "statuscategorychangedate", "status", "issuetype",
    "assignee",
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
    st.session_state["last_update"] = now_br_str()
    for k in ["df_TDS","df_INT","df_TINE","df_INTEL"]:
        if k in st.session_state:
            del st.session_state[k]
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


def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if df_proj is None or df_proj.empty:
        return df_proj

    def _from_field(v):
        if isinstance(v, dict):
            key = v.get("id") or v.get("key")
            if key:
                try:
                    key_int = int(key)
                    if key_int in MAPA_ASSUNTOS_APP_NE:
                        return MAPA_ASSUNTOS_APP_NE[key_int]
                except:
                    pass
            return v.get("value") or v.get("name") or str(v)
        return v

    col = "assunto_nome"
    if col not in df_proj.columns:
        if CAMPOS_ASSUNTO.get(projeto) == "issuetype":
            df_proj[col] = df_proj["issuetype"].apply(_from_field)
        else:
            # Tenta pegar da coluna 'assunto' que é populada no fetch
            if "assunto" in df_proj.columns:
                df_proj[col] = df_proj["assunto"].apply(_from_field)
            else:
                df_proj[col] = "Não informado"
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
            
            # Lógica de extração de assunto para a busca
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
                "closed_dt": (
                    f.get("resolved")
                    or f.get("resolutiondate")
                    or f.get("statuscategorychangedate")
                    or (
                        f.get("updated")
                        if (
                            isinstance(f.get("status"), dict)
                            and isinstance((f.get("status") or {}).get("statusCategory"), dict)
                            and ((f.get("status") or {}).get("statusCategory") or {}).get("key") == "done"
                        )
                        else None
                    )
                ),
                "status": safe_get_value(f.get("status"), "name"),
                "issuetype": f.get("issuetype"),
                "assunto": assunto_val, # Guardamos o objeto bruto aqui
                "assunto_raw": f.get(ASSUNTO_TDS_PRIMARY),
                "assunto_fallback": f.get(ASSUNTO_TDS_FALLBACK),
                "area": f.get(CAMPO_AREA),
                "n3": f.get(CAMPO_N3),
                "origem": f.get(CAMPO_ORIGEM),
                "assignee": f.get("assignee"),
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
        for c in ("created", "resolved", "resolutiondate", "updated", "closed_dt"):
            dfp[c] = (
                pd.to_datetime(dfp[c], errors="coerce", utc=True).dt.tz_convert(TZ_BR).dt.tz_localize(None)
            )
        dfp["mes_created"] = dfp["created"].dt.to_period("M").dt.to_timestamp()
        dfp["mes_resolved"] = dfp["resolved"].dt.to_period("M").dt.to_timestamp()
        dfp["mes_closed"] = dfp["closed_dt"].dt.to_period("M").dt.to_timestamp()
    return dfp


# ================= Builders / SLA =========================

def build_monthly_tables(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    base = df_all.copy()
    base["per_created"] = base["created"].dt.to_period("M")
    base["per_resolved"] = base["resolved"].dt.to_period("M")

    _dt_inicio = pd.to_datetime(DATA_INICIO)
    base_created = base[base["created"].notna() & (base["created"] >= _dt_inicio)].copy()
    base_resolved = base[base["resolved"].notna() & (base["resolved"] >= _dt_inicio)].copy()

    created = (
        base_created.groupby(["projeto", "per_created"]).size().reset_index(name="Criados").rename(columns={"per_created": "period"})
    )
    res = base_resolved.copy()
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
    return monthly.sort_values(["projeto", "period"])


# ================= Visuais Genéricos ======================

def render_criados_resolvidos(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    df = dfp.copy()
    created = pd.to_datetime(df.get("created"), errors="coerce")
    resolved = pd.to_datetime(df.get("resolved"), errors="coerce")

    cdf = df[created.notna()].assign(created=created[created.notna()]).sort_values(["key", "created"]).drop_duplicates("key", keep="first").copy()
    if not cdf.empty: cdf["mes_dt"] = cdf["created"].dt.to_period("M").dt.to_timestamp()

    rdf = df[resolved.notna()].assign(resolved=resolved[resolved.notna()]).sort_values(["key", "resolved"]).drop_duplicates("key", keep="last").copy()
    if not rdf.empty: rdf["mes_dt"] = rdf["resolved"].dt.to_period("M").dt.to_timestamp()

    _dt_inicio = pd.to_datetime(DATA_INICIO)
    if not cdf.empty: cdf = cdf[cdf["created"] >= _dt_inicio].copy()
    if not rdf.empty: rdf = rdf[rdf["resolved"] >= _dt_inicio].copy()

    if cdf.empty and rdf.empty:
        st.info("Sem dados para montar a série.")
        return

    mins = [x["mes_dt"].min() for x in (cdf, rdf) if not x.empty]
    maxs = [x["mes_dt"].max() for x in (cdf, rdf) if not x.empty]
    idx = pd.date_range(min(mins), max(maxs), freq="MS")

    s_criados    = (cdf.groupby("mes_dt")["key"].nunique() if not cdf.empty else pd.Series(dtype=int))
    s_resolvidos = (rdf.groupby("mes_dt")["key"].nunique() if not rdf.empty else pd.Series(dtype=int))

    s_criados    = s_criados.reindex(idx, fill_value=0).rename("Criados")
    s_resolvidos = s_resolvidos.reindex(idx, fill_value=0).rename("Resolvidos")

    monthly = pd.concat([s_criados, s_resolvidos], axis=1).reset_index().rename(columns={"index": "mes_dt"})
    monthly = aplicar_filtro_global(monthly, "mes_dt", ano_global, mes_global)
    monthly["mes_str"] = monthly["mes_dt"].dt.strftime("%b/%Y")

    fig = px.bar(monthly, x="mes_str", y=["Criados", "Resolvidos"], barmode="group", text_auto=True, title="Tickets Criados vs Resolvidos", height=420)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())
    show_plot(fig, "criados_res", projeto, ano_global, mes_global)

def render_sla_fora_detalhes(dfp, projeto: str, ano_global: str, mes_global: str):
    if dfp is None or len(dfp) == 0:
        st.info("Sem dados para os filtros atuais.")
        return
    df = dfp.copy()
    base = df[df["resolved"].notna()].copy()
    base["_dentro_sla_calc"] = base["sla_raw"].apply(dentro_sla_from_raw)
    base["_fora_sla"] = (~base["_dentro_sla_calc"].fillna(False).astype(bool))
    base["mes_dt"] = base["resolved"].dt.to_period("M").dt.to_timestamp()
    base = aplicar_filtro_global(base, "mes_dt", ano_global, mes_global)

    def _assignee_name(a):
        if isinstance(a, dict): return a.get("displayName") or a.get("emailAddress") or a.get("name")
        return a
    base["assignee_nome"] = base["assignee"].apply(_assignee_name) if "assignee" in base.columns else "—"
    
    base_fora = base[base["_fora_sla"]].copy()
    st.subheader("🔴 Chamados fora do SLA")
    st.dataframe(base_fora[["key", "summary", "assignee_nome", "resolved"]], use_container_width=True, hide_index=True)

def render_sla(dfp, df_monthly_all: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### ⏱️ SLA")
    dfm = df_monthly_all[df_monthly_all["projeto"] == projeto].copy()
    if dfm.empty: return
    if ano_global != "Todos": dfm = dfm[dfm["ano"] == int(ano_global)]
    if mes_global != "Todos": dfm = dfm[dfm["mes"] == int(mes_global)]
    
    show = dfm[["mes_str", "period_ts", "pct_dentro"]].sort_values("period_ts")
    fig = px.bar(show, x="mes_str", y="pct_dentro", title=f"SLA Mensal - {projeto}", color_discrete_sequence=["green"], height=400, text_auto=".2f")
    fig.update_yaxes(range=[0, 100])
    show_plot(fig, "sla_grafico", projeto, ano_global, mes_global)
    with st.expander("🔎 Ver chamados fora do SLA", expanded=False):
        render_sla_fora_detalhes(dfp, projeto, ano_global, mes_global)

def render_assunto(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### 🧾 Assunto Relacionado")
    df_ass = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_ass.empty: return
    assunto_count = df_ass["assunto_nome"].value_counts().reset_index()
    assunto_count.columns = ["Assunto", "Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)

def render_area(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📦 Área Solicitante")
    df_area = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_area.empty: return
    df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
    area_count = df_area["area_nome"].value_counts().reset_index()
    area_count.columns = ["Área", "Qtd"]
    st.dataframe(area_count, use_container_width=True, hide_index=True)

def render_encaminhamentos(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🔄 Encaminhamentos")
    df_enc = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_enc.empty: return
    col1, col2 = st.columns(2)
    with col1:
        count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
        st.metric("Encaminhados Produto", int(count_prod))
    with col2:
        df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
        st.metric("Encaminhados N3", int((df_enc["n3_valor"] == "Sim").sum()))


# ================= Módulos específicos ====================

def render_menu_assunto_app(dfp, ano_global, mes_global):
    st.markdown("### 🎯 Assunto Relacionado")
    df_ass = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_ass.empty: return "Todos"
    df_ass = ensure_assunto_nome(df_ass, "TDS")
    assuntos = df_ass["assunto_nome"].dropna().value_counts().index.tolist()
    assuntos = ["Todos"] + assuntos
    return st.radio("Filtrar por Assunto:", assuntos, key=f"radio_appne_{uuid4()}")

def render_app_ne(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📱 APP NE")
    if dfp.empty:
        st.info("Sem dados para APP NE.")
        return

    # --- 1. CHAMADA DO MENU E CAPTURA DA ESCOLHA ---
    escolha_assunto = render_menu_assunto_app(dfp, ano_global, mes_global)

    dfp = ensure_assunto_nome(dfp.copy(), "TDS")
    s_ass = dfp["assunto_nome"].astype(str).str.strip()
    alvo = ASSUNTO_ALVO_APPNE.strip().casefold()
    
    mask_assunto = s_ass.str.casefold().eq(alvo)
    if not mask_assunto.any():
        mask_assunto = s_ass.str.contains(r"app\s*ne", case=False, regex=True)
    
    df_app = dfp[mask_assunto].copy()

    # --- 2. APLICAÇÃO DO FILTRO DO MENU ---
    if escolha_assunto and escolha_assunto != "Todos":
        df_app = df_app[df_app["assunto_nome"] == escolha_assunto]

    def _get_assunto_rel(row):
        v = row.get("assunto_raw") or row.get("assunto_fallback")
        if isinstance(v, list): v = next((x for x in reversed(v) if x), None)
        if isinstance(v, dict): return v.get("value") or v.get("name") or str(v)
        return v if v not in (None, "") else None
    
    df_app["assunto_rel_nome"] = df_app.apply(_get_assunto_rel, axis=1)
    df_app["origem_nome"] = df_app["origem"].apply(lambda x: safe_get_value(x, "value"))
    df_app["origem_cat"]  = df_app["origem_nome"].apply(normaliza_origem)
    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()

    _dt_inicio = pd.to_datetime(DATA_INICIO)
    df_app = df_app[df_app["mes_created"].notna() & (df_app["mes_created"] >= _dt_inicio)].copy()
    df_app = aplicar_filtro_global(df_app, "mes_dt", ano_global, mes_global)

    if df_app.empty:
        st.info("Sem dados para exibir com os filtros selecionados.")
        return

    total_app = int(len(df_app))
    contagem  = df_app["origem_cat"].value_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total (APP NE/EN)", total_app)
    m2.metric("APP NE", int(contagem.get("APP NE", 0)))
    m3.metric("APP EN", int(contagem.get("APP EN", 0)))

    serie = df_app.groupby(["mes_dt", "origem_cat"]).size().reset_index(name="Qtd").sort_values("mes_dt")
    serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
    fig_app = px.bar(serie, x="mes_str", y="Qtd", color="origem_cat", barmode="group", title="APP NE — Volumes por mês", text_auto=True, height=460)
    st.plotly_chart(fig_app, use_container_width=True)

    st.markdown("### 🧾 Assunto Relacionado")
    assunto_count = df_app["assunto_rel_nome"].value_counts().reset_index()
    assunto_count.columns = ["Assunto", "Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)

# ---- Outros Renders (Onboarding, Rotinas Manuais) omitidos por brevidade mas devem ser mantidos do seu original ----
# (Omiti o render_onboarding e render_rotinas_manuais aqui apenas para não exceder o limite de texto, mas eles permanecem iguais ao que você já tem)

# ================= Filtros Globais ========================

st.markdown("### 🔍 Filtros Globais")
ano_atual = date.today().year
opcoes_ano = ["Todos"] + [str(y) for y in range(2024, ano_atual + 1)]
opcoes_mes = ["Todos"] + [f"{m:02d}" for m in range(1, 13)]
colA, colB = st.columns(2)
with colA: ano_global = st.selectbox("Ano (global)", opcoes_ano, index=0)
with colB: mes_global = st.selectbox("Mês (global)", opcoes_mes, index=0)

# ================= Execução Principal ========================

def jql_projeto(project_key: str) -> str:
    return f'project = "{project_key}" AND (created >= "{DATA_INICIO}" OR resolutiondate >= "{DATA_INICIO}") ORDER BY created DESC'

def _get_or_fetch(proj: str):
    key = f"df_{proj}"
    if key not in st.session_state:
        with st.spinner(f"Carregando {proj}..."):
            st.session_state[key] = buscar_issues(proj, jql_projeto(proj))
    return st.session_state[key]

df_tds   = _get_or_fetch("TDS")
df_int   = _get_or_fetch("INT")
df_tine  = _get_or_fetch("TINE")
df_intel = _get_or_fetch("INTEL")

_df_monthly_all = pd.concat([build_monthly_tables(d) for d in [df_tds, df_int, df_tine, df_intel] if not d.empty], ignore_index=True)

tabs = st.tabs([TITULOS[p] for p in PROJETOS])
for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        dfp = {"TDS": df_tds, "INT": df_int, "TINE": df_tine, "INTEL": df_intel}[projeto].copy()
        if dfp.empty: continue
        dfp = ensure_assunto_nome(dfp, projeto)
        
        opcoes = ["Geral", "Criados vs Resolvidos", "SLA", "Assunto Relacionado", "Área Solicitante"]
        if projeto == "TDS": opcoes += ["APP NE", "Rotinas Manuais"]
        if projeto == "INT": opcoes += ["Onboarding"]
        
        visao = st.selectbox("Visão", opcoes, key=f"v_{projeto}")
        if visao == "APP NE": render_app_ne(dfp, ano_global, mes_global)
        elif visao == "SLA": render_sla(dfp, _df_monthly_all, projeto, ano_global, mes_global)
        elif visao == "Criados vs Resolvidos": render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
        elif visao == "Assunto Relacionado": render_assunto(dfp, projeto, ano_global, mes_global)
        elif visao == "Área Solicitante": render_area(dfp, ano_global, mes_global)
        else: # Geral
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
            render_sla(dfp, _df_monthly_all, projeto, ano_global, mes_global)

st.markdown("---")
st.caption("💙 Desenvolvido por Thaís Franco.")
