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
DATA_INICIO = "2024-08-01"

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

# 👉 ADIÇÃO: vamos buscar também o assignee
JIRA_FIELDS_BASE = [
    "key", "summary", "created", "updated", "resolutiondate", "resolved", "statuscategorychangedate", "status", "issuetype",
    "assignee",  # <— NOVO
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
    # limpa cache em memória (session_state) para forçar nova busca no Jira
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
                # ✅ Data de "fechamento" (para volumes) mesmo quando não há resolutiondate:
                # 1) resolutiondate/resolved (se existir)
                # 2) statuscategorychangedate (quando entra em Done)
                # 3) updated (somente se statusCategory = Done)
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
                "assunto": assunto_val,
                "area": f.get(CAMPO_AREA),
                "n3": f.get(CAMPO_N3),
                "origem": f.get(CAMPO_ORIGEM),
                "assignee": f.get("assignee"),  # <— NOVO
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

    # ✅ Mantém o SLA como no arquivo original, mas respeita DATA_INICIO no recorte do painel:
    # - "Criados": só a partir de DATA_INICIO
    # - "Resolvidos"/SLA: só a partir de DATA_INICIO
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
    created  = pd.to_datetime(df.get("created"),  errors="coerce")
    resolved = pd.to_datetime(df.get("closed_dt"), errors="coerce")

    cdf = (
        df[created.notna()]
        .assign(created=created[created.notna()])
        .sort_values(["key", "created"])
        .drop_duplicates("key", keep="first")
        .copy()
    )
    if not cdf.empty:
        cdf["mes_dt"] = cdf["created"].dt.to_period("M").dt.to_timestamp()

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

    # ✅ Respeita DATA_INICIO no eixo: criados por created, resolvidos por closed_dt
    _dt_inicio = pd.to_datetime(DATA_INICIO)
    if not cdf.empty:
        cdf = cdf[cdf["created"].notna() & (cdf["created"] >= _dt_inicio)].copy()
    if not rdf.empty:
        rdf = rdf[rdf["resolved"].notna() & (rdf["resolved"] >= _dt_inicio)].copy()
    if cdf.empty and rdf.empty:
        st.info("Sem dados a partir de DATA_INICIO para montar a série.")
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
        title=f"Tickets Criados vs Resolvidos",
        height=420,
    )
    fig.update_traces(textangle=0, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())
    show_plot(fig, f"{projeto.lower().replace(' ', '_')}_criados_resolvidos_all", projeto, ano_global, mes_global)

def render_sla_table(df_monthly_all: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### ⏱️ SLA (legado)")
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
    show = show.rename(columns={"pct_dentro": "% Dentro SLA", "% Fora SLA": "pct_fora"})
    fig = px.bar(
        show,
        x="mes_str",
        y=["% Dentro SLA", "pct_fora"],
        barmode="group",
        title=titulo,
        color_discrete_map={"% Dentro SLA": "green", "pct_fora": "red"},
        height=440,
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    show_plot(fig, "sla_legacy", projeto, ano_global, mes_global)


# ================== SLA – gráfico + submenu "Chamados fora do SLA" ==================
def render_sla_fora_detalhes(dfp, projeto: str, ano_global: str, mes_global: str):
    """
    Lista de tickets fora do SLA alinhada ao gráfico:
      - Apenas tickets com 'resolved'
      - Fora do SLA = not(dentro_sla_from_raw(sla_raw)) com fillna(False)
      - Filtro por mês/ano aplicado sobre 'resolved'
      - ADIÇÃO: coluna 'assignee_nome'
      - REMOÇÃO: não exibimos mais o card da mediana de horas
    """
    import numpy as np
    import pandas as pd
    import streamlit as st

    if dfp is None or len(dfp) == 0:
        st.info("Sem dados para os filtros atuais.")
        return

    df = dfp.copy()
    df["created"]  = pd.to_datetime(df.get("created"),  errors="coerce")
    df["resolved"] = pd.to_datetime(df.get("resolved"), errors="coerce")

    base = df[df["resolved"].notna()].copy()
    base["_dentro_sla_calc"] = base["sla_raw"].apply(dentro_sla_from_raw)
    base["_fora_sla"] = (~base["_dentro_sla_calc"].fillna(False).astype(bool))  # None => Fora

    base["mes_dt"] = base["resolved"].dt.to_period("M").dt.to_timestamp()
    base = aplicar_filtro_global(base, "mes_dt", ano_global, mes_global)

    # area_nome
    if "area_nome" not in base.columns:
        def _get_area(x):
            if isinstance(x, dict):
                return x.get("value") or x.get("name") or str(x)
            return x
        if "area" in base.columns:
            base["area_nome"] = base["area"].apply(_get_area)
        else:
            base["area_nome"] = np.nan

    # assunto_nome
    if "assunto_nome" not in base.columns:
        if "assunto" in base.columns:
            base["assunto_nome"] = base["assunto"].apply(
                lambda x: x.get("value") if isinstance(x, dict) else x
            )
        else:
            base["assunto_nome"] = np.nan

    # 👉 assignee_nome (displayName → emailAddress → name → accountId)
    def _assignee_name(a):
        if isinstance(a, dict):
            return a.get("displayName") or a.get("emailAddress") or a.get("name") or a.get("accountId")
        return a
    base["assignee_nome"] = base.get("assignee").apply(_assignee_name) if "assignee" in base.columns else np.nan

    # tempo em horas para ordenar/exibir
    if {"resolved", "created"}.issubset(base.columns):
        base["tempo_resolucao_horas"] = (
            (base["resolved"] - base["created"]).dt.total_seconds() / 3600.0
        )

    base_fora = base[base["_fora_sla"]].copy()

    st.subheader("🔴 Chamados fora do SLA")
    total_fora = int(base_fora["key"].nunique() if "key" in base_fora.columns else len(base_fora))
    st.metric("Total fora do SLA (período filtrado)", total_fora)

    prefer = [
        "key", "created", "resolved", "summary",
        "assignee_nome", "area_nome", "assunto_nome", "tempo_resolucao_horas"
    ]
    cols = [c for c in prefer if c in base_fora.columns]
    if not cols:
        cols = [c for c in ["key", "created", "resolved", "summary"] if c in base_fora.columns]

    if "tempo_resolucao_horas" in base_fora.columns:
        base_fora = base_fora.sort_values("tempo_resolucao_horas", ascending=False)
        base_fora["tempo_resolucao_horas"] = base_fora["tempo_resolucao_horas"].astype(float).round(1)

    # ✅ AQUI é a correção (Arrow): criamos um df "limpo" antes de passar pro st.dataframe
    show_df = base_fora[cols].copy()

    # 1) datetime com timezone -> sem timezone (Arrow costuma quebrar com tz)
    for c in show_df.columns:
        if pd.api.types.is_datetime64tz_dtype(show_df[c]):
            show_df[c] = show_df[c].dt.tz_convert(TZ_BR).dt.tz_localize(None)

    # 2) qualquer dict/list/objeto vira string (Arrow não aceita dict em célula)
    for c in show_df.columns:
        if show_df[c].dtype == "object":
            show_df[c] = show_df[c].apply(
                lambda x: x if (x is None or isinstance(x, (str, int, float, bool))) else str(x)
            )

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    csv = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Baixar CSV (fora SLA)",
        data=csv,
        file_name=f"{projeto.lower().replace(' ','_')}_fora_sla.csv",
        mime="text/csv",
    )

def render_sla(dfp, df_monthly_all: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    """Gráfico de SLA + submenu (expander) com lista fora do SLA."""
    import plotly.express as px
    import streamlit as st

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

    with st.expander("🔎 Ver chamados fora do SLA (lista detalhada)", expanded=False):
        render_sla_fora_detalhes(dfp, projeto, ano_global, mes_global)
# ==================/ SLA – gráfico + submenu ==================


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
        mask_assunto = s_ass.str.contains(r"app\s*ne", case=False, regex=True)

    df_app = dfp[mask_assunto].copy()
    if df_app.empty:
        st.info(f"Não há chamados para '{ASSUNTO_ALVO_APPNE}'.")
        return

    df_app["origem_nome"] = df_app["origem"].apply(lambda x: safe_get_value(x, "value"))
    df_app["origem_cat"]  = df_app["origem_nome"].apply(normaliza_origem)

    df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()
    # ✅ Respeita DATA_INICIO no APP NE (comparando pelo created)
    _dt_inicio = pd.to_datetime(DATA_INICIO)
    df_app = df_app[df_app["mes_created"].notna() & (df_app["mes_created"] >= _dt_inicio)].copy()
    df_app = aplicar_filtro_global(df_app, "mes_dt", ano_global, mes_global)
    if df_app.empty:
        st.info("Sem dados para exibir com os filtros selecionados.")
        return

    total_app = int(len(df_app))
    contagem  = df_app["origem_cat"].value_counts()

    m1,m2,m3 = st.columns(3)
    m1.metric("Total (APP NE/EN)", total_app)
    m2.metric("APP NE", int(contagem.get("APP NE", 0)))
    m3.metric("APP EN", int(contagem.get("APP EN", 0)))

    serie = (df_app.groupby(["mes_dt","origem_cat"]).size()
             .reset_index(name="Qtd").sort_values("mes_dt"))
    serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
    cats = serie["mes_str"].dropna().unique().tolist()
    serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

    fig_app = px.bar(
        serie, x="mes_str", y="Qtd", color="origem_cat", barmode="group",
        title="APP NE — Volumes por mês e Origem do problema",
        color_discrete_map={"APP NE":"#2ca02c","APP EN":"#1f77b4","Outros/Não informado":"#9ca3af"},
        text="Qtd", height=460,
        category_orders={"origem_cat": ["APP NE", "APP EN", "Outros/Não informado"]},
    )
    fig_app.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=16, cliponaxis=False)
    max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
    if max_qtd > 0:
        fig_app.update_yaxes(range=[0, max_qtd * 1.25])
    fig_app.update_layout(yaxis_title="Qtd", xaxis_title="Mês",
                          uniformtext_minsize=14, uniformtext_mode="show",
                          bargap=0.15, margin=dict(t=70, r=20, b=60, l=50))
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
    # ✅ Respeita DATA_INICIO no Onboarding
    _dt_inicio = pd.to_datetime(DATA_INICIO)
    # usa mes_created (já normalizado no pipeline) para evitar qualquer divergência de timezone/string
    if "mes_created" in df_onb.columns:
        df_onb = df_onb[df_onb["mes_created"].notna() & (df_onb["mes_created"] >= _dt_inicio)].copy()
    else:
        df_onb = df_onb[df_onb["created"].notna() & (df_onb["created"] >= _dt_inicio)].copy()

    total_clientes_novos = int((df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO).sum())
    df_erros = df_onb[df_onb["assunto_nome"].isin(ASSUNTOS_ERROS)].copy()
    pend_mask = df_onb["status"].isin(STATUS_PENDENCIAS)

    tickets_pendencias = int(pend_mask.sum())
    possiveis_clientes = int(pend_mask.sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tickets clientes novos", total_clientes_novos)
    c2.metric("Erros onboarding", int(len(df_erros)))
    c3.metric("Tickets com pendências", tickets_pendencias)
    c4.metric("Possíveis clientes", possiveis_clientes)

    # 1) Cliente novo mensal com variação
    df_cli_novo = df_onb[df_onb["assunto_nome"].astype(str).str.contains("cliente novo", case=False, na=False)].copy()
    if not df_cli_novo.empty:
        serie = (
            df_cli_novo
            .groupby(pd.Grouper(key="created", freq="MS"))
            .size()
            .rename("qtd")
            .reset_index()
        )
        if not serie.empty:
            _dt_inicio = pd.to_datetime(DATA_INICIO)
            _min = max(serie["created"].min(), _dt_inicio)
            idx = pd.date_range(_min, serie["created"].max(), freq="MS")
            serie = (
                serie.set_index("created")
                     .reindex(idx)
                     .fillna(0.0)
                     .rename_axis("created")
                     .reset_index()
            )
            serie["qtd"] = serie["qtd"].astype(int)
            serie["pct"] = serie["qtd"].pct_change() * 100.0
            serie["pct"].replace([float("inf"), float("-inf")], float("nan"), inplace=True)

            def _ann(v):
                import math
                try:
                    if v is None: return ""
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return ""
                    v2 = int(round(v))
                    if v2 > 0:  return f"▲ {v2}%"
                    if v2 < 0:  return f"▼ {abs(v2)}%"
                    return "0%"
                except Exception:
                    return ""

            serie["annot"] = serie["pct"].map(_ann)
            serie["mes_str"] = serie["created"].dt.strftime("%Y %b")

            fig_cli = px.bar(serie, x="mes_str", y="qtd", text="qtd", title="Tickets – Cliente novo", height=420)
            fig_cli.update_traces(textposition="outside", cliponaxis=False)
            fig_cli.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            for _, r in serie.iterrows():
                txt = r.get("annot") or ""
                if not txt:
                    continue
                color = "blue" if (r.get("pct") or 0) >= 0 else "red"
                fig_cli.add_annotation(
                    x=r["mes_str"], y=1.02, xref="x", yref="paper",
                    text=txt, showarrow=False, font=dict(size=12, color=color), yanchor="bottom"
                )

    # 2) Tipo de Integração
    def _tipo_from_assunto(s: str) -> str:
        s = (s or "").strip().lower()
        if "cliente novo" in s: return "Cliente novo"
        if "alteração" in s and "plataforma" in s: return "Alteração de plataforma"
        if "conta filho" in s: return "Conta filho"
        return "Outros"

    tipo_counts = (
        df_onb.assign(tipo=df_onb["assunto_nome"].map(_tipo_from_assunto))
              .groupby("tipo").size().rename("Qtd").reset_index()
    )
    priority = ["Cliente novo","Outros","Alteração de plataforma","Conta filho"]
    tipo_counts["ord"] = tipo_counts["tipo"].apply(lambda x: priority.index(x) if x in priority else len(priority)+1)
    tipo_counts = tipo_counts.sort_values(["ord","Qtd"], ascending=[True, False])

    fig_tipo = px.bar(tipo_counts, x="Qtd", y="tipo", orientation="h", text="Qtd", title="Tipo de Integração", height=420)
    fig_tipo.update_traces(textposition="outside", cliponaxis=False)
    fig_tipo.update_layout(margin=dict(l=10, r=90, t=45, b=10))
    try:
        _max_q = float(tipo_counts["Qtd"].max())
        fig_tipo.update_xaxes(range=[0, _max_q * 1.12])
    except Exception:
        pass

    if "fig_cli" in locals():
        show_plot(fig_cli, "onb_cli_novo", "INT", ano_global, mes_global)
    else:
        st.info("Sem dados para 'Cliente novo' com os filtros atuais.")
    show_plot(fig_tipo, "onb_tipo_int", "INT", ano_global, mes_global)

    if not df_erros.empty:
        cont_erros = (df_erros["assunto_nome"].value_counts()
                      .reindex(ASSUNTOS_ERROS, fill_value=0).reset_index())
        cont_erros.columns = ["Categoria","Qtd"]

        fig_onb = px.bar(cont_erros, x="Qtd", y="Categoria", orientation="h",
                         text="Qtd", title="Erros Onboarding", height=420)
        fig_onb.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                              textfont_size=16, cliponaxis=False)
        max_q = int(cont_erros["Qtd"].max()) if not cont_erros.empty else 0
        if max_q > 0:
            fig_onb.update_xaxes(range=[0, max_q*1.25])
        fig_onb.update_layout(margin=dict(t=50, r=20, b=30, l=10), bargap=0.25)
        show_plot(fig_onb, "onboarding", "INT", ano_global, mes_global)

    st.markdown("---")
    st.subheader("💸 Dinheiro perdido (simulação)")
    c_left,c_right = st.columns([1,1])
    with c_left:
        st.number_input("Clientes novos (simulação)", value=possiveis_clientes, disabled=True, key="sim_clientes_onb")
    with c_right:
        receita_cliente = st.slider("Cenário Receita por Cliente (R$)",
                                    min_value=0, max_value=100000, step=500, value=20000,
                                    key="sim_receita_onb")
    dinheiro_perdido = float(possiveis_clientes) * float(receita_cliente)
    st.markdown(f"### **R$ {dinheiro_perdido:,.2f}**",
                help="Cálculo: Clientes novos (simulação) × Cenário Receita por Cliente")


# ---- Rotinas Manuais (TDS) — 100% por Jira
def render_rotinas_manuais(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    try:
        from unidecode import unidecode as _unidecode
    except Exception:
        _unidecode = lambda s: s

    def _canon(s: str) -> str:
        return " ".join(_unidecode(str(s or "")).lower().split())

    def _parse_dt_col(s):
        x = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        if x.notna().sum() == 0:
            x = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)
        return x

    def discover_tech_support_areas(df):
        if "area_nome" not in df.columns:
            return []
        areas = sorted({str(a) for a in df["area_nome"].dropna().unique()})
        tech = []
        for a in areas:
            c = _canon(a)
            if ("tech" in c and ("support" in c or "suporte" in c)) or \
               ("suporte" in c and ("tecnico" in c or "ti" in c)) or \
               c.startswith("tech support") or c.startswith("it suporte"):
                tech.append(a)
        if not tech:
            tech = [a for a in areas if "tech support" in _canon(a) or "suporte tecnico" in _canon(a)]
        return sorted(set(tech))

    OPS_AREAS = [
        "Ops - Conferência", "Ops - Cubagem", "Ops - Logística",
        "Ops - Coletas", "Ops - Expedição", "Ops - Divergências",
    ]

    MANUAL_TS_ASSUNTOS = [
        "Volumetria - Tabela Divergência",
        "Volumetria - Tabela Erro",
        "Volumetria - Cotação/Grafana",
        "Volumetria - IE / Qliksense",
        "Volumetria - Painel sem registro",
    ]

    MANUAL_TS_AREAS_EXTRA = ["Suporte - Infra", "Outra / Não Encontrada"]

    SUBJECT_GROUPS = {
        "volumetria - ie / qliksense": "Inscrição Estadual",
        "erro no processamento - inscricao estadual": "Inscrição Estadual",
        "erro no processamento - cte": "CTE",
        "volumetria - tabela erro": "CTE",
        "volumetria - tabela divergencia": "Divergência",
        "volumetria - cotação/grafana": "Cotação",
        "volumetria - cotacao/grafana": "Cotação",
        "volumetria - painel sem registro": "Outros",
    }

    st.markdown("### 🛠️ Rotinas Manuais")

    if dfp.empty:
        st.info("Sem tickets para o período.")
        return

    df = dfp.copy()
    df = ensure_assunto_nome(df, "TDS")
    df["area_nome"] = df["area"].apply(lambda x: safe_get_value(x, "value"))

    df["qtd_encomendas"] = df[CAMPO_QTD_ENCOMENDAS].apply(parse_qtd_encomendas)
    df = df[df["qtd_encomendas"] > 0].copy()
    if df.empty:
        st.info("Sem tickets com 'Quantidade de encomendas' > 0.")
        return

    df["resolved"] = _parse_dt_col(df.get("resolved"))
    df["created"]  = _parse_dt_col(df.get("created"))
    df["updated"]  = _parse_dt_col(df.get("updated"))

    df["_best_dt"] = df["resolved"]
    m = df["_best_dt"].isna() & df["created"].notna()
    df.loc[m, "_best_dt"] = df.loc[m, "created"]
    m = df["_best_dt"].isna() & df["updated"].notna()
    df.loc[m, "_best_dt"] = df.loc[m, "updated"]
    df = df[df["_best_dt"].notna()].copy()

    df = (
        df.sort_values(["key", "_best_dt"])
          .groupby("key", as_index=False)
          .agg({
              "_best_dt": "min",
              "qtd_encomendas": "max",
              "assunto_nome": "first",
              "summary": "first",
              "area_nome": "first",
          })
          .rename(columns={"_best_dt": "resolved"})
          .copy()
    )

    # ✅ Respeita DATA_INICIO em Rotinas Manuais (pelo resolved)
    _dt_inicio = pd.to_datetime(DATA_INICIO)
    df = df[df["resolved"].notna() & (df["resolved"] >= _dt_inicio)].copy()
    df["mes_dt"] = df["resolved"].dt.to_period("M").dt.to_timestamp()
    df["assunto_nome"] = df["assunto_nome"].astype(str)
    df["assunto_canon"] = df["assunto_nome"].apply(_canon)

    tech_areas_auto = discover_tech_support_areas(df)
    extras_canon = {_canon(a) for a in MANUAL_TS_AREAS_EXTRA}
    tech_areas = sorted(
        set(tech_areas_auto) |
        {a for a in df["area_nome"].dropna().unique() if _canon(a) in extras_canon}
    )

    base_ops = df[df["area_nome"].isin(OPS_AREAS)].copy()
    base_ts  = df[df["area_nome"].isin(tech_areas)].copy()
    if base_ops.empty and base_ts.empty:
        st.info("Sem tickets nas áreas Ops/Tech Support para os filtros atuais.")
        return

    full_ops = base_ops.copy()
    full_ts  = base_ts.copy()

    if not base_ops.empty:
        base_ops = aplicar_filtro_global(base_ops, "mes_dt", ano_global, mes_global)
    if not base_ts.empty:
        base_ts  = aplicar_filtro_global(base_ts,  "mes_dt", ano_global, mes_global)

    if base_ops.empty and base_ts.empty:
        st.info("Sem dados para exibir com os filtros atuais.")
        return

    assuntos_canon = {_canon(a) for a in MANUAL_TS_ASSUNTOS}

    monthly_tds = (
        base_ops.groupby("mes_dt")["qtd_encomendas"].sum().rename("Encomendas TDS")
        if not base_ops.empty else pd.Series(dtype=float, name="Encomendas TDS")
    )

    if not base_ts.empty and assuntos_canon:
        ts_mask_manual = base_ts["assunto_canon"].isin(assuntos_canon)
        ts_manuais = base_ts[ts_mask_manual].copy()
        monthly_manual = (
            ts_manuais.groupby("mes_dt")["qtd_encomendas"].sum().rename("Encomendas manuais")
        )
    else:
        ts_manuais = base_ts.iloc[0:0].copy()
        monthly_manual = pd.Series(dtype=float, name="Encomendas manuais")

    min_m = pd.concat([
        base_ops["mes_dt"] if not base_ops.empty else pd.Series(dtype="datetime64[ns]"),
        base_ts["mes_dt"]  if not base_ts.empty  else pd.Series(dtype="datetime64[ns]")
    ]).min()
    max_m = pd.concat([
        base_ops["mes_dt"] if not base_ops.empty else pd.Series(dtype="datetime64[ns]"),
        base_ts["mes_dt"]  if not base_ts.empty  else pd.Series(dtype="datetime64[ns]")
    ]).max()
    _dt_inicio = pd.to_datetime(DATA_INICIO)
    min_m = max(min_m, _dt_inicio)
    idx = pd.date_range(min_m, max_m, freq="MS")

    s_tds    = monthly_tds.reindex(idx, fill_value=0.0)
    s_manual = monthly_manual.reindex(idx, fill_value=0.0)

    monthly = pd.concat([
        s_manual.rename("Encomendas manuais"),
        s_tds.rename("Encomendas TDS")
    ], axis=1).reset_index().rename(columns={"index": "mes_dt"})
    monthly["mes_str"] = monthly["mes_dt"].dt.strftime("%b/%Y")

    fig = px.bar(
        monthly,
        x="mes_str",
        y=["Encomendas manuais", "Encomendas TDS"],
        barmode="group",
        text_auto=True,
        title="Encomendas manuais vs Encomendas TDS",
        height=420,
    )
    fig.update_traces(textangle=0, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=monthly["mes_str"].tolist())
    show_plot(fig, "rotinas_manual_ts_assunto_vs_tds_ops_extras", "TDS", ano_global, mes_global)

    total_tds    = float(s_tds.sum())
    total_manual = float(s_manual.sum())

    df_donut = pd.DataFrame(
        {"tipo": ["Encomendas manuais", "Encomendas TDS"],
         "qtd":  [total_manual, total_tds]}
    )
    fig_donut = px.pie(
        df_donut, values="qtd", names="tipo", hole=0.6,
        title="Participação — Manuais vs TDS"
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    show_plot(fig_donut, "rotinas_manual_ts_assunto_donut_vs_tds_ops_extras", "TDS", ano_global, mes_global)

    if not ts_manuais.empty:
        def _map_subject(canon_name: str) -> str:
            return SUBJECT_GROUPS.get(canon_name, SUBJECT_GROUPS.get("volumetria - painel sem registro", "Outros"))

        ts_manuais["assunto_grupo"] = ts_manuais["assunto_canon"].apply(_map_subject)
        df_ass = (
            ts_manuais.groupby("assunto_grupo")["qtd_encomendas"]
            .sum()
            .reset_index()
            .sort_values("qtd_encomendas", ascending=True)
        )

        fig_ass = px.bar(
            df_ass, x="qtd_encomendas", y="assunto_grupo",
            orientation="h", text="qtd_encomendas",
            title="Manual: Assuntos relacionados", height=380
        )
        fig_ass.update_traces(textposition="outside")
        fig_ass.update_layout(yaxis_title="", xaxis_title="Qtd")
        show_plot(fig_ass, "rotinas_manual_por_assunto", "TDS", ano_global, mes_global)

    with st.expander("📤 Exportar / diagnóstico", expanded=False):
        def _prep_export(dd: pd.DataFrame, origem: str, somente_assuntos: bool = False) -> pd.DataFrame:
            if dd.empty:
                return pd.DataFrame(columns=["key","resolved","mes_dt","area_nome","assunto_nome","summary","qtd_encomendas","origem"])
            tmp = dd.copy()
            tmp["assunto_canon"] = tmp["assunto_nome"].apply(_canon)
            if somente_assuntos and len(assuntos_canon) > 0:
                tmp = tmp[tmp["assunto_canon"].isin(assuntos_canon)].copy()
            tmp["origem"] = origem
            return tmp[["key","resolved","mes_dt","area_nome","assunto_nome","summary","qtd_encomendas","origem"]]

        exp_ops = _prep_export(full_ops, "Ops (TDS total)")
        exp_ts  = _prep_export(full_ts,  "TS + extras (manuais | assunto)", somente_assuntos=True)
        df_export = pd.concat([exp_ops, exp_ts], ignore_index=True).sort_values("resolved")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tickets únicos (Ops + TS+extras)", int(df_export["key"].nunique()))
        c2.metric("Soma TDS (Ops)", int(total_tds))
        c3.metric("Soma Manuais (TS+extras | assunto)", int(total_manual))

        csv = df_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV", data=csv, file_name="rotinas_ops_tds_vs_manual_ts_extras_por_assunto.csv", mime="text/csv")
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

# ================= Coleta de dados (CORRIGIDO) ========================

def jql_projeto(project_key: str, ano_sel: str, mes_sel: str) -> str:
    base = f'project = "{project_key}"'

    # ✅ Sempre respeita DATA_INICIO para o recorte do painel, mas sem perder tickets
    # fechados depois (mesmo que criados antes).
    if ano_sel != "Todos" and mes_sel != "Todos":
        a = int(ano_sel)
        m = int(mes_sel)
        ini = date(a, m, 1)
        fim = date(a + 1, 1, 1) if m == 12 else date(a, m + 1, 1)
        base += (
            f' AND ('
            f' (created >= "{ini:%Y-%m-%d}" AND created < "{fim:%Y-%m-%d}")'
            f' OR ' 
            f' (resolutiondate >= "{ini:%Y-%m-%d}" AND resolutiondate < "{fim:%Y-%m-%d}")'
            f' OR ' 
            f' (statusCategoryChangedDate >= "{ini:%Y-%m-%d}" AND statusCategoryChangedDate < "{fim:%Y-%m-%d}")'
            f' OR ' 
            f' (statusCategory = Done AND updated >= "{ini:%Y-%m-%d}" AND updated < "{fim:%Y-%m-%d}")'
            f' )'
        )
    else:
        base += (
            f' AND ('
            f' created >= "{DATA_INICIO}"'
            f' OR resolutiondate >= "{DATA_INICIO}"'
            f' OR statusCategoryChangedDate >= "{DATA_INICIO}"'
            f' OR (statusCategory = Done AND updated >= "{DATA_INICIO}")'
            f' )'
        )

    return base + " ORDER BY created DESC"

# Agora passamos os estados globais para a JQL para que a busca seja cirúrgica
JQL_TDS = jql_projeto("TDS", ano_global, mes_global)
JQL_INT = jql_projeto("INT", ano_global, mes_global)
JQL_TINE = jql_projeto("TINE", ano_global, mes_global)
JQL_INTEL = jql_projeto("INTEL", ano_global, mes_global)

def _get_or_fetch(proj: str, jql: str):
    key = f"df_{proj}"
    if key in st.session_state and isinstance(st.session_state.get(key), pd.DataFrame):
        return st.session_state[key]
    with st.spinner(f"Carregando {proj}..."):
        dfp = buscar_issues(proj, jql)
    st.session_state[key] = dfp
    return dfp

df_tds   = _get_or_fetch("TDS",   JQL_TDS)
df_int   = _get_or_fetch("INT",   JQL_INT)
df_tine  = _get_or_fetch("TINE",  JQL_TINE)
df_intel = _get_or_fetch("INTEL", JQL_INTEL)

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
        if "closed_dt" in dfp.columns:
            dfp["mes_closed"] = pd.to_datetime(dfp["closed_dt"], errors="coerce")
        dfp = ensure_assunto_nome(dfp, projeto)

        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")

        if visao == "Criados vs Resolvidos":
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
        elif visao == "SLA":
            render_sla(dfp, _df_monthly_all, projeto, ano_global, mes_global)
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
            render_sla(dfp, _df_monthly_all, projeto, ano_global, mes_global)
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
