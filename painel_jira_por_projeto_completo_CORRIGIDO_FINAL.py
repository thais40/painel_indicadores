# === BLOCO A: Imports + Config + Jira helpers + jql_projeto ===
from __future__ import annotations

import os
import json
import math
from datetime import date, datetime
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st

# ----------------- Config Jira via st.secrets -----------------
JIRA_URL = st.secrets["JIRA_URL"].rstrip("/")
EMAIL    = st.secrets["EMAIL"]
TOKEN    = st.secrets["TOKEN"]

# Data mínima global
DATA_INICIO = "2024-01-01"

# ----------------- Campos que queremos do Jira ----------------
# Adapte a lista conforme os painéis que você já tem
JIRA_FIELDS = [
    "summary",
    "created",
    "updated",
    "resolutiondate",
    # SLA resolução (SUP) — informe o customfield correto por projeto se necessário
    "customfield_13744",
    # Origem do problema (se usar em APP NE/EN)
    "customfield_13628",
    # Quantidade de encomendas (Rotinas Manuais)
    "customfield_13666",
    # Outros campos que seu arquivo já usa...
]

# ----------------- Nova chamada (API v3 /search/jql) -----------------
def _jira_post_search_jql(jql: str, start_at: int = 0, max_results: int = 100) -> Dict[str, Any]:
    """
    Usa a API nova do Jira: POST /rest/api/3/search/jql
    """
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    headers = {"Content-Type": "application/json"}
    payload = {
        "jql": jql,
        "startAt": start_at,
        "maxResults": max_results,
        "fields": JIRA_FIELDS,
    }

    r = requests.post(url, headers=headers, auth=(EMAIL, TOKEN), data=json.dumps(payload))
    if r.status_code >= 400:
        raise RuntimeError(f"Erro Jira {r.status_code}: {r.text[:400]}")
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def jira_search_all(jql: str) -> List[Dict[str, Any]]:
    """
    Busca todas as páginas de issues para um JQL.
    """
    issues: List[Dict[str, Any]] = []
    start = 0
    page = _jira_post_search_jql(jql, start_at=start, max_results=100)
    total = page.get("total", 0)
    issues.extend(page.get("issues", []))

    while len(issues) < total:
        start = len(issues)
        page = _jira_post_search_jql(jql, start_at=start, max_results=100)
        issues.extend(page.get("issues", []))

    return issues

def normalize_issues(raw_issues: List[Dict[str, Any]], project_key: str) -> pd.DataFrame:
    """
    Converte issues em DataFrame e normaliza campos usados nos seus gráficos.
    (Mantém o resto do seu pipeline igual.)
    """
    if not raw_issues:
        return pd.DataFrame()

    rows = []
    for it in raw_issues:
        f = it.get("fields", {}) or {}
        rows.append({
            "key": it.get("key"),
            "project": project_key,
            "summary": f.get("summary"),
            "created": f.get("created"),
            "updated": f.get("updated"),
            "resolutiondate": f.get("resolutiondate"),
            # SLA (ex.: customfield_13744) — mantenha como string/numero conforme você já trata depois
            "sla_sup": f.get("customfield_13744"),
            # Origem do problema
            "origem_problema": f.get("customfield_13628"),
            # Quantidade de encomendas (Rotinas Manuais)
            "qtd_encomendas": f.get("customfield_13666"),
        })

    df = pd.DataFrame(rows)
    # Datas para datetime
    for col in ["created", "updated", "resolutiondate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Campo numérico de encomendas (se vier string)
    if "qtd_encomendas" in df.columns:
        df["qtd_encomendas"] = pd.to_numeric(df["qtd_encomendas"], errors="coerce")

    # Month label (mes_str) — seus gráficos usam isso
    if "created" in df.columns:
        df["mes_str"] = df["created"].dt.strftime("%b/%Y")

    return df

# ----------------- Montagem robusta de JQL -----------------
def jql_projeto(project_key: str, ano_selecionado=None, mes_selecionado=None) -> str:
    """
    Monta um JQL seguro e com recorte:
      - Sempre traz a partir de DATA_INICIO
      - Se Mês != 'Todos': corta até 1º dia do mês seguinte (created < next_month)
      - Se Mês == 'Todos': não corta (traz até hoje)
      - Protege o nome do projeto com aspas (ex.: "INT") para evitar conflito com palavra reservada
    """
    if ano_selecionado is None:
        ano_selecionado = "Todos"
    if mes_selecionado is None:
        mes_selecionado = "Todos"

    base = f'project = "{project_key}" AND created >= "{DATA_INICIO}"'

    if mes_selecionado != "Todos" and ano_selecionado != "Todos":
        try:
            a = int(ano_selecionado)
            m = int(mes_selecionado)
            # 1º dia do mês seguinte
            if m == 12:
                next_month_first = date(a + 1, 1, 1)
            else:
                next_month_first = date(a, m + 1, 1)
            base += f' AND created < "{next_month_first:%Y-%m-%d}"'
        except Exception:
            # Se algo vier inválido, não aplica corte final
            pass

    # Ordenação padrão
    return base + " ORDER BY created ASC"
# === BLOCO B: Header + Filtros Globais + JQL + Coleta ===

st.markdown("## 📊 Painel de Indicadores — Jira")

# Botão de atualizar cache
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🔄 Atualizar dados"):
        st.cache_data.clear()
        st.rerun()
with c2:
    agora_brt = pd.Timestamp.utcnow().tz_convert("America/Sao_Paulo").strftime("%d/%m/%Y %H:%M:%S")
    st.caption(f"🕒 Última atualização (BRT): {agora_brt}")

# --------- Filtros Globais (devem vir ANTES de montar JQL) ----------
col_ano, col_mes = st.columns(2)

anos_opcoes = ["Todos"] + [str(y) for y in range(2024, date.today().year + 1)]
meses_opcoes = ["Todos"] + [f"{m:02d}" for m in range(1, 12 + 1)]

with col_ano:
    # Mantém o padrão "Todos" se quiser sempre ver tudo até hoje
    ano_global = st.selectbox("Ano (global)", anos_opcoes, index=anos_opcoes.index(str(date.today().year)))
with col_mes:
    mes_global = st.selectbox("Mês (global)", meses_opcoes, index=0)  # "Todos"

# --------- Monta os JQLs (somente AGORA, após filtros) ----------
JQL_TDS   = jql_projeto("TDS",   ano_global, mes_global)
JQL_INT   = jql_projeto("INT",   ano_global, mes_global)
JQL_TINE  = jql_projeto("TINE",  ano_global, mes_global)
JQL_INTEL = jql_projeto("INTEL", ano_global, mes_global)

# --------- Baixa e normaliza os dados ----------
with st.spinner("Carregando TDS..."):
    df_tds  = normalize_issues(jira_search_all(JQL_TDS),  "TDS")
with st.spinner("Carregando INT..."):
    df_int  = normalize_issues(jira_search_all(JQL_INT),  "INT")
with st.spinner("Carregando TINE..."):
    df_tine = normalize_issues(jira_search_all(JQL_TINE), "TINE")
with st.spinner("Carregando INTEL..."):
    df_intel = normalize_issues(jira_search_all(JQL_INTEL), "INTEL")

# ===================
# Filtros Globais UI
# ===================
st.markdown("### 🔍 Filtros Globais")
if all(d.empty for d in [df_tds, df_int, df_tine, df_intel]):
    st.warning("Sem dados do Jira em nenhum projeto.")
    st.stop()

anos_glob = sorted(pd.Series(
    pd.concat([d["mes_created"] for d in [df_tds, df_int, df_tine, df_intel] if not d.empty])
      .dt.year.dropna().unique()
).astype(int).tolist())

meses_glob = list(range(1,13))

c1, c2 = st.columns(2)
with c1:
    ano_global = st.selectbox("Ano (global)", ["Todos"] + [str(a) for a in anos_glob], key="ano_global")
with c2:
    mes_global = st.selectbox("Mês (global)", ["Todos"] + [f"{m:02d}" for m in meses_glob], key="mes_global")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

# =====================
# Pré-agregados mensais
# =====================
@st.cache_data(show_spinner=False)
def build_monthly_tables(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    base = df_all.copy()
    base["per_created"]  = base["created"].dt.to_period("M")
    base["per_resolved"] = base["resolved"].dt.to_period("M")

    # Criados por mês (created)
    created = (base.groupby(["projeto","per_created"])
                    .size().reset_index(name="Criados")
                    .rename(columns={"per_created":"period"}))

    # Resolvidos por mês (resolved) + SLA (JSM)
    res = base[base["resolved"].notna()].copy()
    res["dentro_sla"] = res["sla_raw"].apply(dentro_sla_from_raw).fillna(False)
    resolved = (res.groupby(["projeto","per_resolved"])
                  .agg(Resolvidos=("key","count"), Dentro=("dentro_sla","sum"))
                  .reset_index()
                  .rename(columns={"per_resolved":"period"}))

    monthly = pd.merge(created, resolved, how="outer", on=["projeto","period"]).fillna(0)
    monthly["period"]   = monthly["period"].astype("period[M]")
    monthly["period_ts"]= monthly["period"].dt.to_timestamp()
    monthly["ano"]      = monthly["period"].dt.year.astype(int)
    monthly["mes"]      = monthly["period"].dt.month.astype(int)
    monthly["mes_str"]  = monthly["period_ts"].dt.strftime("%b/%Y")

    monthly["Dentro"]     = monthly["Dentro"].astype(int)
    monthly["Resolvidos"] = monthly["Resolvidos"].astype(int)
    monthly["Fora"]       = (monthly["Resolvidos"] - monthly["Dentro"]).clip(lower=0)

    # Percentuais com 2 casas
    monthly["pct_dentro"] = monthly.apply(
        lambda r: (r["Dentro"]/r["Resolvidos"]*100) if r["Resolvidos"]>0 else 0.0, axis=1
    ).round(2)
    monthly["pct_fora"] = monthly.apply(
        lambda r: (r["Fora"]/r["Resolvidos"]*100) if r["Resolvidos"]>0 else 0.0, axis=1
    ).round(2)

    return monthly.sort_values(["projeto","period"])

df_monthly_all = pd.concat(
    [build_monthly_tables(d) for d in [df_tds, df_int, df_tine, df_intel] if not d.empty],
    ignore_index=True
) if not all(d.empty for d in [df_tds, df_int, df_tine, df_intel]) else pd.DataFrame()
# ===================
# Renderizadores base
# ===================
def aplicar_filtro_global(df_in: pd.DataFrame, col_dt: str, ano: str, mes: str) -> pd.DataFrame:
    out = df_in.copy()
    if ano != "Todos":
        out = out[out[col_dt].dt.year == int(ano)]
    if mes != "Todos":
        out = out[out[col_dt].dt.month == int(mes)]
    return out

def ensure_assunto_nome(df_proj: pd.DataFrame, projeto: str) -> pd.DataFrame:
    if "assunto_nome" not in df_proj.columns:
        if CAMPOS_ASSUNTO[projeto] == "issuetype":
            df_proj["assunto_nome"] = df_proj["issuetype"].apply(lambda x: safe_get_value(x, "name"))
        else:
            df_proj["assunto_nome"] = df_proj["assunto"].apply(lambda x: safe_get_value(x, "value"))
    return df_proj

def render_criados_resolvidos(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
    st.markdown("### 📈 Tickets Criados vs Resolvidos")
    dfm = df_monthly_all[df_monthly_all["projeto"] == projeto].copy()
    if dfm.empty:
        st.info("Sem dados para esta visão.")
        return
    if ano_global != "Todos":
        dfm = dfm[dfm["ano"] == int(ano_global)]
    if mes_global != "Todos":
        dfm = dfm[dfm["mes"] == int(mes_global)]
    if ano_global != "Todos" and mes_global != "Todos":
        alvo = pd.Period(f"{int(ano_global)}-{int(mes_global):02d}", freq="M")
        dfm = dfm[dfm["period"] == alvo]
    dfm = dfm.sort_values("period_ts")
    show = dfm[["mes_str","period_ts","Criados","Resolvidos"]].copy()
    show["Criados"] = show["Criados"].astype(int)
    show["Resolvidos"] = show["Resolvidos"].astype(int)
    fig = px.bar(show, x="mes_str", y=["Criados","Resolvidos"], barmode="group", text_auto=True, height=440)
    fig.update_traces(textangle=0, textfont_size=14, cliponaxis=False)
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    st.plotly_chart(fig, use_container_width=True)

def render_sla(dfp: pd.DataFrame, projeto: str, ano_global: str, mes_global: str):
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
    show = dfm[["mes_str","period_ts","pct_dentro","pct_fora"]].sort_values("period_ts")
    show = show.rename(columns={"pct_dentro": "% Dentro SLA", "pct_fora":"% Fora SLA"})
    fig = px.bar(show, x="mes_str", y=["% Dentro SLA","% Fora SLA"], barmode="group",
                 title=titulo, color_discrete_map={"% Dentro SLA":"green","% Fora SLA":"red"}, height=440)
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", textfont_size=14, cliponaxis=False)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(categoryorder="array", categoryarray=show["mes_str"].tolist())
    st.plotly_chart(fig, use_container_width=True)

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
    assunto_count.columns = ["Assunto","Qtd"]
    st.dataframe(assunto_count, use_container_width=True, hide_index=True)

def render_area(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📦 Área Solicitante")
    df_area = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_area.empty:
        st.info("Sem dados para Área Solicitante nos filtros atuais.")
        return
    df_area["area_nome"] = df_area["area"].apply(lambda x: safe_get_value(x, "value"))
    area_count = df_area["area_nome"].value_counts().reset_index()
    area_count.columns = ["Área","Qtd"]
    st.dataframe(area_count, use_container_width=True, hide_index=True)

def render_encaminhamentos(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🔄 Encaminhamentos")
    df_enc = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
    if df_enc.empty:
        st.info("Sem dados para Encaminhamentos nos filtros atuais.")
        return
    col1,col2 = st.columns(2)
    with col1:
        count_prod = df_enc["status"].astype(str).str.contains("Produto", case=False, na=False).sum()
        st.metric("Encaminhados Produto", int(count_prod))
    with col2:
        df_enc["n3_valor"] = df_enc["n3"].apply(lambda x: safe_get_value(x, "value", None))
        st.metric("Encaminhados N3", int((df_enc["n3_valor"] == "Sim").sum()))
def render_app_ne(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 📱 APP NE — Origem do problema")
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
    fig_app = px.bar(serie, x="mes_str", y="Qtd", color="origem_cat", barmode="group",
                     title="APP NE — Volumes por mês e Origem do problema",
                     color_discrete_map={"APP NE":"#2ca02c","APP EN":"#1f77b4","Outros/Não informado":"#9ca3af"},
                     text="Qtd", height=460)
    fig_app.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
    max_qtd = int(serie["Qtd"].max()) if not serie.empty else 0
    if max_qtd > 0:
        fig_app.update_yaxes(range=[0, max_qtd * 1.25])
    fig_app.update_layout(yaxis_title="Qtd", xaxis_title="Mês",
                          uniformtext_minsize=14, uniformtext_mode="show",
                          bargap=0.15, margin=dict(t=70, r=20, b=60, l=50))
    st.plotly_chart(fig_app, use_container_width=True)

def render_rotinas_manuais(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🛠️ Rotinas Manuais")
    if dfp.empty:
        st.info("Sem dados do Jira para Rotinas Manuais.")
        return
    base = dfp[["key","resolved",CAMPO_QTD_ENCOMENDAS,"summary"]].copy()
    base = base[base["summary"].astype(str).str.Contains(TITULO_ROTINA, case=False, na=False)] if hasattr(pd.Series, "Contains") else base[base["summary"].astype(str).str.contains(TITULO_ROTINA, case=False, na=False)]
    if base.empty:
        st.info("Nenhum ticket de Rotinas Manuais encontrado (por título).")
        return
    base[CAMPO_QTD_ENCOMENDAS] = base[CAMPO_QTD_ENCOMENDAS].apply(parse_qtd_encomendas).astype("Int64")
    base["resolved"] = pd.to_datetime(base["resolved"], errors="coerce")
    base = base.dropna(subset=["resolved"])
    base = base[base[CAMPO_QTD_ENCOMENDAS].notna() & (base[CAMPO_QTD_ENCOMENDAS] > 0)]
    base = base.sort_values("resolved").drop_duplicates(subset=["key"], keep="last")
    base["period"]    = base["resolved"].dt.to_period("M")
    base["period_ts"] = base["period"].dt.to_timestamp()
    base["ano"]       = base["period"].dt.year
    base["mes"]       = base["period"].dt.month
    base["mes_str"]   = base["period_ts"].dt.strftime("%b/%Y")
    if ano_global != "Todos":
        base = base[base["ano"] == int(ano_global)]
    if mes_global != "Todos":
        if ano_global != "Todos":
            alvo = pd.Period(f"{int(ano_global)}-{int(mes_global):02d}", freq="M")
            base = base[base["period"] == alvo]
        else:
            base = base[base["mes"] == int(mes_global)]
    if base.empty:
        st.info("Sem dados de Rotinas Manuais no período filtrado.")
        return
    agg = (base.groupby(["period","period_ts","mes_str"], as_index=False)[CAMPO_QTD_ENCOMENDAS]
           .sum().rename(columns={CAMPO_QTD_ENCOMENDAS:"Qtd encomendas"}).sort_values("period_ts"))
    agg["label"] = agg["Qtd encomendas"].map(lambda x: f"{x:,.0f}".replace(",", "."))
    fig = px.bar(agg, x="mes_str", y="Qtd encomendas", text="label", height=420)
    fig.update_traces(textangle=0, textfont_size=14, cliponaxis=False)
    fig.update_yaxes(title_text="Qtd encomendas", tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("🔎 Tickets usados (amostra)"):
        st.dataframe(base[["key","summary","resolved",CAMPO_QTD_ENCOMENDAS]]
                     .sort_values("resolved"), use_container_width=True, hide_index=True)

def render_onboarding(dfp: pd.DataFrame, ano_global: str, mes_global: str):
    st.markdown("### 🧭 Onboarding")
    if dfp.empty:
        st.info("Sem dados de Onboarding.")
        return
    ASSUNTO_CLIENTE_NOVO = "Nova integração - Cliente novo"
    ASSUNTOS_ERROS = ["Erro durante Onboarding - Frete","Erro durante Onboarding - Pedido","Erro durante Onboarding - Rastreio","Erro durante Onboarding - Teste"]
    STATUS_PENDENCIAS = ["Aguardando informações adicionais","Em andamento","Aguardando pendências da Triagem","Aguardando validação do cliente","Aguardando Comercial"]
    dfp = ensure_assunto_nome(dfp.copy(), "INT")
    df_onb = aplicar_filtro_global(dfp.copy(), "mes_created", ano_global, mes_global)
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
    st.markdown("---")
    if not df_erros.empty:
        cont_erros = (df_erros["assunto_nome"].value_counts().reindex(ASSUNTOS_ERROS, fill_value=0).reset_index())
        cont_erros.columns = ["Categoria","Qtd"]
        fig_onb = px.bar(cont_erros, x="Qtd", y="Categoria", orientation="h", text="Qtd", title="Erros Onboarding", height=420)
        fig_onb.update_traces(texttemplate="%{text:.0f}", textposition="outside", textfont_size=16, cliponaxis=False)
        max_q = int(cont_erros["Qtd"].max()) if not cont_erros.empty else 0
        if max_q > 0:
            fig_onb.update_xaxes(range=[0, max_q*1.25])
        fig_onb.update_layout(margin=dict(t=50, r=20, b=30, l=10), bargap=0.25)
        st.plotly_chart(fig_onb, use_container_width=True)
    df_cli = df_onb[df_onb["assunto_nome"] == ASSUNTO_CLIENTE_NOVO].copy()
    if not df_cli.empty:
        serie_cli = (df_cli.groupby(df_cli["mes_created"].dt.to_period("M")).size().reset_index(name="ClientesNovos"))
        serie_cli["mes_dt"] = serie_cli["mes_created"].dt.to_timestamp()
        serie_cli = serie_cli.sort_values("mes_dt")
        serie_cli["mes_str"] = serie_cli["mes_dt"].dt.strftime("%Y %b")
        serie_cli["MoM"] = serie_cli["ClientesNovos"].pct_change() * 100
        fig_cli = px.bar(serie_cli, x="mes_str", y="ClientesNovos", title="Tickets - Cliente novo", text="ClientesNovos", height=380)
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
                    fig_cli.add_annotation(x=x, y=yb + (y_top*0.20), text=f"{arrow} {mom_abs:.0f}%", showarrow=False, font=dict(size=12, color=color))
        fig_cli.update_layout(margin=dict(t=50, r=20, b=35, l=40), bargap=0.18, xaxis_title=None, yaxis_title="ClientesNovos")
        st.plotly_chart(fig_cli, use_container_width=True)
    st.markdown("---")
    st.subheader("💸 Dinheiro perdido (simulação)")
    c_left,c_right = st.columns([1,1])
    with c_left:
        st.number_input("Clientes novos (simulação)", value=possiveis_clientes, disabled=True, key="sim_clientes_onb")
    with c_right:
        receita_cliente = st.slider("Cenário Receita por Cliente (R$)", min_value=0, max_value=100000, step=500, value=20000, key="sim_receita_onb")
    dinheiro_perdido = float(possiveis_clientes) * float(receita_cliente)
    st.markdown(f"### **R$ {dinheiro_perdido:,.2f}**", help="Cálculo: Clientes novos (simulação) × Cenário Receita por Cliente")

# ======================
# Abas por Projeto/Visão
# ======================
tabs = st.tabs([TITULOS[p] for p in PROJETOS])

for projeto, tab in zip(PROJETOS, tabs):
    with tab:
        st.subheader(f"📂 Projeto: {TITULOS[projeto]}")
        if projeto == "TDS":
            dfp = df_tds.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante","APP NE"]
        elif projeto == "INT":
            dfp = df_int.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante","Onboarding"]
        elif projeto == "TINE":
            dfp = df_tine.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado","Área Solicitante"]
        else:
            dfp = df_intel.copy()
            opcoes = ["Geral","Criados vs Resolvidos","SLA","Assunto Relacionado"]
        if dfp.empty:
            st.info("Sem dados carregados para este projeto.")
            continue
        dfp = ensure_assunto_nome(dfp, projeto)
        visao = st.selectbox("Visão", opcoes, key=f"visao_{projeto}")
        if visao == "Criados vs Resolvidos":
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
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
            render_criados_resolvidos(dfp, projeto, ano_global, mes_global)
            render_sla(dfp, projeto, ano_global, mes_global)
            render_assunto(dfp, projeto, ano_global, mes_global)
            if projeto != "INTEL":
                render_area(dfp, ano_global, mes_global)
            if projeto in ("TDS","INT"):
                render_encaminhamentos(dfp, ano_global, mes_global)
            if projeto == "TDS":
                render_app_ne(dfp, ano_global, mes_global)
                with st.expander("🛠️ Rotinas Manuais", expanded=False):
                    render_rotinas_manuais(dfp, ano_global, mes_global)
            if projeto == "INT":
                with st.expander("🧭 Onboarding", expanded=False):
                    render_onboarding(dfp, ano_global, mes_global)

# Rodapé
st.markdown("---")
st.caption("Feito com 💙 — layout completo preservado, com endpoint Jira /search/jql e paginação por nextPageToken.")
