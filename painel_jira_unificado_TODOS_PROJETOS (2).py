
import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from sla_utils import extrair_sla_millis
import plotly.graph_objects as go

st.set_page_config(layout='wide')
st.title('Painel de Indicadores - Todos os Projetos')

JIRA_URL = st.secrets['JIRA_URL']
EMAIL = st.secrets['EMAIL']
TOKEN = st.secrets['TOKEN']
auth = HTTPBasicAuth(EMAIL, TOKEN)

PROJETOS = ['TDS', 'INT', 'TINE', 'INTEL']
CUTOFF_DATE = '2024-01-01'

@st.cache_data(show_spinner=True)
def buscar_issues():
    headers = {"Accept": "application/json"}
    all_issues = []
    for projeto in PROJETOS:
        start_at = 0
        while True:
            jql = f'project = "{projeto}" AND created >= "{CUTOFF_DATE}" ORDER BY created ASC'
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": 100,
                "fields": "summary,created,resolutiondate,status,customfield_13686,customfield_13719,customfield_13712,customfield_13643,customfield_13699,customfield_13659"
            }
            res = requests.get(f"{JIRA_URL}/rest/api/3/search", headers=headers, auth=auth, params=params)
            if res.status_code != 200:
                break
            data = res.json()
            issues = data.get("issues", [])
            if not issues:
                break
            for issue in issues:
                f = issue["fields"]
                sla_millis = extrair_sla_millis(f.get("customfield_13686", {}))
                all_issues.append({
                    "projeto": projeto,
                    "created": f["created"],
                    "resolved": f.get("resolutiondate"),
                    "status": f.get("status", {}).get("name", ""),
                    "sla_millis": sla_millis,
                    "customfield_13719": f.get("customfield_13719"),
                    "customfield_13712": f.get("customfield_13712"),
                    "customfield_13643": f.get("customfield_13643"),
                    "customfield_13699": f.get("customfield_13699"),
                    "customfield_13659": f.get("customfield_13659")
                })
            start_at += 100
    return pd.DataFrame(all_issues)

df = buscar_issues()
df['created'] = pd.to_datetime(df['created'])
df['resolved'] = pd.to_datetime(df['resolved'])
df['mes_str'] = df['created'].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")
df['escalado_n3_valor'] = df['customfield_13659'].apply(lambda x: x.get('value') if isinstance(x, dict) else None)

aba1, aba2, aba3, aba4 = st.tabs(['Visão Geral', 'Área Solicitante', 'Assunto Relacionado', 'Encaminhamentos'])

with aba1:
    st.subheader("Criados vs Resolvidos por Mês")
    criados = df.groupby('mes_str')['created'].count().reset_index(name='Criados')
    resolvidos = df[df['resolved'].notna()].groupby('mes_str')['resolved'].count().reset_index(name='Resolvidos')
    grafico = pd.merge(criados, resolvidos, how="outer", on="mes_str").fillna(0).sort_values("mes_str")
    fig = go.Figure()
    fig.add_bar(x=grafico['mes_str'], y=grafico['Criados'], name='Criados')
    fig.add_bar(x=grafico['mes_str'], y=grafico['Resolvidos'], name='Resolvidos')
    fig.update_layout(barmode='group', xaxis_title='Mês', yaxis_title='Chamados')
    st.plotly_chart(fig, use_container_width=True)

with aba2:
    st.subheader("Área Solicitante por Mês")
    if 'customfield_13719' in df.columns:
        df_area = df.dropna(subset=['resolved', 'customfield_13719']).copy()
        df_area['mes_resolvido'] = df_area['resolved'].dt.to_period("M").dt.to_timestamp()
        df_area['mes_str'] = df_area['mes_resolvido'].dt.strftime("%b/%Y")
        df_area['area'] = df_area['customfield_13719'].apply(lambda x: x.get('value') if isinstance(x, dict) else str(x))
        meses = sorted(df_area['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
        mes_sel = st.selectbox("Selecione o mês", meses, index=len(meses)-1, key="area")
        dados_filtrados = df_area[df_area['mes_str'] == mes_sel]
        dados_area = dados_filtrados['area'].value_counts().reset_index()
        dados_area.columns = ['Área', 'Qtd. Chamados']
        fig_area = go.Figure(go.Bar(
            y=dados_area['Área'],
            x=dados_area['Qtd. Chamados'],
            orientation='h',
            text=dados_area['Qtd. Chamados'],
            textposition='outside',
            marker_color='skyblue',
        ))
        fig_area.update_layout(title=f'Tickets por Área - {mes_sel}', xaxis_title='Qtd. Chamados')
        st.plotly_chart(fig_area, use_container_width=True)

with aba3:
    st.subheader("Assunto Relacionado por Projeto e Mês")
    campos_assunto = {
        'TDS': 'customfield_13712',
        'INT': 'customfield_13643',
        'TINE': 'customfield_13699'
    }
    projeto = st.selectbox("Selecione o projeto", list(campos_assunto.keys()))
    campo_assunto = campos_assunto[projeto]
    df_assunto = df[df['projeto'] == projeto].copy()
    df_assunto['mes_str'] = df_assunto['created'].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")
    df_assunto['assunto'] = df_assunto[campo_assunto].apply(lambda x: x.get('value') if isinstance(x, dict) else str(x))
    meses_assunto = sorted(df_assunto['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
    mes_assunto = st.selectbox("Selecione o mês", meses_assunto, index=len(meses_assunto)-1, key="assunto")
    dados_assunto = df_assunto[df_assunto['mes_str'] == mes_assunto]['assunto'].value_counts().reset_index()
    dados_assunto.columns = ['Assunto Relacionado', 'Qtd. Chamados']
    fig_assunto = go.Figure(go.Bar(
        y=dados_assunto['Assunto Relacionado'],
        x=dados_assunto['Qtd. Chamados'],
        orientation='h',
        text=dados_assunto['Qtd. Chamados'],
        textposition='outside',
        marker_color='mediumpurple'
    ))
    fig_assunto.update_layout(title=f'Assunto Relacionado - {projeto} | {mes_assunto}', xaxis_title='Qtd. Chamados')
    st.plotly_chart(fig_assunto, use_container_width=True)

with aba4:
    st.subheader("Encaminhamentos por Projeto e Mês")
    for projeto_nome in ['TDS', 'INT']:
        st.markdown(f"### Projeto: {projeto_nome}")
        df_proj = df[df['projeto'] == projeto_nome]
        meses_disp = sorted(df_proj['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
        mes_selecionado = st.selectbox(f"Selecione o mês ({projeto_nome})", meses_disp, index=len(meses_disp)-1, key=f"mes_{projeto_nome}")
        df_mes = df_proj[df_proj['mes_str'] == mes_selecionado]

        col1, col2 = st.columns(2)
        with col1:
            count_produto = df_mes['status'].str.contains("Produto", case=False, na=False).sum()
            st.metric("Encaminhados Produto", count_produto)
        with col2:
            count_n3 = df_mes['escalado_n3_valor'] == "Sim"
            st.metric("Encaminhados N3", count_n3.sum())
