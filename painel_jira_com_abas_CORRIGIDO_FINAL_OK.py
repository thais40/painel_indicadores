
import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from sla_utils import extrair_sla_millis
import plotly.graph_objects as go

st.set_page_config(layout='wide')
st.title('Painel de Indicadores')

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
                "fields": "summary,created,resolutiondate,status,customfield_13686,customfield_13719,customfield_13712,customfield_13643,customfield_13699"
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
                    "sla_millis": sla_millis,
                    "customfield_13719": f.get("customfield_13719"),
                    "customfield_13712": f.get("customfield_13712"),
                    "customfield_13643": f.get("customfield_13643"),
                    "customfield_13699": f.get("customfield_13699")
                })
            start_at += 100
    return pd.DataFrame(all_issues)

df = buscar_issues()
df['created'] = pd.to_datetime(df['created'])
df['resolved'] = pd.to_datetime(df['resolved'])

aba1, aba2, aba3 = st.tabs(['Resumo', 'Área Solicitante', 'Assunto Relacionado'])

with aba2:
    st.subheader("Área Solicitante por mês")
    if 'customfield_13719' in df.columns:
        df_area = df.dropna(subset=['resolved', 'customfield_13719']).copy()
        df_area['mes_resolvido'] = df_area['resolved'].dt.to_period("M").dt.to_timestamp()
        df_area['mes_str'] = df_area['mes_resolvido'].dt.strftime("%b/%Y")
        df_area['area'] = df_area['customfield_13719'].apply(lambda x: x.get('value') if isinstance(x, dict) else str(x))
        meses = sorted(df_area['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
        mes_selecionado = st.selectbox("Selecione o mês", meses, index=len(meses)-1)
        dados_filtrados = df_area[df_area['mes_str'] == mes_selecionado]
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
        fig_area.update_layout(title=f'Tickets por Área - {mes_selecionado}', xaxis_title='Qtd. Chamados')
        st.plotly_chart(fig_area, use_container_width=True)

with aba3:
    st.subheader("Assunto Relacionado por projeto e mês")
    campos_assunto = {
        'TDS': 'customfield_13712',
        'INT': 'customfield_13643',
        'TINE': 'customfield_13699'
    }
    projeto_selecionado = st.selectbox("Selecione o projeto", list(campos_assunto.keys()))
    campo_assunto = campos_assunto[projeto_selecionado]
    df_assunto = df[df['projeto'] == projeto_selecionado].copy()
    df_assunto['mes_referencia'] = df_assunto['created']
    df_assunto['mes_str'] = df_assunto['mes_referencia'].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")
    df_assunto['assunto'] = df_assunto[campo_assunto].apply(lambda x: x.get('value') if isinstance(x, dict) else str(x))
    meses_disponiveis = sorted(df_assunto['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
    mes_assunto = st.selectbox("Selecione o mês do assunto", meses_disponiveis, index=len(meses_disponiveis)-1)
    df_filtrado = df_assunto[df_assunto['mes_str'] == mes_assunto]
    dados_assunto = df_filtrado['assunto'].value_counts().reset_index()
    dados_assunto.columns = ['Assunto Relacionado', 'Qtd. Chamados']
    fig_assunto = go.Figure(go.Bar(
        y=dados_assunto['Assunto Relacionado'],
        x=dados_assunto['Qtd. Chamados'],
        orientation='h',
        text=dados_assunto['Qtd. Chamados'],
        textposition='outside',
        marker_color='mediumpurple'
    ))
    fig_assunto.update_layout(title=f'Assunto Relacionado - {projeto_selecionado} | {mes_assunto}', xaxis_title='Qtd. Chamados')
    st.plotly_chart(fig_assunto, use_container_width=True)
