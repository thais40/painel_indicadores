
import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import plotly.graph_objects as go
from sla_utils import extrair_sla_millis

st.set_page_config(layout='wide')
st.title('Painel de Indicadores')

JIRA_URL = st.secrets['JIRA_URL']
EMAIL = st.secrets['EMAIL']
TOKEN = st.secrets['TOKEN']
auth = HTTPBasicAuth(EMAIL, TOKEN)

PROJETOS = ['TDS', 'INT', 'TINE', 'INTEL']
CUTOFF_DATE = '2024-01-01'
SLA_METAS = {'TDS': 98, 'INT': 96, 'TINE': 96, 'INTEL': 96}

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
            issues = res.json().get("issues", [])
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
df['mes_str'] = pd.to_datetime(df['created']).dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m")
df['escalado_n3_valor'] = df['customfield_13659'].apply(lambda x: x.get('value') if isinstance(x, dict) else None)

abas = st.tabs(PROJETOS)

for projeto, aba in zip(PROJETOS, abas):
    with aba:
        st.header(f'📊 Projeto: {projeto}')
        df_proj = df[df['projeto'] == projeto]
        meses = sorted(df_proj['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%Y-%m'))

        st.subheader("📈 Criados vs Resolvidos")
        criados = df_proj.groupby('mes_str')['created'].count().reset_index(name='Criados')
        resolvidos = df_proj[df_proj['resolved'].notna()].groupby('mes_str')['resolved'].count().reset_index(name='Resolvidos')
        grafico = pd.merge(criados, resolvidos, how="outer", on="mes_str").fillna(0).sort_values("mes_str")
        fig = go.Figure()
        fig.add_bar(x=grafico['mes_str'], y=grafico['Criados'], name='Criados', text=grafico['Criados'], textposition='outside')
        fig.add_bar(x=grafico['mes_str'], y=grafico['Resolvidos'], name='Resolvidos', text=grafico['Resolvidos'], textposition='outside')
        fig.update_layout(barmode='group', xaxis_title='Mês', yaxis_title='Chamados')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⏱️ SLA por Mês")
        df_sla = df_proj.dropna(subset=['sla_millis']).copy()
        df_sla['mes_str'] = df_sla['created'].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m")
        df_sla['sla_horas'] = df_sla['sla_millis'] / (1000 * 60 * 60)
        sla_meta = SLA_METAS[projeto]
        df_sla['dentro_sla'] = df_sla['sla_horas'] <= 40
        sla_mensal = df_sla.groupby('mes_str')['dentro_sla'].agg(['mean', 'count']).reset_index()
        sla_mensal['percentual'] = (sla_mensal['mean'] * 100).round(1)
        sla_mensal['fora'] = 100 - sla_mensal['percentual']
        fig_sla = go.Figure(data=[
            go.Bar(name='Dentro SLA', x=sla_mensal['mes_str'], y=sla_mensal['percentual'], text=sla_mensal['percentual'], textposition='outside', marker_color='green'),
            go.Bar(name='Fora SLA', x=sla_mensal['mes_str'], y=sla_mensal['fora'], text=sla_mensal['fora'], textposition='outside', marker_color='red')
        ])
        fig_sla.update_layout(barmode='group', yaxis_title='%', xaxis_title='Mês', title=f'SLA Mensal ({sla_meta}%)')
        st.plotly_chart(fig_sla, use_container_width=True)

        st.subheader("📦 Área Solicitante")
        if 'customfield_13719' in df_proj.columns:
            df_area = df_proj.dropna(subset=['customfield_13719']).copy()
            df_area['area'] = df_area['customfield_13719'].apply(lambda x: x.get('value') if isinstance(x, dict) else str(x))
            dados_area = df_area['area'].value_counts().reset_index()
            dados_area.columns = ['Área', 'Qtd. Chamados']
            st.dataframe(dados_area)

        st.subheader("🧠 Assunto Relacionado")
        campos_assunto = {
            'TDS': 'customfield_13712',
            'INT': 'customfield_13643',
            'TINE': 'customfield_13699'
        }
        if projeto in campos_assunto:
            campo_assunto = campos_assunto[projeto]
            df_assunto = df_proj.dropna(subset=[campo_assunto]).copy()
            df_assunto['assunto'] = df_assunto[campo_assunto].apply(lambda x: x.get('value') if isinstance(x, dict) else str(x))
            dados_assunto = df_assunto['assunto'].value_counts().reset_index()
            dados_assunto.columns = ['Assunto Relacionado', 'Qtd. Chamados']
            st.dataframe(dados_assunto)

        if projeto in ['TDS', 'INT']:
            st.subheader("🔄 Encaminhamentos")
            mes_enc = st.selectbox(f"Selecione o mês - {projeto} (Encaminhamentos)", meses, index=len(meses)-1, key=f"mes_enc_{projeto}")
            df_mes = df_proj[df_proj['mes_str'] == mes_enc]
            col1, col2 = st.columns(2)
            with col1:
                count_produto = df_mes['status'].str.contains("Produto", case=False, na=False).sum()
                st.metric("Encaminhados Produto", count_produto)
            with col2:
                count_n3 = df_mes['escalado_n3_valor'] == "Sim"
                st.metric("Encaminhados N3", count_n3.sum())
