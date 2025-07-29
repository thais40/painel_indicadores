
import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

st.set_page_config(layout='wide')
st.title('Painel de Encaminhamentos - INT e TDS')

JIRA_URL = st.secrets['JIRA_URL']
EMAIL = st.secrets['EMAIL']
TOKEN = st.secrets['TOKEN']
auth = HTTPBasicAuth(EMAIL, TOKEN)

PROJETOS = ['TDS', 'INT']
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
                "fields": "summary,created,resolutiondate,status,customfield_13659"
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
                all_issues.append({
                    "projeto": projeto,
                    "created": f["created"],
                    "resolved": f.get("resolutiondate"),
                    "status": f.get("status", {}).get("name", ""),
                    "escalado_n3": f.get("customfield_13659")
                })
            start_at += 100
    return pd.DataFrame(all_issues)

df = buscar_issues()
df['created'] = pd.to_datetime(df['created'])
df['mes_str'] = df['created'].dt.to_period("M").dt.to_timestamp().dt.strftime("%b/%Y")

aba_int, aba_tds = st.tabs(['INT', 'TDS'])

for projeto_nome, aba in zip(['INT', 'TDS'], [aba_int, aba_tds]):
    with aba:
        st.subheader(f"Projeto: {projeto_nome}")
        df_proj = df[df['projeto'] == projeto_nome]
        meses = sorted(df_proj['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
        mes_sel = st.selectbox(f"Selecione o mês ({projeto_nome})", meses, key=projeto_nome)
        df_mes = df_proj[df_proj['mes_str'] == mes_sel]

        col1, col2 = st.columns(2)
        with col1:
            encaminhados_produto = df_mes[df_mes['status'].isin(["Priorizar com Produto", "Priorizado com Produto"])]
            st.metric("Encaminhados Produto", encaminhados_produto.shape[0])
        with col2:
            encaminhados_n3 = df_mes[df_mes["escalado_n3"] == "Sim"]
            st.metric("Encaminhados N3", encaminhados_n3.shape[0])
