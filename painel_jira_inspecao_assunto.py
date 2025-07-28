import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json

JIRA_URL = st.secrets['JIRA_URL']
EMAIL = st.secrets['EMAIL']
TOKEN = st.secrets['TOKEN']
auth = HTTPBasicAuth(EMAIL, TOKEN)

st.title('🔍 Inspeção de Chamados - Assunto Relacionado')
projetos = ['TDS', 'INT', 'TINE']
projeto = st.selectbox('Selecione o projeto', projetos)

headers = {"Accept": "application/json"}
params = {
    "jql": f"project = '{projeto}' ORDER BY created DESC",
    "maxResults": 3,
    "fields": "summary,created,resolutiondate,customfield_13712,customfield_13643,customfield_13699"
}

res = requests.get(f"{JIRA_URL}/rest/api/3/search", headers=headers, auth=auth, params=params)
if res.status_code == 200:
    issues = res.json().get('issues', [])
    for issue in issues:
        st.markdown(f"### {issue['key']} - {issue['fields'].get('summary', '')}")
        st.json(issue['fields'])
else:
    st.error(f"Erro ao buscar issues: {res.status_code}")
