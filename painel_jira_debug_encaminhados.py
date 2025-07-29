
import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

st.set_page_config(layout='wide')
st.title('🔍 Debug: Encaminhados Produto e N3')

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

projeto_sel = st.selectbox("Selecione o projeto", df['projeto'].unique())
df_proj = df[df['projeto'] == projeto_sel]

meses = sorted(df_proj['mes_str'].unique(), key=lambda x: pd.to_datetime(x, format='%b/%Y'))
mes_sel = st.selectbox("Selecione o mês", meses)

df_mes = df_proj[df_proj['mes_str'] == mes_sel]

st.write("📌 Amostra de registros (10 linhas):")
st.dataframe(df_mes[['created', 'status', 'escalado_n3']].head(10))

st.write("🔢 Contagem de Status:")
st.dataframe(df_mes['status'].value_counts())

st.write("📊 Contagem Escalado N3:")
st.dataframe(df_mes['escalado_n3'].value_counts(dropna=False))
