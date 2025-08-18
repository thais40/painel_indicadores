# 👉 Submenu APP NE — só para TDS (com filtros e meses ordenados)
if projeto == "TDS":
    with st.expander("📱 APP NE — Origem do problema", expanded=False):
        # filtra assunto alvo
        df_app = dfp.copy()
        df_app["assunto_nome"] = df_app["assunto"].apply(lambda x: x.get("value") if isinstance(x, dict) else str(x))
        df_app = df_app[df_app["assunto_nome"] == ASSUNTO_ALVO_APPNE].copy()

        if df_app.empty:
            st.info(f"Não há chamados com Assunto '{ASSUNTO_ALVO_APPNE}'.")
        else:
            # campos auxiliares
            df_app["origem_nome"] = df_app["origem"].apply(
                lambda x: x.get("value") if isinstance(x, dict) else (str(x) if x is not None else "—")
            )
            df_app["mes_dt"] = df_app["mes_created"].dt.to_period("M").dt.to_timestamp()

            # ===== Filtros (Ano / Mês) iguais aos outros =====
            anos_app = sorted(df_app["mes_dt"].dt.year.dropna().unique())
            meses_app = sorted(df_app["mes_dt"].dt.month.dropna().unique())
            col_app1, col_app2 = st.columns(2)
            with col_app1:
                ano_app = st.selectbox("Ano (APP NE)", ["Todos"] + [str(a) for a in anos_app], key=f"ano_app_{projeto}")
            with col_app2:
                mes_app = st.selectbox("Mês (APP NE)", ["Todos"] + [str(m).zfill(2) for m in meses_app], key=f"mes_app_{projeto}")

            df_app_f = df_app.copy()
            if ano_app != "Todos":
                df_app_f = df_app_f[df_app_f["mes_dt"].dt.year == int(ano_app)]
            if mes_app != "Todos":
                df_app_f = df_app_f[df_app_f["mes_dt"].dt.month == int(mes_app)]

            if df_app_f.empty:
                st.info("Sem dados para exibir com os filtros selecionados.")
            else:
                # métricas (já considerando filtros)
                total_app = len(df_app_f)
                contagem = df_app_f["origem_nome"].value_counts(dropna=False).to_dict()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total (APP NE/EN)", total_app)
                c2.metric("APP NE", contagem.get("APP NE", 0))
                c3.metric("APP EN", contagem.get("APP EN", 0))

                # ===== Série mensal por origem (meses ordenados) =====
                serie = (
                    df_app_f.groupby(["mes_dt", "origem_nome"])
                            .size()
                            .reset_index(name="Qtd")
                            .sort_values("mes_dt")
                )
                serie["mes_str"] = serie["mes_dt"].dt.strftime("%b/%Y")
                # categoria ordenada para o eixo X
                cats = serie["mes_str"].dropna().unique().tolist()
                serie["mes_str"] = pd.Categorical(serie["mes_str"], categories=cats, ordered=True)

                # Gráfico com rótulos de quantidade nas barras (lado a lado)
                fig_app = px.bar(
                    serie,
                    x="mes_str",
                    y="Qtd",
                    color="origem_nome",
                    barmode="group",
                    title="APP NE — Volumes por mês e Origem do problema",
                    color_discrete_map={"APP NE": "#2ca02c", "APP EN": "#1f77b4"},
                    text="Qtd"
                )
                fig_app.update_traces(texttemplate="%{text}", textposition="outside")
                fig_app.update_layout(yaxis_title="Qtd", xaxis_title="Mês")
                st.plotly_chart(fig_app, use_container_width=True)

                # Tabela detalhada (filtrada)
                df_app_f["mes_str"] = df_app_f["mes_dt"].dt.strftime("%b/%Y")
                cols_show = ["key", "created", "mes_str", "assunto_nome", "origem_nome", "status"]
                cols_show = [c for c in cols_show if c in df_app_f.columns]
                st.dataframe(df_app_f[cols_show], use_container_width=True, hide_index=True)
