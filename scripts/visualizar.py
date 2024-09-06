import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st


def visualizar(predict, orig):
    def _analizar():
        uno = 0
        cero = 0
        for i in orig[1:-1].split(" "):
            if i == "1":
                uno += 1
                continue
            if i == "":
                continue
            cero += 1

        return cero, uno

    def _contar():
        unos = 0
        zeros = 0
        for i in predict:
            if int(i) == 1:
                unos += 1
                continue
            zeros += 1
        return zeros, unos

    cant_zeros_p, cant_unos_p = _contar()
    cant_zeros, cant_unos = _analizar()

    df_predict = pd.DataFrame(
        {
            "cantidad de tumores benignos predichos": cant_unos_p,
            "cantidad de tumores malignos predichos": cant_zeros_p,
        },
        index=["Predicciones"],
    )
    df_orig = pd.DataFrame(
        {
            "cantidad de tumores benignos originales": cant_unos,
            "cantidad de tumores malignos origiales": cant_zeros,
        },
        index=["Originales"],
    )
    fig = px.bar(df_predict)
    fig.update_layout(title="Predicciones", yaxis_title="valor")
    fig1 = px.bar(df_orig)
    fig1.update_layout(title="Originales", yaxis_title="valor")

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig1, use_container_width=True)
