import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit_extras.annotated_text import annotated_text
from annotated_text import annotation
from streamlit_extras.stoggle import stoggle
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)

# locales
from scripts.autores import autores
from scripts.modelo import modelo, modelo_sin_svd

st.set_page_config(layout="wide")

st.logo("data/logo/ matcom.jpeg")


def inicio():
    espacio = st.empty()
    with espacio.container(border=True):
        st.markdown(
            """<h1 class = 'titulos'>Cáncer de mama</h1> <style>
                .titulos{
                font-size: 60px;
                text-align: center;
                }
            </style>""",
            unsafe_allow_html=True,
        )
        st.code(
            """
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_squared_error,r2_score,mean_absolute_error,confusion_matrix,ConfusionMatrixDisplay

df = pd.read_csv(" breast-cancer.csv.xls",index_col=0)
print(df)""",
            line_numbers=True,
        )
        df = pd.read_csv("data/ breast-cancer.csv.xls", index_col=0)
        with st.expander("dataframe"):
            st.dataframe(df)
        st.markdown(
            """<h1 class = 'aed'>Análisis Exploratorio</h1> <style>
                .aed{
                font-size: 30px;
                text-align: center;
                }
            </style>""",
            unsafe_allow_html=True,
        )
        st.code(
            "df.info()",
            line_numbers=True,
            language="python",
        )

        with st.expander("Resultado: df.info()"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            a = buffer.getvalue()
            st.text(a)

        st.code("df.duplicated().sum()", line_numbers=True)
        st.write(df.duplicated().sum())

        st.code("df.describe()", line_numbers=True)
        st.write(df.describe())

        st.code("df.isna().sum()", line_numbers=True)
        with st.expander("Resultado: df.isna().sum()"):
            st.write(df.isna().sum())

        st.code(
            """
        df1 = df.drop(["diagnosis"],axis=1)
        corr = df1.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0)""",
            line_numbers=True,
        )
        df1 = df.drop(["diagnosis"], axis=1)
        matriz = plt.figure(figsize=(4, 4))
        corr = df1.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0)
        with st.expander("Matriz de correlación"):
            st.pyplot(matriz, use_container_width=True)

        U, s, V_transp = linalg.svd(df1)
        st.code(
            """U, s, V_transp = linalg.svd(df1)
print(U)
print(s)""",
            line_numbers=True,
        )
        with st.expander("print(U) y  print(s)"):
            annotated_text(annotation("U", color="#FF4B4B"))
            st.text(U)

            annotated_text(annotation("s", color="#FF4B4B"))
            st.text(s)

        st.code(
            """sigma_s=[]
for i in range(len(s)):
    a=(s[i]/(sum(s)))*100
    sigma_s.append(a)

print(sigma_s)""",
            line_numbers=True,
        )
        sigma_s = []
        for i in range(len(s)):
            a = (s[i] / (sum(s))) * 100
            sigma_s.append(a)
        with st.expander("print(sigma_s)"):
            sigma_s
        st.markdown(
            """<h1 class = 'graf'>Gráfico de los valores singulares</h1> <style>
                .graf{
                font-size: 30px;
                text-align: center;
                }
            </style>""",
            unsafe_allow_html=True,
        )
        st.code(
            """fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 31)), 
                         y=sigma_s,
                         mode="lines+markers"))
fig.update_layout(
        title="valores sigulares",
        xaxis_title="cantidad",
        yaxis_title="valor",
        xaxis=dict(tickmode="linear"),
)
# en la grafica se observa que los 4 primeros valores singulares son los más significativos.
    """,
            line_numbers=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, 31)), y=sigma_s, mode="lines+markers"))
        fig.update_layout(
            title="porciento que representan los valores sigulares",
            xaxis_title="cantidad",
            yaxis_title="valor",
            xaxis=dict(tickmode="linear"),
        )
        st.plotly_chart(fig)
        with st.expander("Regresión logistica sin SVD truncado"):
            st.markdown(
                """<h1 class = 'model'>Modelo de Regresión Logística</h1> <style>
                .model{
                font-size: 30px;
                text-align: center;
                }
            </style>""",
                unsafe_allow_html=True,
            )
            st.code(
                f"""
    X = df1
    # se llevó los benignos a 1 y los malignos a 0
    # para hacer la clasificación binaria
    y = (df['diagnosis'] == 'B').astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    a=model.predict(X)
    predict = model.predict(X_test)
    cm = confusion_matrix(y_test, predict)
    report = classification_report(y_test, predict)
    r2 = r2_score(y_test, predict)
    mean_absolute = mean_absolute_error(y_test, predict)
    mean_squared = mean_squared_error(y_test, predict)

    print(model.score(X_test,y_test))
    print(a)""",
                line_numbers=True,
            )
            modelo_sin_svd(df1, df)

        st.markdown(
            """<h1 class = 'model'>Modelo de Regresión Logística con SVD truncado</h1> <style>
                .model{
                font-size: 30px;
                text-align: center;
                }
            </style>""",
            unsafe_allow_html=True,
        )
        st.code(
            f"""
    X = df1
    # se llevó los benignos a 1 y los malignos a 0
    # para hacer la clasificación binaria
    y = (df['diagnosis'] == 'B').astype(int)


    # usando los 4 valores singulares(n_components = 4) se obtenia un resultado de 92%
    # usando los 3 valores singulares(n_components = 3) se obtenia un resultado de 94%
    # usando los 2 valores singulares(n_components = 2) se obtenia un resultado de 94%
    # usando los 1 valores singulares(n_components = 1) se obtenia un resultado de 91%

    # entonces se obtienen mejores resultados con n_components = 2,3 
    # nos decidimos por n_components = 2, ya que esto implicaria menor coste computacional(Menos columnas)
    # y obtendriamos mismos resultados.
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2,random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    a=model.predict(X_reduced)
    predict = model.predict(X_test)
    cm = confusion_matrix(y_test, predict)
    report = classification_report(y_test, predict)
    r2 = r2_score(y_test, predict)
    mean_absolute = mean_absolute_error(y_test, predict)
    mean_squared = mean_squared_error(y_test, predict)

    print(model.score(X_test,y_test))
    print(a)""",
            line_numbers=True,
        )
        st.info("Seleccione uno a la vez")
        if st.checkbox("para n = 1"):
            modelo(df1, df, 1)
        elif st.checkbox("para n = 2 "):
            modelo(df1, df, 2)
        elif st.checkbox("para n = 3 "):
            modelo(df1, df, 3)
        elif st.checkbox("para n = 4 "):
            modelo(df1, df, 4)


if __name__ == "__main__":
    inicio()
    check = st.button("¿Quiénes somos?")

    if check:
        with st.spinner():
            autores()
