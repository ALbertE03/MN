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
import streamlit as st

from scripts.visualizar import visualizar


def modelo(df1, df, n):

    X = df1
    y = (df["diagnosis"] == "B").astype(int)

    svd = TruncatedSVD(n_components=n)
    X_reduced = svd.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=0
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    a = model.predict(X_reduced)
    predict = model.predict(X_test)
    report = classification_report(y_test, predict)
    r2 = r2_score(y_test, predict)
    mean_absolute = mean_absolute_error(y_test, predict)
    mean_squared = mean_squared_error(y_test, predict)

    stoggle(
        "Resultado del modelo (click)",
        str((model.score(X_test, y_test) * 100)) + " % de precisión",
    )
    aux = "["
    for i in y:
        aux += str(i)
        aux += " "
    aux += "]"

    stoggle(
        "Originales: ",
        aux,
    )
    stoggle(
        "Predicciones: ",
        str(a),
    )
    if st.checkbox("Reportes"):
        st.text(report)

    if st.checkbox("Errores"):
        st.text("Error medio absoluto: " + str(mean_absolute))
        st.text("Eror medio cuadrado: " + str(mean_squared))
        st.text("r2_score: " + str(r2))

    if st.checkbox("Visualizar"):
        visualizar(a, aux)
