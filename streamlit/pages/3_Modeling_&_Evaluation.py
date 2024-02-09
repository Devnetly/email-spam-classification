import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer,
    roc_curve,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from joblib import load, dump


# @st.cache_data
def load_data(path):
    data = pd.read_csv(path, sep=";")
    return data


@st.cache_data
def load_model(path):
    return load(path)


def plot_confusion_matrix(conf_matrix):
    fig = go.Figure(
        data=go.Heatmap(z=conf_matrix, colorscale="Blues", colorbar=dict(title="Count"))
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted label"),
        yaxis=dict(title="True label"),
        annotations=[
            dict(
                text=str(conf_matrix[i, j]),
                x=j,
                y=conf_matrix.shape[0] - 1 - i,  # Reverse the row index
                font=dict(color="black"),
                showarrow=False,
            )
            for i in range(conf_matrix.shape[0])
            for j in range(conf_matrix.shape[1])
        ],
    )
    st.plotly_chart(fig)


model_choice = st.radio(
    "Choose a model", ["Self Training", "Label Propagation", "Label Spreading"]
)

model_name = ""

if model_choice == "Label Propagation":
    # choose hyperparameters (kernel, vectroizer, truncated svd or not)
    st.write("Label Propagation")
    st.write(
        "Label Propagation is a semi-supervised learning algorithm that assigns labels to previously unlabeled data points. It does this by propagating the labels of known data points to unknown data points. The algorithm works by modeling the data as a graph and using the graph to predict the labels of unknown data points."
    )
    st.write(
        "The algorithm has a few hyperparameters that can be tuned. The kernel hyperparameter is used to determine the similarity between data points. The vectorizer hyperparameter is used to convert the text data into a format that can be used by the algorithm. The truncated svd hyperparameter is used to reduce the dimensionality of the data."
    )

    kernel = st.selectbox("Kernel", ["knn", "rbf"])
    vectorizer = st.selectbox("Vectorizer", ["Tfidf", "Count"])
    truncated_svd = st.checkbox("Truncated SVD")

    params = {
        "kernel": kernel,
        "vectorizer": "cv" if vectorizer == "Count" else "tfid",
        "truncated_svd": "_svd" if truncated_svd else "",
    }

    model_name = (
        "lp_" + params["vectorizer"] + "_" + params["kernel"] + params["truncated_svd"]
    )
    lp_evaluation = load_data("data/lp_evaluation_df.csv")

elif model_choice == "Label Spreading":
    st.write("Label Spreading")
    st.write(
        "Label Spreading is a semi-supervised learning algorithm that assigns labels to previously unlabeled data points. It does this by propagating the labels of known data points to unknown data points. The algorithm works by modeling the data as a graph and using the graph to predict the labels of unknown data points."
    )
    st.write(
        "The algorithm has a few hyperparameters that can be tuned. The kernel hyperparameter is used to determine the similarity between data points. The vectorizer hyperparameter is used to convert the text data into a format that can be used by the algorithm. The truncated svd hyperparameter is used to reduce the dimensionality of the data."
    )

    kernel = st.selectbox("Kernel", ["knn", "rbf"])
    vectorizer = st.selectbox("Vectorizer", ["Tfidf", "Count"])
    truncated_svd = st.checkbox("Truncated SVD")

    params = {
        "kernel": kernel,
        "vectorizer": "cv" if vectorizer == "Count" else "tfid",
        "truncated_svd": "_svd" if truncated_svd else "",
    }

    model_name = (
        "ls_" + params["vectorizer"] + "_" + params["kernel"] + params["truncated_svd"]
    )
    ls_evaluation = load_data("data/ls_evaluation_df.csv")


elif model_choice == "Self Training":
    st.write("Self Training")
    st.write(
        "Self Training is a semi-supervised learning algorithm that assigns labels to previously unlabeled data points. It does this by training a supervised learning model on the labeled data and using the model to predict the labels of unknown data points. The algorithm works by modeling the data as a graph and using the graph to predict the labels of unknown data points."
    )
    st.write(
        "The algorithm has a few hyperparameters that can be tuned. The vectorizer hyperparameter is used to convert the text data into a format that can be used by the algorithm. The truncated svd hyperparameter is used to reduce the dimensionality of the data."
    )

    base_estimator = st.selectbox(
        "Base Estimator",
        ["Logistic Regression", "SGD Classifier", "Naive Bayes", "SVM"],
    )

    param = (
        "lr"
        if base_estimator == "Logistic Regression"
        else (
            "sgd"
            if base_estimator == "SGD Classifier"
            else "nb" if base_estimator == "Naive Bayes" else "svm"
        )
    )

    model_name = "st_" + param
    st_evaluation = load_data("data/st_evaluation_df.csv")

# button to train the model
st.write("Click the button below to train the model")

if st.button("Show Metrics", type="primary"):
    # loading the model
    spam_model = load_model("models/" + model_name + "_model.joblib")
    if model_choice == "Label Propagation":
        row = lp_evaluation[lp_evaluation["Unnamed: 0"] == model_name]
        two_d_array = json.loads(row["Confusion Matrix"].values[0])
        plot_confusion_matrix(np.array(two_d_array))
    elif model_choice == "Label Spreading":
        row = ls_evaluation[ls_evaluation["Unnamed: 0"] == model_name]
        two_d_array = json.loads(row["Confusion Matrix"].values[0])
        plot_confusion_matrix(np.array(two_d_array))
    elif model_choice == "Self Training":
        row = st_evaluation[st_evaluation["Unnamed: 0"] == model_name]
        two_d_array = json.loads(row["Confusion Matrix"].values[0])
        plot_confusion_matrix(np.array(two_d_array))
