import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
import nltk
import os
import zipfile
from wordcloud import WordCloud

st.session_state['toggle'] = 0
st.session_state['download'] = 0
st.session_state['slider'] = 0
st.session_state['df'] = pd.DataFrame()

st.title("Exploratory Data Analysis")

st.markdown("### Data Loading")

buffer = None

with st.expander(label="Data Loading", expanded=True):
    buffer = st.file_uploader(label="Upload your dataframe", type="csv")
    if buffer:
        st.success("Dataset was uploaded with success")

if buffer:
    if st.session_state.get('df') is None:
        df = pd.read_csv(buffer)
        st.session_state['df'] = df
    else:
        df = st.session_state.get('df')

st.markdown("### Basic EDA")

with st.expander("Examine the top & the tail of the dataframe", expanded=False):
    st.markdown("")
    if buffer:
        st.dataframe(df.head())
        st.dataframe(df.tail())

with st.expander("Examine the dimensions of the dataset", expanded=False):
    st.markdown("")
    if buffer:
        m, n = df.shape
        st.info(f"The number of columns is : {n}")
        st.info(f"The number of data points is : {n}")

with st.expander("Examine data types and missing values", expanded=False):
    st.markdown("")
    if buffer:
        info = pd.DataFrame()
        info["Column"] = df.columns
        info["Non-Null Count"] = (~df.isna()
                                  ).sum().apply(lambda x: str(x) + ' non-null').values
        info["Dtype"] = df.dtypes.values
        st.dataframe(info)

with st.expander("Examine basic statistics about the dataset", expanded=False):
    st.markdown("")
    if buffer:
        st.dataframe(df.describe())
        st.dataframe(df.select_dtypes(bool).describe())

st.markdown("### Visualization")

figures = {}


def fig_download_button(fig: plt.Figure, name: str):
    download = st.session_state['download']
    st.session_state['download'] += 1
    bytes = io.BytesIO()
    fig.savefig(fname=bytes)
    st.download_button(
        label="Export as png",
        data=bytes,
        file_name=f"{name}.png",
        mime="image/png",
        key=f"btn{download}"
    )
    return bytes


with st.expander("Labels distribution", expanded=False):
    st.markdown("")
    if buffer:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(15, 7.5)

        sns.countplot(data=df, x="class", ax=ax1)
        df["class"].value_counts().plot.pie(
            labels=["unlabled", "ham", "spam"], autopct="%0.2f", ax=ax2)

        ax1.set_title("class counts by label (Bar plot)")
        ax2.set_title("class counts by label (Pie plot)")

        st.pyplot(fig=fig)

        figures["Labels_distribution"] = fig_download_button(
            fig, name="lables_distribution")

with st.expander("Unlabled vs labeled", expanded=False):
    st.markdown("")
    if buffer:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(15, 7.5)

        is_labeled = (df["class"] == -1).map({
            True: "unlabled",
            False: "labeled"
        })

        sns.countplot(x=is_labeled, ax=ax1)
        is_labeled.value_counts().plot.pie(
            labels=["unlabled", "labled"], autopct="%0.2f", ax=ax2)

        ax1.set_title("Unalebed vs labled (Bar plot)")
        ax2.set_title("Unalebed vs labled (Pie plot)")

        st.pyplot(fig=fig)

        figures["unlabled_vs_labled"] = fig_download_button(
            fig, name="unlabled_vs_labled")

labled_df = df[df["class"] != -1]

with st.expander("Ham vs Spam", expanded=False):
    st.markdown("")
    if buffer:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(15, 7.5)

        sns.countplot(data=labled_df, x="class", ax=ax1)
        # labled_df["class"].value_counts().plot.pie(labels = ["ham", "spam"], autopct="%0.2f", ax=ax2)

        ax1.set_title("class counts by label (Bar plot)")
        ax2.set_title("class counts by label (Pie plot)")

        figures["class_counts_lables_bar_plot"] = ax1
        figures["class_counts_lables_pie_plot"] = ax2

        st.pyplot(fig=fig)

        figures["ham_vs_spam"] = fig_download_button(fig, name="ham_vs_spam")


def hist_plot(label: str, column: str, fig_name: str):
    with st.expander(label, expanded=False):
        st.markdown("")
        if buffer:

            selectbox = st.session_state['toggle']
            slider = st.session_state['slider']

            st.session_state['toggle'] += 1
            st.session_state['slider'] += 1

            log_scale = st.toggle(
                label="Log Scale",
                key=f"toggle{selectbox}",
                value=False
            )

            eps = 0

            if log_scale and df[column].min() <= 0:
                st.warning(
                    "Can not scale when data contains element with value less or equal to 0")
                eps = st.slider("Specify a value for Epsilon",
                                min_value=10e-10,
                                max_value=1.,
                                value=10e-1,
                                key=f"slider{slider}"
                                )

            fig, ax = plt.subplots()

            try:
                sns.histplot(
                    x=df[column] + eps,
                    hue=df["class"],
                    kde=True,
                    ax=ax,
                    stat="density",
                    common_norm=False,
                    log_scale=log_scale
                )

                fig.set_size_inches(5, 5)
                st.pyplot(fig=fig)

                figures[fig_name] = fig_download_button(fig, name=fig_name)
            except np.linalg.LinAlgError as error:
                st.error(
                    f"Log scaling has failed due to the following error : {error}")


# hist_plot(label="Urls Count", column="Number of urls", fig_name="urls_count")
# hist_plot(label="Digits Count", column="Number of digit strings", fig_name="digits_count")
# hist_plot(label="Length", column="Length", fig_name="length")


st.markdown("### Word frequency analysis")


def plot_topk_results(dist, ax, k=20):
    most_common = dist.most_common(k)
    tokens = np.array([word for word, count in most_common])
    counts = np.array([count for word, count in most_common])
    sns.barplot(x=tokens, y=counts, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80)


def top_k():
    with st.expander("Top k most frequent words"):
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

        k = st.number_input(label="K", min_value=5, max_value=50, value=20)

        fig.set_size_inches(15, 7.5)

        plot_topk_results(grouped_df["frequencies"].loc[0], ax=ax1, k=k)
        plot_topk_results(grouped_df["frequencies"].loc[1], ax=ax2, k=k)

        ax1.set_title(f"Top {k} most frequent words in ham emails")
        ax2.set_title(f"Top {k} most frequent words in spam emails")

        st.pyplot(fig=fig)

        figures["top_k"] = fig_download_button(fig, name="top_k")


def word_clouds():
    with st.expander("Wordclouds", expanded=False):
        wc1, wc2 = WordCloud(), WordCloud()

        im1 = wc1.generate(grouped_df["content"].loc[0])
        im2 = wc2.generate(grouped_df["content"].loc[1])

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

        fig.set_size_inches(15, 7.5)

        ax1.imshow(im1)
        ax2.imshow(im2)

        st.pyplot(fig=fig)

        figures["word_clouds"] = fig_download_button(fig, name="word_clouds")


if buffer:

    grouped_df = labled_df[["class", "content"]
                           ].groupby(by="class").agg(" ".join)
    grouped_df["tokens"] = grouped_df["content"].apply(nltk.word_tokenize)
    grouped_df["frequencies"] = grouped_df["tokens"].apply(
        nltk.probability.FreqDist)

    top_k()
    word_clouds()

st.markdown("### Correlations analysis")


def heatmap():
    with st.expander("Heatmap", expanded=False):
        method = st.selectbox("Pick heatmap type", options=[
                              'pearson', 'kendall', 'spearman'])
        data = labled_df.select_dtypes(
            include=np.number).corr(method=method).abs()
        data.fillna(0, inplace=True)
        fig, ax = plt.subplots()
        sns.heatmap(data=data, vmin=0, vmax=1, annot=True, ax=ax)
        fig.set_size_inches(8, 8)
        st.pyplot(fig=fig)

        figures["heatmap"] = fig_download_button(fig, name="heatmap")


heatmap()
