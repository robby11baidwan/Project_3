import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv(
    "https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv"
)

st.title("Interactive K-Means Clustering")

st.write("Here is the dataset used in this analysis:")

sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value=True)

if df_display:
    st.write(df)

n_clusters = sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
)

def run_kmeans(df, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[["Age", "Income"]])

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.grid(False)
    #ax.set_facecolor("#FFF")
    #ax.spines[["left", "bottom"]].set_visible(True)
    #ax.spines[["left", "bottom"]].set_color("#4a4a4a")
    ax.tick_params(labelcolor="#4a4a4a")
    ax.yaxis.label.set(color="#4a4a4a", fontsize=20)
    ax.xaxis.label.set(color="#4a4a4a", fontsize=20)
    # --------------------------------------------------

    # Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df.Age,
        y=df.Income,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters)
        #legend=None,
    )

    # Annotate cluster centroids
    for ix, [age, income] in enumerate(kmeans.cluster_centers_):
        ax.scatter(age, income, s=200, c="#a8323e")
        ax.annotate(
            f"Cluster #{ix + 1}",
            (age, income),
            fontsize=25,
            color="#a8323e",
            xytext=(age + 5, income + 3),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a8323e", lw=2),
            ha="center",
            va="center",
        )

    return fig

st.write(run_kmeans(df, n_clusters=n_clusters))

def credits(content):
    st.markdown(
        f'<p style="color:{"#f63366"};">{content}</p>',
        unsafe_allow_html=True)


st.text("")
st.text("")
credits("Made By:")
credits("Brijeshwar Singh")
credits("101803170")
credits("COE-9")
