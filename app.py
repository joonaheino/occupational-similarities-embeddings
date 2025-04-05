# File: app.py
import streamlit as st
from openai import OpenAI
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import scipy.stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Error: OPENAI_API_KEY not found in environment. Please add it to your .env file.")
    st.stop()

# Constants
REVERSE_METRICS = ["euclidean_distance", "angular_distance"]  # Metrics where lower values mean more similar

# Instantiate the new OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Occupation-Theme Cosine Similarity", layout="wide")
st.title("Occupation & Theme Comparison via Cosine Similarity")
st.markdown("""
This app lets you upload two text files:
- **Occupations:** A list of occupations (one per line)
- **Themes:** A list of themes/words or short phrases (one per line)

It uses the OpenAI embeddings API (`text-embedding-3-large`) to compute embeddings for each entry, calculates the cosine similarity between every occupation and theme, and displays the results. You can also download the resulting CSV.
""")

# --- Helper Functions ---

def get_embeddings(texts, model="text-embedding-3-large"):
    """
    Fetch embeddings for a list of texts using the new OpenAI client interface.
    Returns both the embeddings and a dictionary mapping text to embedding.
    """
    # Ensure texts is a list (if a single string is passed, wrap it in a list)
    if isinstance(texts, str):
        texts = [texts]
    try:
        response = client.embeddings.create(input=texts, model=model)
    except Exception as e:
        st.error(f"Error fetching embeddings: {e}")
        return None, None
    # Extract the embeddings from the response and convert to NumPy arrays
    embeddings = [np.array(item.embedding) for item in response.data]
    # Create a dictionary mapping text to embedding
    embedding_dict = {text: emb for text, emb in zip(texts, embeddings)}
    return embeddings, embedding_dict

def save_embeddings(filename, texts, embedding_dict):
    """
    Save embeddings to a numpy file.
    """
    save_dict = {
        'texts': texts,
        'embeddings': np.array([embedding_dict[text] for text in texts])
    }
    np.save(f"embeddings_{filename}", save_dict)

def load_embeddings(filename):
    """
    Load embeddings from a numpy file.
    Returns texts and embedding dictionary.
    """
    try:
        data = np.load(f"embeddings_{filename}", allow_pickle=True).item()
        texts = data['texts']
        embeddings = data['embeddings']
        embedding_dict = {text: emb for text, emb in zip(texts, embeddings)}
        return texts, embedding_dict
    except FileNotFoundError:
        return None, None

def compute_similarities(vec1, vec2):
    """
    Compute multiple similarity metrics between two vectors.
    """
    # Normalize vectors
    vec1_norm = normalize(vec1.reshape(1, -1))
    vec2_norm = normalize(vec2.reshape(1, -1))
    
    # Cosine similarity
    cos_sim = np.dot(vec1_norm[0], vec2_norm[0])
    
    # Euclidean distance (smaller means more similar)
    eucl_dist = np.linalg.norm(vec1 - vec2)
    
    # Pearson correlation
    pearson_corr = scipy.stats.pearsonr(vec1, vec2)[0]
    
    # Dot product of raw vectors
    dot_prod = np.dot(vec1, vec2)
    
    # Angular distance (in radians, smaller means more similar)
    angular_dist = np.arccos(min(cos_sim, 1.0))
    
    return {
        'cosine_similarity': cos_sim,
        'euclidean_distance': eucl_dist,
        'pearson_correlation': pearson_corr,
        'dot_product': dot_prod,
        'angular_distance': angular_dist
    }

def load_and_process_files(occupations_text, themes_text, occ_filename, theme_filename):
    """
    Initial loading and processing of files, returns embeddings and dataframe
    """
    # Clean and split the input texts
    occupations = [line.strip() for line in occupations_text.splitlines() if line.strip()]
    themes = [line.strip() for line in themes_text.splitlines() if line.strip()]

    st.info(f"Found {len(occupations)} occupations and {len(themes)} themes.")

    # Try to load existing embeddings first
    saved_occupations, occ_emb_dict = load_embeddings(occ_filename)
    if occ_emb_dict is None:
        with st.spinner("Fetching embeddings for occupations..."):
            _, occ_emb_dict = get_embeddings(occupations)
            if occ_emb_dict is None:
                st.error("Failed to fetch embeddings for occupations.")
                return None, None, None
            save_embeddings(occ_filename, occupations, occ_emb_dict)
            st.success(f"Saved occupations embeddings to embeddings_{occ_filename}")

    saved_themes, theme_emb_dict = load_embeddings(theme_filename)
    if theme_emb_dict is None:
        with st.spinner("Fetching embeddings for themes..."):
            _, theme_emb_dict = get_embeddings(themes)
            if theme_emb_dict is None:
                st.error("Failed to fetch embeddings for themes.")
                return None, None, None
            save_embeddings(theme_filename, themes, theme_emb_dict)
            st.success(f"Saved themes embeddings to embeddings_{theme_filename}")

    return occupations, occ_emb_dict, theme_emb_dict

def compute_similarity_df(occupations, occ_emb_dict, theme_emb_dict, metric_name):
    """
    Compute similarity dataframe for a specific metric
    """
    rows = []
    with st.spinner("Computing similarities..."):
        for occ in occupations:
            occ_vec = occ_emb_dict[occ]
            for theme in theme_emb_dict.keys():
                theme_vec = theme_emb_dict[theme]
                similarities = compute_similarities(occ_vec, theme_vec)
                rows.append({
                    "Occupation": occ,
                    "Theme": theme,
                    **similarities
                })

    df = pd.DataFrame(rows)
    
    return df.sort_values(
        metric_name, 
        ascending=metric_name in REVERSE_METRICS
    )

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def visualize_embeddings(df, embeddings_dict, selected_words=None):
    """
    Create t-SNE visualization of the embeddings
    """
    # Combine all embeddings
    words = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[word] for word in words])
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Perform K-means clustering
    n_clusters = min(5, len(words))  # Adjust number of clusters based on data size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'word': words,
        'cluster': clusters
    })
    
    # Create interactive plot
    fig = px.scatter(
        plot_df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['word'],
        title='t-SNE visualization of embeddings with K-means clusters'
    )
    
    # Highlight selected words if any
    if selected_words:
        mask = plot_df['word'].isin(selected_words)
        highlight_df = plot_df[mask]
        fig.add_trace(
            go.Scatter(
                x=highlight_df['x'],
                y=highlight_df['y'],
                text=highlight_df['word'],
                mode='markers+text',
                marker=dict(size=15, symbol='star'),
                name='Selected Words'
            )
        )
    
    return fig

def analyze_similarity_distribution(df, metric):
    """
    Analyze the distribution of similarity scores
    """
    fig = px.histogram(
        df, 
        x=metric,
        title=f'Distribution of {metric}',
        marginal='box'  # adds a box plot above the histogram
    )
    return fig

def create_similarity_chart(filtered_df, x_col, y_col, metric):
    """
    Create a horizontal bar chart using Altair
    """
    return alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X(f"{metric}:Q", title=metric.replace('_', ' ').title()),
        y=alt.Y(f"{y_col}:N", sort='-x', title=y_col),
        tooltip=[y_col, metric]
    ).properties(width=700, height=400)

def create_heatmap(df, metric):
    """
    Create a heatmap visualization using Altair
    """
    # Pivot the data
    pivot = df.pivot(index="Occupation", columns="Theme", values=metric)
    pivot_long = pivot.reset_index().melt(
        id_vars="Occupation", 
        var_name="Theme", 
        value_name=metric
    )
    
    return alt.Chart(pivot_long).mark_rect().encode(
        x=alt.X("Theme:N", title="Theme", sort=None),
        y=alt.Y("Occupation:N", title="Occupation", sort=None),
        color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme='viridis')),
        tooltip=["Occupation", "Theme", metric]
    ).properties(width=900, height=600)

# --- UI Elements ---
st.sidebar.header("File Uploads")
occupation_file = st.sidebar.file_uploader("Upload Occupations Text File", type=["txt"], key="occ")
theme_file = st.sidebar.file_uploader("Upload Themes Text File", type=["txt"], key="theme")

if occupation_file and theme_file:
    # Read file contents
    occupations_text = occupation_file.read().decode("utf-8")
    themes_text = theme_file.read().decode("utf-8")

    # Get filenames for saving embeddings
    occ_filename = occupation_file.name
    theme_filename = theme_file.name

    # Optional: Expanders to view file content
    with st.expander("Show Occupations File Content"):
        st.text(occupations_text)
    with st.expander("Show Themes File Content"):
        st.text(themes_text)

    if "embeddings_loaded" not in st.session_state:
        st.session_state.embeddings_loaded = False

    if not st.session_state.embeddings_loaded:
        if st.button("Process Files"):
            occupations, occ_emb_dict, theme_emb_dict = load_and_process_files(
                occupations_text, themes_text, occ_filename, theme_filename
            )
            if occupations is not None:
                st.session_state.occupations = occupations
                st.session_state.occ_emb_dict = occ_emb_dict
                st.session_state.theme_emb_dict = theme_emb_dict
                st.session_state.embeddings_loaded = True
                st.rerun()

    if st.session_state.embeddings_loaded:
        st.success("Files processed! Choose visualization options below.")
        
        # Add metric selector
        metric = st.selectbox(
            "Choose similarity metric:",
            [
                "cosine_similarity",
                "euclidean_distance",
                "pearson_correlation",
                "dot_product",
                "angular_distance"
            ]
        )
        
        # Show explanation of the selected metric
        metric_explanations = {
            "cosine_similarity": "Measures the cosine of the angle between vectors (1 = most similar, -1 = most different)",
            "euclidean_distance": "Measures the straight-line distance between vectors (smaller = more similar)",
            "pearson_correlation": "Measures linear correlation between vectors (1 = perfect positive correlation, -1 = perfect negative)",
            "dot_product": "Raw similarity without normalization (larger magnitude = more similar)",
            "angular_distance": "Actual angle between vectors in radians (smaller = more similar)"
        }
        st.info(metric_explanations[metric])

        # Compute similarities for current metric
        if "current_metric" not in st.session_state or st.session_state.current_metric != metric:
            df = compute_similarity_df(
                st.session_state.occupations,
                st.session_state.occ_emb_dict,
                st.session_state.theme_emb_dict,
                metric
            )
            st.session_state.df = df
            st.session_state.current_metric = metric
        else:
            df = st.session_state.df

        # Provide a download button for the CSV results
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f'similarities_{metric}.csv',
            mime='text/csv'
        )

        # Add visualization options
        st.subheader("Visualizations")
        
        viz_option = st.radio(
            "Choose visualization:",
            ["Per Occupation", "Per Theme", "Top Pairs", "Heatmap", "Similarity Distribution", "Embedding Space"]
        )

        if viz_option == "Similarity Distribution":
            dist_fig = analyze_similarity_distribution(df, metric)
            st.plotly_chart(dist_fig)
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            stats_df = df[metric].describe()
            st.dataframe(stats_df)
            
        elif viz_option == "Embedding Space":
            # Combine both dictionaries from session state
            all_embeddings = {
                **st.session_state.occ_emb_dict, 
                **st.session_state.theme_emb_dict
            }
            
            # Get top pairs for highlighting
            top_n = st.slider("Number of top pairs to highlight", 1, 10, 5)
            top_pairs = df.nlargest(top_n, metric)[['Occupation', 'Theme']]
            selected_words = pd.concat([top_pairs['Occupation'], top_pairs['Theme']]).unique()
            
            tsne_fig = visualize_embeddings(df, all_embeddings, selected_words)
            st.plotly_chart(tsne_fig)
            
        elif viz_option == "Top Pairs":
            top_n = st.slider("Number of top pairs to show", 1, 20, 10)
            st.subheader(f"Top {top_n} Most Similar Pairs")
            
            cols = st.columns(2)
            with cols[0]:
                st.write("Most Similar")
                top_pairs = df.nlargest(top_n, metric)[
                    ['Occupation', 'Theme', metric]
                ]
                st.dataframe(top_pairs)
            
            with cols[1]:
                st.write("Most Different")
                bottom_pairs = df.nsmallest(top_n, metric)[
                    ['Occupation', 'Theme', metric]
                ]
                st.dataframe(bottom_pairs)

        elif viz_option == "Per Occupation":
            occupation = st.selectbox("Select Occupation", sorted(df["Occupation"].unique()))
            filtered = df[df["Occupation"] == occupation].copy()
            filtered = filtered.sort_values(metric, ascending=metric in REVERSE_METRICS)
            
            st.subheader(f"Themes for {occupation} (sorted by {metric})")
            chart = create_similarity_chart(filtered, metric, "Theme", metric)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(filtered.reset_index(drop=True))

        elif viz_option == "Per Theme":
            theme = st.selectbox("Select Theme", sorted(df["Theme"].unique()))
            filtered = df[df["Theme"] == theme].copy()
            filtered = filtered.sort_values(metric, ascending=metric in REVERSE_METRICS)
            
            st.subheader(f"Occupations for {theme} (sorted by {metric})")
            chart = create_similarity_chart(filtered, metric, "Occupation", metric)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(filtered.reset_index(drop=True))

        elif viz_option == "Heatmap":
            st.subheader(f"Heatmap: Occupations vs. Themes ({metric})")
            heatmap = create_heatmap(df, metric)
            st.altair_chart(heatmap, use_container_width=True)
else:
    st.info("Please upload both an Occupations file and a Themes file via the sidebar.")
