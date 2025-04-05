# File: visualize_app.py
import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Occupation & Theme Similarity", layout="wide")

# Add footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #262730;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
.footer a {
    color: white;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    Built by <a href="https://x.com/joonaheino" target="_blank">@joonaheino</a>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Visualization", "About"])

if page == "About":
    st.title("About the Occupation & Theme Similarity Visualizer")
    st.markdown("""
    ## Exploring Semantic Connections

    Ever wondered how closely related the concept of 'Doctor' is to the theme of 'Healing'? Or how 'Entrepreneurship' aligns with 'Innovation'? This interactive tool allows you to explore the semantic connections between various occupations and underlying themes.

    It uses **cosine similarity scores**, derived from analyzing vast amounts of text data (using language models), to measure how closely related these concepts are in meaning. A higher score suggests a stronger semantic link.

    This visualizer stems from a personal exploration project aiming to uncover and understand these fascinating relationships in a quantifiable way.

    ## What Can You Discover?

    By visualizing these similarities, you might uncover interesting patterns and insights:

    *   **Identify Core Themes:** See which themes are most strongly associated with specific occupations (e.g., 'Craftsmanship' for 'Artisan').
    *   **Uncover Surprises:** Discover less obvious or counter-intuitive connections between certain jobs and themes.
    *   **Compare Occupations:** See how different occupations relate to the same theme (e.g., how do 'CEO', 'Artist', and 'Scientist' relate to 'Visionary'?).
    *   **Theme Resonance:** Observe which themes resonate strongly across a wide range of professions versus those specific to a few.
    *   **Data Reflections:** Consider how these patterns might reflect common associations, language use, or even societal biases present in the underlying text data.

    ## How to Use the Visualizer

    1.  **Load Data:**
        *   Use the **default dataset** (check the box) to start exploring immediately.
        *   Or, **upload your own CSV file**. Ensure it has the columns: `Occupation`, `Theme`, and `Cosine_Similarity`.
    2.  **Select a View (in the sidebar):**
        *   **Heatmap:** Get a bird's-eye view of all occupation-theme relationships. Colors indicate similarity strength (often, brighter/darker means stronger - check the legend!). This view uses a square root color scale (viridis scheme by default) to better distinguish values.
        *   **Per Occupation:** Choose an occupation to see a ranked list and bar chart of its most similar themes.
        *   **Per Theme:** Select a theme to see which occupations align most closely with it, again shown as a ranked list and chart.

    ## About the Data & Scores

    *   The scores represent the **cosine similarity** between vector representations (embeddings) of occupations and themes. These embeddings capture semantic meaning learned from large language datasets.
    *   The **default dataset** contains pre-calculated similarities, offering a ready-to-use example.
    *   **Important Note:** Similarity scores reflect patterns learned from text data. They can be influenced by how concepts are discussed, potentially including common associations or societal biases. Interpret the results with this context in mind.

    ## Technical Details

    *   Built with: Python, Streamlit, Pandas, Altair
    *   Visualization uses Altair charts, including a heatmap with a perceptually uniform color scale (viridis) and square root scaling for better differentiation.
    *   Built by [@joonaheino](https://x.com/joonaheino)
    """)

else:  # Visualization page
    # ... (rest of your visualization code remains the same) ...
    st.title("Occupation & Theme Cosine Similarity Visualizations")

    # Try to load the default CSV file
    default_csv_path = "cosine_similarities (1).csv"
    default_df = None

    if os.path.exists(default_csv_path):
        try:
            default_df = pd.read_csv(default_csv_path)
        except Exception as e:
            st.warning(f"Could not load default CSV file: {e}")

    # File uploader with option to use default
    if default_df is not None:
        use_default = st.checkbox("Use default dataset", value=True)
    else:
        st.warning("Default dataset not found. Please upload a CSV.")
        use_default = False # Force user to upload if default isn't there

    if use_default and default_df is not None:
        df = default_df
    else:
        uploaded_file = st.file_uploader("Upload your own CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                st.stop()
        elif use_default and default_df is None:
             st.error("Default dataset selected but could not be loaded. Please upload a file.")
             st.stop()
        else:
            st.info("Please upload a CSV file or select the default dataset to get started.")
            st.stop()


    # Check that the CSV has the expected columns
    expected_cols = {'Occupation', 'Theme', 'Cosine_Similarity'}
    # Handle potential leading/trailing spaces in column names from uploaded CSVs
    df.columns = df.columns.str.strip()
    if not expected_cols.issubset(df.columns):
        st.error(f"CSV must contain columns named: {', '.join(expected_cols)}. Found: {', '.join(df.columns)}")
        st.stop()

    # --- Rest of your visualization code ---
    # Sidebar: Choose visualization type
    st.sidebar.header("Visualization Options")
    view_option = st.sidebar.radio("View by", ["Heatmap", "Per Occupation", "Per Theme"], index=0)  # Set Heatmap as default

    if view_option == "Per Occupation":
        # Let the user select an occupation
        occupation_list = sorted(df["Occupation"].unique())
        if not occupation_list:
             st.warning("No occupations found in the data.")
             st.stop()
        occupation = st.sidebar.selectbox("Select Occupation", occupation_list)
        filtered = df[df["Occupation"] == occupation].copy()
        filtered = filtered.sort_values("Cosine_Similarity", ascending=False)
        st.subheader(f"Themes for **{occupation}** (sorted by similarity)")

        if not filtered.empty:
            # Create a horizontal bar chart using Altair
            chart = alt.Chart(filtered).mark_bar().encode(
                x=alt.X("Cosine_Similarity:Q", title="Cosine Similarity"),
                y=alt.Y("Theme:N", sort='-x', title="Theme"),
                tooltip=["Theme", "Cosine_Similarity"]
            ).properties(width=700, height=400)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(filtered.reset_index(drop=True))
        else:
            st.info("No themes found for the selected occupation.")


    elif view_option == "Per Theme":
        # Let the user select a theme
        theme_list = sorted(df["Theme"].unique())
        if not theme_list:
             st.warning("No themes found in the data.")
             st.stop()
        theme = st.sidebar.selectbox("Select Theme", theme_list)
        filtered = df[df["Theme"] == theme].copy()
        filtered = filtered.sort_values("Cosine_Similarity", ascending=False)
        st.subheader(f"Occupations for **{theme}** (sorted by similarity)")

        if not filtered.empty:
            # Create a horizontal bar chart using Altair
            chart = alt.Chart(filtered).mark_bar().encode(
                x=alt.X("Cosine_Similarity:Q", title="Cosine Similarity"),
                y=alt.Y("Occupation:N", sort='-x', title="Occupation"),
                tooltip=["Occupation", "Cosine_Similarity"]
            ).properties(width=700, height=400)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(filtered.reset_index(drop=True))
        else:
            st.info("No occupations found for the selected theme.")

    elif view_option == "Heatmap":
        st.subheader("Heatmap: Occupations vs. Themes")
        if df.empty:
            st.warning("The dataset is empty. Cannot generate heatmap.")
            st.stop()
        if df['Occupation'].nunique() == 0 or df['Theme'].nunique() == 0:
             st.warning("No occupations or themes found in the data to create a heatmap.")
             st.stop()

        # Pivot the data so that each cell is the cosine similarity
        # Handle potential duplicate Occupation/Theme pairs - take the mean similarity
        pivot_table = pd.pivot_table(df, index="Occupation", columns="Theme", values="Cosine_Similarity", aggfunc='mean')

        # Reset index and melt to long format for Altair
        pivot_long = pivot_table.reset_index().melt(id_vars="Occupation", var_name="Theme", value_name="Cosine_Similarity")

        # Drop rows where Cosine_Similarity is NaN (can happen if aggfunc='mean' results in NaN or original data had NaNs)
        pivot_long.dropna(subset=['Cosine_Similarity'], inplace=True)

        if pivot_long.empty:
             st.warning("Data could not be processed into a valid format for the heatmap (possibly due to missing values or structure issues).")
             st.stop()

        # Calculate min and max for better color scaling
        min_val = pivot_long["Cosine_Similarity"].min()
        max_val = pivot_long["Cosine_Similarity"].max()
        # Ensure min_val and max_val are valid numbers
        if pd.isna(min_val) or pd.isna(max_val):
             st.warning("Could not determine valid min/max similarity values for heatmap scaling.")
             st.stop()
        mid_val = (min_val + max_val) / 2

        # Determine appropriate dimensions based on data size
        num_occupations = pivot_table.shape[0]
        num_themes = pivot_table.shape[1]
        # Basic heuristic for sizing - adjust as needed
        heatmap_width = min(max(num_themes * 20, 600), 1200)
        heatmap_height = min(max(num_occupations * 15, 400), 800)


        # Create a heatmap with Altair using a steeper color gradient
        heatmap = alt.Chart(pivot_long).mark_rect().encode(
            x=alt.X("Theme:N", title="Theme", sort=None, axis=alt.Axis(labelAngle=-45)), # Angle labels if many themes
            y=alt.Y("Occupation:N", title="Occupation", sort=None),
            color=alt.Color(
                "Cosine_Similarity:Q",
                scale=alt.Scale(
                    domain=[min_val, mid_val, max_val],
                    range=['#440154', '#21918c', '#fde725'],  # viridis color scheme
                    type='sqrt'  # Use square root scale for steeper gradient
                ),
                title="Cosine Similarity"
            ),
            tooltip=["Occupation", "Theme", alt.Tooltip("Cosine_Similarity", format=".3f")] # Format tooltip
        ).properties(
             width=heatmap_width,
             height=heatmap_height
        ).interactive() # Enable zooming and panning

        st.altair_chart(heatmap, use_container_width=True)
        # Optionally show the full dataframe below the heatmap
        with st.expander("Show Raw Data"):
             st.dataframe(df)