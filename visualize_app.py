# File: visualize_app.py
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Cosine Similarity Visualizations", layout="wide")
st.title("Occupation & Theme Cosine Similarity Visualizations")
st.markdown("""
Upload your CSV file with columns `Occupation`, `Theme`, and `Cosine_Similarity`.  
This app provides different ways to visualize the results:  
- **Per Occupation**: See which themes are most similar/dissimilar for a selected occupation.  
- **Per Theme**: See which occupations best align with a selected theme.  
- **Heatmap**: Get an overview of all cosine similarity values.
""")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Check that the CSV has the expected columns
    expected_cols = {'Occupation', 'Theme', 'Cosine_Similarity'}
    if not expected_cols.issubset(df.columns):
        st.error(f"CSV must contain these columns: {expected_cols}")
        st.stop()

    # Sidebar: Choose visualization type
    st.sidebar.header("Visualization Options")
    view_option = st.sidebar.radio("View by", ["Per Occupation", "Per Theme", "Heatmap"])

    if view_option == "Per Occupation":
        # Let the user select an occupation
        occupation = st.sidebar.selectbox("Select Occupation", sorted(df["Occupation"].unique()))
        filtered = df[df["Occupation"] == occupation].copy()
        filtered = filtered.sort_values("Cosine_Similarity", ascending=False)
        st.subheader(f"Themes for **{occupation}** (sorted by similarity)")
        
        # Create a horizontal bar chart using Altair
        chart = alt.Chart(filtered).mark_bar().encode(
            x=alt.X("Cosine_Similarity:Q", title="Cosine Similarity"),
            y=alt.Y("Theme:N", sort='-x', title="Theme"),
            tooltip=["Theme", "Cosine_Similarity"]
        ).properties(width=700, height=400)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(filtered.reset_index(drop=True))

    elif view_option == "Per Theme":
        # Let the user select a theme
        theme = st.sidebar.selectbox("Select Theme", sorted(df["Theme"].unique()))
        filtered = df[df["Theme"] == theme].copy()
        filtered = filtered.sort_values("Cosine_Similarity", ascending=False)
        st.subheader(f"Occupations for **{theme}** (sorted by similarity)")
        
        # Create a horizontal bar chart using Altair
        chart = alt.Chart(filtered).mark_bar().encode(
            x=alt.X("Cosine_Similarity:Q", title="Cosine Similarity"),
            y=alt.Y("Occupation:N", sort='-x', title="Occupation"),
            tooltip=["Occupation", "Cosine_Similarity"]
        ).properties(width=700, height=400)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(filtered.reset_index(drop=True))

    elif view_option == "Heatmap":
        st.subheader("Heatmap: Occupations vs. Themes")
        # Pivot the data so that each cell is the cosine similarity
        pivot = df.pivot(index="Occupation", columns="Theme", values="Cosine_Similarity")
        # Reset index and melt to long format for Altair
        pivot_long = pivot.reset_index().melt(id_vars="Occupation", var_name="Theme", value_name="Cosine_Similarity")
        # Create a heatmap with Altair
        heatmap = alt.Chart(pivot_long).mark_rect().encode(
            x=alt.X("Theme:N", title="Theme", sort=None),
            y=alt.Y("Occupation:N", title="Occupation", sort=None),
            color=alt.Color("Cosine_Similarity:Q", scale=alt.Scale(scheme='viridis')),
            tooltip=["Occupation", "Theme", "Cosine_Similarity"]
        ).properties(width=900, height=600)
        st.altair_chart(heatmap, use_container_width=True)
        st.dataframe(df)
else:
    st.info("Please upload your CSV file to get started.")
