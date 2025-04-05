# Occupation & Theme Cosine Similarity Visualizer

A Streamlit application for visualizing cosine similarity between occupations and themes using embeddings.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run visualize_app.py
```

2. Upload a CSV file with the following columns:
   - `Occupation`
   - `Theme`
   - `Cosine_Similarity`

3. Choose your visualization type:
   - Per Occupation: See which themes are most similar/dissimilar for a selected occupation
   - Per Theme: See which occupations best align with a selected theme
   - Heatmap: Get an overview of all cosine similarity values

## Project Structure

- `visualize_app.py`: Main Streamlit application
- `requirements.txt`: Project dependencies
- `.gitignore`: Specifies files to be ignored by git 