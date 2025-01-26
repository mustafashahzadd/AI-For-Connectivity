from groq import Groq
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# API key for Groq
GROQ_API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load the filtered CSV data
filtered_data_path = r"D:\Desktop\computer science\Backend Developement\API\AI-For-Connectivity\filtered_measurements.csv"
try:
    filtered_df = pd.read_csv(filtered_data_path, encoding='utf-8')
    filtered_df.dropna(subset=['school_name', 'country'], inplace=True)
    filtered_df['school_name'] = filtered_df['school_name'].astype(str)
    filtered_df['country'] = filtered_df['country'].astype(str)
except Exception as e:
    st.error(f"Error loading the CSV file: {str(e)}")
    st.stop()

# List of allowed countries for the dropdown
allowed_countries = [
    "Botswana", "Mongolia", "Uzbekistan", "Fiji", "Namibia",
    "Kenya", "Bosnia Hergovina", "Kazakhstan",
    "Saint Vincent and the Grenadines", "Srilanka", "Rawansa"
]

# Filter data to include only allowed countries
filtered_df = filtered_df[filtered_df['country'].isin(allowed_countries)]

# FAISS index creation
def create_faiss_index(dataframe):
    try:
        dataframe['combined_text'] = dataframe['school_name'] + " " + dataframe['country']
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(dataframe['combined_text'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, model
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        st.stop()

# Create FAISS index
faiss_index, embedding_model = create_faiss_index(filtered_df)

# Function to retrieve data from FAISS
def retrieve_data_faiss(query, num_results=5):
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding, num_results)
        results = filtered_df.iloc[indices[0]]
        return results
    except Exception as e:
        st.error(f"Error retrieving data: {str(e)}")
        return pd.DataFrame()

# Analyze data and calculate priority score
def analyze_data(country, school_name):
    query = f"{school_name} {country}"
    retrieved_data = retrieve_data_faiss(query)

    if retrieved_data.empty:
        return None, None

    selected_school = retrieved_data.iloc[0]
    download_speed = selected_school['download_speed']
    upload_speed = selected_school['upload_speed']
    latency = selected_school['latency']

    # Priority score calculation
    priority_score = (
        (1 - download_speed) * 0.2 +
        (1 - upload_speed) * 0.2 +
        latency * 0.2
    )

    # Development levels
    development = (
        "Low Development" if priority_score > 0.7 else
        "Medium Development" if priority_score > 0.5 else
        "High Development"
    )

    return {
        "download_speed": download_speed,
        "upload_speed": upload_speed,
        "latency": latency,
        "priority_score": priority_score,
        "development": development
    }, retrieved_data

# Function to use Groq for AI/ML solutions
def suggest_solutions(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content if hasattr(response, "choices") else "No valid response."
    except Exception as e:
        return f"Error generating solution: {str(e)}"

# Streamlit UI
st.title("School Network Performance Analyzer with Groq API")

# Country dropdown
selected_country = st.selectbox("Select a Country", allowed_countries)
schools_in_country = filtered_df[filtered_df['country'] == selected_country]['school_name'].unique()

if schools_in_country.size > 0:
    selected_school = st.selectbox("Select a School", schools_in_country)
else:
    st.error("No schools available for the selected country.")
    st.stop()

if st.button("Analyze"):
    result, retrieved_data = analyze_data(selected_country, selected_school)

    if result:
        st.subheader("Current Metrics")
        st.write(f"**Download Speed:** {result['download_speed']} Mbps")
        st.write(f"**Upload Speed:** {result['upload_speed']} Mbps")
        st.write(f"**Latency:** {result['latency']} ms")
        st.write(f"**Priority Score:** {result['priority_score']:.2f} ({result['development']})")

        # Include retrieved data in the prompt for RAG
        solution_prompt = f"""
You are a predictive analyst. Based on the following school network data:

- **Country:** {selected_country}
- **School:** {selected_school}
- **Download Speed:** {result['download_speed']} Mbps
- **Upload Speed:** {result['upload_speed']} Mbps
- **Latency:** {result['latency']} ms

### Step 1:
Analyze the network performance and provide insights on its impact on the school's development in rural or urban areas.

### Step 2:
Predict a development score based on the following criteria:
- If the average speed (download + upload / 2) is low, and latency is high, classify as:
  - **Low Development** if the score is below 0.7
  - **Medium Development** if the score is between 0.7 and 0.8
  - **High Development** if the score is above 0.8

### Step 3:
Suggest solutions to improve the development score to Medium or High Development levels.
Provide estimates for the required download speed, upload speed, and latency improvements to achieve this.

### Step 4:
Summarize the findings and include actionable steps for improving the network performance and boosting development metrics.

"""
        solution = suggest_solutions(solution_prompt)

        st.subheader("Suggested Solution")
        st.write(solution)
    else:
        st.error("No data available for the selected country and school.")
