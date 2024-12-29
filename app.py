# Import Library
import streamlit as st
from dotenv import load_dotenv
import os
import vertexai
from google.cloud import bigquery
from setup import (
    multiturn_generate_content, 
    get_available_dataset,
    get_columns_from_datasets, 
    generate_sql_query, 
    clean_generated_query, 
    execute_generated_query
)

# Load environment variables from .env file
load_dotenv()

# Access the variables
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Authenticate with GCP
vertexai.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
client = bq_client

# Set Streamlit layout and header
st.set_page_config(layout="wide")
st.header("Ask a Question")

# User input for the question
user_request = st.text_area(
    "Enter your question"
    )

# Set up two columns
left_col, right_col = st.columns(2)

# Handle logic
if st.button('Submit', key='main_submit'):
    # Prepare datasets and columns
    get_available_dataset(PROJECT_ID)

    # Define constants
    folder_path = "json-files"
    json_file = "json-files/dataset.json"

    get_columns_from_datasets(json_file, 
                            PROJECT_ID, 
                            output_folder="json-files", 
                            output_file="columns.json")
    
    # Generate and clean SQL query
    generated_query = generate_sql_query(folder_path, user_request)
    cleaned_query = clean_generated_query(generated_query)
    
    # Left column: Display the query
    with left_col:
        st.subheader("Generated Query")
        if cleaned_query:
            st.code(cleaned_query)
        else:
            st.warning("Failed to generate a valid query.")
    
    # Right column: Display the DataFrame
    with right_col:
        st.subheader("Extracted Data")
        try:
            df = execute_generated_query(PROJECT_ID, cleaned_query)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error executing query: {e}")