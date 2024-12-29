# Import Library
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from google.cloud import bigquery
import os
import pandas as pd
import json
import re

# Setup GCP project and location

PROJECT_ID = "eikon-dev-ai-team"  
LOCATION = "us-central1"  
DATASET_ID = 'test1' 

vertexai.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
client = bq_client

def multiturn_generate_content(prompt,question):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=prompt
    )
    chat = model.start_chat()

    generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        ]
    
    result = chat.send_message(
        question,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # Extract and return the text content
    if result and hasattr(result, "candidates"):
        return [candidate.content.parts[0].text for candidate in result.candidates]
    return None

def get_available_dataset(project_id, output_folder="json-files", output_file="dataset.json"):
    """
    Retrieves and saves the list of available dataset in the specified project as a JSON file.
    The INFORMATION_SCHEMA.SCHEMATA view provides information about the datasets in a project or region. The view returns one row for each dataset.

    Parameters:
        project_id (str): The Google Cloud project ID.
        output_file (str): The path to the JSON file where the dataset will be saved.

    Returns:
        pandas.DataFrame: A DataFrame containing the available dataset.
    """
    client = bq_client

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Full path to the output file
    output_path = os.path.join(output_folder, output_file)

    # Query to get the list of dataset
    query_schema = f"""
    SELECT schema_name
    FROM `{project_id}.region-us.INFORMATION_SCHEMA.SCHEMATA`;
    """

    # Execute the query and fetch the results into a DataFrame
    job = client.query(query_schema)
    list_of_dataset = job.to_dataframe()

    # Save the DataFrame as a JSON file
    list_of_dataset.to_json(output_path, orient="records", indent=4)

    # Print confirmation
    print(f"Available dataset saved to {output_path}")

    return list_of_dataset

def get_columns_from_datasets(json_file, project_id, output_folder="json-files", output_file="columns.json"):
    """
    Retrieves column names and types for all tables in the datasets from the JSON file
    and saves them in one JSON file.

    Parameters:
        json_file (str): Path to the JSON file containing dataset names.
        project_id (str): Google Cloud project ID.
        output_file (str): Path to the output JSON file where column information will be saved.

    Returns:
        dict: A dictionary containing dataset names, their tables, and column information.
    """
    client = bq_client

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Full path to the output file
    output_path = os.path.join(output_folder, output_file)

    # Load the datasets from the JSON file
    datasets_df = pd.read_json(json_file, orient="records")

    # Dictionary to store dataset, table, and column information
    dataset_columns = {}

    # Iterate over each dataset
    for dataset in datasets_df["schema_name"]:
        dataset_columns[dataset] = {}

        # Query to get table names
        tables_query = f"""
        SELECT table_name
        FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.TABLES`;
        """
        tables_job = client.query(tables_query)
        tables = [row["table_name"] for row in tables_job.result()]

        # For each table, fetch column information
        for table in tables:
            columns_query = f"""
            SELECT column_name, data_type
            FROM `{project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table}';
            """
            columns_job = client.query(columns_query)
            columns = {row["column_name"]: row["data_type"] for row in columns_job.result()}

            # Store the column information under the table name
            dataset_columns[dataset][table] = columns

    # Save the dataset-tables-columns mapping to a JSON file
    with open(output_path, "w") as file:
        json.dump(dataset_columns, file, indent=4)

    print(f"Columns saved to {output_path}")

    return dataset_columns

def generate_sql_query(folder_path, user_request):
    """
    Generates a natural language prompt for an LLM to create a SQL query based on saved schemas, tables, and columns.

    Parameters:
        schemas_file (str): Path to the JSON file containing schemas.
        tables_file (str): Path to the JSON file containing tables.
        columns_file (str): Path to the JSON file containing columns.
        user_request (str): The user's request in plain English.

    Returns:
        str: A detailed prompt for the LLM to create the desired SQL query.
    """

    # Construct file paths
    schemas_file = os.path.join(folder_path, "dataset.json")
    columns_file = os.path.join(folder_path, "columns.json")
    
    # Load data from JSON files
    try:
        with open(schemas_file, "r") as file:
            schemas = json.load(file)
        with open(columns_file, "r") as file:
            columns = json.load(file)
    except FileNotFoundError as e:
        return f"Error: {e}. Ensure the required JSON files exist in {folder_path}."
    except json.JSONDecodeError as e:
        return f"Error: Failed to parse JSON. {e}"

    # Prepare the prompt
    prompt = f"""
    You are an SQL expert. Based on the following schema, table, and column metadata, generate an SQL query to fulfill the user's request.

    ### Available Schemas
    The following schemas are available:
    {', '.join([schema['schema_name'] for schema in schemas])}

    ### Columns in Each Table
    Here are the columns and their types for each table:
    {json.dumps(columns, indent=4)}

    ### User Request
    {user_request}

    ### Your Task
    1. Identify the appropriate schema and table(s) based on the user's request.
    2. Select the columns and apply any filtering, sorting, or aggregation mentioned.
    3. Write the SQL query, formatted and ready to run.

    ### Example Output
    #### Request: Retrieve the `id` and `name` from `table1` in `dataset1`.
    ```sql
    SELECT id, name
    FROM `dataset1.table1`;
    ```
    """
    result = multiturn_generate_content(prompt,user_request)
    return result

def clean_generated_query(generated_query):
    # Join the list into a single string if it's in list format
    query_text = "".join(generated_query)
    # Use regex to extract the SQL code inside ```sql and ```
    cleaned_query = re.sub(r"```sql\n|```", "", query_text).strip()
    return cleaned_query

def execute_generated_query(project_id, query):
  """
  Executes the generated SQL query using BigQuery.

  Parameters:
    project_id (str): The Google Cloud project ID.
    query (str): The SQL query to execute.

  Returns:
    pandas.DataFrame: A DataFrame containing the query results.
  """
    # Initialize BigQuery client
  client = bigquery.Client(project=project_id)

  try:
      # Execute the query
      job = client.query(query)
      results = job.to_dataframe()

      print(f"Query executed successfully. {len(results)} rows retrieved.")
      return results

  except Exception as e:
      print(f"Error executing query: {e}")
      return None