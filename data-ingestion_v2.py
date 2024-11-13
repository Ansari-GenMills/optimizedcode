#Data Ingestion_V2
import argparse
import pandas as pd
import re
import yaml
import os
import sys

try:
    from gmi_gds_logging import console_logger, file_logger
    from gmi_gds_data_read_write.reader import bq_reader
    from gmi_gds_data_read_write.writer import gcs_writer
except ModuleNotFoundError as e:
    print("Environment not set up correctly, internal libraries not found in kernel")
    raise e
    
class DataIngestion:
    def __init__(self, project_id, required_columns, queries):
        self.project_id = project_id
        self.required_columns = required_columns
        self.queries = queries
        self.console_log = console_logger.ConsoleLogger("console")
        self.combined_logger = file_logger.FileLogger(name="Data_Ingestion", logger=self.console_log)
        self.db_reader = bq_reader.BQReader(self.project_id, self.combined_logger)

    def _extract_file_name(self, sql_query):
        """Extracts a simplified file name from an SQL query."""
        match = re.search(r'FROM `([^`]+)`', sql_query)
        if match:
            full_table_name = match.group(1)
            simplified_table_name = full_table_name.split('.')[-1]
            return f"{simplified_table_name}.csv"
        else:
            raise ValueError("Table name could not be extracted from the SQL query.")

    def _process_query(self, query):
        """Executes the query and processes the DataFrame."""
        try:
            df = self.db_reader.read_data(query)
            if df is not None:
                # Check for required columns
                if not set(self.required_columns).issubset(df.columns):
                    missing_columns = [col for col in self.required_columns if col not in df.columns]
                    print(f"Query: {query} - Missing columns: {missing_columns}, skipping this file.")
                    return None
                # Ensure only required columns are included
                df = df[self.required_columns]
                return df
            else:
                print(f"No data returned for query: {query}, skipping.")
                return None
        except Exception as e:
            print(f"Failed to read or process data for query: {query}. Error: {e}")
            return None

    def ingest_data(self):
        """Main method to process all queries and merge the results."""
        data_frames = []
        for query in self.queries:
            df = self._process_query(query)
            if df is not None:
                data_frames.append(df)
                print(f"Successfully read and processed data for query: {query}")

        # Merge all DataFrames if available
        if data_frames:
            final_df = pd.concat(data_frames, ignore_index=True)
            print("Successfully merged all data frames.")
            return final_df
        else:
            print("No data frames available for merging.")
            return None

# Function to load configuration from a YAML file
def load_config(config_file):
    """
    Loads configuration from a YAML file.
    
    Args:
    - config_file: Path to the YAML configuration file.
    
    Returns:
    - config: Dictionary containing the loaded configuration.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to parse command line arguments
def parse_args():
    """
    Parse command line arguments, but detect if running in Jupyter Notebook.
    
    Returns:
    - args: Namespace object with parsed arguments or defaults for notebook.
    """
    # Detect if running inside a notebook
    if 'ipykernel' in sys.modules:
        print("Detected notebook environment, skipping argument parsing.")
        return argparse.Namespace(config='utils/utils_config.yml') 
    else:
        parser = argparse.ArgumentParser(description="Data Ingestion Script")
        # Command-line argument for the YAML configuration file
        parser.add_argument('--config', required=True, help="utils/utils_config.yml")
        return parser.parse_args()

# Main function to handle the ingestion process
def run_data_ingestion(config_file):
    """Runs the data ingestion process."""
    # Load config from the YAML file passed as argument
    config = load_config(config_file)
    
    # Access configuration values from the YAML file
    project_id = config['project_id']
    required_columns = config['required_columns']
    queries = config['queries']
    gcs_bucket = config['gcs_bucket_name']  # GCS bucket name
    gcs_path = config['source_input_path']  # Path in GCS where the data will be saved

    # Initialize and run the data ingestion
    data_ingestion = DataIngestion(project_id=project_id, required_columns=required_columns, queries=queries)
    input_data = data_ingestion.ingest_data()
    
    # Save the resulting DataFrame to GCS using GCSWriter
    if input_data is not None:
        gcs_writer_obj = gcs_writer.GCSWriter(project_id, data_ingestion.combined_logger)
        gcs_writer_obj.write_data(input_data, gcs_bucket, gcs_path, is_overwrite=True)
        print(f"Data written to GCS: gs://{gcs_bucket}/{gcs_path}")
    else:
        print("No data ingested, nothing to save.")

# Command-line parser trigger at the end
if __name__ == "__main__":
    args = parse_args()
    run_data_ingestion(args.config)

# Notebook execution section
def notebook_run(config_file):
    """Run data ingestion from a Jupyter Notebook."""
    run_data_ingestion(config_file)

# Example Usage in a Jupyter Notebook
# If you want to run the script in a Jupyter notebook, you can call the function directly:
# notebook_run('/path/to/DataIngestion_config.yaml')
