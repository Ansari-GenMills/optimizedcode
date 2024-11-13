#same# Final Preprocessing.py
import yaml
#from dtype_handler import DataTypeConverter
#from date_gap_check import MonthGapChecker
#from missingvalueimputer import TimeSeriesMissingValueHandler
#from time_series_outlier_handler import TimeSeriesOutlierHandler
#from categorical_binner import CategoricalBinner
#from boxcox_transformation import BoxCox
from utils import DataTypeConverter, MonthGapChecker, TimeSeriesMissingValueHandler,TimeSeriesOutlierHandler,CategoricalBinner,BoxCox

from gmi_gds_data_read_write.reader import gcs_reader
from gmi_gds_data_read_write.writer import gcs_writer
from gmi_gds_logging import console_logger, file_logger

def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as e:
        console_logger.error(f"Configuration file not found: {e}")
        return None
    except yaml.YAMLError as e:
        console_logger.error(f"Error reading configuration file: {e}")
        return None

def save_to_gcs(writer, data, file_paths, bucket_name, logger):
    """Save multiple files to GCS in a loop."""
    for data_df, output_path in zip(data, file_paths):
        if data_df is not None and not data_df.empty:
            try:
                writer.write_data(data_df, bucket_name, output_path, is_overwrite=True)
                logger.info(f"Data successfully written to GCS: gs://{bucket_name}/{output_path}")
            except RuntimeError as e:
                logger.error(f"Failed to write data to GCS: {e}")
        else:
            logger.warning(f"No data generated for {output_path}, skipping save.")

def remove_unnamed_columns(df):
    """Remove all columns with names starting with 'Unnamed:'."""
    unnamed_columns = df.columns[df.columns.str.contains('^Unnamed:')]
    return df.drop(columns=unnamed_columns, errors='ignore')

def main():
    # Load configuration from the YAML file
    config_file = "utils/utils_config.yml"  # Path to your configuration file
    config = load_config(config_file)
    
    if config is None:
        console_logger.error("Configuration loading failed. Terminating the pipeline.")
        return

    # Access configuration variables
    project_id = config.get('project_id')
    column_types = config.get('column_types', {})
    gcs_bucket_name = config.get('gcs_bucket_name')
    dtype_output_path = config.get('dtype_output_path')
    gap_check_output_path = config.get('date_gap_check_output_path')
    imputed_df_output_path = config.get('imputed_df_output_path')
    outlier_treated_df_output_path = config.get("outlier_treated_df_output_path")
    group_by_columns = config.get('group_by_columns', [])
    month_variable = config.get('month_variable')
    imputation_method = config.get('imputation_method', 'linear')
    k_neighbors = config.get('k_neighbors', 5)
    columns_to_impute = config.get('columns_to_impute', [])
    outlier_columns_to_process = config.get('outlier_columns', [])
    outlier_method = config.get('outlier_method')
    outlier_threshold = config.get('outlier_threshold')
    binned_df_output_path = config.get('binned_df_output_path')
    boxcox_transform_df_output_path=config['boxcox_transform_df_output_path']
    lambda_df_output_path=config['lambda_df_output_path']
    value_column=config["value_column"]
    category_column=config["category_column"]
    subcategory_column=config["subcategory_column"]

    # Initialize logger (Console logger and File logger)
    console_log = console_logger.ConsoleLogger("console")
    combined_logger = file_logger.FileLogger(name="data_preprocessing", logger=console_log)

    # Initialize GCS Reader and Writer
    gcs_reader_obj = gcs_reader.GCSReader(project_id=project_id, logger=combined_logger)
    gcs_writer_obj = gcs_writer.GCSWriter(project_id=project_id, logger=combined_logger)

    # Read input data from GCS
    dtype_input_path = config.get('source_input_path')
  
    try:
        input_data = gcs_reader_obj.read_data(dtype_input_path, gcs_bucket_name)
    except Exception as e:
        combined_logger.error(f"Failed to read data from GCS: {e}")

    if input_data is None:
        combined_logger.error("No input data found. Terminating the pipeline.")
        return

    # Remove all unnamed columns
    input_data = remove_unnamed_columns(input_data)

    # Step 1: Data Type Conversion
    try:
        converter = DataTypeConverter(project_id=project_id, logger=combined_logger)
        converted_df = converter.convert_dataframe(input_data, column_types)
    except Exception as e:
        combined_logger.error(f"Error during data type conversion: {e}")
        
    # Step 2: Month Gap Check
    try:
        month_gap_checker = MonthGapChecker(project_id=project_id, logger=combined_logger)
        month_gap_check = month_gap_checker.generate_results(df=converted_df, group_by_columns=group_by_columns, month_variable=month_variable)
    except Exception as e:
        combined_logger.error(f"Error during month gap check: {e}")
  
    # Step 3: Missing Value Imputation
    try:
        missing_value_handler = TimeSeriesMissingValueHandler(project_id=project_id, logger=combined_logger, k_neighbors=k_neighbors)
        imputed_data = missing_value_handler.impute_missing_values(converted_df, outlier_columns_to_process, group_by=group_by_columns, method=imputation_method)
    except Exception as e:
        combined_logger.error(f"Error during missing value imputation: {e}")

    # Step 4: Outlier Handling
    try:
        outlier_handler = TimeSeriesOutlierHandler(project_id=project_id, method=outlier_method, threshold=outlier_threshold)
        outlier_treated_df= outlier_handler.handle_outliers(imputed_data, columns=outlier_columns_to_process, group_by=group_by_columns)

    except Exception as e:
        combined_logger.error(f"Error during missing value imputation: {e}")
        
    '''#step 5: Categorical binning
    try:
        binner_categorical = CategoricalBinner(project_id=project_i,logger=combined_logger,column_names=config['column_names'],
                                   default_bins=config['default_bins'],rules=config['rules'])  # Make sure this is included
        df_binned =binner_categorical.bin_categorical_variables(outlier_treated_df)  
    except Exception as e:
        combined_logger.error(f"Error during missing value imputation: {e}")
        ''' 
    #step 6: boxcox Transformation
    try:
        boxcox_transform = BoxCox(project_id=project_id, logger=combined_logger)
        boxcox_transform_df, lambda_df= boxcox_transform.apply_boxcox(outlier_treated_df,value_column, category_column, subcategory_column)
    except Exception as e:
        combined_logger.error(f"Error during missing value imputation: {e}")                     
    
    # Save files to GCS in a loop
    data_to_save = [converted_df, month_gap_check, imputed_data, outlier_treated_df,boxcox_transform_df,lambda_df]#,df_binned,binned_df_output_path
    file_paths = [dtype_output_path, gap_check_output_path, imputed_df_output_path,outlier_treated_df_output_path,boxcox_transform_df_output_path,lambda_df_output_path] #, 
    save_to_gcs(gcs_writer_obj, data_to_save, file_paths, gcs_bucket_name, combined_logger)

    combined_logger.info("Data processing pipeline completed successfully.")

if __name__ == "__main__":
    main()