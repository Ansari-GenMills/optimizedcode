import pandas as pd

class CategoricalBinner:
    def __init__(self, project_id, logger, column_names, default_bins, rules):
        self.project_id = project_id
        self.logger = logger
        self.column_names = column_names
        self.default_bins = default_bins
        self.rules = rules

    def bin_categorical_variables(self, df):
        """
        Bins multiple categorical variables in a DataFrame according to the provided binning rules.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with new binned columns added.
        """
        # Iterate through each column to be binned
        for idx, column_name in enumerate(self.column_names):
            default_bin = self.default_bins[idx]
            
            # Create a new column name for the binned variable
            new_column_name = f"{column_name}_binned"
            
            # Initialize the new column with the default bin value
            df[new_column_name] = default_bin
            
            # Get the rules for the current column
            column_rules = self.rules.get(column_name, {})
            
            # Loop through the rules to assign new categories
            for new_bin, categories in column_rules.items():
                df.loc[df[column_name].isin(categories), new_column_name] = new_bin
            
            # Log the binning process for each column
            self.logger.info(f"Binned column '{column_name}' with rules: {column_rules}")

        return df
