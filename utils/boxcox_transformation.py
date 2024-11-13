#BoxCox Transforamtion and saving Lamda Value
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox


class BoxCox:
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger  # Logger instance for logging operations
        
    #def apply_boxcox(self, df: pd.DataFrame, value_column: str , category_column: str  , subcategory_column: str ) -> tuple(pd.Series, pd.Series):   
    def apply_boxcox(self, df, value_column, category_column, subcategory_column):
        lambda_values = {}  # Dictionary to store lambda values for each group

        for _, group in df.groupby([category_column, subcategory_column]):
            values = group[value_column].values

            # Check if values are constant
            if np.all(values == values[0]):
                self.logger.warning(f"Skipping Box-Cox transformation for constant data in group {group}")
                transformed_values = values  # Keep values as they are
                lam = None  # No lambda for constant data
            else:
                # Apply Box-Cox transformation if values are non-constant
                transformed_values, lam = boxcox(values)

            # Store the transformed values back in the DataFrame
            df.loc[group.index, value_column] = transformed_values

            # Record lambda value used for transformation
            lambda_values[(group[category_column].iloc[0], group[subcategory_column].iloc[0])] = lam

        # Return the transformed DataFrame and lambda values as a new DataFrame
        lambda_df = pd.DataFrame.from_dict(lambda_values, orient='index', columns=['Lambda'])
        return df, lambda_df

    
         
    def inverse_boxcox(self, df: pd.DataFrame, transformed_column: str , category_column: str  , subcategory_column: str , lambda_df: pd.DataFrame) -> pd.Series:   
    #def inverse_boxcox(df, transformed_column, category_column, subcategory_column, lambda_df):
        """
        Apply inverse Box-Cox transformation to the data using stored lambda values for each subcategory.

        :param df: pandas DataFrame containing the transformed data
        :param transformed_column: str, name of the column to be inverse transformed
        :param category_column: str, name of the column that represents categories
        :param subcategory_column: str, name of the column that represents subcategories
        :param lambda_df: pandas DataFrame containing lambda values with columns ['category', 'subcategory', 'lambda']
        :return: DataFrame with the inverse transformed values
        """
        df = df.copy()

        # Merge the lambda values into the original DataFrame
        df = df.merge(lambda_df, on=[category_column, subcategory_column], how='left')

        # Apply inverse Box-Cox transformation using the stored lambda values
        for (category, subcategory), group in df.groupby([category_column, subcategory_column]):
            lam = group['lambda'].iloc[0]  # Get the lambda value for this subcategory
            if lam is not None:
                inverse_transformed_values = inv_boxcox(group[transformed_column], lam)
                df.loc[group.index, transformed_column] = inverse_transformed_values
            else:
                raise ValueError(f"Missing lambda value for category '{category}' and subcategory '{subcategory}'.")

        # Drop lambda column used for merging
        df = df.drop(columns=['lambda'])

        return df

    
# Example usage
###if __name__ == "__main__":
###    # Sample data with category and subcategory
###    np.random.seed(0)
###    df = pd.DataFrame({
###        'category': ['A'] * 50 + ['B'] * 50,
###        'subcategory': ['X'] * 25 + ['Y'] * 25 + ['X'] * 25 + ['Y'] * 25,
###        'value': np.random.exponential(scale=2, size=100) + 1  # Ensure values are positive
###    })
###
###    # Apply Box-Cox transformation and get lambda values
###    transformed_df, lambda_df = apply_boxcox(df, value_column='value', category_column='category', subcategory_column='subcategory')
###
###    print("Transformed DataFrame:")
###    print(transformed_df.head())
###
###    print("\nLambda Values DataFrame:")
###    print(lambda_df)