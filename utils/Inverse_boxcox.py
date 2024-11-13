import pandas as pd
import numpy as np
from scipy.special import inv_boxcox

def inverse_boxcox(df, transformed_column, category_column, subcategory_column, lambda_df):
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
if __name__ == "__main__":
    # Apply inverse Box-Cox transformation
    inverse_transformed_df = inverse_boxcox(transformed_df, transformed_column='value',                       category_column='category',subcategory_column='subcategory',lambda_df=lambda_df)

    print("Inverse Transformed DataFrame:")
    print(inverse_transformed_df.head())
