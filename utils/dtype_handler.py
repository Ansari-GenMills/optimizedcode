# datatype_handler

import pandas as pd
from typing import Dict, Any

class DataTypeConverter:
    def __init__(self, project_id: str, logger):
        self.project_id = project_id
        self.logger = logger  # Logger instance for logging operations

    def convert_column(self, df: pd.DataFrame, column_name: str, dtype: str) -> pd.Series:
        """Convert a DataFrame column to the specified data type."""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in DataFrame.")
        
        # Apply the conversion function to the column
        return df[column_name].apply(lambda x: self._convert_value(x, dtype))

    def convert_dataframe(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Convert multiple DataFrame columns based on a dictionary of column names and desired data types."""
        for column_name, dtype in column_types.items():
            try:
                df[column_name] = self.convert_column(df, column_name, dtype)
            except ValueError as e:
                print(f"Warning: {e}")
        return df

    def _convert_value(self, value: Any, dtype: str) -> Any:
        """Convert a single value to the specified data type."""
        conversion_functions = {
            'int': int,
            'float': float,
            'str': str,
            'bool': self._to_bool,
            'datetime': pd.to_datetime
        }
        try:
            if dtype in conversion_functions:
                return conversion_functions[dtype](value)
            else:
                raise ValueError(f"Unsupported dtype '{dtype}'")
        except Exception as e:
            raise ValueError(f"Error converting value '{value}' to {dtype}: {e}")

    def _to_bool(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value  # If the value is already a boolean, return it directly

        if isinstance(value, str):
            value = value.lower()
            if value in ['true', '1']:
                return True
            elif value in ['false', '0']:
                return False
            else:
                raise ValueError(f"Cannot convert value '{value}' to bool.")
        return bool(value)
