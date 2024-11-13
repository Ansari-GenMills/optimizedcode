import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

class MonthGapChecker:
    def __init__(self, project_id: str, logger=None):
        self.project_id = project_id
        self.logger = logger  # Logger instance for logging operations

    def convert_to_datetime(self, df, month_variable):
        """Convert Month Variable to Datetime"""
        df[month_variable] = pd.to_datetime(df[month_variable])
        return df

    def group_by_combination(self, df, comb_variables_list, month_variable='months'):
        """Group Data by Specified Variables"""
        df_grouped = df.groupby(by=comb_variables_list, as_index=False).agg(
            first_month=(month_variable, "min"),
            last_month=(month_variable, "max")
        )
        return df_grouped

    def calculate_month_gap(self, row):
        """Calculate the Gap Between First and Last Month"""
        return (row['last_month'].year - row['first_month'].year) * 12 + row['last_month'].month - row['first_month'].month + 1

    def calculate_actual_last_month(self):
        """Calculate the Actual Last Month"""
        today = date.today()
        d = today - relativedelta(months=1)
        first_day = date(d.year, d.month, 1)
        return pd.to_datetime(first_day)

    def calculate_actual_month_gap(self, row, actual_last_month):
        """Calculate the Gap Between First Month and Actual Last Month"""
        return (actual_last_month.year - row['first_month'].year) * 12 + actual_last_month.month - row['first_month'].month + 1

    def calculate_delta_month_gap(self, row):
        """Calculate the Delta Between Actual Month Gap and Found Month Gap"""
        return row['actual_month_gap'] - row['month_gap']

    def check_warnings(self, row, expected_month_gap):
        """Check for Warnings Based on Gaps"""
        warnings = []
        if row['month_gap'] < expected_month_gap[row.name]:
            warnings.append("In-between months are missing")
        if row['actual_month_gap'] > row['month_gap']:
            warnings.append("Tail-end data is missing")
        if len(warnings) == 0:
            return "Pass"
        else:
            return "; ".join(warnings)

    def generate_results(self, df, group_by_columns, month_variable='months'):
        """Main Function to Generate Warnings and Pass/Fail Status"""
        df = self.convert_to_datetime(df, month_variable)
        df_grouped = self.group_by_combination(df, group_by_columns, month_variable)
        
        # Calculate the month gaps
        df_grouped['month_gap'] = df_grouped.apply(self.calculate_month_gap, axis=1)
        actual_last_month = self.calculate_actual_last_month()
        df_grouped['actual_month_gap'] = df_grouped.apply(self.calculate_actual_month_gap, actual_last_month=actual_last_month, axis=1)
        
        # Calculate the delta month gap
        df_grouped['delta_month_gap'] = df_grouped.apply(self.calculate_delta_month_gap, axis=1)
        
        # Calculate expected month gap for pass/fail criteria
        expected_month_gap = df_grouped.apply(lambda row: self.calculate_month_gap({'first_month': row['first_month'], 'last_month': actual_last_month}), axis=1)
        
        # Generate warnings and pass/fail status
        df_grouped['Status'] = df_grouped.apply(self.check_warnings, expected_month_gap=expected_month_gap, axis=1)
        df_grouped['Pass/Fail'] = df_grouped['Status'].apply(lambda x: 'Fail' if 'missing' in x else 'Pass')
        
        return df_grouped
