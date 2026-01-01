"""
Data Cleaning Module for Marketing Campaign Performance Analysis

This module handles loading, validating, cleaning, and preprocessing
raw marketing campaign data before feature engineering and analysis.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class MarketingDataCleaner:
    """
    A class to clean and preprocess marketing campaign data.
    """

    def __init__(self, file_path="data/marketing_data.csv"):
        """
        Initialize the cleaner with the data file path.

        Parameters
        ----------
        file_path : str
            Path to the raw CSV data file
        """
        self.file_path = file_path
        self.df = None

    # ------------------------------------------------------------------
    # Data Loading & Validation
    # ------------------------------------------------------------------

    def load_data(self):
        """Load raw marketing data from CSV."""
        print("Loading data...")
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"ERROR: File not found at '{self.file_path}'")
            return None

    def check_data_quality(self):
        """Perform basic data quality checks."""
        if self.df is None:
            print("No data loaded.")
            return

        print("\n=== DATA QUALITY CHECK ===")
        print(f"Rows: {len(self.df)}")
        print(f"Columns: {len(self.df.columns)}")

        print("\nMissing values:")
        print(self.df.isnull().sum())

        print("\nDuplicate rows:", self.df.duplicated().sum())
        print("\nData types:")
        print(self.df.dtypes)

    # ------------------------------------------------------------------
    # Cleaning Steps
    # ------------------------------------------------------------------

    def clean_dates(self):
        """Convert date columns to datetime and create campaign duration."""
        print("\nCleaning date columns...")

        date_cols = ["reporting_start", "reporting_end"]
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        if all(col in self.df.columns for col in date_cols):
            self.df["campaign_duration_days"] = (
                self.df["reporting_end"] - self.df["reporting_start"]
            ).dt.days

            negative_durations = (self.df["campaign_duration_days"] < 0).sum()
            if negative_durations > 0:
                print(f"Warning: {negative_durations} negative campaign durations found")

    def handle_missing_values(self):
        """Handle missing values using simple, explainable rules."""
        print("\nHandling missing values...")

        numeric_cols = [
            "impressions",
            "clicks",
            "spent",
            "total_conversion",
            "approved_conversion",
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        categorical_cols = ["age", "gender"]
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("Unknown")

    def validate_numeric_columns(self):
        """Validate numeric columns and remove invalid values."""
        print("\nValidating numeric columns...")

        numeric_cols = [
            "impressions",
            "clicks",
            "spent",
            "total_conversion",
            "approved_conversion",
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    print(f"Warning: {negative_count} negative values in {col}")
                    self.df.loc[self.df[col] < 0, col] = 0

    def clean_categorical_data(self):
        """Standardize categorical columns."""
        print("\nCleaning categorical data...")

        if "gender" in self.df.columns:
            self.df["gender"] = self.df["gender"].str.upper().str.strip()

        if "age" in self.df.columns:
            self.df["age"] = self.df["age"].str.strip()

    def remove_duplicates(self):
        """Remove duplicate rows."""
        print("\nRemoving duplicates...")
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        print(f"Removed {before - after} duplicate rows")

    # ------------------------------------------------------------------
    # Lightweight Derived Fields (Cleaning-Level Only)
    # ------------------------------------------------------------------

    def create_basic_derived_features(self):
        """
        Create lightweight derived features strictly related to cleaning.
        Heavy KPIs are handled in feature_engineering.py.
        """
        print("\nCreating basic derived features...")

        if all(col in self.df.columns for col in ["clicks", "impressions"]):
            self.df["ctr"] = np.where(
                self.df["impressions"] > 0,
                self.df["clicks"] / self.df["impressions"],
                0,
            )

    # ------------------------------------------------------------------
    # Output & Pipeline
    # ------------------------------------------------------------------

    def save_cleaned_data(self, output_path="data/cleaned_marketing_data.csv"):
        """Save cleaned dataset."""
        self.df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to {output_path}")
        print(f"Final shape: {self.df.shape}")

    def run_full_pipeline(self):
        """Run the complete data cleaning pipeline."""
        print("=" * 60)
        print("STARTING DATA CLEANING PIPELINE")
        print("=" * 60)

        self.load_data()
        if self.df is None:
            return None

        self.check_data_quality()
        self.clean_dates()
        self.handle_missing_values()
        self.validate_numeric_columns()
        self.remove_duplicates()
        self.clean_categorical_data()
        self.create_basic_derived_features()
        self.save_cleaned_data()

        print("\nDATA CLEANING PIPELINE COMPLETED")
        print("=" * 60)

        return self.df

def main():
    cleaner = MarketingDataCleaner("data/marketing_data.csv")
    return cleaner.run_full_pipeline()

if __name__ == "__main__":
    cleaned_df = main()
    if cleaned_df is not None:
        print("\nSample cleaned data:")
        print(cleaned_df.head())