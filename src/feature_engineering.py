"""
Feature Engineering Module for Marketing Campaign Performance Analysis

This module creates analytical features and KPIs from cleaned marketing data.
It focuses on business-relevant metrics used for reporting and insight generation.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class MarketingFeatureEngineer:
    """
    A class to engineer analytical features from cleaned marketing data.
    """

    def __init__(self, data_path="data/cleaned_marketing_data.csv"):
        """
        Initialize the feature engineer.

        Parameters
        ----------
        data_path : str
            Path to the cleaned CSV data file
        """
        self.data_path = data_path
        self.df = None

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_cleaned_data(self):
        """Load cleaned marketing data."""
        print("Loading cleaned data...")
        try:
            self.df = pd.read_csv(
                self.data_path,
                parse_dates=["reporting_start", "reporting_end"]
            )
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    # ------------------------------------------------------------------
    # Time-Based Features
    # ------------------------------------------------------------------

    def create_time_features(self):
        """Create time-based features from campaign dates."""
        print("\nCreating time-based features...")

        if "reporting_start" not in self.df.columns:
            print("  reporting_start column not found")
            return

        self.df["campaign_year"] = self.df["reporting_start"].dt.year
        self.df["campaign_month"] = self.df["reporting_start"].dt.month
        self.df["campaign_day_of_week"] = self.df["reporting_start"].dt.day_name()
        self.df["campaign_week"] = self.df["reporting_start"].dt.isocalendar().week

        print("  Created year, month, week, and day-of-week features")

    # ------------------------------------------------------------------
    # Core KPI Engineering
    # ------------------------------------------------------------------

    def create_kpi_features(self):
        """
        Create core marketing KPIs.
        All formulas are aligned with available CSV columns.
        """
        print("\nCreating KPI features...")

        # CTR
        self.df["ctr"] = np.where(
            self.df["impressions"] > 0,
            self.df["clicks"] / self.df["impressions"],
            0
        )

        # Conversion rate (click → conversion)
        self.df["conversion_rate"] = np.where(
            self.df["clicks"] > 0,
            self.df["total_conversion"] / self.df["clicks"],
            0
        )

        # Approved conversion rate
        self.df["approved_conversion_rate"] = np.where(
            self.df["total_conversion"] > 0,
            self.df["approved_conversion"] / self.df["total_conversion"],
            0
        )

        # Cost per acquisition (approved conversions)
        self.df["cost_per_acquisition"] = np.where(
            self.df["approved_conversion"] > 0,
            self.df["spent"] / self.df["approved_conversion"],
            np.nan
        )

        # Cost per 1000 impressions (CPM)
        self.df["cpm"] = np.where(
            self.df["impressions"] > 0,
            (self.df["spent"] / self.df["impressions"]) * 1000,
            0
        )

        print("  Created CTR, conversion rate, CPA, and CPM")

    # ------------------------------------------------------------------
    # Demographic Features
    # ------------------------------------------------------------------

    def create_demographic_features(self):
        """Create demographic-based features."""
        print("\nCreating demographic features...")

        if "age" in self.df.columns:
            self.df["age_group"] = self.df["age"]

        if "gender" in self.df.columns and "age_group" in self.df.columns:
            self.df["demographic_segment"] = (
                self.df["gender"] + "_" + self.df["age_group"]
            )

        print("  Created age_group and demographic_segment features")

    # ------------------------------------------------------------------
    # Aggregated Performance Metrics
    # ------------------------------------------------------------------

    def create_aggregated_features(self):
        """Create campaign- and demographic-level aggregates."""
        print("\nCreating aggregated features...")

        # Campaign-level metrics
        campaign_metrics = self.df.groupby("campaign_id").agg(
            campaign_total_spend=("spent", "sum"),
            campaign_avg_ctr=("ctr", "mean"),
            campaign_avg_cpa=("cost_per_acquisition", "mean"),
            campaign_total_conversions=("approved_conversion", "sum")
        ).reset_index()

        self.df = self.df.merge(
            campaign_metrics,
            on="campaign_id",
            how="left"
        )

        print("  Created campaign-level aggregates")

        # Demographic segment metrics
        if "demographic_segment" in self.df.columns:
            segment_metrics = self.df.groupby("demographic_segment").agg(
                segment_avg_ctr=("ctr", "mean"),
                segment_avg_cpa=("cost_per_acquisition", "mean"),
                segment_total_spend=("spent", "sum")
            ).reset_index()

            self.df = self.df.merge(
                segment_metrics,
                on="demographic_segment",
                how="left"
            )

            print("  Created demographic-level aggregates")

    # ------------------------------------------------------------------
    # Output & Pipeline
    # ------------------------------------------------------------------

    def save_featured_data(self, output_path="data/marketing_data_featured.csv"):
        """Save dataset with engineered features."""
        self.df.to_csv(output_path, index=False)
        print(f"\nFeatured data saved to {output_path}")
        print(f"Final shape: {self.df.shape}")

    def run_full_pipeline(self):
        """Run full feature engineering pipeline."""
        print("=" * 60)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        self.load_cleaned_data()
        if self.df is None:
            return None

        self.create_time_features()
        self.create_kpi_features()
        self.create_demographic_features()
        self.create_aggregated_features()
        self.save_featured_data()

        print("\nFEATURE ENGINEERING PIPELINE COMPLETED")
        print("=" * 60)

        return self.df

def main():
    engineer = MarketingFeatureEngineer()
    return engineer.run_full_pipeline()

if __name__ == "__main__":
    featured_df = main()
    if featured_df is not None:
        print("\nSample featured data:")
        print(featured_df.head())