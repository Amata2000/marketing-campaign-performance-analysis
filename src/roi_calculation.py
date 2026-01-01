import pandas as pd
import numpy as np
import os

class MarketingROICalculator:
    """
    Perform ROI, ROAS, CPA, and performance analysis on marketing campaign data.
    """

    def __init__(self, data_path="marketing_data_featured.csv", conversion_value=100):
        """
        Parameters
        ----------
        data_path : str
            Path to featured marketing dataset
        conversion_value : float
            Assumed revenue per approved conversion
        """
        self.data_path = data_path
        self.conversion_value = conversion_value
        self.df = None
        self.results = {}

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_data(self):
        print("Loading featured marketing data...")
        self.df = pd.read_csv(
            self.data_path,
            parse_dates=["reporting_start", "reporting_end"]
        )
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    # ------------------------------------------------------------------
    # BASIC ROI METRICS (OVERALL)
    # ------------------------------------------------------------------
    def calculate_overall_roi(self):
        print("\nCalculating overall ROI metrics...")

        total_spend = self.df["spent"].sum()
        total_impressions = self.df["impressions"].sum()
        total_clicks = self.df["clicks"].sum()
        total_conversions = self.df["approved_conversion"].sum()
        total_revenue = total_conversions * self.conversion_value

        roi = (total_revenue - total_spend) / total_spend if total_spend > 0 else 0
        roas = total_revenue / total_spend if total_spend > 0 else 0
        cpa = total_spend / total_conversions if total_conversions > 0 else np.nan
        ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0

        metrics = {
            "total_spend": total_spend,
            "total_revenue": total_revenue,
            "total_profit": total_revenue - total_spend,
            "roi": roi,
            "roas": roas,
            "cpa": cpa,
            "ctr": ctr,
            "conversion_rate": conversion_rate
        }

        self.results["overall_metrics"] = metrics
        return metrics

    # ------------------------------------------------------------------
    # CAMPAIGN-LEVEL ROI
    # ------------------------------------------------------------------
    def calculate_campaign_roi(self):
        print("\nCalculating campaign-level ROI...")

        campaign_df = (
            self.df
            .groupby("campaign_id", as_index=False)
            .agg({
                "spent": "sum",
                "impressions": "sum",
                "clicks": "sum",
                "approved_conversion": "sum"
            })
        )

        campaign_df["revenue"] = campaign_df["approved_conversion"] * self.conversion_value
        campaign_df["profit"] = campaign_df["revenue"] - campaign_df["spent"]
        campaign_df["roi"] = campaign_df["profit"] / campaign_df["spent"]
        campaign_df["roas"] = campaign_df["revenue"] / campaign_df["spent"]
        campaign_df["cpa"] = np.where(
            campaign_df["approved_conversion"] > 0,
            campaign_df["spent"] / campaign_df["approved_conversion"],
            np.nan
        )
        campaign_df["ctr"] = campaign_df["clicks"] / campaign_df["impressions"]
        campaign_df["conversion_rate"] = np.where(
            campaign_df["clicks"] > 0,
            campaign_df["approved_conversion"] / campaign_df["clicks"],
            0
        )

        self.results["campaign_metrics"] = campaign_df
        return campaign_df

    # ------------------------------------------------------------------
    # DEMOGRAPHIC ROI
    # ------------------------------------------------------------------
    def calculate_demographic_roi(self):
        print("\nCalculating demographic ROI...")

        demographic_results = {}

        for col in ["gender", "age_group", "demographic_segment"]:
            if col in self.df.columns:
                demographic_results[col] = self._group_roi(col)

        self.results["demographic_metrics"] = demographic_results
        return demographic_results

    def _group_roi(self, group_col):
        grouped = (
            self.df
            .groupby(group_col, as_index=False)
            .agg({
                "spent": "sum",
                "impressions": "sum",
                "clicks": "sum",
                "approved_conversion": "sum"
            })
        )

        grouped["revenue"] = grouped["approved_conversion"] * self.conversion_value
        grouped["profit"] = grouped["revenue"] - grouped["spent"]
        grouped["roi"] = grouped["profit"] / grouped["spent"]
        grouped["roas"] = grouped["revenue"] / grouped["spent"]
        grouped["cpa"] = np.where(
            grouped["approved_conversion"] > 0,
            grouped["spent"] / grouped["approved_conversion"],
            np.nan
        )

        grouped["spend_share"] = grouped["spent"] / grouped["spent"].sum()
        grouped["conversion_share"] = grouped["approved_conversion"] / grouped["approved_conversion"].sum()
        grouped["efficiency_score"] = grouped["conversion_share"] / grouped["spend_share"]

        return grouped

    # ------------------------------------------------------------------
    # TIME-BASED ROI
    # ------------------------------------------------------------------
    def calculate_time_roi(self):
        print("\nCalculating time-based ROI...")

        self.df["month"] = self.df["reporting_start"].dt.to_period("M")
        self.df["day_of_week"] = self.df["reporting_start"].dt.day_name()

        time_results = {
            "monthly": self._group_roi("month"),
            "day_of_week": self._group_roi("day_of_week")
        }

        self.results["time_metrics"] = time_results
        return time_results

    # ------------------------------------------------------------------
    # PERFORMANCE TIERS
    # ------------------------------------------------------------------
    def assign_performance_tiers(self):
        print("\nAssigning performance tiers...")

        campaign_df = self.results["campaign_metrics"].copy()

        def tier(roas):
            if roas < 1:
                return "Loss Making"
            elif roas < 2:
                return "Break Even"
            elif roas < 5:
                return "Profitable"
            return "Highly Profitable"

        campaign_df["performance_tier"] = campaign_df["roas"].apply(tier)
        self.results["campaign_metrics"] = campaign_df
        return campaign_df

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    def save_results(self, output_dir="roi_results"):
        print(f"\nSaving ROI outputs to '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)

        for name, obj in self.results.items():
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(f"{output_dir}/{name}.csv", index=False)
            elif isinstance(obj, dict):
                for sub_name, df in obj.items():
                    df.to_csv(f"{output_dir}/{name}_{sub_name}.csv", index=False)

        print("All results saved successfully.")

    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        print("STARTING ROI ANALYSIS")
        print("=" * 60)

        self.load_data()
        self.calculate_overall_roi()
        self.calculate_campaign_roi()
        self.calculate_demographic_roi()
        self.calculate_time_roi()
        self.assign_performance_tiers()
        self.save_results()

        print("=" * 60)
        print("ROI ANALYSIS COMPLETED")
        print("=" * 60)

        return self.results