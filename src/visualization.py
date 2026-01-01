import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

class MarketingVisualization:
    """
    Generate visual insights from ROI analysis outputs.
    """

    def __init__(self, input_dir="outputs/tables", output_dir="outputs/figures"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------
    # HELPER
    # ---------------------------------------------------
    def _save_plot(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")

    # ---------------------------------------------------
    # OVERALL METRICS
    # ---------------------------------------------------
    def plot_overall_metrics(self):
        df = pd.read_csv(f"{self.input_dir}/overall_metrics.csv")

        metrics = ["roi", "roas", "cpa", "ctr", "conversion_rate"]
        values = df[metrics].iloc[0]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=metrics, y=values)
        plt.title("Overall Marketing Performance Metrics")
        plt.ylabel("Value")

        self._save_plot("overall_metrics.png")

    # ---------------------------------------------------
    # CAMPAIGN PERFORMANCE
    # ---------------------------------------------------
    def plot_campaign_roi(self):
        df = pd.read_csv(f"{self.input_dir}/campaign_metrics.csv")

        top = df.sort_values("roi", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x="roi", y="campaign_id")
        plt.title("Top 10 Campaigns by ROI")

        self._save_plot("top_campaigns_by_roi.png")

    def plot_campaign_cpa(self):
        df = pd.read_csv(f"{self.input_dir}/campaign_metrics.csv")

        top = df.sort_values("cpa").head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x="cpa", y="campaign_id")
        plt.title("Top 10 Campaigns by Lowest CPA")

        self._save_plot("top_campaigns_by_cpa.png")

    # ---------------------------------------------------
    # DEMOGRAPHIC PERFORMANCE
    # ---------------------------------------------------
    def plot_demographic_roi(self, segment="demographic_segment"):
        path = f"{self.input_dir}/demographic_metrics_{segment}.csv"
        if not os.path.exists(path):
            return

        df = pd.read_csv(path)

        plt.figure(figsize=(9, 5))
        sns.barplot(data=df, x="roi", y=segment)
        plt.title(f"ROI by {segment.replace('_', ' ').title()}")

        self._save_plot(f"roi_by_{segment}.png")

    # ---------------------------------------------------
    # TIME-BASED PERFORMANCE
    # ---------------------------------------------------
    def plot_monthly_roi(self):
        df = pd.read_csv(f"{self.input_dir}/time_metrics_monthly.csv")

        df["month"] = df["month"].astype(str)

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df, x="month", y="roi", marker="o")
        plt.xticks(rotation=45)
        plt.title("Monthly ROI Trend")

        self._save_plot("monthly_roi_trend.png")

    def plot_day_of_week_roi(self):
        df = pd.read_csv(f"{self.input_dir}/time_metrics_day_of_week.csv")

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=order, ordered=True)

        plt.figure(figsize=(9, 5))
        sns.barplot(data=df, x="day_of_week", y="roi")
        plt.title("ROI by Day of Week")

        self._save_plot("roi_by_day_of_week.png")

    # ---------------------------------------------------
    # RUN ALL
    # ---------------------------------------------------
    def run_all(self):
        print("=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.plot_overall_metrics()
        self.plot_campaign_roi()
        self.plot_campaign_cpa()
        self.plot_demographic_roi("gender")
        self.plot_demographic_roi("age_group")
        self.plot_demographic_roi("demographic_segment")
        self.plot_monthly_roi()
        self.plot_day_of_week_roi()

        print("=" * 60)
        print("ALL VISUALS GENERATED")
        print("=" * 60)

if __name__ == "__main__":
    viz = MarketingVisualization()
    viz.run_all()