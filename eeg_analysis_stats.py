
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

#SA MODIFICATI CU PATH-UL VOSTRU 
eeg_data = pd.read_csv("/Users/butararon/Desktop/Uvt-Cognitive Science/Year 1 /Semester 2 /Computer science and cognition /Projects/Dots_30_006_data.csv")
behavioral_data = pd.read_csv("/Users/butararon/Desktop/Uvt-Cognitive Science/Year 1 /Semester 2 /Computer science and cognition /Projects/Dots_30_006_trial_info.csv")

data = eeg_data.copy()
data['GForce'] = behavioral_data['GForce']
data['ResponseID'] = behavioral_data['ResponseID']

#Mean diff analysis
mean_columns = [col for col in data.columns if "Mean" in col]
regions_mean = set()

for col in mean_columns:
    parts = col.split()
    if len(parts) == 3:
        _, region, _ = parts
        regions_mean.add(region)

mean_diff_records = []

for region in sorted(regions_mean):
    pre_col = f"Pre-Response {region} Mean"
    post_col = f"Post-Stimulus {region} Mean"
    if pre_col in data.columns and post_col in data.columns:
        diff = data[post_col] - data[pre_col]
        for response, value in zip(data['ResponseID'], diff):
            mean_diff_records.append({
                "Region": region,
                "Mean_Diff": value,
                "ResponseID": response
            })

mean_diff_df = pd.DataFrame(mean_diff_records)

sns.set(style="whitegrid")
g = sns.catplot(
    data=mean_diff_df,
    x="ResponseID", y="Mean_Diff", col="Region",
    kind="box", col_wrap=3,
    palette="Set1", sharey=False
)

g.set_titles("{col_name}")
g.set_axis_labels("ResponseID", "Post - Pre Mean Difference")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("EEG Mean Signal Shift (Post - Pre) Across Brain Regions by Response Type")
g.savefig("all_regions_mean_diff_by_response.png")

#ANOVA
anova_results = []

for region in sorted(mean_diff_df["Region"].unique()):
    region_df = mean_diff_df[mean_diff_df["Region"] == region]
    model = ols('Mean_Diff ~ C(ResponseID)', data=region_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_value = anova_table["PR(>F)"][0]
    anova_results.append({"Region": region, "p_value": round(p_value, 4)})

anova_df = pd.DataFrame(anova_results).sort_values(by="p_value")
anova_df.to_csv("anova_results_mean_diff.csv", index=False)
print("\nANOVA Results (Mean Difference):")
print(anova_df)
