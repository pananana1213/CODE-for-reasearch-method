# CODE-for-reasearch-method
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("Results_21Mar2022.csv")
df = df[df['n_participants'] >= 50].dropna()


metrics = [
    'mean_ghgs', 'mean_land', 'mean_watscar',
    'mean_eut', 'mean_ghgs_ch4', 'mean_ghgs_n2o',
    'mean_bio', 'mean_watuse', 'mean_acid'
]


grouped = df.groupby('age_group')[metrics].mean().reset_index()


scaler = MinMaxScaler()
grouped_scaled = scaler.fit_transform(grouped[metrics])
grouped_scaled_df = pd.DataFrame(grouped_scaled, columns=metrics)
grouped_scaled_df.insert(0, 'age_group', grouped['age_group'])


plt.figure(figsize=(12, 6))


heatmap_data = grouped_scaled_df.set_index('age_group')[metrics]


sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Normalized Environmental Impact (0 = best)'}
)

plt.title("Heatmap: Environmental Impact by Age Group and Metric (Normalized)")
plt.ylabel("Age Group")
plt.xlabel("Environmental Metric")
plt.tight_layout()


plt.show()
