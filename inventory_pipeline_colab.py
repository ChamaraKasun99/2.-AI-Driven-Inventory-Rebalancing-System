
# 📦 AI-Driven Inventory Rebalancing System – Full Pipeline (Google Colab Version)

# ✅ Setup
!pip install prophet pulp scikit-learn matplotlib seaborn openpyxl --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus
from google.colab import drive
import os

# 🔗 Mount Google Drive
drive.mount('/content/drive')
save_path = "/content/drive/MyDrive/inventory_project_outputs"
os.makedirs(save_path, exist_ok=True)

# 📥 Load Dataset
file_path = "/content/drive/MyDrive/inventory_data_large_sample.xlsx"
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])

# 📊 Phase 3: EDA
sns.set(style='whitegrid')
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Date', y='Demand', hue='Warehouse')
plt.title("Warehouse Demand Trends")
plt.savefig(f"{save_path}/warehouse_trends.png")
plt.close()

# 📈 Phase 4: Forecasting
forecast_list = []
forecast_results = []

for (sku, wh), group in df.groupby(['SKU', 'Warehouse']):
    group = group.groupby('Date').agg({'Demand': 'sum'}).reset_index()
    group.columns = ['ds', 'y']
    if len(group) < 30:
        continue  # Skip small data
    model = Prophet()
    model.fit(group)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast['SKU'] = sku
    forecast['Warehouse'] = wh
    forecast_list.append(forecast[['ds', 'yhat', 'SKU', 'Warehouse']])

forecast_df = pd.concat(forecast_list)
forecast_df.to_csv(f"{save_path}/forecast_results.csv", index=False)

# 📉 Accuracy Summary (MAPE Sample)
from sklearn.metrics import mean_absolute_percentage_error
sample = df[(df['SKU'] == 'SKU0096') & (df['Warehouse'] == 'W01')].groupby('Date')['Demand'].sum()
sample_df = sample.reset_index()
sample_df.columns = ['ds', 'y']
m = Prophet()
m.fit(sample_df)
future = m.make_future_dataframe(periods=30)
pred = m.predict(future)
mape = mean_absolute_percentage_error(sample_df['y'], m.predict(sample_df)['yhat'])
print("Sample MAPE:", round(mape, 3))

# 🧠 Phase 5: Clustering
sku_stats = df.groupby('SKU').agg({'Demand': ['sum', 'mean', 'std']}).reset_index()
sku_stats.columns = ['SKU', 'Total_Demand', 'Avg_Demand', 'Demand_STD']
X_sku = StandardScaler().fit_transform(sku_stats[['Total_Demand', 'Avg_Demand', 'Demand_STD']])
kmeans_sku = KMeans(n_clusters=3, random_state=42).fit(X_sku)
sku_stats['Cluster'] = kmeans_sku.labels_
sku_stats.to_csv(f"{save_path}/sku_clusters.csv", index=False)

# 🧮 Phase 6: Optimization
# Create dummy current stock
stock = df.groupby(['SKU', 'Warehouse'])['Demand'].sum().reset_index()
stock.columns = ['SKU', 'Warehouse', 'Current_Stock']

# Join forecast
latest_forecast = forecast_df.groupby(['SKU', 'Warehouse']).agg({'yhat': 'sum'}).reset_index()
latest_forecast.columns = ['SKU', 'Warehouse', 'Forecast_Demand']
data = pd.merge(stock, latest_forecast, on=['SKU', 'Warehouse'], how='inner')
data['Transferable'] = data['Current_Stock'] - data['Forecast_Demand']
data = data[data['Transferable'] > 0]

# Define Optimization
prob = LpProblem("Inventory_Rebalancing", LpMinimize)
vars = {}
warehouses = df['Warehouse'].unique()

for i, row in data.iterrows():
    for wh_target in warehouses:
        if wh_target != row['Warehouse']:
            key = (row['SKU'], row['Warehouse'], wh_target)
            vars[key] = LpVariable(f"Transfer_{key}", lowBound=0)

# Cost: Assume unit transfer cost = 1
prob += lpSum([vars[k]*1 for k in vars])

# Constraints: Ensure transferred does not exceed surplus
for (sku, wh_from, wh_to), var in vars.items():
    transferable = data[(data['SKU'] == sku) & (data['Warehouse'] == wh_from)]['Transferable'].values[0]
    prob += var <= transferable

# Solve
prob.solve()
print("Optimization Status:", LpStatus[prob.status])

# Results
transfer_plan = []
for (sku, src, tgt), var in vars.items():
    qty = var.varValue
    if qty > 0:
        transfer_plan.append([sku, src, tgt, qty])

transfer_df = pd.DataFrame(transfer_plan, columns=['SKU', 'Source', 'Target', 'Quantity'])
transfer_df.to_csv(f"{save_path}/transfer_plan.csv", index=False)
print("Transfer Plan saved!")
