# 2.-AI-Driven-Inventory-Rebalancing-System
An end-to-end automated pipeline built in Python and designed to forecast inventory demand, segment SKUs and warehouses, and optimize inventory redistribution using machine learning and operations research techniques.

Modern supply chains face immense pressure from fluctuating demand, fragmented inventory systems, and high logistics costs. This project builds a full-scale AI-driven inventory rebalancing system designed to help supply chain managers automate demand forecasting, identify inefficiencies, and optimize stock distribution across warehouses.

The pipeline is implemented using Python and Google Colab, combining advanced machine learning, operations research, and data visualization tools for a completely automated, modular, and user-friendly experience.

🎯 Project Objectives
Forecast SKU-level demand across warehouse locations using historical data.

Classify and segment inventory and warehouse behavior to guide decision-making.

Automatically optimize inventory redistribution between warehouses based on forecasted demand, inventory levels, and cost constraints.

Deliver actionable insights through an interactive UI and easy-to-use visual outputs.

🚀 Pipeline Architecture
📌 Phase 1: Data Upload & Preparation
Upload .xlsx dataset via Google Colab UI.

Clean, standardize, and validate the input.

📌 Phase 2: Exploratory Data Analysis (EDA)
Identify demand spikes, volatility, and seasonality.
Visualize inventory trends, bottlenecks, and under/overstocking.
Tools: matplotlib, seaborn, pandas_profiling, plotly.

📌 Phase 3: Demand Forecasting
Use Facebook Prophet to forecast future demand per SKU-location.
Evaluate accuracy using MAPE, RMSE.
Export forecast data for optimization.

📌 Phase 4: SKU & Warehouse Clustering
Use KMeans or DBSCAN to segment:
High-volume vs low-volume SKUs
Strategic vs regional warehouses
Visualize using PCA or t-SNE if needed.
Save clusters for targeted planning.

📌 Phase 5: Inventory Optimization
Use linear programming (PuLP) to solve:
Which SKUs to move between which warehouses
How many units to transfer
Subject to capacity, cost, and lead-time constraints

Generate:
Transfer plan (source → target → SKU → quantity)
Cost breakdown
Constraint satisfaction summary

📌 Phase 6: Reporting & Export
Save all outputs:
Forecasts
Clustering results
Optimization plan
Output formats: .csv, .png
Integrated into Google Drive for easy access

✅ How to Use the Pipeline (In Google Colab)
📥 Upload your .xlsx inventory dataset.
✅ Run the entire pipeline using UI controls.
📊 Download or visualize the results (charts, CSVs).
📦 Use the outputs to make business decisions or feed into your ERP system.
