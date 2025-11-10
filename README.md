# Median Price Growth by Mesh Block

Author: Kevin Thomas  
Project: Australian Residential Property Analysis  

This project calculates median property price growth by Mesh Block (a very small geographic area, usually a few streets) for Australian residential properties. It provides insights for investors, showing micro-level capital growth trends and visualizations.

---

# Overview

The analysis focuses on:

1. Median Price by Mesh Block: Yearly median sale prices for each small area.
2. Year-on-Year Growth: Annual percentage change in median prices.
3. Filtering Low-Volume Data: Mesh Blocks with fewer than 3 sales in a year are ignored to remove outliers.
4. Visualizations:
   - Top 10 growth areas (bar chart)
   - Median price history for the top 3 performing Mesh Blocks (line chart)

This helps investors identify emerging hotspots at a granular level, beyond suburb averages.

---

# Data Requirements

The project uses two data files:

1. GNAF properties (`gnaf_prop.parquet`)
   - Contains property IDs and Mesh Block codes.
2. Transactions (`transactions.parquet`)
   - Contains sale prices, dates, and property IDs.

---

# Setup Instructions

1. Clone this repository:

```bash
git clone https://github.com/kwinthomas/Aussie_Meshblock_Median_Price_Growth.git
cd Aussie_Meshblock_Median_Price_Growth
```

2.	Install required Python packages:

```bash
pip install pandas matplotlib
```

3. Make sure data files are in the project folder:
gnaf_prop.parquet
transactions.parquet

4. Run the analysis script:
python analysis.py

This will generate:
1) median_price_growth_by_mesh_block.csv: the main output with median prices, sales counts, and YoY growth.
2) top_10_growth_areas.png: a bar chart of the top 10 growth areas.
3) top_3_price_history.png: a line chart showing median price history of the top 3 performing Mesh Blocks.

# Configuration Options

In analysis.py, you can adjust:
1) LAST_N_YEARS_FOR_GROWTH: Number of recent years to calculate average growth for visualizations.
2) MINIMUM_SALES_COUNT: Minimum number of sales in a Mesh Block/year to include in analysis.

# License

This project is open for personal and research use. Modify and share responsibly.