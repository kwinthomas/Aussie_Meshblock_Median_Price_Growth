import pandas as pd
import warnings
import sys
import matplotlib.pyplot as plt
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
GNAF_FILE = "gnaf_prop.parquet"
TRANSACTIONS_FILE = "transactions.parquet"
RESULTS_DIR = "Results"

GNAF_PROP_ID_COL = 'gnaf_pid'
GNAF_MESH_BLOCK_COL = 'mb_2016_code'

TRANS_PROP_ID_COL = 'gnaf_pid'
TRANS_PRICE_COL = 'price'
TRANS_DATE_COL = 'date_sold'

LAST_N_YEARS_FOR_GROWTH = 5
MINIMUM_SALES_COUNT = 3


def decode_binary_column(df, col_name):
    """Decode a binary Parquet column to Python strings."""
    if col_name not in df.columns:
        return df
    df[col_name] = df[col_name].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
    df[col_name] = df[col_name].str.strip()
    return df


def load_data():
    """Load GNAF and transaction data."""
    try:
        gnaf_df = pd.read_parquet(GNAF_FILE)
        gnaf_df = decode_binary_column(gnaf_df, GNAF_PROP_ID_COL)
        print(f"Loaded {len(gnaf_df)} properties from GNAF.")

        trans_df = pd.read_parquet(TRANSACTIONS_FILE)
        trans_df = decode_binary_column(trans_df, TRANS_PROP_ID_COL)
        print(f"Loaded {len(trans_df)} transactions.")

        return gnaf_df, trans_df

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def process_transactions(gnaf_df, trans_df):
    """Merge transactions with property data and clean."""
    cols_to_keep = [GNAF_PROP_ID_COL, GNAF_MESH_BLOCK_COL]

    if GNAF_MESH_BLOCK_COL not in gnaf_df.columns:
        print(f"Error: Column '{GNAF_MESH_BLOCK_COL}' not found in GNAF data.")
        sys.exit(1)

    trans_df[TRANS_PROP_ID_COL] = trans_df[TRANS_PROP_ID_COL].astype(str)
    gnaf_df[GNAF_PROP_ID_COL] = gnaf_df[GNAF_PROP_ID_COL].astype(str)
    gnaf_df = gnaf_df.drop_duplicates(subset=[GNAF_PROP_ID_COL])

    merged_data = trans_df.merge(
        gnaf_df[cols_to_keep],
        on=TRANS_PROP_ID_COL,
        how='inner'
    )

    print(f"Successfully merged {len(merged_data)} transactions.")
    if len(merged_data) == 0:
        print("Merge resulted in 0 transactions.")
        sys.exit(1)

    merged_data[TRANS_PRICE_COL] = pd.to_numeric(merged_data[TRANS_PRICE_COL], errors='coerce')
    merged_data[TRANS_DATE_COL] = pd.to_datetime(merged_data[TRANS_DATE_COL], errors='coerce')
    merged_data = merged_data.dropna(subset=[TRANS_PRICE_COL, TRANS_DATE_COL, GNAF_MESH_BLOCK_COL])
    merged_data = merged_data[merged_data[TRANS_PRICE_COL] > 10000]
    merged_data['year'] = merged_data[TRANS_DATE_COL].dt.year

    return merged_data


def aggregate_growth(merged_data):
    """Calculate median price and YoY growth by Mesh Block, filtering low-volume areas."""
    grouped = merged_data.groupby([GNAF_MESH_BLOCK_COL, 'year'])[TRANS_PRICE_COL]
    metrics_with_count = grouped.agg(['median', 'count']).reset_index()

    filtered_metrics = metrics_with_count[metrics_with_count['count'] >= MINIMUM_SALES_COUNT].copy()
    print(f"Removed {len(metrics_with_count) - len(filtered_metrics)} low-volume rows.")

    filtered_metrics = filtered_metrics.sort_values(by=[GNAF_MESH_BLOCK_COL, 'year'])
    filtered_metrics['median_price_growth_yoy'] = filtered_metrics.groupby(GNAF_MESH_BLOCK_COL)['median'].pct_change() * 100

    filtered_metrics = filtered_metrics.rename(columns={'median': 'median_price', 'count': 'sales_count'})
    filtered_metrics['median_price'] = filtered_metrics['median_price'].round(0)
    filtered_metrics['median_price_growth_yoy'] = filtered_metrics['median_price_growth_yoy'].round(2)

    return filtered_metrics


def create_visualizations(metrics_df):
    """Generate charts for top growth areas and their price history."""
    if metrics_df.empty or 'year' not in metrics_df.columns:
        print("No data available for visualization.")
        return

    try:
        max_year = metrics_df['year'].max()

        recent_growth = metrics_df[
            (metrics_df['year'] > max_year - LAST_N_YEARS_FOR_GROWTH) &
            (metrics_df['median_price_growth_yoy'].notna())
        ]

        if not recent_growth.empty:
            avg_growth = recent_growth.groupby('mb_2016_code')['median_price_growth_yoy'].mean()
            top_10 = avg_growth.nlargest(10).sort_values(ascending=True)

            top_10.index = top_10.index.astype(str)

            plt.figure(figsize=(10, 7))
            top_10.plot(kind='barh', color='skyblue')
            plt.title(f'Top 10 Growth Areas (Avg. YoY Growth, Last {LAST_N_YEARS_FOR_GROWTH} Yrs, Min {MINIMUM_SALES_COUNT} Sales/Yr)')
            plt.xlabel('Average Year-on-Year Growth (%)')
            plt.ylabel('Mesh Block (mb_2016_code)')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()

            output_file_1 = os.path.join(RESULTS_DIR, 'top_10_growth_areas.png')
            plt.savefig(output_file_1)
            print(f"Saved Top 10 growth chart to: {output_file_1}")
            plt.clf()
        else:
            print("Not enough recent data to create Top 10 growth chart.")
            top_10 = pd.Series()

        if not top_10.empty:
            top_3_ids_original = avg_growth.nlargest(3).index

            top_3_history = metrics_df[metrics_df['mb_2016_code'].isin(top_3_ids_original)]

            if not top_3_history.empty:
                price_history = top_3_history.pivot(
                    index='year',
                    columns='mb_2016_code',
                    values='median_price'
                )

                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Loop through each column (mb_2016_code) in the pivoted data
                for mb_code in price_history.columns:
                    # Plot each one as a separate line using ax.plot()
                    ax.plot(
                        price_history.index, 
                        price_history[mb_code], 
                        marker='o', 
                        linestyle='--', 
                        label=str(mb_code) # Ensure label is a string
                    )
                
                # Set plot attributes on the axes object
                ax.set_title('Median Price History for Top 3 Performers')
                ax.set_xlabel('Year')
                ax.set_ylabel('Median Price ($)')
                ax.legend(title='Mesh Block (mb_2016_code)')
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Use tight_layout on the figure object
                fig.tight_layout()

                output_file_2 = os.path.join(RESULTS_DIR, 'top_3_price_history.png')
                plt.savefig(output_file_2)
                print(f"Saved Price History chart to: {output_file_2}")
                plt.clf()
            else:
                print("Could not find history for top 3 performers.")
        else:
            print("Skipping Price History chart as no top performers were found.")

    except Exception as e:
        print(f"An error occurred during visualization: {e}")


if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    gnaf_df, trans_df = load_data()
    processed_data = process_transactions(gnaf_df, trans_df)
    final_metrics = aggregate_growth(processed_data)

    output_filename = os.path.join(RESULTS_DIR, "median_price_growth_by_mesh_block.csv")
    final_metrics.to_csv(output_filename, index=False)

    print("\nAnalysis Complete!")
    print(f"Saved metrics to: {output_filename}")
    print(f"Total rows: {len(final_metrics)}")
    print(final_metrics.head(10))

    create_visualizations(final_metrics)