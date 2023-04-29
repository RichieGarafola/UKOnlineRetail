import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose


# Define function to read in CSV files
@st.cache_data
def load_data():
    retail_2009_2010 = pd.read_csv('./Resources/retail_2009_2010.csv')
    retail_2010_2011 = pd.read_csv('./Resources/retail_2010_2011.csv')
    df = pd.concat([retail_2009_2010, retail_2010_2011])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
    return df

# Define function to show summary statistics
def show_summary_stats(df):
    st.write("Summary Statistics")
    st.write(df.describe())

# Define function to show correlation matrix
def show_corr_matrix(df):
    st.write("Correlation Coefficients")
    st.write(df.corr())

# Define function to show top 10 best sellers
def show_top_sellers(df):
    st.write("Top 10 Best Sellers (by Quantity Sold)")
    top_products = df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head(10)
    
    # Create two columns for the plot and the table
    col1, col2 = st.columns(2)
    
    # Add plot to the first column
    with col1:
        # Create a horizontal bar chart
        fig, ax1 = plt.subplots(figsize=(13, 20))

        # Plot the bar chart
        ax1.barh(top_products.index, top_products['Quantity'], color='skyblue')

        # Set chart title and labels
        ax1.set_title('Top 10 Most Popular Products by Quantity', fontsize=32)
        ax1.set_xlabel('Quantity Sold', fontsize=20)
        # Increase font size of xlabels
        ax1.tick_params(axis='x', labelsize=20)
        ax1.set_ylabel('Product Description', fontsize=20)
        # Increase font size of ylabels
        ax1.tick_params(axis='y', labelsize=20)

        # Show the chart
        st.pyplot(fig)

    # Add table to the second column
    with col2:
        # Show the data table
        top_products_formatted = top_products.style.format({"Quantity": "{:.0f}"})
        st.dataframe(top_products_formatted, width=500, height=400)

# Define function to show total price per invoice
def show_total_price_per_invoice(df):
    # Group data by invoice and sum up the total price
    total_price_per_invoice = df.groupby("Invoice").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False).head(10)
    
    # Create two columns for the plot and the table
    col1, col2 = st.columns(2)

    # Add plot to the first column
    with col1:
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(13, 20))
        ax.barh(total_price_per_invoice.index, total_price_per_invoice['TotalPrice'], color='purple')

        # Set chart title and labels
        ax.set_title('Total Price per Invoice', fontsize=32)
        ax.set_xlabel('Total Price', fontsize=20)
        # Increase font size of xlabels
        ax.tick_params(axis='x', labelsize=20)
        # Increase font size of ylabels
        ax.tick_params(axis='y', labelsize=20)        
        ax.set_ylabel('Invoice Number', fontsize=20)
        st.pyplot(fig)
    # Add table to the second column
    with col2:
        # Show the chart and the data table
        st.write("Top 10 Invoices by Total Price")
        st.dataframe(total_price_per_invoice.style.format({"TotalPrice": "${:.2f}"}))


# Define function to show date range
def show_date_range(df):
    st.write("Date Range in the DataFrame")
    st.write("Oldest date:", df["InvoiceDate"].min())
    st.write("Newest date:", df["InvoiceDate"].max())

# Define function to remove outliers
def remove_outliers(df):
    z_scores = stats.zscore(df['Quantity'])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    retail_df = df[filtered_entries]
    return retail_df

def show_time_series_analysis(df):
    st.write("Time Series Analysis")

    # Remove outliers
    retail_df = remove_outliers(df)

    # Group sales by month
    monthly_sales = retail_df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['Quantity'].sum()

    # Create two columns for the plot and the table
    col1, col2 = st.columns(2)

    # Add plot to the first column
    with col1:
        fig, ax = plt.subplots(figsize=(13, 20))
        ax.plot(monthly_sales.index, monthly_sales.values)
        # Set chart title and labels
        ax.set_title('Total Sales per Month', fontsize=32)
        ax.set_xlabel('Date', fontsize=20)
        # Increase font size of xlabels
        ax.tick_params(axis='x', labelsize=20)
        ax.set_ylabel('Total Sales', fontsize=20)
        # Increase font size of ylabels
        ax.tick_params(axis='y', labelsize=20)
        st.pyplot(fig)

    # Add table to the second column
    with col2:
        st.write("Monthly Sales")
        st.dataframe(monthly_sales.reset_index(name="Sales"), width=350, height=550)

# Define function to show customer segmentation
def show_customer_segmentation(df):
    st.write("Customer Segmentation")
    retail_df = remove_outliers(df)
    customer_df = retail_df.groupby('Customer ID').agg({'Invoice': pd.Series.nunique, 'Price': np.sum})
    kmeans = KMeans(n_clusters=4, random_state=42).fit(customer_df)
    fig, ax = plt.subplots()
    sns.scatterplot(x='Invoice', y='Price', hue=kmeans.labels_, data=customer_df, ax=ax)
    ax.set(xlabel="Number of invoices", ylabel="Total Price", title="Customer Segmentation")
    st.pyplot(fig)

# Define a function for Product Analysis
def plot_product_analysis(df):
    product_df = df.groupby('Description').agg({'Quantity': np.sum, 'Price': np.mean})
    top_products = product_df.nlargest(10, 'Quantity')
    pivot_df = df.pivot_table(index='Description', values=['Quantity', 'TotalPrice'], aggfunc=np.sum)
    top_pivot_df = pivot_df.loc[top_products.index]
    sns.heatmap(top_pivot_df, annot=True, fmt='g');

# Define a function for sales forecasting with Prophet
def plot_sales_forecasting(df):
    prophet_df = df.groupby(pd.Grouper(key='InvoiceDate', freq='D'))['Quantity'].sum().reset_index()
    prophet_df.rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'}, inplace=True)
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    m.plot(forecast)
    
    
    
# Define a function for Time Series Decomposition
st.set_option('deprecation.showPyplotGlobalUse', False)
def plot_time_series_decomposition(df):
    decomposition = seasonal_decompose(df.groupby(pd.Grouper(key='InvoiceDate', freq='D'))['Quantity'].sum(), period=365)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(df.groupby(pd.Grouper(key='InvoiceDate', freq='D'))['Quantity'].sum(), label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

def main():
    st.set_page_config(page_title="Online Retail", page_icon=":moneybag:")
    st.title("Online Retail")

    # Create sidebar
    st.sidebar.title("Dashboard Options")
    analysis_type = st.sidebar.selectbox("Select analysis type", ["Summary Statistics", "Correlation Matrix", "Top 10 Best Sellers", "Total Price per Invoice", "Date Range", "Time Series Analysis", "Customer Segmentation", "Product Analysis", "Sales Forecasting", "Time Series Decomposition"])

    # Load data
    df = load_data()

    # Show analysis based on user selection
    if analysis_type == "Summary Statistics":
        show_summary_stats(df)
        st.markdown("""
        These summary statistics provide a quick overview of the central tendency, dispersion, and range of values in each column of the dataframe. They can be used to identify potential outliers or to compare the distribution of different variables in the dataframe.
        """)
    elif analysis_type == "Correlation Matrix":
        show_corr_matrix(df)
        st.markdown("""
        The correlation matrix shows the correlation coefficients between all pairs of variables in the dataframe. The correlation coefficients range between -1 and 1, where values closer to 1 or -1 indicate a strong positive or negative correlation between the two variables, respectively. A value of 0 indicates no correlation between the two variables. The diagonal of the matrix shows the correlation of each variable with itself, which is always 1. This provides information about the degree and direction of the linear relationship between pairs of variables in the dataset.
        """)
    elif analysis_type == "Top 10 Best Sellers":
        show_top_sellers(df)
        st.markdown("""
        Top 10 best selling products based on the sum of their quantities sold, grouped by their descriptions. The "Description" column contains the names of the top 10 best selling products, while the "Quantity" column shows the total quantity sold for each product.
        """)
    elif analysis_type == "Total Price per Invoice":
        show_total_price_per_invoice(df)
        st.markdown("""
        The total price per invoice for the top 10 invoices. The "TotalPrice" column contains the product of the "Quantity" and "UnitPrice" columns, representing the total price of each product in the invoice.
        """)
    elif analysis_type == "Date Range":
        show_date_range(df)
        st.markdown("""
        Oldest and newest date present in the "InvoiceDate".
        """)
    elif analysis_type == "Time Series Analysis":
        show_time_series_analysis(df)
        st.markdown("""
        Sales are relatively stable in the first 3 quarters of the year, in the beginning of Q4 we can see a sharp increase in sales before a sharp decrease near the end of December. The table provides more detailed information on the sales data for each month, allowing us to identify specific months where sales were particularly high or low.
        """)
    elif analysis_type == "Customer Segmentation":
        show_customer_segmentation(df)
        st.markdown("""
        The plot is divided into four different clusters, each represented by a different color.

The customer segmentation was performed using KMeans clustering with four clusters. This allowed the customers to be grouped based on their purchasing behavior. The scatter plot shows that the customer groups are clearly separated, with each group having its own characteristic purchasing behavior.

This provides useful information for the retailer, as it helps to identify the different types of customers and their purchasing behavior. For example, the retailer could use this information to tailor their marketing campaigns to each customer group or to identify high-value customers for loyalty programs.
        """)
    elif analysis_type == "Product Analysis":
        plot_product_analysis(df)
        st.pyplot()
        st.markdown("""
        The heatmap shows the total quantity sold for each product and the total revenue generated, with the values representing the corresponding intersection.

The heatmap provides an easy way to identify which products are selling well and generating the most revenue. By visualizing the data in this way, it becomes easy to identify which products are the most profitable and which ones may require further attention.
        """)
    elif analysis_type == "Sales Forecasting":
        plot_sales_forecasting(df)
        st.pyplot()
        st.markdown("""
        The Prophet model is fitted to the sales data and used to predict the sales for each day in the future.

The plot shows the historical sales data in black dots, the fitted model in blue, and the forecasted sales in light blue with shaded areas representing the uncertainty intervals. The plot allows us to visualize the trend, seasonality, and uncertainty in the sales forecast.
        """)
    elif analysis_type == "Time Series Decomposition":
        plot_time_series_decomposition(df)
        st.pyplot()
        st.markdown("""
        By decomposing the time series into trend, seasonal, and residual components, we can better understand the underlying patterns and trends in the data. The trend component represents the long-term direction of the time series, while the seasonal component represents the repeating patterns within the data. The residual component represents the random fluctuations or noise in the data that cannot be explained by the trend or seasonal components.

The plot allows us to visually inspect each of these components and gain insight into the underlying patterns and trends in the data. We can see if there is a clear upward or downward trend in the data, or if there are any significant seasonal patterns that repeat over time. We can also examine the residuals to see if there are any patterns or trends that have not been accounted for by the other components.
        """)
        
if __name__ == '__main__':
    main()