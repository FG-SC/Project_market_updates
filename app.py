import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config to wide mode
st.set_page_config(layout="wide")

# Function to download and preprocess data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    data = data.asfreq('B')  # Align to business days
    data = data.fillna(method='ffill').fillna(method='bfill')  # Fill missing values
    return data

# Function to plot correlation matrix
def plot_correlation_matrix(data):
    corr_matrix = data.corr()
    fig = px.imshow(corr_matrix, 
                    title="Correlation Matrix",
                    color_continuous_scale=px.colors.diverging.Portland,
                    aspect="auto",
                    labels=dict(x="Assets", y="Assets", color="Correlation"))
    return fig

# Function to plot time series
def plot_time_series(data, title, scale):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", yaxis_type=scale, hovermode="x unified")
    return fig

# Function to plot rolling window correlation
def plot_rolling_correlation(data, asset1, asset2, window_size):
    rolling_corr = data[asset1].rolling(window=window_size).corr(data[asset2])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=rolling_corr, mode='lines', name=f'Rolling Correlation ({asset1} vs {asset2})'))
    fig.update_layout(title=f'Rolling Window Correlation ({asset1} vs {asset2})', xaxis_title="Date", yaxis_title="Correlation", hovermode="x unified")
    return fig

# Streamlit App
def main():
    st.title("Financial Assets Correlation and Time Series Analysis")

    # Define the assets to be analyzed
    assets = {
        "S&P 500": "^GSPC",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "10Y Treasury Bond": "^TNX",
        "Bitcoin": "BTC-USD"
    }

    # Sidebar for settings
    st.sidebar.header("Settings")
    selected_assets = st.sidebar.multiselect("Select assets to analyze", options=list(assets.keys()))

    if selected_assets:
        start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
        end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

        # Time series scale selection
        scale = st.sidebar.radio("Select scale for time series plot", options=["linear", "log"], index=0)

        # Rolling window correlation settings
        st.sidebar.subheader("Rolling Window Correlation")
        asset1 = st.sidebar.selectbox("Asset 1", options=selected_assets)
        asset2 = st.sidebar.selectbox("Asset 2", options=selected_assets)
        window_size = st.sidebar.slider("Rolling Window Size (days)", min_value=5, max_value=365, value=30)

        if start_date >= end_date:
            st.sidebar.error("Error: End date must fall after start date.")
        else:
            # Download and preprocess data
            tickers = [assets[asset] for asset in selected_assets]
            data = download_data(tickers, start_date, end_date)

            # Ensure the correct asset names are assigned to the columns
            data.columns = selected_assets

            # Plot Correlation Matrix
            st.subheader("Correlation Matrix")
            if len(selected_assets) > 1:
                corr_plot = plot_correlation_matrix(data)
                st.plotly_chart(corr_plot)
            else:
                st.write("Please select at least two assets to view the correlation matrix.")

            # Plot Time Series
            st.subheader("Time Series Analysis")
            time_series_plot = plot_time_series(data, "Time Series of Selected Assets", scale)
            st.plotly_chart(time_series_plot)

            # Plot Rolling Window Correlation
            st.subheader(f"Rolling Window Correlation ({asset1} vs {asset2})")
            if asset1 != asset2:
                rolling_corr_plot = plot_rolling_correlation(data, asset1, asset2, window_size)
                st.plotly_chart(rolling_corr_plot)
            else:
                st.write("Please select two different assets to view the rolling window correlation.")
    else:
        st.sidebar.warning("Please select at least one asset to begin the analysis.")

# Run the app
if __name__ == "__main__":
    main()
