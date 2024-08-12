import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#from pypfopt.efficient_frontier import EfficientFrontier
#from pypfopt import risk_models, expected_returns
from prophet import Prophet

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

# Function to plot drawdown
def plot_drawdown(data):
    cumulative_returns = (1 + data.pct_change()).cumprod()
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    fig = go.Figure()
    for col in drawdown.columns:
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown[col], mode='lines', name=col))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", hovermode="x unified")
    return fig

# # Function to optimize portfolio
# def optimize_portfolio(data, method):
#     mu = expected_returns.mean_historical_return(data)
#     S = risk_models.sample_cov(data)

#     ef = EfficientFrontier(mu, S)
    
#     if method == "Maximum Sharpe Ratio":
#         weights = ef.max_sharpe()
#     elif method == "Minimum Volatility":
#         weights = ef.min_volatility()
#     elif method == "Equal Weights":
#         weights = dict.fromkeys(data.columns, 1 / len(data.columns))
#     else:
#         weights = None

#     if weights:
#         cleaned_weights = ef.clean_weights()
#         performance = ef.portfolio_performance(verbose=True)
#         return cleaned_weights, performance
#     return None, None


# Function to download data for forecasting
def download_forecast_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Adj Close']].reset_index()
    data.columns = ['ds', 'y']
    return data

# Function to forecast stock prices
def forecast_stock(data, days):
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# Forecast page
def forecast_page():
    st.header("Stock Price Forecast")
    
    # Sidebar inputs
    selected_stock = st.sidebar.text_input("Enter Yahoo ticker (e.g., AAPL, MSFT, BTC-USD)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    forecast_horizon = st.sidebar.slider("Select forecast horizon (days)", min_value=1, max_value=365, value=30)
    
    if selected_stock:
        # Download historical data
        data = download_forecast_data(selected_stock, start_date, end_date)
        
        # Plot historical data
        st.subheader(f"Historical Prices for {selected_stock}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Historical Prices'))
        st.plotly_chart(fig)
        
        # Forecast future prices
        st.subheader(f"Forecasted Prices for {selected_stock} for next {forecast_horizon} days")
        forecast = forecast_stock(data, forecast_horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Confidence Interval', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Confidence Interval', line=dict(dash='dash')))
        st.plotly_chart(fig)

# Main App
def main():
    st.title("Financial Assets Analysis and Forecasting")

    pages = st.sidebar.selectbox("Select a page", ["Analysis", "Forecast"])

    if pages == "Analysis":
        st.header("Financial Assets Correlation and Time Series Analysis")
        # Define the assets to be analyzed
        assets = {
            "S&P 500": "^GSPC",
            "Crude Oil": "CL=F",
            "Gold": "GC=F",
            "Silver": "SI=F",
            "10Y Treasury Bond": "^TNX",
            "Bitcoin": "BTC-USD",
            "Tesla": "TSLA"
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

                # Plot Drawdown
                st.subheader("Drawdown Analysis")
                drawdown_plot = plot_drawdown(data)
                st.plotly_chart(drawdown_plot)
        else:
            st.sidebar.warning("Please select at least one asset to begin the analysis.")

    elif pages == "Forecast":
        forecast_page()
if __name__ == "__main__":
    main()
