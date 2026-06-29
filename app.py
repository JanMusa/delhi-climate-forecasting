import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go


st.set_page_config(page_title="Delhi Climate Forecast", layout="wide")
st.title("🌡️ Daily Delhi Climate Forecasting Dashboard")
st.subheader("Demonstrating Time-Series Predictive Insights to Stakeholders")


st.sidebar.header("🕹️ Forecast Configurations")
forecast_horizon = st.sidebar.slider(
    "Select Forecast Horizon (Days)", 
    min_value=7, 
    max_value=60, 
    value=30, 
    step=1
)


@st.cache_data
def load_data():
    # Replace 'DailyDelhiClimateTrain.csv' with your actual dataset filename
    try:
        df = pd.read_csv("DailyDelhiClimateTrain.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        # Ensure daily frequency
        df = df.asfreq('D')
        # Fill any minor missing values if present
        df['meantemp'] = df['meantemp'].ffill()
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file 'DailyDelhiClimateTrain.csv' not found in the project directory. Please add it to run the forecast.")
        return None

df = load_data()

if df is not None:
  
    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Historical Records", f"{len(df)} Days")
    col2.metric("Min Recorded Temp", f"{df['meantemp'].min():.2f} °C")
    col3.metric("Max Recorded Temp", f"{df['meantemp'].max():.2f} °C")
    
    
    with st.spinner("Calculating future temperature horizons..."):
        model = ExponentialSmoothing(
            df['meantemp'], 
            trend='add', 
            seasonal='add', 
            seasonal_periods=7
        ).fit()
        
        
        forecast_idx = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_values = model.forecast(forecast_horizon)
        
        forecast_df = pd.DataFrame(data={'Forecasted Temp': forecast_values}, index=forecast_idx)

    
    st.markdown("### 📈 Historical Timeline vs Future Forecast")
    
    fig = go.Figure()
    
    
    recent_history = df.tail(90)
    fig.add_trace(go.Scatter(
        x=recent_history.index, 
        y=recent_history['meantemp'],
        mode='lines',
        name='Historical Temperature',
        line=dict(color='#1f77b4', width=2)
    ))
    
    
    fig.add_trace(go.Scatter(
        x=forecast_df.index, 
        y=forecast_df['Forecasted Temp'],
        mode='lines+markers',
        name=f'Holt-Winters Forecast ({forecast_horizon} Days)',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Mean Temperature (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    
    st.markdown("---")
    st.markdown("### 🔍 Technical Model Performance (Validation Context)")
    st.info(
        "Based on backtesting validation against your original report, the **Holt-Winters Seasonal Model** outperformed ARIMA "
        "by achieving significantly minimized error balances (Lower RMSE, MAE, and MAPE) due to its structural capturing of regular weekly seasonality."
    )
    
    
    with st.expander("📥 View and Download Forecasted Data Points"):
        st.dataframe(forecast_df)