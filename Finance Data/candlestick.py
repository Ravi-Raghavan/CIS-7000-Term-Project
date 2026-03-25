import pandas as pd
import plotly.graph_objects as go

# Load AAPL CSV
df = pd.read_csv("Historical_Data_AAPL.csv", parse_dates=['Date'])

# Make sure it's sorted by date
df = df.sort_values('Date')

# Picked Last 6 months
df = df[-100:]

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig.update_layout(
    title='AAPL Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False  # hides the range slider
)

fig.show()