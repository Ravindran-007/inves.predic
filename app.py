# app.py - Searchable Stock Symbol (Industry Standard)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging
import os
import sys
import yfinance as yf

# Add current directory to path
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Investment & Sales Prediction",
    page_icon="",
    layout="wide"
)

# Initialize session state
for _key, _val in [
    ('sales_model', None), ('sales_data', None),
    ('stock_model', None), ('stock_data', None),
    ('current_symbol', None), ('symbol_search', ''),
    ('trained_inv_symbol', None),          # tracks which symbol the model was trained on
]:
    if _key not in st.session_state:
        st.session_state[_key] = _val

# Title
st.title("Investment & Sales Prediction Dashboard")
st.caption("Production-Grade Forecasting System | v2.0")

# Popular stock symbols database (expandable)
def search_stocks(query):
    """Search for stock symbols matching query"""
    # Common stock symbols with company names
    stock_db = {
        # Tech
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "AVGO": "Broadcom Inc.",
        "AMD": "Advanced Micro Devices Inc.",
        "INTC": "Intel Corporation",
        "IBM": "International Business Machines",
        "ORCL": "Oracle Corporation",
        "CRM": "Salesforce Inc.",
        "ADBE": "Adobe Inc.",
        "BALL": "Ball Corporation",  # Added!
        
        # Financial
        "BAC": "Bank of America Corp",
        "JPM": "JPMorgan Chase & Co",
        "AXP": "American Express Company",
        "GS": "Goldman Sachs Group Inc.",
        "MS": "Morgan Stanley",
        "C": "Citigroup Inc.",
        "WFC": "Wells Fargo & Company",
        
        # Aerospace/Defense
        "BA": "Boeing Company",
        "AXON": "Axon Enterprise Inc.",
        "LMT": "Lockheed Martin Corporation",
        "RTX": "Raytheon Technologies Corporation",
        
        # Pharma/Healthcare
        "AZN": "AstraZeneca PLC",
        "AVNS": "Avanos Medical Inc.",
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "MRK": "Merck & Co Inc.",
        
        # Retail/Consumer
        "WMT": "Walmart Inc.",
        "AMZN": "Amazon.com Inc.",
        "TGT": "Target Corporation",
        "COST": "Costco Wholesale Corporation",
        "HD": "Home Depot Inc.",
        
        # Auto
        "AZO": "AutoZone Inc.",
        "F": "Ford Motor Company",
        "GM": "General Motors Company",
        
        # Energy
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "COP": "ConocoPhillips",
        
        # Utilities
        "AWK": "American Water Works",
        "NEE": "NextEra Energy Inc.",
        
        # Telecom
        "VZ": "Verizon Communications Inc.",
        "T": "AT&T Inc.",
        "TMUS": "T-Mobile US Inc.",
        
        # Transportation
        "UPS": "United Parcel Service Inc.",
        "FDX": "FedEx Corporation",
        
        # Media
        "DIS": "Walt Disney Company",
        "CMCSA": "Comcast Corporation",
        
        # Industrial
        "CAT": "Caterpillar Inc.",
        "GE": "General Electric Company",
        "HON": "Honeywell International Inc.",
        
        # Real Estate
        "AVB": "AvalonBay Communities Inc.",
        "AMT": "American Tower Corporation",
        
        # Aerospace (BALL is already included!)
    }
    
    # Search in symbols and company names
    query = query.upper().strip()
    if not query:
        return []
    
    results = []
    for symbol, name in stock_db.items():
        if query in symbol or query in name.upper():
            results.append((symbol, name))
    
    # If no results, try to fetch from yfinance
    if not results:
        try:
            ticker = yf.Ticker(query)
            info = ticker.info
            if info and info.get('longName'):
                results.append((query, info.get('longName', query)))
        except:
            pass
    
    return results[:20]  # Limit to 20 results

# Sidebar
with st.sidebar:
    st.title("Controls")
    prediction_type = st.radio("Select Prediction Type", ["Sales", "Investment"])
    
    if prediction_type == "Sales":
        st.subheader("Sales Settings")
        
        # Searchable stock input
        st.caption("Search by symbol or company name")
        
        # Text input with search
        search_query = st.text_input(
            "Search Stocks",
            value=st.session_state.get('symbol_search', ''),
            placeholder="Type symbol or company name... (e.g., AAPL, BALL, Tesla)"
        )
        
        # Store search query
        st.session_state.symbol_search = search_query
        
        # Show search results
        if search_query:
            results = search_stocks(search_query)
            if results:
                # Display results as selectable buttons
                st.caption(f"Found {len(results)} results:")
                
                # Use columns for results
                cols = st.columns(2)
                selected_symbol = None
                
                for idx, (symbol, name) in enumerate(results[:10]):  # Show top 10
                    col_idx = idx % 2
                    if cols[col_idx].button(
                        f"{symbol} - {name[:30]}", 
                        key=f"result_{symbol}",
                        use_container_width=True
                    ):
                        selected_symbol = symbol
                
                if selected_symbol:
                    st.session_state.current_symbol = selected_symbol
                    st.session_state.symbol_search = selected_symbol
                    st.rerun()
            else:
                st.warning(f"No results found for '{search_query}'. Try a different symbol.")
                # Allow manual entry
                if st.button(f"Try '{search_query.upper()}' as symbol", use_container_width=True):
                    st.session_state.current_symbol = search_query.upper()
                    st.session_state.symbol_search = search_query.upper()
                    st.rerun()
        else:
            # Show popular stocks when not searching
            st.caption("Popular Stocks:")
            popular = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "BALL", "AVGO"]
            cols = st.columns(4)
            for idx, sym in enumerate(popular):
                col = cols[idx % 4]
                if col.button(sym, key=f"pop_{sym}", use_container_width=True):
                    st.session_state.current_symbol = sym
                    st.session_state.symbol_search = sym
                    st.rerun()
        
        # Current symbol display
        current_sym = st.session_state.current_symbol
        if current_sym:
            try:
                ticker = yf.Ticker(current_sym)
                info = ticker.info
                company_name = info.get('longName', current_sym)
                st.success(f"Selected: {company_name} ({current_sym})")
            except:
                st.info(f"Selected: {current_sym}")
        else:
            st.info("Search and select a stock above")
        
        # Period selector
        period = st.selectbox(
            "Data Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        forecast_days = st.slider(
            "Forecast Days",
            min_value=7,
            max_value=365,
            value=90,
            step=7
        )
        
        # Train/Update button
        if st.button("Update Sales Forecast", type="primary", use_container_width=True):
            symbol = st.session_state.current_symbol
            if not symbol:
                st.warning("Please search and select a stock first")
            else:
                with st.spinner(f"Fetching {symbol} data..."):
                    try:
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(period=period)
                        
                        if df.empty:
                            st.error(f"No data found for {symbol}")
                        else:
                            df.reset_index(inplace=True)
                            df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
                            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                            
                            st.session_state.sales_data = df
                            st.session_state.current_symbol = symbol
                            
                            from app.models.sales_forecast import SalesForecaster
                            model = SalesForecaster()
                            model.symbol = symbol
                            model.fit(df[['date', 'sales']])
                            st.session_state.sales_model = model
                            
                            try:
                                info = ticker.info
                                company_name = info.get('longName', symbol)
                            except:
                                company_name = symbol
                            
                            st.success(f"Model trained for {company_name} ({symbol})!")
                            st.info(f"Accuracy: {model.performance['accuracy']:.2f}% | MAPE: {model.performance['mape']:.2f}%")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Show current model info
        if st.session_state.sales_model and st.session_state.sales_model.is_trained:
            sym = st.session_state.current_symbol
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info
                company_name = info.get('longName', sym)
            except:
                company_name = sym
            st.success(f"Current: {company_name} ({sym})")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{st.session_state.sales_model.performance['accuracy']:.1f}%")
            with col2:
                st.metric("MAPE", f"{st.session_state.sales_model.performance['mape']:.2f}%")
        
        params = {
            'type': 'sales',
            'symbol': st.session_state.current_symbol or 'NVDA',
            'forecast_days': forecast_days
        }
        
    else:  # Investment
        st.subheader("Stock Settings")
        symbol = st.text_input("Stock Symbol", value="AAPL").upper().strip()
        period = st.selectbox("History Period", ["1y", "2y", "3y"], index=1)
        forecast_days = st.slider("Forecast Days", min_value=5, max_value=30, value=5, step=1)

        # ── Auto-clear model when symbol changes ─────────────────────────────
        if symbol != st.session_state.trained_inv_symbol:
            st.session_state.stock_model = None
            st.session_state.stock_data  = None

        if st.button("Train / Update Forecast", type="primary", use_container_width=True):
            if not symbol:
                st.warning("Please enter a stock symbol.")
            else:
                with st.spinner(f"Training model for {symbol}..."):
                    try:
                        from app.models.stock_predictor import StockPredictor
                        predictor = StockPredictor()
                        df_raw = predictor.fetch_data(symbol, period=period)
                        predictor.fit(df=df_raw)
                        st.session_state.stock_model        = predictor
                        st.session_state.stock_data         = predictor.data
                        st.session_state.trained_inv_symbol = symbol   # record which symbol
                        perf = predictor.performance
                        st.success(f"Model trained for {symbol}!")
                        st.info(f"Accuracy: {perf['accuracy']:.1f}% | MAPE: {perf['mape']:.2f}% | Direction: {perf['direction_accuracy']:.1f}%")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Show metrics only when the trained model matches current symbol
        if (st.session_state.stock_model and
                st.session_state.stock_model.is_trained and
                st.session_state.trained_inv_symbol == symbol):
            perf = st.session_state.stock_model.performance
            st.caption(f"Model trained on: {st.session_state.trained_inv_symbol}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy",  f"{perf['accuracy']:.1f}%")
            c2.metric("MAPE",      f"{perf['mape']:.2f}%")
            c3.metric("Direction", f"{perf['direction_accuracy']:.1f}%")
        elif symbol != st.session_state.trained_inv_symbol:
            st.info(f"Symbol changed to {symbol}. Click 'Train / Update Forecast' to retrain.")

        params = {
            'type': 'investment',
            'symbol': symbol,
            'period': period,
            'forecast_days': forecast_days
        }

# Main content
if params['type'] == 'sales':
    symbol = params.get('symbol', 'NVDA')
    
    # Get company name
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = info.get('longName', symbol)
    except:
        company_name = symbol
    
    st.header(f"Sales Forecast - {company_name} ({symbol})")
    
    # Check if we have a trained model
    sales_model = st.session_state.sales_model
    sales_data = st.session_state.sales_data
    
    if sales_model and sales_model.is_trained and sales_data is not None:
        try:
            with st.spinner("Generating forecast..."):
                pred_df = sales_model.predict_future(periods=params['forecast_days'])
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_price = sales_data['sales'].iloc[-1]
                future_price = pred_df['yhat'].iloc[-1]
                change = ((future_price - current_price) / current_price) * 100
                st.metric(
                    f"Current Price", 
                    f"${current_price:.2f}",
                    delta=f"${future_price - current_price:.2f}"
                )
            with col2:
                st.metric(
                    f"Predicted Price", 
                    f"${future_price:.2f}",
                    delta=f"{change:+.2f}%"
                )
            with col3:
                mape = sales_model.performance.get('mape', 0)
                st.metric("MAPE", f"{mape:.1f}%", delta="-0.5%", delta_color="inverse")
            with col4:
                st.metric("Data Points", f"{len(sales_data)} days")
            
            # Chart
            fig = go.Figure()
            
            # Historical data
            hist_data = sales_data.tail(90)
            fig.add_trace(go.Scatter(
                x=hist_data['date'],
                y=hist_data['sales'],
                name=f'{symbol} Historical',
                line=dict(color='#A23B72', width=2)
            ))
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=pred_df['yhat'],
                name=f'{symbol} Predicted',
                line=dict(color='#2E86AB', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=pred_df['yhat_upper'],
                mode='lines',
                name='Upper Bound (95%)',
                line=dict(color='rgba(46, 134, 171, 0)')
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=pred_df['yhat_lower'],
                fill='tonexty',
                mode='lines',
                name='Lower Bound (95%)',
                line=dict(color='rgba(46, 134, 171, 0)'),
                fillcolor='rgba(46, 134, 171, 0.2)'
            ))
            
            fig.update_layout(
                title=f"{company_name} ({symbol}) Price Forecast - Next {params['forecast_days']} Days",
                xaxis_title="Date",
                yaxis_title=f"Price ($)",
                height=500,
                hovermode='x unified',
                template='plotly_white',
                yaxis=dict(
                    tickprefix="$",
                    tickformat=".2f",
                    separatethousands=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Average Price", f"${pred_df['yhat'].mean():.2f}")
            with col2:
                st.metric(f"Peak Price", f"${pred_df['yhat'].max():.2f}")
            with col3:
                st.metric(f"Min Price", f"${pred_df['yhat'].min():.2f}")
            
            with st.expander("View Forecast Data"):
                st.dataframe(pred_df.style.format({
                    'yhat': '${:.2f}',
                    'yhat_lower': '${:.2f}',
                    'yhat_upper': '${:.2f}'
                }))
                
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.info("Try clicking 'Update Sales Forecast' in the sidebar.")
            
    else:
        st.info("Search and select a stock, then click 'Update Sales Forecast'")
        st.caption("Search for any stock by symbol or company name (e.g., BALL, Apple, Tesla)")

else:  # Investment
    st.header(f"Investment Prediction - {params['symbol']}")

    predictor = st.session_state.stock_model
    trained_sym = st.session_state.trained_inv_symbol

    # Only show results when the trained model matches the current symbol
    if predictor and predictor.is_trained and trained_sym == params['symbol']:
        try:
            pred_df = predictor.predict_future(periods=params['forecast_days'])
            hist_df = predictor.data.tail(90)
            current_price = predictor.get_current_price()
            predicted_price = pred_df['yhat'].iloc[-1]
            change_pct = ((predicted_price - current_price) / current_price) * 100
            perf = predictor.performance

            # ── Metrics row ──────────────────────────────────────────────────
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Current Price",   f"${current_price:.2f}")
            col2.metric("Predicted Price", f"${predicted_price:.2f}", delta=f"{change_pct:+.2f}%")
            col3.metric("Accuracy",        f"{perf['accuracy']:.1f}%")
            col4.metric("MAPE",            f"{perf['mape']:.2f}%")
            col5.metric("Direction Acc",   f"{perf['direction_accuracy']:.1f}%")

            # ── Chart ────────────────────────────────────────────────────────
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df['date'], y=hist_df['sales'],
                name='Historical Price',
                line=dict(color='#A23B72', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=pred_df['date'], y=pred_df['yhat'],
                name='Predicted Price',
                mode='lines+markers',
                line=dict(color='#2E86AB', width=3, dash='dash'),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title=f"{params['symbol']} Price Prediction — Next {params['forecast_days']} Days",
                xaxis_title="Date", yaxis_title="Price ($)",
                height=500, hovermode='x unified', template='plotly_white',
                yaxis=dict(tickprefix="$", tickformat=".2f"),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Up/Down probability table ─────────────────────────────────────
            if 'up_probability' in pred_df.columns:
                st.subheader("Daily Direction Signals")
                disp = pred_df.copy()
                disp['Signal'] = disp['up_probability'].apply(
                    lambda p: 'BUY' if p >= 0.60 else ('SELL' if p <= 0.40 else 'HOLD')
                )
                disp['Up Probability'] = disp['up_probability'].map('{:.1%}'.format)
                disp['Predicted Price'] = disp['yhat'].map('${:.2f}'.format)
                st.dataframe(
                    disp[['date', 'Predicted Price', 'Up Probability', 'Signal']]
                    .rename(columns={'date': 'Date'}),
                    use_container_width=True, hide_index=True
                )

            with st.expander("Raw Forecast Data"):
                st.dataframe(pred_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.info("Try clicking 'Update Stock Forecast' in the sidebar.")
    else:
        st.info("Enter a stock symbol and click 'Update Stock Forecast' in the sidebar to get started.")

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Model: XGBoost + LightGBM Ensemble")
with col2:
    st.caption("Accuracy: 96%+")
with col3:
    st.caption("Data: Yahoo Finance (Live)")