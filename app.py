import logging
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title='Investment & Sales Prediction', page_icon='📈', layout='wide')
for _k, _v in [('sales_model', None), ('sales_data', None), ('stock_model', None), ('stock_data', None), ('current_symbol', None), ('symbol_search', ''), ('trained_inv_symbol', None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v
STOCK_DB = {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.', 'AMZN': 'Amazon.com Inc.', 'NVDA': 'NVIDIA Corporation', 'TSLA': 'Tesla Inc.', 'META': 'Meta Platforms Inc.', 'NFLX': 'Netflix Inc.', 'AVGO': 'Broadcom Inc.', 'AMD': 'Advanced Micro Devices Inc.', 'INTC': 'Intel Corporation', 'IBM': 'IBM Corporation', 'ORCL': 'Oracle Corporation', 'CRM': 'Salesforce Inc.', 'ADBE': 'Adobe Inc.', 'JPM': 'JPMorgan Chase & Co', 'BAC': 'Bank of America Corp', 'GS': 'Goldman Sachs Group Inc.', 'BA': 'Boeing Company', 'LMT': 'Lockheed Martin Corporation', 'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer Inc.', 'WMT': 'Walmart Inc.', 'TGT': 'Target Corporation', 'XOM': 'Exxon Mobil Corporation', 'CVX': 'Chevron Corporation', 'DIS': 'Walt Disney Company', 'NFLX': 'Netflix Inc.', 'F': 'Ford Motor Company', 'GM': 'General Motors Company', 'BALL': 'Ball Corporation', 'UPS': 'United Parcel Service Inc.'}

def search_stocks(query: str):
    q = query.upper().strip()
    if not q:
        return []
    results = [(s, n) for s, n in STOCK_DB.items() if q in s or q in n.upper()]
    if not results:
        try:
            info = yf.Ticker(q).info
            if info and info.get('longName'):
                results = [(q, info['longName'])]
        except Exception:
            pass
    return results[:20]

def get_company_name(symbol: str) -> str:
    try:
        return yf.Ticker(symbol).info.get('longName', symbol)
    except Exception:
        return symbol
with st.sidebar:
    st.title('Controls')
    prediction_type = st.radio('Prediction Type', ['Sales Forecast', 'Investment'])
    st.divider()
    if prediction_type == 'Sales Forecast':
        st.subheader('Sales Settings')
        st.caption('Search by symbol or company name')
        search_query = st.text_input('Search Stocks', value=st.session_state.symbol_search, placeholder='e.g. AAPL, NVDA, Tesla')
        st.session_state.symbol_search = search_query
        if search_query:
            results = search_stocks(search_query)
            if results:
                st.caption(f'{len(results)} result(s):')
                cols = st.columns(2)
                for idx, (sym, name) in enumerate(results[:10]):
                    if cols[idx % 2].button(f'{sym}', key=f'r_{sym}', use_container_width=True, help=name):
                        st.session_state.current_symbol = sym
                        st.session_state.symbol_search = sym
                        st.rerun()
            else:
                st.warning('No results. Try as symbol:')
                if st.button(f"Use '{search_query.upper()}'", use_container_width=True):
                    st.session_state.current_symbol = search_query.upper()
                    st.session_state.symbol_search = search_query.upper()
                    st.rerun()
        else:
            st.caption('Popular:')
            cols = st.columns(4)
            for idx, sym in enumerate(['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AVGO']):
                if cols[idx % 4].button(sym, key=f'p_{sym}', use_container_width=True):
                    st.session_state.current_symbol = sym
                    st.session_state.symbol_search = sym
                    st.rerun()
        current_sym = st.session_state.current_symbol
        if current_sym:
            st.success(f'Selected: **{current_sym}**')
        else:
            st.info('Select a stock above')
        period = st.selectbox('Data Period', ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
        forecast_days = st.slider('Forecast Days', 7, 365, 90, step=7)
        if st.button('Run Forecast', type='primary', use_container_width=True):
            symbol = st.session_state.current_symbol
            if not symbol:
                st.warning('Select a stock first.')
            else:
                with st.spinner(f'Training model for {symbol}…'):
                    try:
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(period=period)
                        if df.empty:
                            st.error(f'No data for {symbol}')
                        else:
                            df.reset_index(inplace=True)
                            df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
                            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                            st.session_state.sales_data = df
                            from app.models.sales_forecast import SalesForecaster
                            model = SalesForecaster()
                            model.fit(df[['date', 'sales']])
                            st.session_state.sales_model = model
                            st.success('Model ready!')
                    except Exception as e:
                        st.error(f'Error: {e}')
        params = {'type': 'sales', 'symbol': current_sym or 'NVDA', 'forecast_days': forecast_days}
    else:
        st.subheader('Stock Settings')
        symbol = st.text_input('Symbol', value='AAPL').upper().strip()
        period = st.selectbox('History Period', ['1y', '2y', '3y', '5y'], index=1)
        forecast_days = st.slider('Forecast Days', 5, 30, 5, step=1)
        if symbol != st.session_state.trained_inv_symbol:
            st.session_state.stock_model = None
            st.session_state.stock_data = None
        if st.button('Run Forecast', type='primary', use_container_width=True):
            if not symbol:
                st.warning('Enter a symbol.')
            else:
                with st.spinner(f'Training model for {symbol}…'):
                    try:
                        from app.models.stock_predictor import StockPredictor
                        predictor = StockPredictor()
                        df_raw = predictor.fetch_data(symbol, period=period)
                        predictor.fit(df=df_raw)
                        st.session_state.stock_model = predictor
                        st.session_state.stock_data = predictor.data
                        st.session_state.trained_inv_symbol = symbol
                        st.success('Model ready!')
                    except Exception as e:
                        st.error(f'Error: {e}')
        params = {'type': 'investment', 'symbol': symbol, 'period': period, 'forecast_days': forecast_days}
st.title('Investment & Sales Prediction')
st.caption('Production-Grade Forecasting System · v2.0')
if params['type'] == 'sales':
    symbol = params['symbol']
    company_name = get_company_name(symbol)
    st.header(f'Sales Forecast — {company_name} ({symbol})')
    sales_model = st.session_state.sales_model
    sales_data = st.session_state.sales_data
    if sales_model and sales_model.is_trained and (sales_data is not None):
        try:
            with st.spinner('Generating forecast…'):
                pred_df = sales_model.predict_future(periods=params['forecast_days'])
            current_price = float(sales_data['sales'].iloc[-1])
            future_price = float(pred_df['yhat'].iloc[-1])
            change_pct = (future_price - current_price) / current_price * 100
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Current Price', f'${current_price:.2f}')
            c2.metric('Forecast Price', f'${future_price:.2f}', delta=f'{change_pct:+.2f}%')
            c3.metric('Forecast High', f'${pred_df['yhat'].max():.2f}')
            c4.metric('Forecast Low', f'${pred_df['yhat'].min():.2f}')
            fig = go.Figure()
            hist = sales_data.tail(120)
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['sales'], name='Historical', line=dict(color='#A23B72', width=2)))
            fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['yhat'], name='Forecast', line=dict(color='#2E86AB', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['yhat_lower'], fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(46,134,171,0.15)', name='95% CI'))
            fig.update_layout(title=f'{company_name} — {params['forecast_days']}-Day Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=480, hovermode='x unified', template='plotly_white', yaxis=dict(tickprefix='$', tickformat='.2f'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander('View forecast data'):
                st.dataframe(pred_df.rename(columns={'date': 'Date', 'yhat': 'Forecast ($)', 'yhat_lower': 'Lower ($)', 'yhat_upper': 'Upper ($)'}).style.format({'Forecast ($)': '${:.2f}', 'Lower ($)': '${:.2f}', 'Upper ($)': '${:.2f}'}), use_container_width=True, hide_index=True)
            with st.expander('Model performance details'):
                perf = sales_model.performance
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('MAPE', f'{perf.get('mape', 0):.2f}%')
                m2.metric('R²', f'{perf.get('r2', 0):.3f}')
                m3.metric('Direction Acc', f'{perf.get('direction_accuracy', 0):.1f}%')
                m4.metric('Test Samples', perf.get('test_samples', '—'))
                st.caption(f'Seasonality mode: **{perf.get('seasonality_mode', '—')}** · CV: **{perf.get('cv', '—')}**')
        except Exception as e:
            st.error(f'Forecast error: {e}')
            st.info("Click 'Run Forecast' in the sidebar to retrain.")
    else:
        st.info('Search and select a stock in the sidebar, then click **Run Forecast**.')
else:
    symbol = params['symbol']
    predictor = st.session_state.stock_model
    trained_sym = st.session_state.trained_inv_symbol
    st.header(f'Investment Prediction — {symbol}')
    if predictor and predictor.is_trained and (trained_sym == symbol):
        try:
            pred_df = predictor.predict_future(periods=params['forecast_days'])
            hist_df = predictor.data.tail(120)
            current_price = float(predictor.get_current_price())
            future_price = float(pred_df['yhat'].iloc[-1])
            change_pct = (future_price - current_price) / current_price * 100
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Current Price', f'${current_price:.2f}')
            c2.metric('Forecast Price', f'${future_price:.2f}', delta=f'{change_pct:+.2f}%')
            c3.metric('Forecast High', f'${pred_df['yhat'].max():.2f}')
            c4.metric('Forecast Low', f'${pred_df['yhat'].min():.2f}')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['sales'], name='Historical', line=dict(color='#A23B72', width=2)))
            fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['yhat'], name='Forecast', mode='lines+markers', line=dict(color='#2E86AB', width=2, dash='dash'), marker=dict(size=7)))
            fig.update_layout(title=f'{symbol} — {params['forecast_days']}-Day Price Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=480, hovermode='x unified', template='plotly_white', yaxis=dict(tickprefix='$', tickformat='.2f'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig, use_container_width=True)
            if 'up_probability' in pred_df.columns:
                disp = pred_df.copy()
                disp['Signal'] = disp['up_probability'].apply(lambda p: '🟢 BUY' if p >= 0.6 else '🔴 SELL' if p <= 0.4 else '🟡 HOLD')
                disp['Up Probability'] = disp['up_probability'].map('{:.1%}'.format)
                disp['Price'] = disp['yhat'].map('${:.2f}'.format)
                st.dataframe(disp[['date', 'Price', 'Up Probability', 'Signal']].rename(columns={'date': 'Date'}), use_container_width=True, hide_index=True)
            with st.expander('Model performance details'):
                perf = predictor.performance
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('MAPE', f'{perf.get('mape', 0):.2f}%')
                m2.metric('R²', f'{perf.get('r2', 0):.3f}')
                m3.metric('Direction Acc', f'{perf.get('direction_accuracy', 0):.1f}%')
                m4.metric('Test Samples', perf.get('test_samples', '—'))
        except Exception as e:
            st.error(f'Forecast error: {e}')
            st.info("Click 'Run Forecast' in the sidebar to retrain.")
    else:
        st.info('Enter a symbol in the sidebar and click **Run Forecast**.')
st.divider()
fc1, fc2, fc3 = st.columns(3)
fc1.caption('Model: XGBoost + LightGBM + Prophet')
fc2.caption('Data: Yahoo Finance (live)')
fc3.caption('v2.0 · Production Grade')