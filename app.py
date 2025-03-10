import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
import requests
from stock_analyzer import StockAnalyzer
import json
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

def fetch_market_news(source, limit=5):
    try:
        # Calculate date 6 months ago
        six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        ticker = yf.Ticker(source)
        news = ticker.news
        if news:
            # Sort news by date and filter last 6 months
            filtered_news = [
                n for n in news 
                if datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d') >= six_months_ago
            ]
            return sorted(filtered_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)
        return []
    except Exception:
        return []

def fetch_indian_market_news():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        # Using IndianAPI for market news
        news_url = "https://indianapi.in/api/v1/market/news"
        response = requests.get(news_url, headers=headers)
        if response.status_code == 200:
            return response.json().get('data', [])
        return []
    except Exception:
        return []

def fetch_nse_news():
    try:
        # Using multiple sources to get market news
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json'
        }
        
        # Try to get news from MoneyControl
        try:
            mc_url = "https://www.moneycontrol.com/news/business/markets/page-1"
            response = requests.get(mc_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Parse the HTML content to extract news
                soup = BeautifulSoup(response.text, 'html.parser')
                news_list = []
                for article in soup.select('.clearfix.common-article'):
                    try:
                        title = article.select_one('.clearfix a').text.strip()
                        link = article.select_one('.clearfix a')['href']
                        description = article.select_one('.desc').text.strip() if article.select_one('.desc') else ''
                        date = article.select_one('.article_schedule').text.strip() if article.select_one('.article_schedule') else ''
                        news_list.append({
                            'title': title,
                            'description': description,
                            'url': link,
                            'date': date
                        })
                    except Exception:
                        continue
                return news_list[:10]  # Return top 10 news items
        except Exception:
            pass
        
        # Fallback to Economic Times
        try:
            et_url = "https://economictimes.indiatimes.com/markets/json/market_news.htm"
            response = requests.get(et_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data[:10]  # Return top 10 news items
        except Exception:
            pass
            
        return []
    except Exception:
        return []

def fetch_stocktwits_data(symbol):
    try:
        # StockTwits API endpoints
        base_url = "https://api.stocktwits.com/api/2"
        
        # Get stream data for symbol
        stream_url = f"{base_url}/streams/symbol/{symbol}.json"
        response = requests.get(stream_url)
        
        if response.status_code == 200:
            data = response.json()
            messages = data.get('messages', [])
            
            # Calculate sentiment
            bullish = sum(1 for msg in messages if msg.get('entities', {}).get('sentiment', {}).get('basic') == 'Bullish')
            bearish = sum(1 for msg in messages if msg.get('entities', {}).get('sentiment', {}).get('basic') == 'Bearish')
            
            return {
                'messages': messages[:5],  # Latest 5 messages
                'sentiment': {
                    'bullish': bullish,
                    'bearish': bearish
                }
            }
        return None
    except Exception:
        return None

def fetch_trending_stocks():
    try:
        trending_url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        response = requests.get(trending_url)
        if response.status_code == 200:
            return response.json().get('symbols', [])[:5]  # Top 5 trending
        return []
    except Exception:
        return []

# API key validation
def validate_api_keys():
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        st.error("⚠️ Missing Required API Key")
        st.warning(
            "OpenAI API key is missing from your .env file. "
            "Please add your OPENAI_API_KEY to the .env file to use the analysis features."
        )
        return False
    return True

# Page configuration
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for market indices
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stock-metrics {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .market-header {
        background-color: #aeebc5;
        padding: 1rem 0;
        position: sticky;
        top: 0;
        z-index: 999;
        border-bottom: 1px solid #2d2d2d;
        margin: -6rem -4rem 2rem -4rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }
    .market-ticker {
        text-align: center;
        padding: 0.5rem;
        border-radius: 4px;
        background-color: white;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ticker-price {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0;
        color: #333;
    }
    .ticker-change {
        font-size: 0.9rem;
        margin: 0;
    }
    .ticker-name {
        font-size: 0.9rem;
        color: #555;
        margin: 0;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    if validate_api_keys():
        st.session_state.analyzer = StockAnalyzer()
    else:
        st.stop()

# Sidebar
st.sidebar.title("📊 Stock Analysis Settings")

# Add Market News Section in Sidebar
st.sidebar.markdown("### 📰 Market News")
show_market_news = st.sidebar.checkbox("Show Market Updates")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()

# Time period selection
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

# Analysis type selection
analysis_types = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Fundamental Analysis", "Technical Analysis", "News Analysis"],
    default=["Fundamental Analysis", "Technical Analysis", "News Analysis"]
)

# Market Indices Section at the very top (before page config)
indices = {
    "^NSEI": {"name": "NIFTY 50", "currency": "₹"},
    "^BSESN": {"name": "SENSEX", "currency": "₹"},
    "^NSEBANK": {"name": "BANK NIFTY", "currency": "₹"},
    "^IXIC": {"name": "NASDAQ", "currency": "$"},
    "^GSPC": {"name": "S&P 500", "currency": "$"}
}

# Display Market Indices before the title
st.markdown('<div class="market-header">', unsafe_allow_html=True)
index_cols = st.columns(len(indices))

for i, (index_symbol, index_info) in enumerate(indices.items()):
    try:
        index = yf.Ticker(index_symbol)
        current_price = index.info.get('regularMarketPrice', 'N/A')
        previous_close = index.info.get('previousClose', 'N/A')
        
        if current_price != 'N/A' and previous_close != 'N/A':
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            color = "green" if change >= 0 else "red"
            
            with index_cols[i]:
                st.markdown(f"""
                    <div class="market-ticker">
                        <p class="ticker-name">{index_info['name']}</p>
                        <p class="ticker-price">{index_info['currency']}{current_price:,.2f}</p>
                        <p class="ticker-change" style="color: {color}">
                            {change:+,.2f} ({change_pct:+.2f}%)
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    except Exception:
        with index_cols[i]:
            st.markdown(f"""
                <div class="market-ticker">
                    <p class="ticker-name">{index_info['name']}</p>
                    <p class="ticker-price">Error</p>
                    <p class="ticker-change">--</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main title after market indices
st.title("🚀 AI Stock Analysis Platform")

# Market News Section with Indian Exchange Data
if show_market_news:
    st.markdown("### 📰 Market Updates")
    news_tabs = st.tabs(["NSE News", "StockTwits Trending", "Company News"])
    
    with news_tabs[0]:
        nse_news = fetch_nse_news()
        if nse_news:
            st.markdown("### 📈 Latest Market Updates")
            for news in nse_news:
                # Extract news data safely
                if isinstance(news, dict):
                    title = news.get('title', '')
                    description = news.get('summary', news.get('description', ''))
                    url = news.get('url', news.get('link', '#'))
                    date = news.get('publishedAt', news.get('dateTime', ''))
                    
                    # Skip empty news items
                    if not title and not description:
                        continue
                    
                    # Format date
                    try:
                        if isinstance(date, (int, float)):
                            date = datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        pass
                    
                    # Safely truncate description
                    desc_text = str(description)
                    if len(desc_text) > 300:
                        desc_text = desc_text[:300] + "..."
                    
                    st.markdown(f"""
                        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; background-color: white;">
                            <h4 style="margin: 0; color: #333; font-size: 16px;">{title}</h4>
                            <p style="font-size: 0.8em; color: #666; margin: 5px 0;">{date}</p>
                            <p style="color: #444; font-size: 14px; margin: 10px 0;">{desc_text}</p>
                            <a href="{url}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 14px;">Read more →</a>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No market news available at the moment.")
    
    with news_tabs[1]:
        trending = fetch_trending_stocks()
        if trending:
            st.markdown("### 🔥 Trending Stocks")
            for symbol in trending:
                st.markdown(f"""
                    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px 0; background-color: white;">
                        <h4 style="margin: 0; color: #333;">{symbol['symbol']}</h4>
                        <p style="color: #666;">Watchlist Count: {symbol['watchlist_count']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trending stocks available at the moment.")
    
    with news_tabs[2]:
        if symbol:
            company_news = fetch_market_news(symbol)
            if company_news:
                st.markdown(f"### 📰 Last 6 Months News for {symbol}")
                for news in company_news:
                    # Extract news data safely
                    title = news.get('title', '')
                    summary = news.get('summary', '')
                    link = news.get('link', '#')
                    publish_time = datetime.fromtimestamp(news.get('publishedAt', 0))
                    
                    # Skip empty news items
                    if not title and not summary:
                        continue
                    
                    st.markdown(f"""
                        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; background-color: white;">
                            <h4 style="margin: 0; color: #333; font-size: 16px;">{title}</h4>
                            <p style="font-size: 0.8em; color: #666; margin: 5px 0;">{publish_time.strftime('%Y-%m-%d %H:%M')}</p>
                            <p style="color: #444; font-size: 14px; margin: 10px 0;">{summary}</p>
                            <a href="{link}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 14px;">Read more →</a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No news available for {symbol} in the last 6 months.")
        else:
            st.info("Enter a stock symbol to view company news.")

if symbol:
    try:
        # Display loading message
        with st.spinner(f'Fetching data for {symbol}...'):
            # Get stock info for quick metrics
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Determine currency based on exchange
            currency = "₹" if ".NS" in symbol else "$"
            
            # Quick metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"{currency}{info.get('currentPrice', 'N/A'):,.2f}",
                    delta=f"{info.get('regularMarketChangePercent', 0):.2f}%"
                )
                
            with col2:
                market_cap = info.get('marketCap', 0)
                if market_cap >= 1e12:
                    market_cap_str = f"{market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"{market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_str = f"{market_cap/1e6:.2f}M"
                else:
                    market_cap_str = f"{market_cap:,.2f}"
                st.metric(
                    label="Market Cap",
                    value=f"{currency}{market_cap_str}"
                )
                
            with col3:
                st.metric(
                    label="Volume",
                    value=f"{info.get('volume', 0):,.0f}"
                )
                
            with col4:
                st.metric(
                    label="P/E Ratio",
                    value=f"{info.get('forwardPE', 'N/A'):.2f}"
                )

            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["AI Analysis", "Charts", "Raw Data"])
            
            with tab1:
                if st.button("Generate AI Analysis"):
                    with st.spinner('Generating comprehensive analysis... This may take a few minutes.'):
                        # Get AI analysis
                        analysis_result = st.session_state.analyzer.analyze_stock(symbol, analysis_types)
                        st.markdown(analysis_result)
            
            with tab2:
                # Historical price chart
                hist_data = stock.history(period=time_period)
                
                # Add company business information
                st.subheader("Company Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Business Overview")
                    st.write(info.get('longBusinessSummary', 'No business summary available.'))
                
                with col2:
                    st.markdown("### Key Business Segments")
                    sectors = info.get('sector', 'N/A')
                    industry = info.get('industry', 'N/A')
                    country = info.get('country', 'N/A')
                    st.write(f"**Sector:** {sectors}")
                    st.write(f"**Industry:** {industry}")
                    st.write(f"**Country:** {country}")
                
                # Candlestick chart
                fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                            open=hist_data['Open'],
                            high=hist_data['High'],
                            low=hist_data['Low'],
                            close=hist_data['Close'],
                            name="OHLC")])
                
                # Add moving averages
                fig.add_trace(go.Scatter(x=hist_data.index, 
                                       y=hist_data['Close'].rolling(window=20).mean(),
                                       line=dict(color='orange', width=1),
                                       name="20-day MA"))
                fig.add_trace(go.Scatter(x=hist_data.index, 
                                       y=hist_data['Close'].rolling(window=50).mean(),
                                       line=dict(color='blue', width=1),
                                       name="50-day MA"))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure(data=[go.Bar(x=hist_data.index, 
                                                   y=hist_data['Volume'],
                                                   name="Volume")])
                fig_volume.update_layout(
                    title=f"{symbol} Trading Volume",
                    yaxis_title="Volume",
                    xaxis_title="Date",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_volume, use_container_width=True)
            
            with tab3:
                st.subheader("Historical Data")
                
                # Add P/E ratio comparison
                pe_col1, pe_col2, pe_col3 = st.columns(3)
                with pe_col1:
                    current_pe = info.get('forwardPE', info.get('trailingPE', 'N/A'))
                    st.metric(
                        label="Current P/E Ratio",
                        value=f"{current_pe:.2f}" if isinstance(current_pe, (int, float)) else 'N/A'
                    )
                with pe_col2:
                    try:
                        # Try multiple methods to get median PE
                        median_pe = None
                        try:
                            financials = stock.financials
                            if not financials.empty:
                                earnings = financials.loc['Net Income']
                                if len(earnings) > 0:
                                    avg_earnings = earnings.mean()
                                    current_price = info.get('regularMarketPrice', 0)
                                    if avg_earnings != 0:
                                        median_pe = current_price / (avg_earnings / info.get('sharesOutstanding', 1))
                        except:
                            pass
                        
                        if median_pe is None:
                            median_pe = info.get('fiveYearAvgDividendYield', info.get('trailingPE', 'N/A'))
                        
                        st.metric(
                            label="Median P/E Ratio (5Y)",
                            value=f"{median_pe:.2f}" if isinstance(median_pe, (int, float)) else 'N/A'
                        )
                    except:
                        st.metric(
                            label="Median P/E Ratio (5Y)",
                            value='N/A'
                        )
                with pe_col3:
                    try:
                        # Try to get industry PE from multiple sources
                        industry_pe = info.get('industryPE', None)
                        if industry_pe is None:
                            sector = info.get('sector', '')
                            industry = info.get('industry', '')
                            # Calculate average PE of similar companies
                            similar_companies = yf.Tickers(f"{sector} {industry}").tickers
                            pes = []
                            for comp in similar_companies:
                                try:
                                    pe = comp.info.get('forwardPE', comp.info.get('trailingPE', None))
                                    if pe is not None:
                                        pes.append(pe)
                                except:
                                    continue
                            if pes:
                                industry_pe = sum(pes) / len(pes)
                        
                        st.metric(
                            label="Industry P/E Ratio",
                            value=f"{industry_pe:.2f}" if isinstance(industry_pe, (int, float)) else 'N/A'
                        )
                    except:
                        st.metric(
                            label="Industry P/E Ratio",
                            value='N/A'
                        )
                
                # Add Shareholding Pattern with Pie Charts
                st.subheader("Shareholding Pattern")
                try:
                    # Get shareholding data
                    major_holders = stock.major_holders
                    institutional_holders = stock.institutional_holders
                    
                    # Major Holders Section
                    st.markdown('<div class="shareholding-container">', unsafe_allow_html=True)
                    st.markdown("### Major Holders")
                    
                    if isinstance(major_holders, pd.DataFrame):
                        try:
                            # Convert percentage strings to float values and clean data
                            values = []
                            labels = []
                            
                            # Handle different data formats
                            for index, row in major_holders.iterrows():
                                try:
                                    if len(row) >= 2:
                                        # Clean and convert percentage value
                                        val_str = str(row[0]).replace('%', '').strip()
                                        val = float(val_str)
                                        
                                        # Clean label
                                        label = str(row[1]).strip()
                                        if label and val > 0:  # Only add if label exists and value is positive
                                            values.append(val)
                                            labels.append(label)
                                except (ValueError, TypeError):
                                    continue
                            
                            if values and labels:
                                # Create two columns for chart and table
                                chart_col, table_col = st.columns([2, 1])
                                
                                with chart_col:
                                    colors = px.colors.qualitative.Set3[:len(values)]
                                    fig = go.Figure(data=[go.Pie(
                                        labels=labels,
                                        values=values,
                                        hole=0.5,
                                        marker=dict(
                                            colors=colors,
                                            line=dict(color='white', width=2)
                                        ),
                                        textinfo='label+percent',
                                        textposition='outside',
                                        pull=[0.1 if i == 0 else 0 for i in range(len(values))],
                                        textfont=dict(size=12)
                                    )])
                                    
                                    fig.update_layout(
                                        title={
                                            'text': "Shareholding Distribution",
                                            'y':0.95,
                                            'x':0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top',
                                            'font': dict(size=16)
                                        },
                                        showlegend=True,
                                        height=500,
                                        paper_bgcolor='white',
                                        plot_bgcolor='white',
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=-0.3,
                                            xanchor="center",
                                            x=0.5,
                                            font=dict(size=12)
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with table_col:
                                    st.markdown("#### Detailed Breakdown")
                                    formatted_holders = pd.DataFrame({
                                        'Category': labels,
                                        'Percentage': [f"{v:.2f}%" for v in values]
                                    })
                                    st.dataframe(formatted_holders, use_container_width=True, height=450)
                            else:
                                st.warning("No valid shareholding data available")
                        except Exception as e:
                            st.error(f"Error processing shareholding data: {str(e)}")
                    else:
                        st.info("No major holders data available")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Institutional Holders Section
                    st.markdown('<div class="shareholding-container">', unsafe_allow_html=True)
                    st.markdown("### Institutional Holders")
                    if isinstance(institutional_holders, pd.DataFrame) and not institutional_holders.empty and len(institutional_holders) > 0:
                        try:
                            # Create two columns for chart and table
                            chart_col, table_col = st.columns([2, 1])
                            
                            with chart_col:
                                # Create pie chart for top 10 institutional holders
                                top_10_holders = institutional_holders.head(10)
                                
                                # Convert data types and clean data
                                top_10_holders['Holder'] = top_10_holders['Holder'].astype(str)
                                top_10_holders['Shares'] = pd.to_numeric(top_10_holders['Shares'], errors='coerce')
                                
                                # Remove rows with NaN values
                                top_10_holders = top_10_holders.dropna(subset=['Shares'])
                                
                                if not top_10_holders.empty:
                                    fig = go.Figure(data=[go.Pie(
                                        labels=top_10_holders['Holder'],
                                        values=top_10_holders['Shares'],
                                        hole=0.5,
                                        marker=dict(
                                            colors=px.colors.qualitative.Set3[:len(top_10_holders)],
                                            line=dict(color='white', width=2)
                                        ),
                                        textinfo='label+percent',
                                        textposition='outside',
                                        pull=[0.1 if i == 0 else 0 for i in range(len(top_10_holders))],
                                        textfont=dict(size=12)
                                    )])
                                    
                                    fig.update_layout(
                                        title={
                                            'text': "Top 10 Institutional Holders",
                                            'y':0.95,
                                            'x':0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top',
                                            'font': dict(size=16)
                                        },
                                        showlegend=True,
                                        height=500,
                                        paper_bgcolor='white',
                                        plot_bgcolor='white',
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=-0.3,
                                            xanchor="center",
                                            x=0.5,
                                            font=dict(size=12)
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with table_col:
                                st.markdown("#### Top Holders")
                                formatted_inst = top_10_holders.copy()
                                formatted_inst['Shares'] = formatted_inst['Shares'].apply(
                                    lambda x: f"{float(x):,.0f}" if pd.notnull(x) else 'N/A'
                                )
                                formatted_inst['Value'] = formatted_inst['Value'].apply(
                                    lambda x: f"{currency}{float(x):,.0f}" if pd.notnull(x) else 'N/A'
                                )
                                formatted_inst = formatted_inst[['Holder', 'Shares', 'Value']]
                                formatted_inst.columns = ['Institution', 'Shares Held', 'Value']
                                st.dataframe(formatted_inst, use_container_width=True, height=450)
                        except Exception as e:
                            st.error(f"Error creating institutional holders chart: {str(e)}")
                    else:
                        st.info("No institutional holders data available")
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Unable to fetch shareholding pattern: {str(e)}")
                    st.info("This might be due to data availability restrictions for this stock. Please try another stock symbol.")
                
                # Add StockTwits sentiment data
                st.subheader("Market Sentiment (StockTwits)")
                stocktwits_data = fetch_stocktwits_data(symbol)
                
                if stocktwits_data:
                    sent_col1, sent_col2 = st.columns(2)
                    
                    with sent_col1:
                        # Sentiment pie chart
                        sentiment = stocktwits_data['sentiment']
                        fig = go.Figure(data=[go.Pie(
                            labels=['Bullish', 'Bearish'],
                            values=[sentiment['bullish'], sentiment['bearish']],
                            hole=.3,
                            marker_colors=['#00ff00', '#ff0000']
                        )])
                        fig.update_layout(
                            title="Market Sentiment Distribution",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with sent_col2:
                        # Recent messages
                        st.markdown("### Recent Messages")
                        for msg in stocktwits_data['messages']:
                            st.markdown(f"""
                                <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                    <p><strong>{msg['user']['username']}</strong>: {msg['body']}</p>
                                    <small>{msg['created_at']}</small>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No StockTwits data available")

                # Add Financial Statements
                st.subheader("Financial Statements")
                fin_tabs = st.tabs(["Quarterly Results", "Income Statement", "Balance Sheet", "Cash Flow"])
                
                with fin_tabs[0]:
                    try:
                        # Try multiple methods to fetch quarterly data
                        quarterly_data = None
                        try:
                            quarterly_data = stock.quarterly_financials
                        except Exception:
                            try:
                                quarterly_data = stock.quarterly_earnings
                            except Exception:
                                pass
                        
                        if quarterly_data is not None and not quarterly_data.empty:
                            # Format earnings with currency
                            formatted_quarterly = quarterly_data.copy()
                            
                            # Format all numeric columns
                            for col in formatted_quarterly.columns:
                                try:
                                    def format_value(x):
                                        if isinstance(x, (int, float)):
                                            if abs(x) >= 1e6:
                                                return f"{currency}{x/1e6:,.2f}M"
                                            return f"{currency}{x:,.2f}"
                                        return str(x)
                                    
                                    formatted_quarterly[col] = formatted_quarterly[col].apply(format_value)
                                except Exception:
                                    continue
                            
                            # Display the formatted dataframe
                            st.markdown('<div style="background-color: #1a1a1a; padding: 10px; border-radius: 5px;">', 
                                      unsafe_allow_html=True)
                            st.dataframe(formatted_quarterly, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add download button
                            csv = quarterly_data.to_csv()
                            st.download_button(
                                label="Download Quarterly Results",
                                data=csv,
                                file_name=f"{symbol}_quarterly_results.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No quarterly results available for this stock.")
                    except Exception as e:
                        st.error(f"Unable to fetch quarterly results: {str(e)}")
                
                with fin_tabs[1]:
                    try:
                        income_stmt = stock.income_stmt
                        if not income_stmt.empty:
                            # Format income statement with currency
                            formatted_income = income_stmt.copy()
                            for col in formatted_income.columns:
                                formatted_income[col] = formatted_income[col].apply(
                                    lambda x: f"{currency}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{currency}{x:,.2f}"
                                )
                            st.dataframe(formatted_income)
                        else:
                            st.info("No income statement available")
                    except Exception as e:
                        st.error("Unable to fetch income statement")
                
                with fin_tabs[2]:
                    try:
                        balance_sheet = stock.balance_sheet
                        if not balance_sheet.empty:
                            # Format balance sheet with currency
                            formatted_balance = balance_sheet.copy()
                            for col in formatted_balance.columns:
                                formatted_balance[col] = formatted_balance[col].apply(
                                    lambda x: f"{currency}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{currency}{x:,.2f}"
                                )
                            st.dataframe(formatted_balance)
                        else:
                            st.info("No balance sheet available")
                    except Exception as e:
                        st.error("Unable to fetch balance sheet")
                
                with fin_tabs[3]:
                    try:
                        cash_flow = stock.cashflow
                        if not cash_flow.empty:
                            # Format cash flow with currency
                            formatted_cash = cash_flow.copy()
                            for col in formatted_cash.columns:
                                formatted_cash[col] = formatted_cash[col].apply(
                                    lambda x: f"{currency}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{currency}{x:,.2f}"
                                )
                            st.dataframe(formatted_cash)
                        else:
                            st.info("No cash flow statement available")
                    except Exception as e:
                        st.error("Unable to fetch cash flow statement")
                
                st.subheader("Price History")
                # Format the dataframe with proper currency
                formatted_data = hist_data.copy()
                
                # Format price columns with currency
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    formatted_data[col] = formatted_data[col].apply(lambda x: f"{currency}{x:,.2f}")
                
                # Format volume with commas
                formatted_data['Volume'] = formatted_data['Volume'].apply(lambda x: f"{x:,.0f}")
                
                # Display the formatted dataframe
                st.dataframe(formatted_data)
                
                # Add download buttons for all data
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    csv = hist_data.to_csv(index=True)
                    st.download_button(
                        label="Download Price History",
                        data=csv,
                        file_name=f"{symbol}_historical_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    try:
                        csv = stock.quarterly_earnings.to_csv(index=True)
                        st.download_button(
                            label="Download Quarterly Results",
                            data=csv,
                            file_name=f"{symbol}_quarterly_results.csv",
                            mime="text/csv"
                        )
                    except:
                        pass
                
                with col3:
                    try:
                        csv = stock.income_stmt.to_csv(index=True)
                        st.download_button(
                            label="Download Income Statement",
                            data=csv,
                            file_name=f"{symbol}_income_statement.csv",
                            mime="text/csv"
                        )
                    except:
                        pass
                
                with col4:
                    try:
                        csv = stock.balance_sheet.to_csv(index=True)
                        st.download_button(
                            label="Download Balance Sheet",
                            data=csv,
                            file_name=f"{symbol}_balance_sheet.csv",
                            mime="text/csv"
                        )
                    except:
                        pass

    except Exception as e:
        st.error(f"Error analyzing {symbol}. Please check if the symbol is correct.")
        st.exception(e)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This AI-powered stock analysis platform combines fundamental analysis, "
    "technical indicators, and market sentiment to provide comprehensive insights. "
    "The analysis is powered by OpenAI's GPT-4 and uses real-time market data."
)

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("### Version Info")
st.sidebar.text("v1.0.0") 