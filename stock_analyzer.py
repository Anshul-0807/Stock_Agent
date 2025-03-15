from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool, Tool
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from duckduckgo_search import DDGS
from typing import Any, Optional, Dict, Type
from pydantic import BaseModel, Field
from scipy import stats
from utils import calculate_risk_metrics

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7,
    api_key=openai_api_key
)

# Define input schemas for tools
class StockDataInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")

class WebNewsInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to search news for")

class TechnicalAnalysisInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")

class FinancialRatiosInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")

class SectorAnalysisInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")

class DividendGrowthInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol to analyze")

# Define Tool classes
class StockDataTool(BaseTool):
    name: str = "Stock Fundamental Analysis"
    description: str = "Analyzes fundamental stock data including price, market cap, P/E ratio, and other key metrics"
    args_schema: Type[BaseModel] = StockDataInput
    
    def _run(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Determine currency based on exchange
            currency = "₹" if ".NS" in symbol else "$"
            
            # Format market cap based on size
            market_cap = info.get('marketCap', 0)
            if market_cap >= 1e12:
                market_cap_str = f"{market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"{market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"{market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"{market_cap:,.2f}"
            
            analysis = {
                "Company Name": info.get('longName', 'N/A'),
                "Current Price": f"{currency}{info.get('currentPrice', 'N/A'):,.2f}",
                "Market Cap": f"{currency}{market_cap_str}",
                "P/E Ratio": f"{info.get('forwardPE', 'N/A'):.2f}",
                "52 Week Range": f"{currency}{info.get('fiftyTwoWeekLow', 'N/A'):,.2f} - {currency}{info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
                "Revenue Growth": f"{info.get('revenueGrowth', 'N/A')*100:.2f}%" if info.get('revenueGrowth') else 'N/A',
                "Profit Margins": f"{info.get('profitMargins', 'N/A')*100:.2f}%" if info.get('profitMargins') else 'N/A',
                "Analyst Rating": info.get('recommendationKey', 'N/A').upper(),
                "Currency": currency
            }
            
            return str(analysis)
        except Exception as e:
            return f"Error analyzing stock data: {str(e)}"

    def _arun(self, symbol: str) -> str:
        raise NotImplementedError("Async not implemented")

class WebNewsTool(BaseTool):
    name: str = "Web News Search"
    description: str = "Searches for news from various web sources using DuckDuckGo"
    args_schema: Type[BaseModel] = WebNewsInput
    
    def _run(self, symbol: str) -> str:
        try:
            with DDGS() as ddgs:
                company_name = yf.Ticker(symbol).info.get('longName', symbol)
                # Search for different types of news
                search_terms = [
                    f"{company_name} stock news last 30 days",
                    f"{company_name} earnings report",
                    f"{company_name} company announcements",
                    f"{company_name} market analysis",
                ]
                
                all_results = []
                for term in search_terms:
                    try:
                        results = list(ddgs.text(term, max_results=3))
                        if results:
                            all_results.extend(results)
                    except Exception as e:
                        print(f"Error searching for term '{term}': {str(e)}")
                        continue
                
                if not all_results:
                    return "No recent news found for this stock."
                
                # Remove duplicates based on title
                seen_titles = set()
                news_summary = []
                for result in all_results:
                    if result.get('title') and result['title'] not in seen_titles:
                        seen_titles.add(result['title'])
                        news_summary.append({
                            "title": result.get('title', 'No Title'),
                            "snippet": result.get('body', 'No content available'),
                            "source": result.get('source', 'Unknown Source'),
                            "date": result.get('date', 'Unknown Date'),
                            "url": result.get('link', '#')
                        })
                
                if not news_summary:
                    return "No valid news articles found."
                
                # Sort by date (most recent first)
                news_summary.sort(key=lambda x: x['date'], reverse=True)
                
                # Format the output
                formatted_news = "Recent News and Updates:\n\n"
                for news in news_summary[:5]:  # Show top 5 most recent news
                    formatted_news += f"📰 {news['title']}\n"
                    formatted_news += f"📅 {news['date']}\n"
                    formatted_news += f"🔍 {news['snippet']}\n"
                    formatted_news += f"📌 Source: {news['source']}\n"
                    formatted_news += f"🔗 Link: {news['url']}\n"
                    formatted_news += "-" * 80 + "\n\n"
                
                return formatted_news
        except Exception as e:
            return f"Error fetching news: {str(e)}"

    def _arun(self, symbol: str) -> str:
        raise NotImplementedError("Async not implemented")

class TechnicalAnalysisTool(BaseTool):
    name: str = "Technical Analysis"
    description: str = "Analyzes technical indicators including moving averages, RSI, and Bollinger Bands"
    args_schema: Type[BaseModel] = TechnicalAnalysisInput
    
    def _run(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            # Fix period format
            valid_periods = {
                '1m': '1mo',
                '3m': '3mo',
                '6m': '6mo',
                '1y': '1y',
                '2y': '2y',
                '5y': '5y',
                '10y': '10y',
                'ytd': 'ytd',
                'max': 'max'
            }
            period = valid_periods.get('1y', '1y')
            hist = stock.history(period=period)
            
            if hist.empty:
                return "No historical data available for the specified period"
                
            # Determine currency based on exchange
            currency = "₹" if ".NS" in symbol else "$"
            
            sma_20 = SMAIndicator(close=hist['Close'], window=20)
            sma_50 = SMAIndicator(close=hist['Close'], window=50)
            rsi = RSIIndicator(close=hist['Close'])
            bb = BollingerBands(close=hist['Close'])
            
            # Get the last values using iloc to avoid deprecation warnings
            current_price = hist['Close'].iloc[-1]
            sma_20_val = sma_20.sma_indicator().iloc[-1]
            sma_50_val = sma_50.sma_indicator().iloc[-1]
            rsi_val = rsi.rsi().iloc[-1]
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            
            analysis = {
                "Current Price": f"{currency}{current_price:,.2f}",
                "SMA 20": f"{currency}{sma_20_val:,.2f}",
                "SMA 50": f"{currency}{sma_50_val:,.2f}",
                "RSI": f"{rsi_val:.2f}",
                "Bollinger Upper": f"{currency}{bb_upper:,.2f}",
                "Bollinger Lower": f"{currency}{bb_lower:,.2f}",
                "Currency": currency
            }
            
            return str(analysis)
        except Exception as e:
            return f"Error calculating technical indicators: {str(e)}"

    def _arun(self, symbol: str) -> str:
        raise NotImplementedError("Async not implemented")

class FinancialRatiosTool(BaseTool):
    name: str = "Financial Ratios Analysis"
    description: str = "Calculates key financial ratios including D/E, current ratio, ROE, ROA, and operating margin"
    args_schema: Type[BaseModel] = FinancialRatiosInput
    
    def _run(self, symbol: str) -> str:
        try:
            # Check if we already have calculated ratios in session state
            if hasattr(st.session_state, 'financial_ratios') and st.session_state.financial_ratios is not None:
                print(f"Using cached financial ratios from session state: {st.session_state.financial_ratios}")
                return str(st.session_state.financial_ratios)
            
            print(f"Calculating financial ratios for {symbol}...")
            stock = yf.Ticker(symbol)
            
            # Initialize ratios dictionary
            ratios = {}
            
            # Try to get ratios from info first (most reliable source)
            try:
                info = stock.info
                if info:
                    # Get company name for better identification
                    company_name = info.get('longName', '')
                    print(f"Analyzing {company_name} ({symbol})")
                    
                    # Try to get metrics directly from info
                    if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                        ratios['ROE'] = f"{info['returnOnEquity']*100:.2f}%"
                    
                    if 'returnOnAssets' in info and info['returnOnAssets'] is not None:
                        ratios['ROA'] = f"{info['returnOnAssets']*100:.2f}%"
                    
                    if 'operatingMargins' in info and info['operatingMargins'] is not None:
                        ratios['Operating Margin'] = f"{info['operatingMargins']*100:.2f}%"
                    
                    if 'currentRatio' in info and info['currentRatio'] is not None:
                        ratios['Current Ratio'] = f"{info['currentRatio']:.2f}"
                    
                    # Calculate Debt-to-Equity if we have the components
                    if 'totalDebt' in info and 'totalShareholderEquity' in info:
                        if info['totalShareholderEquity'] > 0:
                            de_ratio = info['totalDebt'] / info['totalShareholderEquity']
                            ratios['Debt-to-Equity'] = f"{de_ratio:.2f}"
                    
                    # Add additional ratios if available
                    if 'priceToBook' in info and info['priceToBook'] is not None:
                        ratios['Price/Book'] = f"{info['priceToBook']:.2f}"
                    
                    if 'forwardPE' in info and info['forwardPE'] is not None:
                        ratios['P/E Ratio'] = f"{info['forwardPE']:.2f}"
                    elif 'trailingPE' in info and info['trailingPE'] is not None:
                        ratios['P/E Ratio'] = f"{info['trailingPE']:.2f}"
                    
                    if 'enterpriseToEbitda' in info and info['enterpriseToEbitda'] is not None:
                        ratios['EV/EBITDA'] = f"{info['enterpriseToEbitda']:.2f}"
                    
                    if 'profitMargins' in info and info['profitMargins'] is not None:
                        ratios['Net Profit Margin'] = f"{info['profitMargins']*100:.2f}%"
                    
                    print(f"Got ratios from info: {ratios}")
            except Exception as e:
                print(f"Error getting ratios from info: {str(e)}")
            
            # Store in session state for future use
            if hasattr(st, 'session_state'):
                st.session_state.financial_ratios = ratios
                print(f"Stored financial ratios in session state: {ratios}")
            
            print(f"Final calculated ratios: {ratios}")
            return str(ratios)
        except Exception as e:
            print(f"Error calculating financial ratios: {str(e)}")
            # Return a dictionary with N/A values for all ratios
            default_ratios = {
                'Debt-to-Equity': 'N/A',
                'Current Ratio': 'N/A',
                'ROE': 'N/A',
                'ROA': 'N/A',
                'Operating Margin': 'N/A',
                'ROCE': 'N/A',
                'Dividend Yield': 'N/A',
                'Net Profit Margin': 'N/A',
                'P/E Ratio': 'N/A',
                'Price/Book': 'N/A'
            }
            return str(default_ratios)

    def _arun(self, symbol: str) -> str:
        raise NotImplementedError("Async not implemented")

class SectorAnalysisTool(BaseTool):
    name: str = "Sector and Peer Analysis"
    description: str = "Analyzes sector performance and compares with peer companies"
    args_schema: Type[BaseModel] = SectorAnalysisInput
    
    def _run(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                return "Unable to fetch company information"
            
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # Get sector performance
            sector_performance = {}
            try:
                # Determine appropriate sector ETF based on sector and exchange
                if ".NS" in symbol:
                    # For Indian stocks, use appropriate sector index
                    if 'financial' in sector.lower():
                        sector_etf = yf.Ticker("^NSEBANK")  # NIFTY BANK for financials
                    elif 'technology' in sector.lower():
                        sector_etf = yf.Ticker("^CNXIT")  # NIFTY IT for technology
                    elif 'healthcare' in sector.lower():
                        sector_etf = yf.Ticker("^CNXPHARMA")  # NIFTY PHARMA for healthcare
                    else:
                        sector_etf = yf.Ticker("^NSEI")  # NIFTY 50 as default
                else:
                    # For US stocks, use appropriate sector ETF
                    if 'technology' in sector.lower():
                        sector_etf = yf.Ticker("XLK")  # Technology Select Sector SPDR
                    elif 'healthcare' in sector.lower():
                        sector_etf = yf.Ticker("XLV")  # Healthcare Select Sector SPDR
                    elif 'financial' in sector.lower():
                        sector_etf = yf.Ticker("XLF")  # Financial Select Sector SPDR
                    elif 'energy' in sector.lower():
                        sector_etf = yf.Ticker("XLE")  # Energy Select Sector SPDR
                    elif 'consumer' in sector.lower():
                        sector_etf = yf.Ticker("XLP")  # Consumer Staples Select Sector SPDR
                    else:
                        sector_etf = yf.Ticker("^GSPC")  # S&P 500 as default
                
                sector_performance['Sector ETF Performance'] = f"{sector_etf.info.get('regularMarketChangePercent', 0):.2f}%"
                sector_performance['Sector ETF Price'] = f"${sector_etf.info.get('regularMarketPrice', 0):,.2f}"
                sector_performance['Sector ETF Volume'] = f"{sector_etf.info.get('regularMarketVolume', 0):,}"
            except Exception as e:
                print(f"Error getting sector performance: {str(e)}")
                sector_performance['Sector ETF Performance'] = 'N/A'
            
            # Get peer comparison
            peers = info.get('recommendedSymbols', [])
            peer_comparison = {}
            
            # Add default peers based on sector if no recommended symbols
            if not peers:
                if ".NS" in symbol:
                    if 'financial' in sector.lower():
                        peers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
                    elif 'technology' in sector.lower():
                        peers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"]
                    elif 'healthcare' in sector.lower():
                        peers = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "BIOCON.NS"]
                else:
                    if 'technology' in sector.lower():
                        peers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                    elif 'healthcare' in sector.lower():
                        peers = ["JNJ", "PFE", "UNH", "MRK", "ABBV"]
                    elif 'financial' in sector.lower():
                        peers = ["JPM", "BAC", "WFC", "GS", "MS"]
                    elif 'energy' in sector.lower():
                        peers = ["XOM", "CVX", "COP", "SLB", "EOG"]
            
            for peer in peers[:5]:  # Compare with top 5 peers
                try:
                    peer_stock = yf.Ticker(peer)
                    peer_info = peer_stock.info
                    
                    # Determine currency based on exchange
                    currency = "₹" if ".NS" in peer else "$"
                    
                    # Get sector-specific metrics
                    metrics = {
                        'Price': f"{currency}{peer_info.get('currentPrice', 'N/A'):,.2f}",
                        'Market Cap': f"{currency}{peer_info.get('marketCap', 0)/1e9:.2f}B",
                        'P/E': f"{peer_info.get('forwardPE', 'N/A'):.2f}",
                        'Change': f"{peer_info.get('regularMarketChangePercent', 0):.2f}%"
                    }
                    
                    # Add sector-specific metrics
                    if 'technology' in sector.lower():
                        metrics['Revenue Growth'] = f"{peer_info.get('revenueGrowth', 'N/A')*100:.2f}%" if peer_info.get('revenueGrowth') else 'N/A'
                        metrics['R&D to Revenue'] = f"{peer_info.get('researchAndDevelopmentToRevenue', 'N/A')*100:.2f}%" if peer_info.get('researchAndDevelopmentToRevenue') else 'N/A'
                    elif 'healthcare' in sector.lower():
                        metrics['Gross Margin'] = f"{peer_info.get('grossMargins', 'N/A')*100:.2f}%" if peer_info.get('grossMargins') else 'N/A'
                        metrics['Operating Margin'] = f"{peer_info.get('operatingMargins', 'N/A')*100:.2f}%" if peer_info.get('operatingMargins') else 'N/A'
                    elif 'financial' in sector.lower():
                        metrics['ROE'] = f"{peer_info.get('returnOnEquity', 'N/A')*100:.2f}%" if peer_info.get('returnOnEquity') else 'N/A'
                        metrics['ROA'] = f"{peer_info.get('returnOnAssets', 'N/A')*100:.2f}%" if peer_info.get('returnOnAssets') else 'N/A'
                        metrics['Net Interest Margin'] = f"{peer_info.get('netInterestMargin', 'N/A')*100:.2f}%" if peer_info.get('netInterestMargin') else 'N/A'
                        metrics['Capital Adequacy Ratio'] = f"{peer_info.get('capitalAdequacyRatio', 'N/A')*100:.2f}%" if peer_info.get('capitalAdequacyRatio') else 'N/A'
                    elif 'energy' in sector.lower():
                        metrics['EBITDA Margin'] = f"{peer_info.get('ebitdaMargins', 'N/A')*100:.2f}%" if peer_info.get('ebitdaMargins') else 'N/A'
                        metrics['Operating Margin'] = f"{peer_info.get('operatingMargins', 'N/A')*100:.2f}%" if peer_info.get('operatingMargins') else 'N/A'
                    
                    peer_comparison[peer] = metrics
                except Exception as e:
                    print(f"Error analyzing peer {peer}: {str(e)}")
                    continue
            
            analysis = {
                'Sector': sector,
                'Industry': industry,
                'Sector Performance': sector_performance,
                'Peer Comparison': peer_comparison
            }
            
            return str(analysis)
        except Exception as e:
            return f"Error analyzing sector and peers: {str(e)}"

    def _arun(self, symbol: str) -> str:
        raise NotImplementedError("Async not implemented")

class DividendGrowthTool(BaseTool):
    name: str = "Dividend and Growth Analysis"
    description: str = "Analyzes dividend metrics and earnings growth rate"
    args_schema: Type[BaseModel] = DividendGrowthInput
    
    def _run(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get historical data for growth calculations
            hist_data = stock.history(period="5y")
            
            # Determine currency based on exchange
            currency = "₹" if ".NS" in symbol else "$"
            
            # Get sector for sector-specific analysis
            sector = info.get('sector', '').lower()
            
            # Dividend Analysis with multiple attempts for each metric
            dividend_metrics = {}
            
            # Dividend Yield
            dividend_yield = (
                info.get('dividendYield',
                info.get('yield',
                info.get('trailingAnnualDividendYield', None))))
            dividend_metrics['Dividend Yield'] = f"{dividend_yield*100:.2f}%" if dividend_yield else 'N/A'
            
            # Dividend Rate
            dividend_rate = (
                info.get('dividendRate',
                info.get('trailingAnnualDividendRate',
                info.get('lastDividendValue', None))))
            dividend_metrics['Dividend Rate'] = f"{currency}{dividend_rate:,.2f}" if dividend_rate else 'N/A'
            
            # Payout Ratio
            payout_ratio = (
                info.get('payoutRatio',
                info.get('dividendPayout',
                info.get('dividendPayoutRatio', None))))
            dividend_metrics['Payout Ratio'] = f"{payout_ratio*100:.2f}%" if payout_ratio else 'N/A'
            
            # 5-Year Average Dividend Yield
            five_year_avg_yield = (
                info.get('fiveYearAvgDividendYield',
                info.get('5yearAverageDividendYield',
                info.get('averageDividendYield', None))))
            dividend_metrics['5-Year Average Dividend Yield'] = f"{five_year_avg_yield:.2f}%" if five_year_avg_yield else 'N/A'
            
            # Earnings Growth Analysis
            earnings_growth = {}
            try:
                # Get quarterly financials
                financials = stock.quarterly_financials
                if not financials.empty:
                    # Calculate year-over-year growth rates
                    if 'Net Income' in financials.index:
                        earnings = financials.loc['Net Income']
                        if len(earnings) >= 4:
                            yoy_growth = ((earnings.iloc[0] / earnings.iloc[4]) - 1) * 100
                        else:
                            earnings_growth['Latest Earnings Growth'] = 'N/A (insufficient data)'
                    
                    # Revenue Growth
                    if 'Total Revenue' in financials.index:
                        revenue = financials.loc['Total Revenue']
                        if len(revenue) >= 4:
                            rev_growth = ((revenue.iloc[0] / revenue.iloc[4]) - 1) * 100
                        else:
                            earnings_growth['Latest Revenue Growth'] = 'N/A (insufficient data)'
                    
                    # Calculate 5-year CAGR if enough data
                    if len(hist_data) >= 1250:  # Approximately 5 years of trading days
                        # Price CAGR
                        start_price = hist_data['Close'].iloc[0]
                        end_price = hist_data['Close'].iloc[-1]
                        price_cagr = ((end_price / start_price) ** (1/5) - 1) * 100
                        earnings_growth['5-Year Price CAGR'] = f"{price_cagr:.2f}%"
                        
                        if len(financials.index) >= 20:  # 5 years of quarterly data
                            if 'Net Income' in financials.index:
                                start_earnings = financials.loc['Net Income'].iloc[-1]
                                end_earnings = financials.loc['Net Income'].iloc[0]
                                if start_earnings > 0 and end_earnings > 0:
                                    earnings_cagr = ((end_earnings / start_earnings) ** (1/5) - 1) * 100
                                    earnings_growth['5-Year Earnings CAGR'] = f"{earnings_cagr:.2f}%"
            except Exception as e:
                print(f"Error calculating growth metrics: {str(e)}")
                earnings_growth['Latest Earnings Growth'] = 'N/A'
                earnings_growth['Latest Revenue Growth'] = 'N/A'
                earnings_growth['5-Year Price CAGR'] = 'N/A'
                earnings_growth['5-Year Earnings CAGR'] = 'N/A'
            
            # Add additional growth metrics
            try:
                # Revenue Growth from info
                rev_growth = info.get('revenueGrowth', None)
                if rev_growth is not None:
                    earnings_growth['Revenue Growth (TTM)'] = f"{rev_growth*100:.2f}%"
                
                # Earnings Growth from info
                earn_growth = info.get('earningsGrowth', None)
                if earn_growth is not None:
                    earnings_growth['Earnings Growth (TTM)'] = f"{earn_growth*100:.2f}%"
                
                # EPS Growth
                eps_growth = info.get('earningsQuarterlyGrowth', None)
                if eps_growth is not None:
                    earnings_growth['EPS Growth (QoQ)'] = f"{eps_growth*100:.2f}%"
            except Exception as e:
                print(f"Error calculating additional growth metrics: {str(e)}")
            
            analysis = {
                'Dividend Metrics': dividend_metrics,
                'Growth Metrics': earnings_growth
            }
            
            return str(analysis)
        except Exception as e:
            print(f"Error in analyze_dividend_and_growth: {str(e)}")
            return "Error analyzing dividend and growth metrics"

    def _arun(self, symbol: str) -> str:
        raise NotImplementedError("Async not implemented")

class StockAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        self.historical_data = None
        self.returns = None
        
    def fetch_data(self, period: str = "1y") -> None:
        """Fetch historical data for the stock."""
        self.historical_data = self.stock.history(period=period)
        self.returns = self.historical_data['Close'].pct_change().dropna()
        
    def get_risk_metrics(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate risk metrics for the stock."""
        if self.returns is None:
            self.fetch_data()
        return calculate_risk_metrics(self.returns.values, confidence_level)
        
    def get_technical_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators for the stock."""
        if self.historical_data is None:
            self.fetch_data()
            
        data = self.historical_data.copy()
        
        # Calculate SMA
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Get latest values
        latest = data.iloc[-1]
        
        return {
            'sma_20': float(latest['SMA_20']),
            'sma_50': float(latest['SMA_50']),
            'rsi': float(latest['RSI']),
            'macd': float(latest['MACD']),
            'signal_line': float(latest['Signal_Line'])
        }
        
    def get_fundamental_metrics(self) -> Dict[str, float]:
        """Get fundamental metrics for the stock."""
        try:
            info = self.stock.info
            return {
                'pe_ratio': float(info.get('forwardPE', 0)),
                'market_cap': float(info.get('marketCap', 0)),
                'dividend_yield': float(info.get('dividendYield', 0)),
                'beta': float(info.get('beta', 0)),
                'fifty_two_week_high': float(info.get('fiftyTwoWeekHigh', 0)),
                'fifty_two_week_low': float(info.get('fiftyTwoWeekLow', 0))
            }
        except Exception as e:
            print(f"Error fetching fundamental metrics: {str(e)}")
            return {
                'pe_ratio': 0.0,
                'market_cap': 0.0,
                'dividend_yield': 0.0,
                'beta': 0.0,
                'fifty_two_week_high': 0.0,
                'fifty_two_week_low': 0.0
            }
            
    def get_stock_info(self) -> Dict[str, str]:
        """Get basic information about the stock."""
        try:
            info = self.stock.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'country': info.get('country', ''),
                'currency': info.get('currency', '')
            }
        except Exception as e:
            print(f"Error fetching stock info: {str(e)}")
            return {
                'name': '',
                'sector': '',
                'industry': '',
                'country': '',
                'currency': ''
            }

# Example usage
if __name__ == "__main__":
    analyzer = StockAnalyzer("AAPL")
    result = analyzer.get_risk_metrics()
    print(result) 