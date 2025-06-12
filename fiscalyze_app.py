import streamlit as st

st.set_page_config(
    page_title="Fiscalyze",
    page_icon="https://raw.githubusercontent.com/willtoyne/fiscalyze/main/images/logo.png",
    layout="wide",
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, Float, String, Date, MetaData, Table, select, delete, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib
from PIL import Image


#################################################
# DATABASE SETUP
#################################################


import matplotlib
matplotlib.use('Agg')

# Use SQLite by default if no DATABASE_URL is specified
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///fiscalyze.db')

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Define database model
class FinancialData(Base):
    __tablename__ = 'financial_data'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    income = Column(Float, nullable=False)
    rent = Column(Float, nullable=True)
    utilities = Column(Float, nullable=True)
    salaries = Column(Float, nullable=True)
    marketing = Column(Float, nullable=True)
    miscellaneous = Column(Float, nullable=True)
    total_expenses = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<FinancialData(date='{self.date}', income='{self.income}', profit='{self.profit}')>"

# Create all tables
def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(engine)

# Function to save DataFrame to database
def save_dataframe_to_db(df):
    """
    Save a pandas DataFrame to the database.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing financial data
    """
    # Convert DataFrame to proper format for database
    for index, row in df.iterrows():
        # Check if this date already exists in the database
        existing = session.query(FinancialData).filter(FinancialData.date == row['Date']).first()
        
        if existing:
            # Update existing record
            existing.income = row['Income']
            
            # Update expense categories if they exist
            if 'Rent' in row and row['Rent'] is not None:
                existing.rent = row['Rent']
            if 'Utilities' in row and row['Utilities'] is not None:
                existing.utilities = row['Utilities']
            if 'Salaries' in row and row['Salaries'] is not None:
                existing.salaries = row['Salaries']
            if 'Marketing' in row and row['Marketing'] is not None:
                existing.marketing = row['Marketing']
            if 'Miscellaneous' in row and row['Miscellaneous'] is not None:
                existing.miscellaneous = row['Miscellaneous']
            
            # Calculate total expenses and profit
            existing.total_expenses = row['Total_Expenses'] if 'Total_Expenses' in row else 0
            existing.profit = row['Profit'] if 'Profit' in row else (existing.income - existing.total_expenses)
        else:
            # Create new record
            financial_data = FinancialData(
                date=row['Date'],
                income=row['Income'],
                rent=row['Rent'] if 'Rent' in row and not pd.isna(row['Rent']) else 0,
                utilities=row['Utilities'] if 'Utilities' in row and not pd.isna(row['Utilities']) else 0,
                salaries=row['Salaries'] if 'Salaries' in row and not pd.isna(row['Salaries']) else 0,
                marketing=row['Marketing'] if 'Marketing' in row and not pd.isna(row['Marketing']) else 0,
                miscellaneous=row['Miscellaneous'] if 'Miscellaneous' in row and not pd.isna(row['Miscellaneous']) else 0,
                total_expenses=row['Total_Expenses'] if 'Total_Expenses' in row else 0,
                profit=row['Profit'] if 'Profit' in row else 0
            )
            session.add(financial_data)
    
    # Commit the session
    session.commit()

# Function to load data from database to DataFrame
def load_dataframe_from_db():
    """
    Load financial data from the database as a pandas DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame containing all financial data from the database
    """
    # Query all data from the financial_data table
    data = session.query(FinancialData).order_by(FinancialData.date).all()
    
    if not data:
        return pd.DataFrame()  # Return empty DataFrame if no data
    
    # Convert to DataFrame
    records = []
    for item in data:
        record = {
            'Date': item.date,
            'Income': item.income,
            'Rent': item.rent,
            'Utilities': item.utilities,
            'Salaries': item.salaries,
            'Marketing': item.marketing,
            'Miscellaneous': item.miscellaneous,
            'Total_Expenses': item.total_expenses,
            'Profit': item.profit
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add Month column for display
    if not df.empty:
        df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%b %Y')
    
    return df

# Function to delete all data
def clear_database():
    """Delete all records from the financial_data table."""
    session.query(FinancialData).delete()
    session.commit()

#################################################
# DATA PROCESSING
#################################################

def load_sample_data():
    """
    Generate sample financial data for demonstration purposes.
    Returns a pandas DataFrame with dates, income, and expense categories.
    """
    # Create date range for the past 12 months
    end_date = datetime.now().replace(day=1) - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create sample data with some seasonal patterns and trends
    n_months = len(dates)
    
    # Income with slight upward trend and Q4 boost
    base_income = 50000 + np.arange(n_months) * 500
    seasonal_factor = np.ones(n_months)
    seasonal_factor[9:12] = 1.2  # Q4 boost
    income = base_income * seasonal_factor * (1 + np.random.normal(0, 0.05, n_months))
    
    # Expenses for different categories
    rent = 10000 * np.ones(n_months) * (1 + np.random.normal(0, 0.01, n_months))
    utilities = 2500 * (1 + 0.1 * np.sin(np.arange(n_months) * (2*np.pi/12))) * (1 + np.random.normal(0, 0.03, n_months))
    salaries = 20000 * np.ones(n_months) * (1 + np.random.normal(0, 0.02, n_months)) * (1 + np.arange(n_months) * 0.005)
    marketing = 5000 * np.ones(n_months) * (1 + np.random.normal(0, 0.15, n_months))
    misc = 3000 * np.ones(n_months) * (1 + np.random.normal(0, 0.1, n_months))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Income': income,
        'Rent': rent,
        'Utilities': utilities,
        'Salaries': salaries,
        'Marketing': marketing,
        'Miscellaneous': misc
    })
    
    # Calculate total expenses and profit
    data['Total_Expenses'] = data[['Rent', 'Utilities', 'Salaries', 'Marketing', 'Miscellaneous']].sum(axis=1)
    data['Profit'] = data['Income'] - data['Total_Expenses']
    
    # Format the date column
    data['Month'] = data['Date'].dt.strftime('%b %Y')
    
    return data

def process_user_data(uploaded_file):
    """
    Process user-uploaded financial data.
    Expected format: CSV with columns for Date and various income/expense categories.
    """
    try:
        data = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_cols = ['Date', 'Income']
        if not all(col in data.columns for col in required_cols):
            st.error("Uploaded file is missing required columns. Please ensure it contains 'Date' and 'Income' columns.")
            return None
        
        # Convert date column to datetime
        try:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Month'] = data['Date'].dt.strftime('%b %Y')
        except:
            st.error("Could not parse the Date column. Please ensure dates are in a standard format (e.g., YYYY-MM-DD).")
            return None
        
        # Calculate total expenses and profit if not already present
        expense_cols = [col for col in data.columns if col not in ['Date', 'Month', 'Income', 'Total_Expenses', 'Profit']]
        
        if 'Total_Expenses' not in data.columns and len(expense_cols) > 0:
            data['Total_Expenses'] = data[expense_cols].sum(axis=1)
        
        if 'Profit' not in data.columns:
            data['Profit'] = data['Income'] - data['Total_Expenses']
        
        return data
    
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

def add_monthly_data(existing_data, new_data_dict):
    """
    Add a new month of financial data to the existing dataset.
    
    Parameters:
    existing_data (pd.DataFrame): Existing financial data
    new_data_dict (dict): Dictionary containing new month's data
    
    Returns:
    pd.DataFrame: Updated dataset with new month's data
    """
    # Create a new DataFrame for the new data
    new_row = pd.DataFrame([new_data_dict])
    
    # Ensure the date is in datetime format
    new_row['Date'] = pd.to_datetime(new_row['Date'])
    new_row['Month'] = new_row['Date'].dt.strftime('%b %Y')
    
    # Calculate total expenses and profit
    expense_cols = [col for col in new_row.columns if col not in ['Date', 'Month', 'Income', 'Total_Expenses', 'Profit']]
    new_row['Total_Expenses'] = new_row[expense_cols].sum(axis=1)
    new_row['Profit'] = new_row['Income'] - new_row['Total_Expenses']
    
    # Combine with existing data
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)
    
    # Sort by date
    updated_data = updated_data.sort_values('Date').reset_index(drop=True)
    
    return updated_data

#################################################
# FORECASTING MODELS
#################################################

def generate_forecasts(data, target_column='Income', forecast_periods=6, model_type='linear'):
    """
    Generate financial forecasts based on historical data.
    
    Parameters:
    data (pd.DataFrame): Historical financial data
    target_column (str): Column to forecast
    forecast_periods (int): Number of periods to forecast
    model_type (str): Forecasting model type ('linear', 'moving_avg', 'seasonal')
    
    Returns:
    pd.DataFrame: DataFrame containing historical and forecasted data
    """
    # Ensure data is sorted by date
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Extract date features
    dates = pd.to_datetime(data['Date'])
    last_date = dates.iloc[-1]
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                 periods=forecast_periods, 
                                 freq='MS')
    
    # Create feature based on month index
    data['Month_Index'] = np.arange(len(data))
    
    if model_type == 'linear':
        # Simple linear regression model
        forecasts = linear_regression_forecast(data, target_column, future_dates, forecast_periods)
    
    elif model_type == 'moving_avg':
        # Moving average forecast
        window_size = min(6, len(data))
        forecasts = moving_average_forecast(data, target_column, future_dates, window_size)
    
    elif model_type == 'seasonal':
        # Seasonal forecast (simple decomposition)
        forecasts = seasonal_forecast(data, target_column, future_dates, forecast_periods)
    
    else:
        st.error(f"Unknown model type: {model_type}")
        return None
    
    return forecasts

def linear_regression_forecast(data, target_column, future_dates, forecast_periods):
    """
    Generate forecasts using linear regression.
    """
    # Prepare training data
    X = data[['Month_Index']]
    y = data[target_column]
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future months indices
    future_indices = np.arange(len(data), len(data) + forecast_periods)
    future_X = pd.DataFrame({'Month_Index': future_indices})
    
    # Predict
    future_y = model.predict(future_X)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Month': future_dates.strftime('%b %Y'),
        target_column: future_y,
        'Type': 'Forecast'
    })
    
    # Mark historical data
    historical_df = data[['Date', 'Month', target_column]].copy()
    historical_df['Type'] = 'Historical'
    
    # Combine historical and forecast
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    return result_df

def moving_average_forecast(data, target_column, future_dates, window_size):
    """
    Generate forecasts using moving average.
    """
    # Calculate moving average
    last_values = data[target_column].tail(window_size).values
    forecast_values = np.ones(len(future_dates)) * np.mean(last_values)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Month': future_dates.strftime('%b %Y'),
        target_column: forecast_values,
        'Type': 'Forecast'
    })
    
    # Mark historical data
    historical_df = data[['Date', 'Month', target_column]].copy()
    historical_df['Type'] = 'Historical'
    
    # Combine historical and forecast
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    return result_df

def seasonal_forecast(data, target_column, future_dates, forecast_periods):
    """
    Generate forecasts using a simple seasonal decomposition.
    """
    # Need at least a year of data for seasonal forecasting
    if len(data) < 12:
        # Fall back to linear regression if not enough data
        return linear_regression_forecast(data, target_column, future_dates, forecast_periods)
    
    # Extract month from dates for seasonality
    data['Month_Num'] = data['Date'].dt.month
    
    # Calculate seasonal indices (average value for each month)
    seasonal_indices = data.groupby('Month_Num')[target_column].mean() / data[target_column].mean()
    
    # Calculate trend using linear regression
    X = data[['Month_Index']]
    y = data[target_column]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future indices
    future_indices = np.arange(len(data), len(data) + forecast_periods)
    future_X = pd.DataFrame({'Month_Index': future_indices})
    
    # Predict trend component
    trend_forecast = model.predict(future_X)
    
    # Add seasonality
    future_months = [(last_date.month + i) % 12 for i, last_date in enumerate(future_dates)]
    future_months = [12 if m == 0 else m for m in future_months]  # Convert 0 to 12 for December
    
    seasonal_factors = [seasonal_indices[month] for month in future_months]
    forecast_values = trend_forecast * seasonal_factors
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Month': future_dates.strftime('%b %Y'),
        target_column: forecast_values,
        'Type': 'Forecast'
    })
    
    # Mark historical data
    historical_df = data[['Date', 'Month', target_column]].copy()
    historical_df['Type'] = 'Historical'
    
    # Combine historical and forecast
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    return result_df

def forecast_all_metrics(data, forecast_periods=6, model_type='linear'):
    """
    Generate forecasts for all key financial metrics.
    
    Parameters:
    data (pd.DataFrame): Historical financial data
    forecast_periods (int): Number of periods to forecast
    model_type (str): Forecasting model type
    
    Returns:
    dict: Dictionary containing forecasts for each metric
    """
    metrics = {
        'Income': generate_forecasts(data, 'Income', forecast_periods, model_type),
        'Total_Expenses': generate_forecasts(data, 'Total_Expenses', forecast_periods, model_type),
        'Profit': generate_forecasts(data, 'Profit', forecast_periods, model_type)
    }
    
    # Generate forecasts for individual expense categories
    expense_columns = [col for col in data.columns if col not in 
                     ['Date', 'Month', 'Month_Index', 'Income', 'Total_Expenses', 'Profit', 'Type']]
    
    for column in expense_columns:
        metrics[column] = generate_forecasts(data, column, forecast_periods, model_type)
    
    return metrics

#################################################
# PDF REPORT GENERATION
#################################################

def create_financial_report(historical_data, forecasted_data, company_name=None, report_period=None):
    """
    Create a PDF financial report based on historical and forecasted data.
    
    Parameters:
    historical_data (pd.DataFrame): Historical financial data
    forecasted_data (dict): Dictionary containing forecasts for different metrics
    company_name (str): Name of the company for the report
    report_period (str): Period covered by the report
    
    Returns:
    bytes: PDF report as bytes object
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Set up styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add custom styles
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading3'],
        textColor=colors.darkblue,
        spaceAfter=0.2*inch
    )
    
    # Company name and report title
    if company_name:
        title = company_name
    else:
        title = "Financial Forecast Report"
    
    elements.append(Paragraph(title, title_style))
    
    # Add report date
    if report_period:
        elements.append(Paragraph(f"Forecast Period: {report_period}", subtitle_style))
    else:
        elements.append(Paragraph(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}", subtitle_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Create visualizations
    create_and_add_charts(elements, historical_data, forecasted_data)
    
    # Financial summary
    elements.append(Paragraph("Financial Summary", header_style))
    
    # Create summary table with forecasted values
    if 'Income' in forecasted_data and 'Total_Expenses' in forecasted_data and 'Profit' in forecasted_data:
        create_summary_table(elements, forecasted_data)
    
    # Add forecast details
    elements.append(Paragraph("Forecast Details", header_style))
    
    # Income forecast table
    if 'Income' in forecasted_data:
        create_forecast_detail_table(elements, forecasted_data['Income'], "Income Forecast")
    
    # Expenses forecast table
    if 'Total_Expenses' in forecasted_data:
        create_forecast_detail_table(elements, forecasted_data['Total_Expenses'], "Expenses Forecast")
    
    # Build the PDF document
    doc.build(elements)
    buffer.seek(0)
    
    return buffer

def create_and_add_charts(elements, historical_data, forecasted_data):
    """
    Create charts for the report and add them to the elements list.
    """
    styles = getSampleStyleSheet()
    header_style = styles['Heading3']
    
    # Income vs Expenses Chart
    if 'Income' in forecasted_data and 'Total_Expenses' in forecasted_data:
        elements.append(Paragraph("Income vs Expenses Forecast", header_style))
        
        plt.figure(figsize=(7, 3.5))
        
        # Plot historical data
        hist_dates = pd.to_datetime(historical_data['Date'])
        plt.plot(hist_dates, historical_data['Income'], 'b-', label='Historical Income')
        plt.plot(hist_dates, historical_data['Total_Expenses'], 'r-', label='Historical Expenses')
        
        # Plot forecasted data
        income_forecast = forecasted_data['Income']
        expense_forecast = forecasted_data['Total_Expenses']
        
        # Get forecast data points
        forecast_income = income_forecast[income_forecast['Type'] == 'Forecast']
        forecast_expenses = expense_forecast[expense_forecast['Type'] == 'Forecast']
        
        plt.plot(pd.to_datetime(forecast_income['Date']), forecast_income['Income'], 'b--', label='Forecasted Income')
        plt.plot(pd.to_datetime(forecast_expenses['Date']), forecast_expenses['Total_Expenses'], 'r--', label='Forecasted Expenses')
        
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title('Income vs Expenses Over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure to a bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        # Add the image to the PDF
        img = Image(img_buffer, width=6.5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))
    
    # Profit Chart
    if 'Profit' in forecasted_data:
        elements.append(Paragraph("Profit Forecast", header_style))
        
        plt.figure(figsize=(7, 3.5))
        
        # Plot historical data
        plt.plot(hist_dates, historical_data['Profit'], 'g-', label='Historical Profit')
        
        # Plot forecasted data
        profit_forecast = forecasted_data['Profit']
        forecast_profit = profit_forecast[profit_forecast['Type'] == 'Forecast']
        
        plt.plot(pd.to_datetime(forecast_profit['Date']), forecast_profit['Profit'], 'g--', label='Forecasted Profit')
        
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title('Profit Over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure to a bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        # Add the image to the PDF
        img = Image(img_buffer, width=6.5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.2*inch))

def create_summary_table(elements, forecasted_data):
    """
    Create a summary table of forecasted financial metrics.
    """
    # Extract forecast-only data
    income_forecast = forecasted_data['Income'][forecasted_data['Income']['Type'] == 'Forecast']
    expense_forecast = forecasted_data['Total_Expenses'][forecasted_data['Total_Expenses']['Type'] == 'Forecast']
    profit_forecast = forecasted_data['Profit'][forecasted_data['Profit']['Type'] == 'Forecast']
    
    # Create table data
    table_data = [
        ['Month', 'Forecasted Income', 'Forecasted Expenses', 'Forecasted Profit']
    ]
    
    # Add data for each forecasted month
    for i in range(len(income_forecast)):
        month = income_forecast.iloc[i]['Month']
        income = f"${income_forecast.iloc[i]['Income']:,.2f}"
        expenses = f"${expense_forecast.iloc[i]['Total_Expenses']:,.2f}"
        profit = f"${profit_forecast.iloc[i]['Profit']:,.2f}"
        
        table_data.append([month, income, expenses, profit])
    
    # Create the table
    table = Table(table_data, colWidths=[1.3*inch, 1.7*inch, 1.7*inch, 1.7*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))

def create_forecast_detail_table(elements, forecast_data, title):
    """
    Create a detailed table for a specific forecasted metric.
    """
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    
    elements.append(Paragraph(title, normal_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Extract forecast-only data
    forecast_only = forecast_data[forecast_data['Type'] == 'Forecast']
    
    # Determine the metric column name (might be 'Income', 'Total_Expenses', etc.)
    metric_col = [col for col in forecast_only.columns if col not in ['Date', 'Month', 'Type']][0]
    
    # Create table data
    table_data = [
        ['Month', f'Forecasted {metric_col.replace("_", " ")}']
    ]
    
    # Add data for each forecasted month
    for i in range(len(forecast_only)):
        month = forecast_only.iloc[i]['Month']
        value = f"${forecast_only.iloc[i][metric_col]:,.2f}"
        
        table_data.append([month, value])
    
    # Create the table
    table = Table(table_data, colWidths=[1.5*inch, 2*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))

#################################################
# UI COMPONENTS
#################################################

def plot_income_vs_expenses(data, income_forecast=None, expense_forecast=None):
    """Plot income vs expenses with optional forecast."""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data['Month'],
        y=data['Income'],
        mode='lines+markers',
        name='Historical Income',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Month'],
        y=data['Total_Expenses'],
        mode='lines+markers',
        name='Historical Expenses',
        line=dict(color='red', width=3)
    ))
    
    # Plot forecast if available
    if income_forecast is not None:
        forecast_data = income_forecast[income_forecast['Type'] == 'Forecast']
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Month'],
            y=forecast_data['Income'],
            mode='lines+markers',
            name='Forecasted Income',
            line=dict(color='blue', width=3, dash='dash')
        ))
    
    if expense_forecast is not None:
        forecast_data = expense_forecast[expense_forecast['Type'] == 'Forecast']
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Month'],
            y=forecast_data['Total_Expenses'],
            mode='lines+markers',
            name='Forecasted Expenses',
            line=dict(color='red', width=3, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        margin=dict(l=20, r=20, t=30, b=20),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_profit_trend(data, profit_forecast=None):
    """Plot profit trend with optional forecast."""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data['Month'],
        y=data['Profit'],
        mode='lines+markers',
        name='Historical Profit',
        line=dict(color='green', width=3)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    
    # Plot forecast if available
    if profit_forecast is not None:
        forecast_data = profit_forecast[profit_forecast['Type'] == 'Forecast']
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Month'],
            y=forecast_data['Profit'],
            mode='lines+markers',
            name='Forecasted Profit',
            line=dict(color='green', width=3, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Profit ($)',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        margin=dict(l=20, r=20, t=30, b=20),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_expense_breakdown(data):
    """Plot expense breakdown by category."""
    # Get expense categories (all columns except Date, Month, Income, Total_Expenses, Profit)
    expense_cols = [col for col in data.columns if col not in ['Date', 'Month', 'Income', 'Total_Expenses', 'Profit']]
    
    if not expense_cols:
        st.info("No expense categories found in the data.")
        return
    
    # Get the last few months of data
    recent_data = data.tail(6)
    
    # Create expense breakdown for the most recent month
    last_month = data.iloc[-1]
    expense_values = [last_month[col] for col in expense_cols]
    
    # Pie chart for last month
    fig_pie = px.pie(
        values=expense_values,
        names=expense_cols,
        title=f"Expense Breakdown for {last_month['Month']}",
        hole=0.4
    )
    
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=60, b=60),
        height=350
    )
    
    # Stacked bar chart for recent months
    expense_data = []
    for i, row in recent_data.iterrows():
        for category in expense_cols:
            expense_data.append({
                'Month': row['Month'],
                'Category': category,
                'Amount': row[category]
            })
    
    df_expenses = pd.DataFrame(expense_data)
    
    fig_bar = px.bar(
        df_expenses,
        x='Month',
        y='Amount',
        color='Category',
        title="Expense Categories Over Time"
    )
    
    fig_bar.update_layout(
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        legend_title='Expense Category',
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20),
        height=350
    )
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)

def display_forecast_preview(forecasted_data, metric):
    """Display a preview of the forecast for a specific metric."""
    if metric not in forecasted_data:
        st.error(f"No forecast data found for {metric}")
        return
    
    forecast_df = forecasted_data[metric]
    
    # Split historical and forecast data
    historical = forecast_df[forecast_df['Type'] == 'Historical']
    forecast = forecast_df[forecast_df['Type'] == 'Forecast']
    
    st.subheader(f"{metric.replace('_', ' ')} Forecast")
    
    # Show the forecast numbers in a table
    st.markdown("#### Forecasted Values")
    
    # Create a more readable table
    display_df = forecast[['Month', metric]].copy()
    display_df.columns = ['Month', f'{metric.replace("_", " ")}']
    
    # Format the values
    display_df[f'{metric.replace("_", " ")}'] = display_df[f'{metric.replace("_", " ")}'].map('${:,.2f}'.format)
    
    st.table(display_df)
    
    # Show a chart of historical + forecast
    st.markdown("#### Visualization")
    
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical['Month'],
        y=historical[metric],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Plot forecast data
    fig.add_trace(go.Scatter(
        x=forecast['Month'],
        y=forecast[metric],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=f'{metric.replace("_", " ")} ($)',
        hovermode='x unified',
        margin=dict(l=20, r=20, t=30, b=20),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth calculation
    if len(historical) > 0 and len(forecast) > 0:
        last_historical = historical[metric].iloc[-1]
        last_forecast = forecast[metric].iloc[-1]
        
        growth_pct = ((last_forecast / last_historical) - 1) * 100
        
        st.metric(
            label=f"Projected Growth",
            value=f"{growth_pct:.2f}%",
            delta=f"{growth_pct:.2f}%",
            delta_color="normal" if growth_pct >= 0 else "inverse" if metric != 'Total_Expenses' else "normal" if growth_pct <= 0 else "inverse"
        )

def upload_data():
    """Allow users to upload financial data from CSV or Excel."""
    st.header("Upload Financial Data")
    
    st.markdown("""
    Upload your financial data from a CSV or Excel file.
    
    The file should contain the following columns:
    - `Date`: Date of the financial record (in format YYYY-MM-DD)
    - `Income`: Total income for the period
    - Any expense categories (e.g., `Rent`, `Utilities`, `Salaries`, etc.)
    
    Example:
    ```
    Date,Income,Rent,Utilities,Salaries,Marketing
    2023-01-01,50000,10000,2500,20000,5000
    2023-02-01,52000,10000,2700,20000,5500
    ...
    ```
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file based on file type
            if uploaded_file.name.endswith('.csv'):
                data = process_user_data(uploaded_file)
            else:  # Excel file
                data = pd.read_excel(uploaded_file)
                data = process_user_data(io.StringIO(data.to_csv(index=False)))
            
            if data is not None:
                st.success("Data uploaded successfully!")
                
                # Show preview of the data
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Option to replace or append data
                action = st.radio(
                    "What would you like to do with this data?",
                    options=["Replace existing data", "Append to existing data"]
                )
                
                if st.button("Confirm"):
                    if action == "Replace existing data":
                        st.session_state.financial_data = data
                        # Save to database
                        clear_database()
                        save_dataframe_to_db(data)
                        st.session_state.pop('forecasted_data', None)  # Clear any existing forecasts
                        st.success("Existing data replaced successfully!")
                    else:
                        # Append data - need to handle potential duplicates
                        combined_data = pd.concat([st.session_state.financial_data, data])
                        combined_data = combined_data.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
                        st.session_state.financial_data = combined_data
                        # Save to database
                        save_dataframe_to_db(combined_data)
                        st.session_state.pop('forecasted_data', None)  # Clear any existing forecasts
                        st.success("Data appended successfully!")
                    
                    # Redirect to dashboard
                    if st.button("Go to Dashboard"):
                        st.switch_page("01_Dashboard")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def manual_data_entry():
    """Allow users to manually enter financial data for a month."""
    st.header("Manual Data Entry")
    
    # Get the existing data to determine the next month for entry
    existing_data = st.session_state.financial_data
    
    # Determine the next month after the last entry
    if not existing_data.empty:
        last_date = pd.to_datetime(existing_data['Date'].iloc[-1])
        next_month = last_date + pd.DateOffset(months=1)
    else:
        next_month = datetime.now().replace(day=1)
    
    # Create a form for data entry
    with st.form("financial_data_form"):
        st.subheader("Enter Financial Data for a Month")
        
        # Date selection
        date = st.date_input(
            "Month",
            value=next_month,
            help="Select the month for which you're entering data"
        )
        
        # Ensure date is the first of the month for consistency
        date = date.replace(day=1)
        
        # Income input
        income = st.number_input(
            "Income ($)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Total income for the month"
        )
        
        # Get expense categories from existing data
        expense_cols = [col for col in existing_data.columns if col not in ['Date', 'Month', 'Income', 'Total_Expenses', 'Profit']]
        
        # If no expense categories exist, create default ones
        if not expense_cols:
            expense_cols = ['Rent', 'Utilities', 'Salaries', 'Marketing', 'Miscellaneous']
        
        # Create expense inputs
        st.subheader("Expenses")
        expenses = {}
        
        for expense in expense_cols:
            # Get the last value for this expense category if available
            default_value = 0.0
            if not existing_data.empty and expense in existing_data.columns:
                default_value = float(existing_data[expense].iloc[-1])
            
            expenses[expense] = st.number_input(
                f"{expense} ($)",
                min_value=0.0,
                value=default_value,
                step=100.0
            )
        
        # Add new expense category option
        new_category = st.text_input("Add New Expense Category (optional)")
        
        if new_category and new_category not in expense_cols:
            new_amount = st.number_input(
                f"{new_category} ($)",
                min_value=0.0,
                value=0.0,
                step=100.0
            )
            if new_amount > 0:
                expenses[new_category] = new_amount
        
        submitted = st.form_submit_button("Add Financial Data")
        
        if submitted:
            # Check if the date already exists in the data
            if not existing_data.empty and pd.to_datetime(date) in pd.to_datetime(existing_data['Date']).values:
                st.error(f"Data for {date.strftime('%B %Y')} already exists. Please edit the existing data or choose a different month.")
                return
            
            # Create a new data entry
            new_data = {
                'Date': date,
                'Income': income
            }
            
            # Add expenses
            for category, amount in expenses.items():
                new_data[category] = amount
            
            # Add the new data to the existing dataset
            updated_data = add_monthly_data(existing_data, new_data)
            
            # Update session state
            st.session_state.financial_data = updated_data
            
            # Save to database
            save_dataframe_to_db(updated_data)
            
            # Clear any existing forecasts since we've added new data
            if 'forecasted_data' in st.session_state:
                st.session_state.pop('forecasted_data')
            
            st.success(f"Financial data for {date.strftime('%B %Y')} added successfully!")
            
            # Redirect to dashboard
            if st.button("Go to Dashboard"):
                st.switch_page("01_Dashboard")

def view_current_data():
    """View and potentially edit the current financial data."""
    st.header("Current Financial Data")
    
    if st.session_state.financial_data.empty:
        st.info("No financial data available. Please add data using the Upload or Manual Entry tabs.")
        return
    
    # Get and display current data
    data = st.session_state.financial_data.copy()
    
    # Format date columns for display
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    
    # Display the data
    st.dataframe(data, use_container_width=True)
    
    # Option to download the data
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="fiscalyze_financial_data.csv",
        mime="text/csv"
    )
    
    # Option to clear all data
    if st.button("Clear All Data", help="Warning: This will remove all your financial data"):
        clear_data = st.checkbox("I understand this will delete all my data", value=False)
        
        if clear_data and st.button("Confirm Deletion"):
            # Load sample data
            sample_data = load_sample_data()
            st.session_state.financial_data = sample_data
            
            # Clear database and save sample data
            clear_database()
            save_dataframe_to_db(sample_data)
            
            # Clear any existing forecasts
            if 'forecasted_data' in st.session_state:
                st.session_state.pop('forecasted_data')
            
            st.success("Data cleared and replaced with sample data!")
            st.rerun()

#################################################
# MAIN APP
#################################################

# Function to initialize the database and load data on startup
def initialize_app():
    # Initialize database
    init_db()
    
    # Check if we have data in the database
    db_data = load_dataframe_from_db()
    
    # If no data in database, load sample data and save it
    if db_data.empty:
        sample_data = load_sample_data()
        save_dataframe_to_db(sample_data)
        return sample_data
    else:
        return db_data

# Set page configuration
st.set_page_config(
    page_title="Fiscalyze - Financial Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define app pages
def landing_page():
    # Initialize session state for data storage if not exists
    if 'financial_data' not in st.session_state:
        # Load data from database or initialize with sample data
        st.session_state.financial_data = initialize_app()
    
    if 'forecasted_data' not in st.session_state:
        st.session_state.forecasted_data = None
    
    # Landing page content
    st.title("Welcome to Fiscalyze")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Financial Forecasting for Small Businesses
        
        Fiscalyze helps small business owners and entrepreneurs make data-driven decisions 
        by providing simple yet powerful financial forecasting tools.
        
        ### Key Features:
        - 📊 Visualize your historical financial data
        - 📈 Generate accurate financial forecasts
        - 💰 Track income and expenses with ease
        - 📄 Create professional financial reports
        - 📱 Access your data anywhere, anytime
        
        Get started by navigating to the Dashboard or entering your financial data!
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### How It Works
        
        1. **Input your financial data** - Enter your monthly financial information or use our sample data
        2. **View your dashboard** - See your financial history and forecasted performance visualized
        3. **Generate reports** - Create PDF reports to share with stakeholders or for your records
        
        Fiscalyze uses simple but effective forecasting models like linear regression and moving averages 
        to help you understand where your business finances are headed.
        """)
    
    with col2:
        # Use an emoji as logo
        st.markdown("# 📊")
        
        st.markdown("### Get Started")
        
        if st.button("Go to Dashboard →", help="View your financial dashboard with visualizations"):
            st.session_state.page = "dashboard"
            st.rerun()
        
        if st.button("Enter Financial Data →", help="Input or update your financial information"):
            st.session_state.page = "data_input"
            st.rerun()
        
        if st.button("Generate Reports →", help="Create downloadable PDF reports"):
            st.session_state.page = "reports"
            st.rerun()

def dashboard_page():
    st.title("Financial Dashboard")
    
    # Get data from session state
    data = st.session_state.financial_data
    
    # Sidebar for forecast settings
    with st.sidebar:
        st.header("Forecast Settings")
        
        forecast_periods = st.slider(
            "Forecast Period (Months)",
            min_value=1,
            max_value=12,
            value=6,
            help="Number of months to forecast into the future"
        )
        
        forecast_model = st.selectbox(
            "Forecasting Model",
            options=["linear", "moving_avg", "seasonal"],
            format_func=lambda x: {
                "linear": "Linear Regression",
                "moving_avg": "Moving Average",
                "seasonal": "Seasonal (if enough data)"
            }.get(x),
            help="Choose the statistical model for forecasting"
        )
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecasts..."):
                # Generate forecasts for all metrics
                forecasts = forecast_all_metrics(data, forecast_periods, forecast_model)
                st.session_state.forecasted_data = forecasts
                st.success("Forecasts generated successfully!")
        
        # Navigation buttons
        st.markdown("---")
        st.markdown("### Navigation")
        
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            st.rerun()
        
        if st.button("Data Input"):
            st.session_state.page = "data_input"
            st.rerun()
        
        if st.button("Reports"):
            st.session_state.page = "reports"
            st.rerun()
    
    # Create dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tabs = st.tabs(["Income vs Expenses", "Profit Trend", "Expense Breakdown"])
        
        with tabs[0]:
            st.subheader("Income vs Expenses")
            
            if 'forecasted_data' in st.session_state and st.session_state.forecasted_data is not None:
                plot_income_vs_expenses(
                    data, 
                    st.session_state.forecasted_data.get('Income'),
                    st.session_state.forecasted_data.get('Total_Expenses')
                )
            else:
                plot_income_vs_expenses(data)
            
        with tabs[1]:
            st.subheader("Profit Trend")
            
            if 'forecasted_data' in st.session_state and st.session_state.forecasted_data is not None:
                plot_profit_trend(data, st.session_state.forecasted_data.get('Profit'))
            else:
                plot_profit_trend(data)
            
        with tabs[2]:
            st.subheader("Expense Breakdown")
            plot_expense_breakdown(data)
    
    with col2:
        st.subheader("Financial Summary")
        
        # Most recent month summary
        last_month = data.iloc[-1]
        current_month = last_month['Month']
        
        st.metric(
            label=f"Income ({current_month})", 
            value=f"${last_month['Income']:,.2f}",
            delta=f"{((last_month['Income'] / data.iloc[-2]['Income']) - 1) * 100:.1f}%" if len(data) > 1 else None
        )
        
        st.metric(
            label=f"Expenses ({current_month})", 
            value=f"${last_month['Total_Expenses']:,.2f}",
            delta=f"{((last_month['Total_Expenses'] / data.iloc[-2]['Total_Expenses']) - 1) * 100:.1f}%" if len(data) > 1 else None,
            delta_color="inverse"
        )
        
        st.metric(
            label=f"Profit ({current_month})", 
            value=f"${last_month['Profit']:,.2f}",
            delta=f"{((last_month['Profit'] / data.iloc[-2]['Profit']) - 1) * 100:.1f}%" if len(data) > 1 else None
        )
        
        st.divider()
        
        # Forecasted summary if available
        if 'forecasted_data' in st.session_state and st.session_state.forecasted_data is not None:
            st.subheader("Forecast Summary")
            
            income_forecast = st.session_state.forecasted_data['Income']
            expense_forecast = st.session_state.forecasted_data['Total_Expenses']
            profit_forecast = st.session_state.forecasted_data['Profit']
            
            # Filter to forecasted data only
            future_income = income_forecast[income_forecast['Type'] == 'Forecast']
            future_expenses = expense_forecast[expense_forecast['Type'] == 'Forecast']
            future_profit = profit_forecast[profit_forecast['Type'] == 'Forecast']
            
            # Calculate average forecasted values
            avg_income = future_income['Income'].mean()
            avg_expenses = future_expenses['Total_Expenses'].mean()
            avg_profit = future_profit['Profit'].mean()
            
            # Calculate growth rates
            income_growth = (future_income['Income'].iloc[-1] / last_month['Income'] - 1) * 100
            expense_growth = (future_expenses['Total_Expenses'].iloc[-1] / last_month['Total_Expenses'] - 1) * 100
            profit_growth = (future_profit['Profit'].iloc[-1] / last_month['Profit'] - 1) * 100 if last_month['Profit'] != 0 else 0
            
            # Display forecast metrics
            st.metric(
                label=f"Avg. Monthly Income (Next {len(future_income)} months)", 
                value=f"${avg_income:,.2f}",
                delta=f"{income_growth:.1f}% by end of period"
            )
            
            st.metric(
                label=f"Avg. Monthly Expenses (Next {len(future_expenses)} months)", 
                value=f"${avg_expenses:,.2f}",
                delta=f"{expense_growth:.1f}% by end of period",
                delta_color="inverse"
            )
            
            st.metric(
                label=f"Avg. Monthly Profit (Next {len(future_profit)} months)", 
                value=f"${avg_profit:,.2f}",
                delta=f"{profit_growth:.1f}% by end of period"
            )
            
            # Show forecast details in expander
            with st.expander("Monthly Forecast Details"):
                forecast_df = pd.DataFrame({
                    'Month': future_income['Month'],
                    'Income': future_income['Income'].round(2),
                    'Expenses': future_expenses['Total_Expenses'].round(2),
                    'Profit': future_profit['Profit'].round(2)
                })
                
                st.dataframe(forecast_df, use_container_width=True)

def data_input_page():
    st.title("Financial Data Input")
    
    # Create tabs for different data input methods
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Manual Entry", "View Current Data"])
    
    with tab1:
        upload_data()
    
    with tab2:
        manual_data_entry()
    
    with tab3:
        view_current_data()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        
        if st.button("← Back to Home", key="back_from_data"):
            st.session_state.page = "landing"
            st.rerun()
        
        if st.button("Dashboard", key="to_dashboard_from_data"):
            st.session_state.page = "dashboard"
            st.rerun()
        
        if st.button("Reports", key="to_reports_from_data"):
            st.session_state.page = "reports"
            st.rerun()

def reports_page():
    st.title("Financial Reports")
    
    # Get data and forecasts
    historical_data = st.session_state.financial_data
    
    # Check if forecasts exist, if not, prompt user to generate them
    if 'forecasted_data' not in st.session_state or st.session_state.forecasted_data is None:
        st.info("No forecasts available. Please generate forecasts from the Dashboard page first.")
        if st.button("Go to Dashboard", key="to_dashboard_for_forecast"):
            st.session_state.page = "dashboard"
            st.rerun()
        return
    
    forecasted_data = st.session_state.forecasted_data
    
    # Sidebar for report options
    with st.sidebar:
        st.header("Report Options")
        
        company_name = st.text_input("Company Name (optional)")
        
        # Last month of historical data
        last_month = pd.to_datetime(historical_data['Date'].iloc[-1])
        
        # First and last month of forecast
        forecast_start = pd.to_datetime(forecasted_data['Income'][forecasted_data['Income']['Type'] == 'Forecast']['Date'].iloc[0])
        forecast_end = pd.to_datetime(forecasted_data['Income'][forecasted_data['Income']['Type'] == 'Forecast']['Date'].iloc[-1])
        
        # Format for display
        forecast_period = f"{forecast_start.strftime('%b %Y')} to {forecast_end.strftime('%b %Y')}"
        
        st.write(f"Forecast Period: {forecast_period}")
        
        # Generate report button
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                # Create the PDF report
                pdf_buffer = create_financial_report(
                    historical_data,
                    forecasted_data,
                    company_name=company_name if company_name else "Fiscalyze Report",
                    report_period=forecast_period
                )
                
                # Create download link
                b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                
                # Generate filename
                current_date = datetime.now().strftime("%Y%m%d")
                filename = f"fiscalyze_report_{current_date}.pdf"
                
                # Store in session state for download
                st.session_state.pdf_download = {
                    'b64': b64_pdf,
                    'filename': filename
                }
                
                st.success("PDF report generated successfully!")
        
        # Navigation
        st.markdown("---")
        st.markdown("### Navigation")
        
        if st.button("← Back to Home", key="back_from_reports"):
            st.session_state.page = "landing"
            st.rerun()
        
        if st.button("Dashboard", key="to_dashboard_from_reports"):
            st.session_state.page = "dashboard"
            st.rerun()
        
        if st.button("Data Input", key="to_data_from_reports"):
            st.session_state.page = "data_input"
            st.rerun()
    
    # Main content area
    st.header("Financial Report Preview")
    
    # Display a preview of what's in the report
    tabs = st.tabs(["Income Forecast", "Expense Forecast", "Profit Forecast"])
    
    with tabs[0]:
        display_forecast_preview(forecasted_data, 'Income')
    
    with tabs[1]:
        display_forecast_preview(forecasted_data, 'Total_Expenses')
    
    with tabs[2]:
        display_forecast_preview(forecasted_data, 'Profit')
    
    # Show download link if report has been generated
    if 'pdf_download' in st.session_state:
        st.markdown("### Download Report")
        
        href = f'<a href="data:application/pdf;base64,{st.session_state.pdf_download["b64"]}" download="{st.session_state.pdf_download["filename"]}">Click here to download your PDF report</a>'
        
        st.markdown(href, unsafe_allow_html=True)
        
        st.info("The PDF report includes visualizations of your financial data, forecasted metrics, and financial summary tables.")

# Main app logic
if __name__ == "__main__":
    # Initialize the page state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # Display the appropriate page based on the state
    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "data_input":
        data_input_page()
    elif st.session_state.page == "reports":
        reports_page()
        import streamlit as st

st.sidebar.markdown("---")
st.sidebar.markdown("🔒 [Privacy Policy](Privacy_Policy)")
st.sidebar.markdown("📜 [Terms of Service](Terms_of_Service)")
st.sidebar.markdown("📧 [Contact Us](mailto:will.toyne@fiscalyze.co.uk)")
