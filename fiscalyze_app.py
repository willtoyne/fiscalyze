import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="Fiscalyze", layout="centered")

st.title("📊 Fiscalyze: Financial Forecasting for Entrepreneurs")

st.markdown("Input your monthly income and expenses to forecast your financial future.")

use_sample = st.checkbox("Use sample data?", value=True)

if use_sample:
    months = pd.date_range(start="2023-01-01", periods=12, freq="M")
    income = [3000 + np.random.randint(-200, 200) for _ in range(12)]
    expenses = [2000 + np.random.randint(-150, 150) for _ in range(12)]
    data = pd.DataFrame({"Month": months, "Income": income, "Expenses": expenses})
else:
    uploaded_file = st.file_uploader("Upload CSV with 'Month', 'Income', 'Expenses'", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data["Month"] = pd.to_datetime(data["Month"])

if "data" in locals():
    st.subheader("📈 Financial Overview")
    st.dataframe(data)

    months_num = np.arange(len(data)).reshape(-1, 1)
    future_months = np.arange(len(data) + 6).reshape(-1, 1)

    model_income = LinearRegression().fit(months_num, data["Income"])
    model_expenses = LinearRegression().fit(months_num, data["Expenses"])

    forecast_income = model_income.predict(future_months)
    forecast_expenses = model_expenses.predict(future_months)

    future_dates = pd.date_range(start=data["Month"].iloc[0], periods=len(future_months), freq="M")
    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecasted Income": forecast_income,
        "Forecasted Expenses": forecast_expenses
    })

    st.subheader("📉 Forecast Chart")
    fig, ax = plt.subplots()
    ax.plot(data["Month"], data["Income"], label="Income (Actual)", marker="o")
    ax.plot(data["Month"], data["Expenses"], label="Expenses (Actual)", marker="o")
    ax.plot(forecast_df["Month"], forecast_df["Forecasted Income"], label="Income (Forecast)", linestyle="--")
    ax.plot(forecast_df["Month"], forecast_df["Forecasted Expenses"], label="Expenses (Forecast)", linestyle="--")
    ax.legend()
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("📄 Download Forecast Report (PDF)")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Fiscalyze Forecast Report", ln=1, align="C")
        pdf.ln(10)

        for i in range(len(forecast_df)):
            row = forecast_df.iloc[i]
            line = f"{row['Month'].strftime('%B %Y')}: Income = ${row['Forecasted Income']:.2f}, Expenses = ${row['Forecasted Expenses']:.2f}"
            pdf.cell(200, 10, txt=line, ln=1)

        pdf_output = BytesIO()
        pdf.output(pdf_output)
        st.download_button("Download PDF", data=pdf_output.getvalue(), file_name="fiscalyze_forecast.pdf", mime="application/pdf")
