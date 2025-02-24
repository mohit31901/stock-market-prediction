# Stock Market Prediction using LSTM, XGBoost, and Random Forest

## Overview
This project implements a **stock price prediction model** using a hybrid approach that combines **LSTM (Long Short-Term Memory), XGBoost, and Random Forest**. The goal is to predict the **next-day closing price** of NSE-listed stocks by leveraging historical stock data and sentiment analysis.

## Features
- Fetches **historical stock data** for multiple NSE-listed stocks.
- Uses **financial news sentiment analysis** as an additional feature.
- Implements a **hybrid model** with:
  - **LSTM (Deep Learning for time-series forecasting)**
  - **XGBoost (Gradient Boosting Trees for structured data)**
  - **Random Forest (Ensemble Learning for non-linear dependencies)**
- Evaluates model performance using **MAE, MSE, RMSE, and R² Score**.
- Generates **visualizations of actual vs predicted stock prices**.

## Methodology
1. **Data Collection**
   - Retrieves stock price data using `get_stock_data(ticker)`.
   - Computes a **sentiment score** using `get_news_sentiment(ticker)`.
   
2. **Data Preprocessing**
   - Uses a **time step of 15** (past 15 days data used for prediction).
   - Performs **feature scaling**.
   - Splits data into **80% training** and **20% testing**.

3. **Model Training**
   - **LSTM Model:** Trained for **10 epochs**, batch size **16**.
   - **XGBoost:** Uses **200 estimators**, learning rate **0.05**.
   - **Random Forest:** Uses **200 estimators**.

4. **Prediction & Evaluation**
   - Averages predictions from all three models.
   - Uses **inverse transformation** to convert scaled values back to INR.
   - Computes **MAE, MSE, RMSE, and R² Score**.

5. **Next-Day Price Prediction**
   - Takes last **15 days of stock data + sentiment score**.
   - Predicts the **next trading day's closing price**.

6. **Visualization**
   - Plots **actual vs predicted** stock prices for better analysis.

## Technologies Used
- **Python** (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
- **TensorFlow/Keras** (For LSTM model)
- **XGBoost & RandomForest** (For ensemble learning)
- **FinBERT** (For sentiment analysis)
- **Yahoo Finance API** (For stock data retrieval)
