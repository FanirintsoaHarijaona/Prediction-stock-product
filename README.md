# Stock Forecasting and Recommendation App

## Overview

This application predicts the monthly quantity of products per zone using a trained XGBoost model and provides stock recommendations using an LLM. It is built with Python, Streamlit, and containerized with Docker.

---

## Data Description

- **Dataset**: in a format csv with the following column
- **Columns**:
  - `Product`: Name of the products
  - `Purchase Address`: the zone of the purchase
  - `Quantity ordered`: the quantity of the products ordered
  - `Order Date`: Date of the order in format dd/mm/yyyy

The model uses lag features and rolling averages over previous months to forecast the next 5 months of sales per product and zone.

---

## Technology Stack

- **Python 3.12**
- **XGBoost** for regression forecasting
- **Scikit-learn** for preprocessing and evaluation
- **Streamlit** for the web interface
- **Hugging Face Transformers** for LLM stock recommendations (offline, no API key)
- **LangChain** (optional, if needed for LLM workflow)
- **Docker** to containerize the app

---

## Files

- `app.py` — main Streamlit application, handles prediction and LLM recommendations
- `model_encoder/` — folder containing saved model and encoders
  - `model.pkl` — trained XGBoost model
  - `product.pkl` — LabelEncoder for product names
  - `le_zone.pkl` — LabelEncoder for zones
- `Dockerfile` — Docker configuration to build and run the app

---

## How to Run

### 1. Build Docker Image

```bash
docker build -t stock-forecast-app .
```
### 2. Run the app 
```bash 
docker run -p 8501:8501 stock-forecast-app
```

---

