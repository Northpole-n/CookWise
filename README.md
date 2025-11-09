# 🍳 CookWise — Smart Leftover Meal Planner

**CookWise** is an AI-powered meal planning application that minimizes food waste by recommending recipes based on available ingredients, expiry dates, and user nutrition preferences.  
Built using **Python**, **Streamlit**, and **Machine Learning**, the app predicts recipe suitability using a **regression model** (Random Forest or LightGBM).

---

## 🚀 Features
- 🧠 Predicts recipe suitability using ingredient availability and expiry data.  
- 🥗 Supports *balanced*, *high-protein*, and *low-carb* nutrition goals.  
- 📊 Displays model accuracy through RMSE, MAE, R², and regression plots.  
- 🧾 Exports grocery suggestions automatically to `grocery_suggestion.csv`.  
- 🧮 Interactive Streamlit interface for real-time plan generation.  

---

## ⚙️ Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/CookWise.git
   cd CookWise

Install Dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py
