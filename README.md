# solar-power-prediction-ML-Streamlit

#  Solar Power Prediction
  
Machine Learning model to predict solar power generation (kWh) using weather conditions with a deployed Streamlit web application.

---

##  1. Overview
This project predicts solar power generation based on weather parameters such as temperature, humidity, wind speed, sky cover, and distance to solar noon.  
A user-friendly *Streamlit web app* is provided to make real-time predictions from user inputs.

---

##  2. Problem Statement
Solar energy output fluctuates heavily due to weather changes.  
Solar plant operators and consumers face uncertainty in daily power generation.  
This project aims to build a regression model that accurately predicts solar power generation using historical weather data so operators can plan and optimize energy usage.

---

##  3. Dataset
- **File:** `solarpowergeneration.csv`  
- **Rows:** ~2920  
- **Features include:**  
  - `distance-to-solar-noon` (radians)  
  - `temperature` (°C)  
  - `wind-direction` (degrees)  
  - `wind-speed` (m/s)  
  - `sky-cover` (0–4 scale)  
  - `visibility` (km)  
  - `humidity` (%)  
  - `average-wind-speed-(period)` (m/s)  
  - `average-pressure-(period)` (in mercury inches)  

- **Target Variable:** `power-generated (kWh)`
---

##  4. Tools and Technologies
- **Python (Jupyter Notebook)**  
- **Pandas, NumPy** (data handling)  
- **Matplotlib, Seaborn, Plotly** (visualization)  
- **Scikit-Learn** (models, scaler, CV)  
- **XGBoost** (optional model)  
- **Joblib** (save/load model & scaler)  
- **Streamlit** (web app / deployment)

---

##  5. Methods
1. **Data Cleaning**: filled missing numeric values with column mean; inspected and handled outliers using IQR.  
2. **Exploratory Data Analysis (EDA)**: boxplots, histograms, scatterplots, and correlation heatmap to understand relationships.  
3. **Feature Scaling**: StandardScaler applied to input features before training models.  
4. **Model Training**:
   - Linear Regression (baseline)  
   - Lasso & Ridge (regularized linear models)  
   - Decision Tree Regressor  
   - Random Forest Regressor (primary model)  
   - Hyperparameter tuning for Random Forest (GridSearchCV)  
   - XGBoost Regressor (comparison)  
5. **Model Evaluation**: MAE, MSE, R², and cross-validation (5-fold) to compare stability & performance.  
6. **Deployment**: Save tuned model and scaler with `joblib` and build a Streamlit app (`app.py`) for interactive predictions.

---

##  6. Key Insights
- **Temperature** — strong positive correlation with power generated: higher temperature (as proxy for sunlight) usually means higher generation.  
- **Sky Cover** — strong negative correlation: more cloud cover → less power.  
- **Humidity** — moderate negative correlation: higher humidity scatters sunlight, reducing generation.  
- **Distance to solar noon** — closer to solar noon corresponds to higher power generation.  
- Random Forest consistently performed best on this dataset (R² ≈ 0.91 on test set), balancing bias–variance well.

---

##  7. Model / Output
- **EDA Visuals**: boxplots, histograms, scatterplots, and a correlation heatmap.  
- **Feature Importance**: plotted from Decision Tree / Random Forest to identify the most influential features.  
- **Model Comparison Table**: R², MAE, MSE for Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Tuned Random Forest, and XGBoost.  
- **Interactive App**: Streamlit app where a user inputs weather variables and receives a predicted power output (kWh) with a gauge and a short explanation (Low / Moderate / High).

---

##  8. Results & Conclusion

### **Model Performance Summary**

| Model | R² Score | MAE | MSE |
|-------|----------:|----:|----:|
| Linear Regression | 0.70 | 5016.80 | 3.65e+07 |
| Lasso | 0.70 | 5016.79 | 3.65e+07 |
| Ridge | 0.70 | 5016.26 | 3.65e+07 |
| Decision Tree | 0.87 | 1675.26 | 1.60e+07 |
| Random Forest | **0.92** | **1374.54** | **1.01e+07** |
| Tuned Random Forest | 0.9157 | 1404.28 | 1.03e+07 |
| XGBoost | 0.9152 | 1419.70 | 1.04e+07 |

### **Conclusion**
The Random Forest Regressor achieved the highest accuracy with an R² score of ~0.92, outperforming all other models.  
It captured non-linear relationships between environmental factors and power output effectively.  

Some predictions returned **zero** due to the dataset containing many low/zero power values during nighttime or low solar exposure.  
Overall, the model is reliable for estimating solar power generation and suitable for deployment through a Streamlit application.

## Contact

**Author:** Priyanka Yadav  
**Email:** priyankayadav9822@gmail.com
