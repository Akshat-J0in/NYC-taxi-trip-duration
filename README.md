# NYC Taxi Trip Duration — EDA, Feature Engineering & Prediction

Predicting NYC taxi trip duration from 1.4 million real trips using exploratory analysis, thoughtful feature engineering, and gradient boosting.

## 🎯 Key Findings

- **Thursday 4pm is the worst time** — 35% slower than Sunday morning for the same route
- **LaGuardia has more taxi trips than JFK** — despite being smaller, likely due to lack of direct subway
- **Bearing (direction) outranks hour of day** as a predictor — the model learns airport patterns implicitly
- **Passenger count & vendor ID have zero impact** — validated from EDA
- **Feature engineering drove 34.6% improvement** — from 0.4891 RMSLE baseline to 0.3200 with LightGBM

## 📊 Model Performance

| Model | RMSLE | Notes |
|---|---|---|
| Linear Regression | 0.4891 | Baseline — assumes linear relationships |
| LightGBM | 0.3200 | 34.6% improvement with engineered features |

RMSLE of 0.32 means predictions are within ±38% of actual duration on average.

## 🚀 Quick Start — Run the App

**Prerequisites:** Python 3.8+

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Enter trip details:**
- Date & time of pickup
- Pickup and dropoff coordinates (drag on the map or enter manually)
- Passenger count and vendor ID

**Get instant predictions:**
- Predicted duration in minutes
- Straight-line distance
- Whether it's an airport trip
- Engineered features used in the prediction

## 📖 Full Analysis

See the **[Kaggle Notebook](https://www.kaggle.com/code/akshtjin/nyc-taxi-trip-duration-lightgbm)** for the complete analysis:

1. **Data Cleaning** — removed 0.77% of 1.4M rows (outliers, unrealistic values)
2. **Univariate EDA** — distributions of distance, duration, passenger count, time
3. **Bivariate EDA** — relationships between features and target
4. **Temporal Analysis** — hour, day of week, month patterns (heatmaps included)
5. **Spatial Analysis** — pickup/dropoff density maps showing Manhattan & airport clusters
6. **Feature Engineering** — 20 features from 10 raw columns (distance, bearing, time, location, airport flags)
7. **Modelling** — Linear Regression baseline → LightGBM with SHAP explainability
8. **SHAP Analysis** — why each prediction was made (feature contributions)

## 🔍 Feature Engineering Details

**Groups of features engineered:**

- **Distance** (3): distance_km, manhattan_distance, bearing
- **Time** (6): hours, month, day_of_week, is_weekend, is_rush_hour, hour_day_interaction
- **Airport** (3): is_jfk, is_lga, is_airport
- **Location** (2): pickup_distance_from_center, dropoff_distance_from_center
- **Raw** (6): vendor_id, passenger_count, pickup/dropoff coordinates

Each feature was motivated by a finding in the EDA.

## 🛠️ Tech Stack

- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **ML Model:** LightGBM (gradient boosting)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Deployment:** Streamlit, Pydeck (interactive map)
- **Environment:** Python 3.8+

## 💡 Key Insights

### Distance Dominates
Distance is 16x more important than any other feature. How far you travel fundamentally determines how long it takes.

### Time Matters More Than You Think
- Peak at 3-4pm (not 5-6pm as commonly believed)
- Thursday is 35% worse than Sunday
- Rush hour window is continuous 8am-5pm on weekdays
- Early morning (6am) is fastest across all days

### Direction Captures Airport Patterns
The explicit airport flags (is_jfk, is_lga) ranked last in feature importance. The bearing feature already captured airport directions implicitly — southeast bearing = JFK trips, northeast = LaGuardia trips.

### Features That Don't Matter
- Vendor ID — both vendors operate identically
- Passenger count — whether 1 or 6 people, duration unchanged
- Store & forward flag — 99.5% same value

## 📈 Model Explainability

Every prediction is fully explainable via SHAP analysis. For any trip, you can see:
- How much distance pushed the prediction up/down
- How much time of day contributed
- Which direction pushed it toward a longer trip
- Exact feature contributions for complete transparency

This is not a black box — it's a decision tree that learned from data.

## 📚 Learning Resources Used

- Kaggle EDA best practices
- Feature engineering principles (domain knowledge + data insights)
- LightGBM documentation and examples
- SHAP paper and implementation
- Streamlit documentation

## 👤 Author

Akshat Jain  
[GitHub](https://github.com/Akshat-J0in) | [LinkedIn](https://www.linkedin.com/in/akshat-jain-8a6864257/) | [Kaggle](https://www.kaggle.com/akshtjin)

## 📝 License

MIT License — feel free to use this code for learning, projects, or interviews.

## 🤝 Feedback & Contributions

Have improvements? Found a bug? Open an issue or submit a PR!

---

**Dataset:** [NYC Taxi Trip Duration on Kaggle](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration)  
**Notebook:** Published on Kaggle  
**App:** Live Streamlit deployment
