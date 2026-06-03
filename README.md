# Flight Fare Prediction

A machine learning project to predict flight ticket prices across major Indian routes using regression models trained on airline, route, timing, and stop information.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Author](#author)

---

## Project Overview

This project builds an end-to-end ML regression pipeline to predict flight ticket prices based on attributes such as airline, source, destination, departure time, arrival time, duration, number of stops, and additional flight information. The dataset contains 10,683 flight records across 12 airlines, 5 source cities, and 6 destination cities for journeys in 2019.

The target variable is `Price` — the ticket fare in INR. The pipeline covers the full ML workflow: data loading, exploratory data analysis, missing value handling, feature extraction from date and time columns, categorical encoding, outlier treatment, feature scaling, feature selection, model training, and R² score evaluation across 7 regression models.

---

## Repository Structure

```
Flight-Fare-Prediction/
├── Flight_Fare.ipynb       # End-to-end ML pipeline notebook
└── Flight_Fare.csv         # Dataset (10,683 flight records)
```

---

## Dataset

**File:** `Flight_Fare.csv`  
**Source:** Kaggle  
**Total Records:** 10,683  
**Target Variable:** `Price` — Flight ticket fare in INR

| Feature | Type | Description |
|---|---|---|
| `Airline` | Categorical | Name of the airline (12 airlines) |
| `Date_of_Journey` | String | Date of the flight journey |
| `Source` | Categorical | Departure city (Bangalore, Kolkata, Delhi, Chennai, Mumbai) |
| `Destination` | Categorical | Arrival city (New Delhi, Bangalore, Cochin, Kolkata, Delhi, Hyderabad) |
| `Route` | String | Full route with intermediate stops |
| `Dep_Time` | String | Departure time of the flight |
| `Arrival_Time` | String | Arrival time of the flight |
| `Duration` | String | Total flight duration |
| `Total_Stops` | Categorical | Number of stops (non-stop, 1 stop, 2 stops, 3 stops, 4 stops) |
| `Additional_Info` | Categorical | Extra details (meal info, baggage, layover type, class) |
| `Price` | Integer | Target — ticket fare in INR |

**Airlines covered:** IndiGo, Air India, Jet Airways, SpiceJet, Multiple Carriers, GoAir, Vistara, Air Asia, Vistara Premium Economy, Jet Airways Business, Multiple Carriers Premium Economy, Trujet

---

## ML Pipeline

### 1. Data Loading and Exploration

- Loaded dataset using `pandas`
- Inspected shape, data types, null values using `df.info()` and `df.isnull().sum()`
- Reviewed value counts for categorical columns and price distribution

### 2. Feature Extraction

- Extracted `Journey_Day` and `Journey_Month` from `Date_of_Journey`
- Extracted `Dep_Hour` and `Dep_Min` from `Dep_Time`
- Extracted `Arrival_Hour` and `Arrival_Min` from `Arrival_Time`
- Extracted `Duration_Hours` and `Duration_Mins` from `Duration`
- Dropped original raw columns after extraction

### 3. Missing Value Handling

- **Numerical columns** — imputed using mean via `SimpleImputer(strategy='mean')`
- **Categorical columns** — imputed using mode via `SimpleImputer(strategy='most_frequent')`

### 4. Categorical Encoding

- Applied **One-Hot Encoding** using `pd.get_dummies()` with `drop_first=True` for `Airline`, `Source`, `Destination`, `Additional_Info`
- Applied **Label Encoding** on `Total_Stops` (ordinal: non-stop=0, 1 stop=1, 2 stops=2, 3 stops=3, 4 stops=4)
- Dropped `Route` column — high cardinality (129 unique routes), information captured via other features

### 5. Outlier Treatment

- Used **IQR (Interquartile Range)** method for all numerical columns
- Capped values below `Q1 - 1.5 * IQR` to the lower bound
- Capped values above `Q3 + 1.5 * IQR` to the upper bound

### 6. Feature Selection

- Applied `SelectKBest` with `f_regression` to identify the most relevant features for predicting `Price`
- Visualized feature correlations using a **heatmap** (`seaborn`)

### 7. Feature Scaling

- Applied `StandardScaler` to normalize all feature values before model training

### 8. Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 30% |
| Train size | 70% |
| Random state | 50 |

---

## Model Results

All 7 regression models were trained on the same preprocessed dataset and evaluated using **R² Score**.

| # | Model | R² Score |
|---|---|---|
| 1 | **Random Forest** | **0.9169** |
| 2 | Decision Tree | 0.8872 |
| 3 | Ridge Regression | 0.8670 |
| 4 | Linear Regression | 0.8641 |
| 5 | Gradient Boosting | 0.8104 |
| 6 | K-Nearest Neighbors | 0.4855 |
| 7 | Support Vector Regression | -0.0061 |

**Best performing model: Random Forest — R² Score of 0.9169**

> SVR's negative R² score indicates it performed worse than a simple mean baseline on this dataset. This is expected because SVR is sensitive to feature scaling and high-dimensional sparse data created by One-Hot Encoding of categorical variables with many unique values such as Airline and Route.

---

## Author

**Jaishri Vijayakumar**  
B.Sc. Data Science | PSGR Krishnammal College for Women, Coimbatore  
LinkedIn: linkedin.com/in/jaishri-vijayakumar  
GitHub: github.com/jaishrivijayakumar
