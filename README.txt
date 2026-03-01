# 🏏 Cricket Score Predictor

A machine learning project that predicts the final score of a T20 cricket innings using in-match features. The project includes exploratory data analysis, custom gradient descent implementation, and a scikit-learn pipeline with feature engineering.

---

## 📁 Project Structure

```
cricket_score_predicter.ipynb   # Main notebook
Dataset.csv                     # Match-level cricket dataset
```

---

## 📊 Dataset

The dataset (`Dataset.csv`) contains T20 match data with the following features:

| Feature | Description |
|---|---|
| `batting_team` | Team currently batting |
| `bowling_team` | Team currently bowling |
| `venue` | Match location |
| `current_score` | Runs scored so far |
| `balls_left` | Balls remaining in the innings |
| `wickets_left` | Wickets remaining |
| `current_run_rate` | Runs per over at current point |
| `last_five` | Runs scored in the last 5 overs |
| `is_powerplay` | Whether match is in powerplay phase |
| `Top_Order / Middle_Order / Lower_Order / Tail` | Current batting order (one-hot) |
| `Death_Overs` | Whether match is in death overs phase |
| `final_score` | **Target variable** — total innings score |

---

## 🔧 Feature Engineering

Two new features were derived to improve model performance:

- **`projected_score`** — Extrapolates the current run rate over 20 overs: `current_run_rate × 20`
- **`wickets_fallen`** — A more intuitive representation of dismissals: `10 - wickets_left`

Two pressure-related binary flags were also created:

- **`Pressure`** — Set to `1` when `current_run_rate <= 7.0`
- **`aggressive`** — Set to `1` when `last_five >= 45`

---

## 🧠 Models

### 1. Custom Multivariate Linear Regression (from scratch)
A gradient descent-based linear regression model implemented without any ML libraries:

- Forward pass using dot product: `ŷ = Xw + b`
- Cost function: Mean Squared Error (MSE)
- Iterative weight updates via gradient descent
- Training loss plotted over iterations

### 2. Scikit-learn Pipeline (Linear Regression)
A production-style pipeline using `ColumnTransformer`:

- **Numeric features** → `StandardScaler`
- **Categorical features** → `OneHotEncoder(handle_unknown='ignore')`
- Final estimator: `LinearRegression`

The pipeline was retrained after feature engineering to incorporate `projected_score` and `wickets_fallen`.

---

## 📈 Evaluation Metrics

Model performance is evaluated using:

| Metric | Description |
|---|---|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R²** | Coefficient of determination |

---

## 📉 Visualizations

- **Training Loss Curve** — Cost vs. iteration for the custom gradient descent model
- **Actual vs. Predicted Scatter Plot** — Visual comparison of ground truth and predictions on the test set
- **Highest vs. Lowest Score Bar Chart** — Team-level scoring extremes

---

## 🔍 Exploratory Data Analysis

The notebook answers 15 analytical questions, including:

1. Away venues where Sri Lanka played the most matches
2. Team and opponent involved in the highest recorded score
3. Team with the highest run rate
4. Best-performing batting order (Top/Middle/Lower/Tail) by average score
5. Average score under pressure when the Tail is batting
6. Teams with the most aggressive innings (last 5 overs ≥ 45 runs)
7. India's average runs in death overs
8. Highest and average scores at Johannesburg
9. Highest and average target set against Pakistan
10. Average wickets fallen in death overs
11. Pakistan's average target at home (including UAE venues)
12. Team-wise average runs under pressure
13. Team-wise average runs in death overs
14. Team-wise average targets set

---

## 🚀 Getting Started

### Requirements

```bash
pip install pandas scikit-learn matplotlib numpy
```

### Run

Open the notebook in Jupyter and run all cells in order:

```bash
jupyter notebook cricket_score_predicter.ipynb
```

Make sure `Dataset.csv` is in the same directory as the notebook.

---

## 🔮 Single Match Prediction

The notebook includes a single-prediction example. Provide a match state as a dictionary and the trained pipeline will return the predicted final score:

```python
single_match = pd.DataFrame([{
    'batting_team': 'Pakistan',
    'bowling_team': 'South Africa',
    'venue': 'Dubai',
    'current_score': 110,
    'balls_left': 48,
    'wickets_left': 5,
    'current_run_rate': 6.5,
    'Middle_Order': 1,
    ...
}])

predicted_score = pipeline.predict(single_match)
```

