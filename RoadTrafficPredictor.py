import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
file_traffic_counts = '/Users/xingwenbo/Documents/CS2704/Average_Daily_Traffic_Counts_-_Map.csv'
file_traffic_congestion = '/Users/xingwenbo/Documents/CS2704/Chicago_Traffic_Tracker_-_Congestion_Estimates_by_Segments.csv'

df_traffic_counts = pd.read_csv(file_traffic_counts)
df_traffic_congestion = pd.read_csv(file_traffic_congestion)

# Data cleaning and preprocessing
df_traffic_counts['Total Passing Vehicle Volume'] = df_traffic_counts['Total Passing Vehicle Volume'].str.replace(',', '').astype(float)
df_traffic_congestion[' CURRENT_SPEED'] = pd.to_numeric(df_traffic_congestion[' CURRENT_SPEED'], errors='coerce')

df_merged = pd.merge(df_traffic_counts, df_traffic_congestion, left_on='Street', right_on='STREET', how='inner')
df_merged[' CURRENT_SPEED'].replace(-1, float('nan'), inplace=True)
df_merged = df_merged.dropna(subset=[' CURRENT_SPEED'])

# Feature selection
features = ['Total Passing Vehicle Volume', 'LENGTH']
target = ' CURRENT_SPEED'
X = df_merged[features]
y = df_merged[target]

# Divide training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature normalization and model pipeline
scaler = StandardScaler()
pipeline = make_pipeline(scaler, RandomForestRegressor(random_state=0))

# Parameter optimization using RandomizedSearchCV
param_dist = {
    'randomforestregressor__n_estimators': [50, 100, 150],
    'randomforestregressor__max_depth': [None, 10, 20],
    'randomforestregressor__min_samples_split': [2, 4],
    'randomforestregressor__min_samples_leaf': [1, 2]
}
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=0)
random_search.fit(X_train, y_train)

# Optimum parameters and model performance evaluation
best_model = random_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# output
print(" MSE:", mse_optimized)
print(" R-squared:", r2_optimized)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimized, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel('Actual Speed')
plt.ylabel('Predicted Speed')
plt.title('Actual vs Predicted Speed - Optimized Random Forest Model')
plt.show()
