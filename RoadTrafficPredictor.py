import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


f1 = '/Users/xingwenbo/Documents/CS2704/Average_Daily_Traffic_Counts_-_Map.csv'
f2 = '/Users/xingwenbo/Documents/CS2704/Chicago_Traffic_Tracker_-_Congestion_Estimates_by_Segments.csv'


df_traffic_counts = pd.read_csv(f1)
df_traffic_congestion = pd.read_csv(f2)


df_traffic_counts['Total Passing Vehicle Volume'] = df_traffic_counts['Total Passing Vehicle Volume'].str.replace(',', '').astype(float)


df_traffic_congestion['CURRENT_SPEED'] = pd.to_numeric(df_traffic_congestion[' CURRENT_SPEED'], errors='coerce')

df_merged = pd.merge(df_traffic_counts, df_traffic_congestion, left_on='Street', right_on='STREET', how='inner')

features = ['Total Passing Vehicle Volume', 'LENGTH']
target = 'CURRENT_SPEED'  


df_merged['LENGTH'] = pd.to_numeric(df_merged['LENGTH'], errors='coerce')


df_merged = df_merged.dropna(subset=['LENGTH'])

X = df_merged[features]
y = df_merged[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 模型开发
# 使用简单线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)


cv_scores = cross_val_score(model, X, y, cv=5)  # 5折交叉验证

# 预测测试集结果
y_pred = model.predict(X_test)

# 性能指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 残差分析
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red')
plt.xlabel('Actual Speed')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

#模型预测结果
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R²) value: {r2}')