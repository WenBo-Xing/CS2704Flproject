import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 加载数据
file_path = '/Users/xingwenbo/Documents/CS2704/Chicago_Traffic_Tracker_-_Congestion_Estimates_by_Segments.csv'

# 加载CSV文件
df = pd.read_csv(file_path)

# 选择特征和目标变量
# 假设 'STREET' 是一个分类特征
X = df[['STREET', 'LENGTH', 'START_LONGITUDE']]
y = df[' CURRENT_SPEED']

# 使用独热编码来转换 'STREET'
column_transformer = ColumnTransformer(
    [('ohe', OneHotEncoder(), ['STREET'])],
    remainder='passthrough'
)

X_transformed = column_transformer.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
