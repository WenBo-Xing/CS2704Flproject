import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import geopandas as gpd
from folium.plugins import HeatMap
from shapely.geometry import Point
from scipy.stats import pearsonr
from sklearn.neighbors import KDTree
from scipy.stats import pearsonr

#file address
file_path1 = '/Users/xingwenbo/Documents/CS2704/Chicago_Traffic_Tracker_-_Congestion_Estimates_by_Segments.csv'
df = pd.read_csv(file_path1)

# ' CURRENT_SPEED'
current_speed_distribution = df[' CURRENT_SPEED'].value_counts()

#when speed = -1
proportion_congested = (current_speed_distribution.get(-1, 0) / df.shape[0]) * 100

positive_speeds = df[df[' CURRENT_SPEED'] > 0][' CURRENT_SPEED']
plt.figure(figsize=(10, 6))
plt.hist(positive_speeds, bins=range(0, 60, 5), edgecolor='black')
plt.title('Distribution of Positive Current Speed Values')
plt.xlabel('Speed')
plt.ylabel('Number of Segments')
plt.grid(True)
plt.show()

print(f"Proportion of segments with traffic congestion (speed = -1): {proportion_congested:.2f}%")
print( )

positive_speeds_descriptive_stats = positive_speeds.describe()
print( )
print(positive_speeds_descriptive_stats)

congested_segments = df[df[' CURRENT_SPEED'] == -1].copy()

# 计算拥堵路段的平均经纬度
congested_segments['avg_longitude'] = (congested_segments['START_LONGITUDE'] + congested_segments['END_LONGITUDE']) / 2
congested_segments['avg_latitude'] = (congested_segments[' START_LATITUDE'] + congested_segments[' END_LATITUDE']) / 2

# 绘制拥堵路段的地理分布散点图
plt.figure(figsize=(10, 6))
plt.scatter(congested_segments['avg_longitude'], congested_segments['avg_latitude'], c='red', alpha=0.5)
plt.title('Geographical Distribution of Congested Segments')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

congested_segments['last_updated_time'] = pd.to_datetime(congested_segments[' LAST_UPDATED']).dt.time

time_distribution = congested_segments['last_updated_time'].value_counts()
print(time_distribution.head())  

street_counts = df['STREET'].value_counts()
print( )
print(street_counts.head()) 

most_frequent_street = street_counts.idxmax()

plt.figure(figsize=(10, 6))
sns.barplot(x=street_counts.index[:10], y=street_counts.values[:10])  # 只展示前10个街道作为示例
plt.title('Frequency of Streets')
plt.xlabel('Street')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

print(f"The most frequent street is: {most_frequent_street}")
print(' ')


file_path2 = '/Users/xingwenbo/Documents/CS2704/Average_Daily_Traffic_Counts_-_Map.csv'

congestion_data = pd.read_csv(file_path1)
traffic_counts_data = pd.read_csv(file_path2)

#print(congestion_data.columns)


congestion_data[' CURRENT_SPEED'] = congestion_data[' CURRENT_SPEED'].replace(-1, pd.NA)
congestion_data.dropna(subset=[' CURRENT_SPEED'], inplace=True)
traffic_counts_data['Total Passing Vehicle Volume'] = traffic_counts_data['Total Passing Vehicle Volume'].str.replace(',', '').astype(int)


#cleaned_congestion_data = congestion_data[congestion_data[' CURRENT_SPEED'] != -1]

traffic_volume = traffic_counts_data['Total Passing Vehicle Volume']
congestion_speed = congested_segments[' CURRENT_SPEED']

# Two database are same
min_length = min(len(traffic_volume), len(congestion_speed))
traffic_volume = traffic_volume.head(min_length)
congestion_speed = congestion_speed.head(min_length)

# P value
correlation, p_value = pearsonr(traffic_volume, congestion_speed)

print("相关系数:", correlation)
print("p 值:", p_value)


#heatmap creat
df = pd.read_csv(file_path2)
df['Total Passing Vehicle Volume'] = df['Total Passing Vehicle Volume'].str.replace(',', '').astype(int)

map_hooray = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

# creat heatmap datas
heat_df = df[['Latitude', 'Longitude', 'Total Passing Vehicle Volume']]
heat_data = [[row['Latitude'], row['Longitude'], row['Total Passing Vehicle Volume']] for index, row in heat_df.iterrows()]

HeatMap(heat_data).add_to(map_hooray)


# Heatmap Html dile
heat_map_path = '/Users/xingwenbo/Documents/CS2704/Average_traffic_heatmap.html'
map_hooray.save(heat_map_path)
print(f"Heatmap saved to: {heat_map_path}")
