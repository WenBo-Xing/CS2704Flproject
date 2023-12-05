import pandas as pd

file_path1 = '/Users/xingwenbo/Documents/CS2704/Chicago_Traffic_Tracker_-_Congestion_Estimates_by_Segments.csv'
#file_path2 = '/Users/xingwenbo/Documents/CS2704/Average_Daily_Traffic_Counts_-_Map.csv'
df = pd.read_csv(file_path1)
#df = pd.read_csv(file_path2)
print(df.columns)

print()

# To choose correct list name
#  if correct name is ' CURRENT_SPEED'
# else to use df[' CURRENT_SPEED'] to find
