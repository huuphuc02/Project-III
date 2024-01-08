import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:/Code/2023-2EVRP-DOAN/scenarios/kmeans_silhouette/transship_vehicles_distances.csv'
df = pd.read_csv(file_path, header=None)

bins = [100, 150, 200, 250, 300, 350, 400]

df['Distance Group'] = pd.cut(df[1], bins=bins)

count_by_distance_group = df['Distance Group'].value_counts().sort_index()
print(count_by_distance_group)

plt.figure(figsize=(10, 6))
count_by_distance_group.plot(kind='bar', color='green', width=0.6)
plt.xlabel('Distance Group')
plt.ylabel('Number of Vehicles')
plt.title('Number of Vehicles by distance group in transship phase')
plt.xticks(rotation=0)
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Dữ liệu từ bảng
# methods = ['Kmeans silhouette', 'Kmeans', 'Meanshift', 'DBSCAN', 'Hierar single',
#            'Hierar average', 'Hierar ward', 'Hierar complete', 'Equal-size spectral']
#
# distance = [4059.66, 4119.59, 4050.33, 4472.77, 4121.33, 4116.05, 4205.81, 4173.94, 4190.69]
# computing_time = [5, 1, 4, 0.5, 0.5, 0.15, 3, 3, 5]
# # fill_level = [44.96, 47.55, 43.58, 45.2, 45.81, 46.23, 46.25, 46.22, 45.98]
# # cost = [127, 192, 157, 197, 126, 146, 198, 72, 132]
#
# new_distance = []
# for i in distance:
#     i = i / 1000
#     new_distance.append(i)
#
# distance = new_distance
#
# data = list(zip(methods, distance, computing_time))
#
# sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
#
# methods, distance, computing_time = zip(*sorted_data)
#
# bar_width = 0.33
# index = np.arange(len(methods))
#
# plt.figure(figsize=(12, 6))
#
# plt.bar(index, distance, color='tab:blue', width=bar_width, label='Distance')
# plt.bar(index + bar_width, computing_time, color='tab:orange', width=bar_width, label='Computing Time', alpha=0.7)
# # plt.bar(index + 2 * bar_width, fill_level, color='tab:green', width=bar_width, label='Fill Level', alpha=0.7)
# # plt.bar(index + 3 * bar_width, cost, color='tab:purple', width=bar_width, label='Cost', alpha=0.7)
#
# plt.xlabel('Methods')
# plt.ylabel('Values')
# plt.title('Comparison of Distance, Computing Time for Each Method in phase 2')
# plt.xticks(index + bar_width + 0.5, methods, rotation=30, ha='right')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
