import pandas as pd

total_total_distance = 0
total_total_time = 0
total_average_percentage = 0
total_total_cost = 0

pickup_file_path = "scenarios/hierarchical_complete/pickup_hierarchical_complete.csv"

df = pd.read_csv(pickup_file_path, header=None)

total_distance = df.iloc[:, 2].sum()
total_total_distance += total_distance

total_time = df.iloc[:, 3].sum()
total_total_time += total_time

average_percentage = df.iloc[:, 4].mean()
total_average_percentage += average_percentage

total_cost = df.iloc[:, 5].sum()
total_total_cost += total_cost

print("Total Distance Pickup:", total_distance)
print("Total Time Pickup:", total_time)
print("Fill level Pickup:", average_percentage * 100)
print("Total Cost Pickup:", total_cost)

transship_file_path = "scenarios/hierarchical_complete/transship_hierarchical_complete.csv"

df = pd.read_csv(transship_file_path, header=None)

total_distance = df.iloc[:, 2].sum()
total_total_distance += total_distance

total_time = df.iloc[:, 3].sum()
total_total_time += total_time

# average_percentage = df.iloc[:, 4].mean()
total_average_percentage += average_percentage
# total_cost = df.iloc[:, 5].sum()

print("\n\nTotal Distance Transship:", total_distance)
print("Total Time Transship:", total_time)
# print("Fill level Transship:", average_percentage * 100)
# print("Total Cost Transship:", total_cost)

delivery_file_path = "scenarios/hierarchical_complete/delivery_hierarchical_complete.csv"

df = pd.read_csv(delivery_file_path, header=None)

total_distance = df.iloc[:, 2].sum()
total_total_distance += total_distance

total_time = df.iloc[:, 3].sum()
total_total_time += total_time

average_percentage = df.iloc[:, 4].mean()
total_average_percentage += average_percentage

total_cost = df.iloc[:, 5].sum()
total_total_cost += total_cost

print("\n\nTotal Distance Delivery:", total_distance)
print("Total Time Delivery:", total_time)
print("Fill level Delivery:", average_percentage * 100)
print("Total Cost Delivery:", total_cost)

print("\n\nDistance: ", total_total_distance)
print("Time: ", total_total_time)
print("Fill level: ", total_average_percentage/3 * 100)
print("Cost: ", total_total_cost)
