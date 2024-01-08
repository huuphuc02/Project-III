from time import time
import copy
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics.local_search import solve_tsp_local_search
from python_tsp.heuristics.simulated_annealing import solve_tsp_simulated_annealing
from references.spectral_equal_size_clustering import SpectralEqualSizeClustering
from scipy.spatial.distance import cdist


class Pickup:
    def __init__(self, pickup_nodes: dict[str, list], return_node: dict[str, list], vehicle_list: dict[str, list],
                 distance_matrix, linkage, clustering_type):
        self.pickup_nodes = pickup_nodes
        self.return_node = return_node
        self.vehicle_list = vehicle_list
        self.distance_matrix = np.array(distance_matrix)
        self.linkage = linkage
        self.clustering_type = clustering_type
        return

    def find_best_silhouette_score(self, X, k_max):
        num_clusters = range(2, min(k_max, len(X)))
        silhouette_scores = []

        for n in num_clusters:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=5)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))

        optimal_n = num_clusters[np.argmax(silhouette_scores)]
        return optimal_n

    def find_best_gap(self, X, k_max=10, n_refs=5):
        gaps = np.zeros((k_max - 1,))
        results_dispersion = np.zeros((k_max - 1, n_refs))

        for k in range(1, k_max):
            for i in range(n_refs):
                # Fit KMeans model on random data
                random_data = np.random.random_sample(size=X.shape)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans.fit(random_data)

                # Calculate dispersion for random data
                results_dispersion[k - 1, i] = np.sum(
                    pairwise_distances_argmin_min(random_data, kmeans.cluster_centers_)[1])

                # Fit KMeans model on actual data
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans.fit(X)

                # Calculate dispersion for actual data
                actual_dispersion = np.sum(pairwise_distances_argmin_min(X, kmeans.cluster_centers_)[1])

                # Calculate gap statistic
                gaps[k - 1] += np.log(results_dispersion[k - 1, i]) - np.log(actual_dispersion)

        gaps = gaps / n_refs

        # Find the optimal number of clusters
        optimal_k = np.argmax(gaps) + 1
        return optimal_k

    def clustering(self, X, n_cluster: int, strategy='kmeans', linkage='ward'):
        if strategy == 'kmeans':
            model = KMeans(n_clusters=n_cluster, random_state=42, n_init=5)
            model.fit(X)
            output = model.predict(X)
        elif strategy == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
            output = model.fit_predict(X)
        elif strategy == 'dbscan':
            model = DBSCAN(eps=0.05, min_samples=5)
            output = model.fit_predict(X)
            n_cluster = len(np.unique(output[output != -1]))
            for i in range(len(output)):
                if output[i] == -1:
                    output[i] = n_cluster
            print(n_cluster + 1)
        elif strategy == 'meanshift':
            bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)
            model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            output = model.fit_predict(X)

            n_cluster = len(model.cluster_centers_)
        elif strategy == 'spectral':
            min_range, max_range = 5, 17  # desired number of points per cluster
            npoints = len(X)
            avg_range = (max_range + min_range) / 2.
            nclusters = int(npoints / avg_range)
            if nclusters <= 1:
                return np.zeros(npoints), 1
            n_cluster = nclusters
            eq_fr = 1 - ((avg_range - min_range) / avg_range)
            nn_fr = avg_range / npoints
            nneighbors = int(npoints * nn_fr)
            model = SpectralEqualSizeClustering(nclusters=nclusters,
                                                nneighbors=nneighbors,
                                                equity_fraction=eq_fr,
                                                seed=1234)
            output = model.fit(self.distance_matrix)
        return output, n_cluster

    def tsp(self, distance_matrix, tpe=None):
        if tpe is None or tpe == 'bitmasking':
            return solve_tsp_dynamic_programming(distance_matrix)
        elif tpe == 'local_search':
            return solve_tsp_local_search(distance_matrix)

    def tsp_no_clustering(self, distance_matrix):
        return solve_tsp_simulated_annealing(distance_matrix)

    def tsp_no_clustering2(self, distance_matrix):
        return solve_tsp_simulated_annealing(distance_matrix)

    def to_array(self, a_dict: dict[str, list]):
        res = []
        self.converse_map = []  # map index với code
        self.code_map = {}  # Map code với index
        for i, code in enumerate(a_dict):
            res.append(a_dict[code])
            self.code_map[code] = i
            self.converse_map.append(code)
        return np.array(res)

    def calculate_average_delivery_time(self, listOrders):
        total_delivery_before_time = 0
        total_orders = 0

        for order in listOrders:
            for order_infor in order.values():
                total_delivery_before_time += order_infor[0]
                total_orders += 1

        return total_delivery_before_time / total_orders

    def find_nearest_node(self, current_node: str, candidate_nodes: list[str], all_node_info: dict[str, list]):
        current_node_vector = all_node_info[current_node]
        best = 1e9
        best_code = ''
        for code in candidate_nodes:
            candidate_distance = self.distance_matrix[int(self.code_map[current_node])][int(
                self.code_map[code])]  # np.linalg.norm(np.array(current_node_vector) - np.array(all_node_info[code]))
            if candidate_distance <= best and candidate_distance != 0:
                best = candidate_distance
                best_code = code
        return best_code, best

    def get_location(self, all_node_info):
        result = {}
        for code in all_node_info:
            result[code] = all_node_info[code][:2]
        self.all_node_location = result
        return result

    def find_best_fit_vehicle(self, remain_route: list[str], demand_by_route: list[float], max_distance=200):
        best, best_code = 1e9, ''
        current_node = remain_route[0]
        vehicle_manager_node_index = 1
        vehicle_status_index = 2
        for v_code, v_array in self.vehicle_list.items():
            if v_array[vehicle_status_index] == 0: continue
            if str(v_array[vehicle_manager_node_index]) not in self.all_node_location: continue
            # print(v_array)
            # print("get route length: ",self.get_route_length(remain_route, v_array[vehicle_manager_node_index]))
            # print(self.distance_matrix[int(self.code_map[str(current_node)])][
            #     int(self.code_map[str(v_array[vehicle_manager_node_index])])])
            if self.distance_matrix[int(self.code_map[str(current_node)])][
                int(self.code_map[str(v_array[vehicle_manager_node_index])])] <= best:
                best = self.distance_matrix[int(self.code_map[str(current_node)])][
                    int(self.code_map[str(v_array[vehicle_manager_node_index])])]
                best_code = v_array[vehicle_manager_node_index]

        # if best == 1e9:
        #     # Nếu không còn xe thì đặt trạng thái của tất cả các xe về available, chạy lại find_best_fit_vehicle
        #     for v_code, v_array in self.vehicle_list.items():
        #         v_array[vehicle_status_index] = 1
        #     return self.find_best_fit_vehicle(remain_route, demand_by_route)

        epsilon = 20
        n_try = 0
        while True:
            res = []
            capacity = []
            # print(f"Best: {best}")
            for v_code, v_array in self.vehicle_list.items():
                if v_array[vehicle_status_index] == 0: continue
                if v_array[vehicle_manager_node_index] not in self.all_node_location:
                    continue
                if self.distance_matrix[int(self.code_map[current_node])][
                    int(self.code_map[str(v_array[vehicle_manager_node_index])])] <= best + epsilon:
                    res.append(v_code)
                    capacity.append(v_array[0])
            if len(capacity) == 0:
                epsilon += 20
                n_try += 1
                continue
            zipped_list = zip(capacity, res)
            sorted_pairs = sorted(zipped_list)
            tuples = zip(*sorted_pairs)
            capacity, res = [list(t) for t in tuples]
            if self.vehicle_list[res[-1]][0] > demand_by_route[0]: break
            if n_try == 5:
                print(res[-1], 0)
                return res[-1], 0
            epsilon += 20
            n_try += 1
        if capacity[-1] < np.sum(demand_by_route):  # Xe có tải trọng lớn nhất vẫn nhỏ hơn demand của route
            self.vehicle_list[res[-1]][vehicle_status_index] = 0
            self.vehicle_list[res[-1]][vehicle_manager_node_index] = str(list(self.return_node.keys())[0])
            demand_for_check = 0
            for i, val in enumerate(demand_by_route):
                demand_for_check += val
                if demand_for_check > capacity[-1]: break
            return res[-1], i - 1
        else:
            for i in range(len(capacity)):
                if capacity[len(capacity) - 1 - i] < np.sum(demand_by_route): break
            if i > len(capacity): i = len(capacity)
            self.vehicle_list[res[len(capacity) - i - 1]][vehicle_status_index] = 0
            return res[len(capacity) - i - 1], len(demand_by_route) - 1

    def get_route_length(self, route: list[str], start_node: str):
        res = self.distance_matrix[self.code_map[start_node]][self.code_map[route[0]]]
        for i in range(len(route) - 1):
            res += self.distance_matrix[int(self.code_map[str(route[i])])][int(self.code_map[str(route[i + 1])])]
        return res

    def split_route(self, all_route, all_node):
        # Update demand_by_route
        route_list = []
        demand_by_routes = []
        for r in all_route:
            route_list += r
        for code in route_list:
            # print(code)
            if len(all_node[code]) >= 5:
                demand_by_routes.append(all_node[code][4])
            else:
                demand_by_routes.append(0)
        distance_res = []
        percentage_res = []
        cost = []
        vehicle_route = {}
        remain_route = route_list.copy()
        remain_demand = demand_by_routes.copy()
        vehicle_list = copy.deepcopy(self.vehicle_list)
        # limit = 200
        # n_try = 0
        # k = 0
        print('Split route')
        while True:
            if len(remain_route) == 0: break
            # print(remain_route)
            # Tìm điểm cận trên cho 1 route có thể (<200km)
            idx = -1
            for i in range(len(remain_demand)):
                # print(self.get_route_length(remain_route[:i + 1], remain_route[0]))
                if self.get_route_length(remain_route[:i + 1], remain_route[0]) > 100:
                    idx = i
                    # idx -= k
                    break
            if idx == 0:
                idx = 1
            elif idx == -1:
                idx = len(remain_route)
            # print('idx', idx)
            # print('limit: ', limit)
            vehicle_id, end_index = self.find_best_fit_vehicle(remain_route[:idx], remain_demand[:idx])
            print(vehicle_id, end_index)
            percentage = []
            current_demand = 0
            child_routes = remain_route[: end_index + 1]
            # print(f"Route list: {child_routes}")
            dis = self.get_route_length(child_routes, vehicle_list[vehicle_id][1])
            dis += self.distance_matrix[int(self.code_map[child_routes[-1]])][
                int(self.code_map[list(self.return_node.keys())[0]])]
            # if dis > 200 and n_try < 2:
            #     limit = limit - 50
            #     n_try += 1
            #     continue
            # print(f'idx: {idx}')
            distance_res.append(dis)
            vehicle_route[vehicle_id] = child_routes.copy()
            vehicle_route[vehicle_id].append(str(list(self.return_node.keys())[0]))
            for j in range(end_index + 1):
                current_demand += remain_demand[j]
                if self.vehicle_list[vehicle_id][0] > 0:
                    percentage.append(current_demand / self.vehicle_list[vehicle_id][0])
                else:
                    percentage.append(0)
                if percentage[-1] > 1: percentage[-1] = 1
            percentage_res.append(np.mean(percentage))
            if percentage_res[-1] < 0.01:
                cost.append(distance_res[-1] * vehicle_list[vehicle_id][0] / 0.01)
            else:
                cost.append(distance_res[-1] * vehicle_list[vehicle_id][0] / percentage_res[-1])
            if end_index >= len(remain_demand) - 1: break
            # if remain_demand[end_index] == demand_by_routes[-1]: break
            remain_route = remain_route[end_index + 1:]
            remain_demand = remain_demand[end_index + 1:]
            print('dis: ', dis)
        for v in self.vehicle_list:
            self.vehicle_list[v][2] = 1
        return np.array(distance_res), np.array(percentage_res), np.array(cost), vehicle_route

    def execute2(self, province_code, write_type):
        # Các biến trả về
        distance_res = []
        routes_res = []
        time_res = []
        n_clusters = int(len(self.pickup_nodes) // 15 + 1)

        all_node = self.return_node.copy()
        all_node.update(self.pickup_nodes)
        # all_node_2 = {key: value if key == list(all_node.keys())[0] else value[:-1] for key, value in all_node.items()}
        #
        # average_delivery_times = {}
        # for node_id, node_info in all_node.items():
        #     if(len(node_info) < 3):
        #         average_delivery_times[node_id] = 0
        #     else:
        #         if node_id not in average_delivery_times:
        #             average_delivery_times[node_id] = self.calculate_average_delivery_time(node_info[2])
        #         else:
        #             average_delivery_times[node_id] += self.calculate_average_delivery_time(node_info[2])
        # print(average_delivery_times)
        # distance_time = np.zeros_like(self.distance_matrix)
        # for i, node_i in enumerate(average_delivery_times):
        #     for j, node_j in enumerate(average_delivery_times):
        #         distance_time[i, j] = abs(average_delivery_times[node_i] - average_delivery_times[node_j])
        #
        # weighted_distance = 0.7 * self.distance_matrix + 0.3 * distance_time
        # print(weighted_distance)
        all_node_array = self.to_array(all_node)
        X = all_node_array[:, :2]
        scaler = preprocessing.MinMaxScaler()
        X_normalized = scaler.fit_transform(X)

        # optimal_n = self.find_best_silhouette_score(X_normalized)
        # optimal_n = self.find_best_gap(X_normalized, 10, 5)
        # print('Số cụm tối ưu là: ', optimal_n)
        output, n_clusters = self.clustering(X_normalized, n_clusters, strategy=self.clustering_type,
                                             linkage=self.linkage)
        print(n_clusters)
        output = np.array(output)
        print(output)
        reverse = {}
        for i in range(n_clusters):
            reverse[i] = []
        for i, o in enumerate(output):
            reverse[int(o)].append(i)
        current_node = list(self.return_node.keys())[0]
        candidate_nodes = list(self.pickup_nodes.keys()) + list(self.return_node.keys())
        X_location = self.get_location(all_node)
        current_all_node = X_location.copy()
        for index in range(n_clusters):
            nearest_node, dis = self.find_nearest_node(current_node, candidate_nodes, current_all_node)
            print(dis)
            distance_res.append(dis)
            i = int(output[int(self.code_map[nearest_node])])  # Lấy cluster label của nearest_node làm index
            if len(reverse[i]) > 1 and reverse[i].index(self.code_map[nearest_node]) != 0:
                tmp = reverse[i][0]
                reverse[i].remove(self.code_map[nearest_node])
                reverse[i][0] = self.code_map[nearest_node]
                reverse[i].append(tmp)
            i_distance_matrix = np.zeros((len(reverse[i]), len(reverse[i])))
            for j in range(len(reverse[i])):
                for k in range(len(reverse[i])):
                    i_distance_matrix[j][k] = self.distance_matrix[int(reverse[i][j])][int(reverse[i][k])]
            for j in range(len(reverse[i])):
                i_distance_matrix[j][0] = 0
            if len(reverse[i]) <= 17:
                tpe = 'bitmasking'
            else:
                tpe = 'local_search'
            time1 = time()
            print(tpe)
            routes, distance = self.tsp(i_distance_matrix, tpe)
            time_res.append(time() - time1)
            distance_res.append(distance)
            # print(routes_res)
            tmp = []
            for r in routes:
                tmp.append(self.converse_map[int(reverse[i][int(r)])])
            routes_res.append(tmp)
            current_node = self.converse_map[int(reverse[i][int(routes[-1])])]
            for r in range(len(reverse[i])):
                try:
                    candidate_nodes.remove(self.converse_map[int(reverse[i][r])])
                except:
                    print(int(reverse[i][r]))
                    print(self.converse_map[int(reverse[i][r])])
                    print(self.converse_map[int(reverse[i][r])] in candidate_nodes)
                    raise Exception()
            # print('-'*100)
        r_l = []
        for r in routes_res:
            r_l += r
        distance_res, percentage_res, cost_res, vehicle_routes = self.split_route(routes_res, all_node)
        out_fname_vehicles_distances = f'scenarios/{self.clustering_type}/pickup_vehicles_distances.csv'
        i = 0
        with open(out_fname_vehicles_distances, write_type) as f:
            for vehicle_id in vehicle_routes.keys():
                f.write(f"{vehicle_id},{distance_res[i]}\n")
                i += 1
        # # print(vehicle_routes)
        # # print(distance_res)
        out_fname = f'scenarios/{self.clustering_type}/pickup_{self.clustering_type}.csv'
        dict_return = [province_code, len(self.pickup_nodes) + 1, np.sum(distance_res), np.sum(time_res),
                       np.mean(percentage_res), np.sum(cost_res)]
        with open(out_fname, write_type) as f:
            f.write(
                f"{dict_return[0]},{dict_return[1]},{dict_return[2]},{dict_return[3]},{dict_return[4]}, {dict_return[5]}\n")
        # time1 = time()
        # route_no_clustering, distance_no_clustering = self.tsp_no_clustering(self.distance_matrix)
        # route_list = [self.converse_map[int(i)] for i in route_no_clustering]
        # distance_no_clustering, percentage_res, cost_res, vehicle_routes_2 = self.split_route([route_list], all_node)
        # with open('scenarios/pickup_no_clustering_local_search.csv', write_type) as f:
        #     f.write(f"{dict_return[0]},{dict_return[1]},{np.sum(distance_no_clustering)},{time()-time1},{np.mean(percentage_res)},{np.sum(cost_res)}\n")
        return vehicle_routes, distance_res
