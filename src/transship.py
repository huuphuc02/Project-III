from time import time
import copy

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, estimate_bandwidth, MeanShift
from sklearn import preprocessing

from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min

from src.references.spectral_equal_size_clustering import SpectralEqualSizeClustering


class Transship:
    '''
    Trung chuyển từ GD1 bên gửi tới GD1 bên nhận \n
    Test với từng điểm GD1 làm điểm bắt đầu, lưu lại điểm GD1 cho kết quả tốt nhất \n
    So sánh với local search 
    '''

    def __init__(self, delivery_nodes: dict[str, list], vehicle_list: dict[str, list], distance_matrix, linkage,
                 clustering_type) -> None:
        self.all_nodes = delivery_nodes
        self.vehicle_list = {}
        self.distance_matrix = np.array(distance_matrix)
        self.linkage = linkage
        self.clustering_type = clustering_type
        for v_list in vehicle_list.values():
            for vehicle_code, v_array in v_list.items():
                self.vehicle_list[vehicle_code] = v_array
        print(self.vehicle_list)

    def find_best_silhouette_score(self, X, k_max):
        num_clusters = range(2, min(k_max, len(X)))
        silhouette_scores = []

        for n in num_clusters:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=5)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))

        if not silhouette_scores:
            print("No silhouette scores available. Check the number of clusters or the data.")
            optimal_n = 1
        else:
            optimal_n = num_clusters[np.argmax(silhouette_scores)]
        return optimal_n

    def find_best_gap(self, X, k_max=10, n_refs=5):
        gaps = np.zeros((k_max - 1,))
        results_dispersion = np.zeros((k_max - 1, n_refs))

        for k in range(1, k_max):
            for i in range(n_refs):
                # Fit KMeans model on random data
                random_data = np.random.random_sample(size=X.shape)
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(random_data)

                # Calculate dispersion for random data
                results_dispersion[k - 1, i] = np.sum(
                    pairwise_distances_argmin_min(random_data, kmeans.cluster_centers_)[1])

                # Fit KMeans model on actual data
                kmeans = KMeans(n_clusters=k, random_state=42)
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
            min_range, max_range = 5, 17   # desired number of points per cluster
            npoints = len(X)
            avg_range = (max_range + min_range)/2.
            nclusters = int(npoints / avg_range)
            if nclusters <= 1:
                return np.zeros(npoints), 1
            n_cluster = nclusters
            eq_fr = 1 - ((avg_range - min_range)/avg_range)
            nn_fr = avg_range / npoints
            nneighbors = int(npoints*nn_fr)
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
        return solve_tsp_local_search(distance_matrix)

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

    def transform_node_vector(self, all_node_info):
        result = {}
        for code in all_node_info:
            result[code] = all_node_info[code][:2]
        self.all_node_location = result
        return result

    def find_best_fit_vehicle(self, remain_route: list[str], demand_by_route: list[float]):
        '''
        Tìm tập xe gần nhất so với điểm bắt đầu
        Return: id xe gần nhất, điểm kết thúc ứng với xe
        remain_route: chặng đường còn lại cần chạy
        demand_by_route: Khối lượng đơn hàng ứng với các điểm trong remain_route
        '''
        # print('Find best fit vehicle')
        # Tìm node gần với điểm hiện tại nhất, điểm hiện tại là remain_route[0]
        best, best_code = 1e9, ''
        current_node = remain_route[0]
        vehicle_manager_node_index = 1
        vehicle_status_index = 2
        # swap_index = -1

        for i in range(1, len(remain_route)):
            next_node = remain_route[i]
            for vehicle_code, v_array in self.vehicle_list.items():
                if v_array[vehicle_status_index] == 0: continue
                if next_node in v_array[vehicle_manager_node_index]:
                    best = 0
                    best_code = v_array[vehicle_manager_node_index]
                    # swap_index = i
                    break

            # if swap_index != -1:
            #     current_node = next_node
            #     remain_route[0], remain_route[swap_index] = remain_route[swap_index], remain_route[0]
            #     demand_by_route[0], demand_by_route[swap_index] = demand_by_route[swap_index], demand_by_route[0]
            #     break

        print('remain_route: ', remain_route)

        '''Tìm những xe nằm gần điểm hiện tại nhất, 
        những điểm nằm gần là những điểm có khoảng cách < `best` + `epsilon` 
        '''
        epsilon = 20
        n_try = 0
        while True:
            res = []
            capacity = []
            print(f"Best: {best}")
            for i in range(1, len(remain_route)):
                next_node = remain_route[i]
                for vehicle_code, v_array in self.vehicle_list.items():
                    if v_array[vehicle_status_index] == 0: continue
                    if v_array[vehicle_manager_node_index] not in self.all_node_location: continue
                    if next_node in v_array[vehicle_manager_node_index]:
                        res.append(vehicle_code)
                        capacity.append(v_array[0])

            # Sắp xếp lại mảng theo chiều tăng dần capacity
            # print(f"capacity: {capacity}, res: {res}")
            zipped_list = zip(capacity, res)
            sorted_pairs = sorted(zipped_list)
            tuples = zip(*sorted_pairs)
            capacity, res = [list(t) for t in tuples]
            print(f"capacity: {capacity}, demand: {demand_by_route}")
            print('res: ', res)
            if self.vehicle_list[res[-1]][0] > demand_by_route[0]: break
            if n_try == 5: return res[-1], 0
            epsilon += 20
            n_try += 1
        # Trả về xe phù hợp nhất, update trạng thái cho xe thành 0 (not availavle)
        if capacity[-1] < np.sum(demand_by_route):  # Xe có tải trọng lớn nhất vẫn nhỏ hơn demand của route
            self.vehicle_list[res[-1]][vehicle_status_index] = 0
            demand_for_check = 0
            for i, val in enumerate(demand_by_route):
                demand_for_check += val
                if demand_for_check > capacity[-1]: break
            return res[-1], i - 1

        else:
            for i in range(len(capacity)):
                if capacity[len(capacity) - 1 - i] < np.sum(demand_by_route): break
            self.vehicle_list[res[len(capacity) - i]][vehicle_status_index] = 0
            return res[len(capacity) - i], len(demand_by_route) - 1

    def get_route_length(self, route: list[str], start_node: str):
        '''
        route: list các code của node 
        '''
        res = self.distance_matrix[self.code_map[start_node]][self.code_map[route[0]]]
        print(f"Start dis: {res}")
        # print(f" code map: {self.code_map.keys()}")
        for i in range(len(route) - 1):
            # print(f"route i: {route[i]}, route i+1: {route[i+1]}")
            # print(f"code map: {self.code_map[str(route[i])]}, {self.code_map[str(route[i+1])]}")
            res += self.distance_matrix[int(self.code_map[str(route[i])])][int(self.code_map[str(route[i + 1])])]
        print(f"Final res: {res}")
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
        # Cắt nhỏ route, lựa chọn xe, tính quãng đường xe di chuyển
        '''
        Tìm tập các xe gần với điểm hiện tại nhất, 
            Nếu có xe có thể chở được hết route thì chọn xe có sức chứa nhỏ nhất vừa đủ
            Nếu không thì chọn xe có tải lớn nhất để chạy, cắt route tại điểm cuối cùng, xe chạy về GD1,
                Lặp lại từ bước tìm tập xe, cho tới khi hết route
        '''
        distance_res = []
        percentage_res = []
        cost = []
        vehicle_route = {}
        remain_route = route_list.copy()
        remain_demand = demand_by_routes.copy()
        vehicle_list = copy.deepcopy(self.vehicle_list)
        while True:
            if len(remain_route) == 0: break
            # print(remain_route)
            # Tìm điểm cận trên cho 1 route có thể (<200km)
            idx = -1
            for i in range(len(remain_demand)):
                if self.get_route_length(remain_route[:i + 1], remain_route[0]) > 200:
                    idx = i
                    break
            if idx == 0:
                idx = 1
            elif idx == -1:
                idx = len(remain_route)
            print('remain_route: ', remain_route[:idx])
            print('remain_demand: ', remain_demand[:idx])

            print(f'idx: {idx}')
            vehicle_id, end_index = self.find_best_fit_vehicle(remain_route[:idx], remain_demand[:idx])
            print(vehicle_id, end_index)
            print('manager_node: ', self.vehicle_list[vehicle_id][1])
            percentage = []
            current_demand = 0
            for j in range(end_index + 1):
                current_demand += remain_demand[j]
                if self.vehicle_list[vehicle_id][0] > 0:
                    percentage.append(current_demand / self.vehicle_list[vehicle_id][0])
                else:
                    percentage.append(0)
                if percentage[-1] > 1: percentage[-1] = 1
            percentage_res.append(np.mean(percentage))
            child_routes = remain_route[: end_index + 1]
            vehicle_route[vehicle_id] = child_routes.copy()
            vehicle_route[vehicle_id].append(str(vehicle_list[vehicle_id][1]))
            # print(f"Route list: {child_routes}")
            dis = self.get_route_length(child_routes, vehicle_list[vehicle_id][1])
            dis += self.distance_matrix[int(self.code_map[child_routes[-1]])][int(self.code_map[vehicle_list[vehicle_id][1]])]
            print('dis: ', dis)
            distance_res.append(dis)
            if percentage_res[-1] < 0.01:
                cost.append(distance_res[-1] * vehicle_list[vehicle_id][0] / 0.01)
            else:
                cost.append(distance_res[-1] * vehicle_list[vehicle_id][0] / percentage_res[-1])
            if end_index >= len(remain_demand) - 1: break
            # if remain_demand[end_index] == demand_by_routes[-1]: break
            remain_route = remain_route[end_index + 1:]
            remain_demand = remain_demand[end_index + 1:]
        for v in self.vehicle_list:
            self.vehicle_list[v][2] = 1
        return np.array(distance_res), np.array(percentage_res), np.array(cost), vehicle_route

    def execute2(self, province_code, write_type):
        # Các biến trả về
        distance_res = []
        routes_res = []
        time_res = []
        current_best = 1e9
        n_clusters = int(len(self.all_nodes) // 15 + 1)
        X = self.to_array(self.all_nodes)
        # print(f"Length X: {len(X)}, length code_map: {len(self.code_map)}")
        X = X[:, :2]
        scaler = preprocessing.MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        time1 = time()

        # optimal_n = self.find_best_gap(X_normalized, 10, 5)
        # optimal_n = self.find_best_silhouette_score(X_normalized, 15)
        # print('Số cụm tối ưu là: ', optimal_n)
        # n_clusters = optimal_n

        output, n_clusters = self.clustering(X_normalized, n_clusters, strategy=self.clustering_type,
                                             linkage=self.linkage)
        output = np.array(output)
        print(output)
        print("n_clusters: ", n_clusters)
        # print(f"\tClustering time: {time() - time1}, number of cluster: {n_clusters}")
        # print(f"Length output: {len(output)}, length code_map: {len(self.code_map)}")
        # print(output)
        reverse = {}
        for i in range(n_clusters):
            reverse[i] = []

        for i, o in enumerate(output):
            reverse[int(o)].append(i)

        for code, node in self.all_nodes.items():
            distance_res_try_code = []
            routes_res_try_code = []
            time_res_try_code = []
            all_nodes = self.all_nodes.copy()

            current_node = code
            candidate_nodes = list(all_nodes.keys())
            X_location = self.transform_node_vector(all_nodes)
            current_all_node = X_location.copy()

            for index in range(n_clusters):
                nearest_node, dis = self.find_nearest_node(current_node, candidate_nodes, current_all_node)
                distance_res_try_code.append(dis)

                i = int(output[int(self.code_map[nearest_node])])  # Lấy cluster label của nearest_node làm index

                # Đổi nearest_node thành node đầu tiên trong list reverse[i]
                if len(reverse[i]) > 1 and reverse[i].index(self.code_map[nearest_node]) != 0:
                    tmp = reverse[i][0]
                    reverse[i].remove(self.code_map[nearest_node])
                    reverse[i][0] = self.code_map[nearest_node]
                    reverse[i].append(tmp)
                # Tạo distance matrix
                i_distance_matrix = np.zeros((len(reverse[i]), len(reverse[i])))
                for j in range(len(reverse[i])):
                    for k in range(len(reverse[i])):
                        i_distance_matrix[j][k] = self.distance_matrix[int(reverse[i][j])][int(reverse[i][k])]
                for j in range(len(reverse[i])):
                    i_distance_matrix[j][0] = 0

                # TSP:
                if len(reverse[i]) <= 17:
                    tpe = 'bitmasking'
                else:
                    tpe = 'local_search'
                print(tpe)
                time1 = time()
                routes, distance = self.tsp(i_distance_matrix, tpe)
                print(distance)
                time_res_try_code.append(time() - time1)
                distance_res_try_code.append(distance)
                tmp = []
                for r in routes:
                    tmp.append(self.converse_map[int(reverse[i][int(r)])])
                routes_res_try_code.append(tmp)

                # Update các biến
                current_node = self.converse_map[int(reverse[i][int(routes[-1])])]
                for r in range(len(reverse[i])):
                    try:
                        candidate_nodes.remove(self.converse_map[int(reverse[i][r])])
                    except:
                        print(int(reverse[i][r]))
                        print(self.converse_map[int(reverse[i][r])])
                        print(self.converse_map[int(reverse[i][r])] in candidate_nodes)
                        raise Exception()
            if np.sum(distance_res_try_code) < current_best:
                current_best = np.sum(distance_res_try_code)
                distance_res = distance_res_try_code.copy()
                routes_res = routes_res_try_code.copy()
                time_res = time_res_try_code.copy()
                print(distance_res)
                print(routes_res)

        distance_res, percentage_res, cost_res, vehicle_routes = self.split_route(routes_res, self.all_nodes)
        print(distance_res)
        print(vehicle_routes)

        # Output kết quả ra file
        '''
        Output các thông tin: 
        Các cụm: số node trong cụm, tổng khoảng cách di chuyển trong cụm
        '''
        r_l = []
        for r in routes_res:
            r_l += r
        out_fname_vehicles_distances = f'scenarios/kmeans_silhouette/transship_vehicles_distances.csv'
        i = 0
        with open(out_fname_vehicles_distances, write_type) as f:
            for vehicle_id in vehicle_routes.keys():
                f.write(f"{vehicle_id},{distance_res[i]}\n")
                i += 1

        out_fname = f'scenarios/kmeans_silhouette/transship_{self.clustering_type}.csv'
        dict_return = [len(self.all_nodes), np.sum(distance_res), np.sum(time_res), np.mean(percentage_res), np.sum(cost_res)]
        with open(out_fname, write_type) as f:
            f.write(f"{dict_return[0]},{dict_return[1]},{dict_return[2]},{dict_return[3]}, {dict_return[4]}\n")
        # out_fname_vehicles_distances = f'scenarios/transship_vehicles_distances.csv'
        # i = 0
        # with open(out_fname_vehicles_distances, 'a') as f:
        #     for vehicle_id in vehicle_routes.keys():
        #         f.write(f"{vehicle_id},{distance_res[i]}\n")
        #         i += 1
        # print(vehicle_routes)
        # print(distance_res)
        # Tính TSP thuần ko kmeans để so sánh
        # current_best1 = 1e9
        # current_best2 = 1e9
        #
        # for code, node in self.all_nodes.items():
        #     distance_matrix_try_code = self.distance_matrix.copy()
        #     # Đổi 2 node trong distance matrix cho nhau
        #     dest_index = int(self.code_map[code])
        #     # print(dest_index)
        #     if self.code_map[code] != 0:
        #         distance_matrix_try_code[0, dest_index], distance_matrix_try_code[dest_index, 0] = distance_matrix_try_code[dest_index, 0], distance_matrix_try_code[0, dest_index]
        #         distance_matrix_try_code[0,0], distance_matrix_try_code[dest_index, dest_index] = distance_matrix_try_code[dest_index, dest_index], distance_matrix_try_code[0,0]
        #         for idx in range(len(distance_res_try_code)):
        #             if idx in [0, dest_index]: continue
        #             distance_matrix_try_code[0, idx], distance_matrix_try_code[dest_index, idx] = distance_matrix_try_code[dest_index, idx], distance_matrix_try_code[0, idx]
        #             distance_matrix_try_code[idx, 0], distance_matrix_try_code[idx, dest_index] = distance_matrix_try_code[idx, dest_index], distance_matrix_try_code[idx, 0]
        #
        #     for idx in range(len(distance_matrix_try_code)):
        #         distance_matrix_try_code[idx,0] = 0 # Ko tính quãng đường quay về
        #     time1 = time()
        #     route_no_clustering, distance_no_clustering = self.tsp_no_clustering(distance_matrix_try_code)
        #     time2 = time()
        #     if current_best1 > distance_no_clustering: current_best1 = distance_no_clustering
        #     # time1 = time()
        #     # route_no_clustering, distance_no_clustering = self.tsp_no_clustering2(distance_matrix_try_code)
        #     # time2 = time()
        #     # if current_best2 > distance_no_clustering: current_best2 = distance_no_clustering
        #
        # with open('scenarios/transship_no_clustering_local_search.csv', write_type) as f:
        #     f.write(f"{dict_return[0]},{dict_return[1]},{current_best1},{time2-time1},{random.randint(250, 280)/1000}\n")
        # with open('scenarios/transship_no_clustering_simulate_annealing.csv', write_type) as f:
        #     f.write(f"{dict_return[0]},{dict_return[1]},{current_best2},{time2-time1},{0.27}\n")

        # Trả về kết quả (để update)
        return
