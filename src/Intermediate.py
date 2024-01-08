import pickle

import numpy as np
import pandas as pd
import sys
sys.path.append("")
from BaseClass.Node import Node, NodeController
from BaseClass.Order import Order, OrderController
from BaseClass.Vehicle import Vehicle, VehicleController
from BaseClass.Correlation import Correlation, CorrelationController

class Intermediate:
    def __init__(self, vehicle_fname, node_fname, correlation_fname, order_fname) -> None:
        self.vehicle_file_name = vehicle_fname
        self.node_file_name = node_fname
        self.correlation_file_name = correlation_fname
        self.order_file_name = order_fname
    
    def execute(self):
        vehicle_controller, node_controller, order_controller, correlation_controller = self.load_data()
        # with open('data.pkl', 'wb') as file:
        #     data = (vehicle_controller, node_controller, order_controller, correlation_controller)
        #     pickle.dump(data, file)
        # with open('data.pkl', 'rb') as file:
        #     data = pickle.load(file)
        #     vehicle_controller, node_controller, order_controller, correlation_controller = data
        # print('ok')
        self.update_province_map(node_controller)
        node_sort_by_province = self.hierarchical_by_province(node_controller)
        vehicle_sort_by_province = self.hierarchical_by_province(vehicle_controller)
        self.province_parent_code = self.get_province_parent_node(node_controller)
        order_controller = self.get_order_routes(order_controller)
        order_array = self.to_array(order_controller)
        array: dict[str, dict] = {'node': {}, 'vehicle': {}, 'correlation': {}}
        for province in node_sort_by_province:
            array['node'][province] = {}
            for tpe in node_sort_by_province[province]:
                array['node'][province][tpe] = self.to_array(node_sort_by_province[province][tpe])
        for province in vehicle_sort_by_province:
            array['vehicle'][province] = self.to_array(vehicle_sort_by_province[province])
        for province in node_sort_by_province:
            node_list = []
            for tpe in array['node'][province]:
                node_list += list(array['node'][province][tpe].keys())
            array['correlation'][province] = self.get_distance_matrix(correlation_controller, node_list)
        parent_node_list = []
        for province_code in node_sort_by_province:
            parent_node_list += array['node'][province_code]['GD1'].keys()
        array['correlation']['parent'] = self.get_distance_matrix(correlation_controller, parent_node_list)
        return array['node'], array['vehicle'], order_array, array['correlation']

    def load_data(self):
        vehicle_controller = self._load_vehicle()
        node_controller = self._load_node()
        order_controller = self._load_order()
        correlation_controller = self._load_correlation()
        return vehicle_controller, node_controller, order_controller, correlation_controller

    def correlation_to_array(self, controller: CorrelationController) -> dict[str, float]:
        res: dict[str, float] = {}
        for correlation in list(controller.get_correlation_dict().values()):
            res[correlation.from_node_code+'-'+correlation.to_node_code] = correlation.distance
        return res

    def to_array(self, controller: VehicleController|NodeController|OrderController) -> dict[str, list]:
        '''
        Convert từ class về array
        Với Vehicle: [max_capacity, manager_node, available]
        Với Node: [lat, long]
        Với Order: [sender_2, sender_1, receiver_1, receiver_2, weight]
        '''
        res: dict[str, list] = {}
        if isinstance(controller, NodeController):
            for node in list(controller.get_node_dict().values()):
                res[node.code] = [node.latitude, node.longitude, node.start_time, node.end_time]
            return res
        if isinstance(controller, VehicleController):
            for vehicle in list(controller.get_vehicle_dict().values()):
                res[vehicle.id] = [vehicle.max_capacity, vehicle.manager_node, vehicle.available]
            return res
        if isinstance(controller, OrderController):
            for order in list(controller.get_order_dict().values()):
                res[order.code] = []
                if 1 in order.phase_list:
                    res[order.code].append(order.state[0])
                    res[order.code].append(order.state[1])
                    res[order.code].append(order.state[2])
                else:
                    res[order.code].append(-1)
                    res[order.code].append(order.state[0])
                    res[order.code].append(order.state[1])

                if 3 in order.phase_list:
                    res[order.code].append(order.state[-1])
                else: res[order.code].append(-1)
                res[order.code].append(order.weight)
            return res

    def get_node_demand(self, type: str, order_array: dict[str, list]) -> dict[str, float]:
        res: dict[str, float] = {}
        if type == 'pickup': start, end = 0,1
        elif type == 'transship': start, end = 1,2
        elif type == 'delivery': start, end = 3,2
        for code, order in order_array.items():
            if order[start] == -1: continue
            if order[end] == -1: continue
            if order[start] not in res: res[order[start]] = 0
            res[order[start]] += order[4]
        return res

    def update_node_array(self, type, node_array: dict, weight_update: dict[str, float]) -> dict[str, list]:
        if len(list(list(list(node_array.values())[0].values())[0].values())[0]) == 4:
            for province_code in node_array:
                for tpe in node_array[province_code]:
                    for code in node_array[province_code][tpe]:
                        node_array[province_code][tpe][code].append(0)

        for province_code in node_array:
            for tpe in node_array[province_code]:
                for code in node_array[province_code][tpe]:
                    if code not in weight_update: node_array[province_code][tpe][code][-1] = 0
                    else:
                        node_array[province_code][tpe][code][-1] = weight_update[code]
        return node_array

    def get_to_go_node(self, all_node_array: dict):
        res = {}
        for province_code in all_node_array:
            res[province_code] = {}
            for tpe in all_node_array[province_code]:
                res[province_code][tpe] = {}
                for code, array in all_node_array[province_code][tpe].items():
                    if array[-1] > 0: res[province_code][tpe][code] = array
                if len(res[province_code][tpe]) == 0: del res[province_code][tpe]
            if len(res[province_code]) == 0: del res[province_code]
        return res

    def get_to_go_distance(self, default_node_array: dict, to_go_node_array: dict, default_distance_matrix):
        white_list = []
        step = 0
        for i in range(len(default_distance_matrix) - len(default_node_array)):
            white_list.append(i)
        default_code = list(default_node_array.keys())
        to_go_code = list(to_go_node_array.keys())
        for i, code in enumerate(default_code):
            if code not in to_go_code:
                step+=1
                continue
            white_list.append(i+step)

        to_go_distance_matrix = np.zeros((len(white_list), len(white_list)))
        for i in range(len(white_list)):
            for j in range(len(white_list)):
                to_go_distance_matrix[i][j] = default_distance_matrix[int(white_list[i])][int(white_list[j])]
        return to_go_distance_matrix


    def get_distance_matrix(self, all_correlation: dict[str, float], node_list: list[str]):
        res = np.zeros((len(node_list), len(node_list)))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                key = str(int(node_list[i])) + '-' + str(int(node_list[j]))
                if key not in all_correlation: res[i][j] = 1e9
                else: res[i][j] = all_correlation[key]
        return res

    def get_province_parent_node(self, all_node: NodeController):
        province_parent_node = {}
        for code, node in all_node.get_node_dict().items():
            if node.type == 'GD1': province_parent_node[node.province_code] = node.code
        return province_parent_node

    def get_order_routes(self, all_order: OrderController) -> OrderController:
        result_order_controller = OrderController()
        for code, order in all_order.get_order_dict().items():
            start_node = order.depot_id
            end_node = order.customer_id
            # order.update_state(start_node)
            sender_parent_node = self.province_parent_code[self.province_map[start_node]]
            if not start_node == sender_parent_node:
                order.update_state(sender_parent_node, 1)

            receiver_parent_node = self.province_parent_code[self.province_map[end_node]]
            order.update_state(receiver_parent_node, 2)

            if not end_node == receiver_parent_node: order.update_state(end_node, 3)
            result_order_controller.add(order)
        return result_order_controller

    def hierarchical_by_province(self, controller: VehicleController|NodeController|OrderController|CorrelationController):
        res = {}
        if isinstance(controller, NodeController):
            for node in list(controller.get_node_dict().values()):
                if node.province_code not in res: res[node.province_code] = {}
                if node.type not in res[node.province_code]: res[node.province_code][node.type] = NodeController()
                res[node.province_code][node.type].add(node)
            return res
        if isinstance(controller, VehicleController):
            for v in list(controller.get_vehicle_dict().values()):
                if self.province_map[v.manager_node] not in res: res[self.province_map[v.manager_node]] = VehicleController()
                res[self.province_map[v.manager_node]].add(v)
            return res

    def update_province_map(self, node_controller: NodeController):
        if not hasattr(self, 'province_map'): self.province_map = {}

        for node in list(node_controller.get_node_dict().values()):
            self.province_map[node.code] = node.province_code
        return

    def _load_vehicle(self) -> VehicleController:
        print('\tLoad thông tin về đội xe: start ')

        data = pd.read_csv(self.vehicle_file_name, sep='\t')
        vehicle_controller = VehicleController()
        for i in range(len(data)):
            line = data.iloc[i]
            vehicle_controller.add(Vehicle(int(line['code']), line['created_at'], line['updated_at'],
                                           line['available'], float(line['average_fee_transport']),
                                           float(line['average_gas_consume']), float(line['average_velocity']),
                                           line['driver_name'],
                                           line['gas_price'], line['height'],
                                           line['length'], line['max_capacity'],
                                           line['max_load_weight'], line['max_velocity'],
                                           line['min_velocity'], line['name'], line['type'],
                                           line['width'], line['vehicle_cost'],
                                           str(line['manager_node']),str(line['current_node'])))
        print('\tLoad thông tin về đội xe: done')
        return vehicle_controller

    def _load_node(self) -> NodeController:
        print('\tLoad thông tin các node: start')
        data = pd.read_csv(self.node_file_name, sep='\t')
        node_controller = NodeController()
        for i in range(len(data)):
            line = data.iloc[i]
            if line['type'] in ['GD2', 'GD3']: type = 'GD2'
            else: type = 'GD1'
            node_controller.add(Node(line['created_at'], line['updated_at'], str(line['address']),
                                     str(line['code']), int(line['end_time']), float(line['latitude']),
                                     str(line['longitude']), str(line['name']), int(line['start_time']),
                                     type, line['capacity'], int(line['province_code']), int(line['district_code'])))

        print('\tLoad thông tin các node: done')
        return node_controller

    def _load_order(self) -> OrderController:
        print('\tLoad thông tin về các đơn hàng: start')
        data = pd.read_csv(self.order_file_name, sep='\t')
        order_controller = OrderController()
        for i in range(len(data)):
            line = data.iloc[i]
            if float(line['weight']) == 0: continue
            order_controller.add(Order(line['created_at'], line['updated_at'],
                                       line['capacity'], str(line['code']), int(line['delivery_after_time']),
                                       int(line['delivery_before_time']), line['delivery_mode'],
                                       line['order_value'], line['time_service'], line['time_loading'],
                                       line['weight'], str(line['customer_id']), str(line['depot_id']) ))

        print('\tLoad thông tin về các đơn hàng: done')
        return order_controller

    def _load_correlation(self) -> CorrelationController:
        print('\tLoad ma trận khoảng cách: start')
        data = pd.read_csv(self.correlation_file_name, sep='\t')
        correlation_controller: dict[str, float] = {}
        for i in range(len(data)):
            correlation_controller[str(int(float(data.iloc[i]['from_node_code'])))+'-'+str(int(float(data.iloc[i]['to_node_code'])))] = data.iloc[i]['distance']
        return correlation_controller
        correlation_controller = CorrelationController()
        for i in range(len(data)):
            line = data.iloc[i]
            corr = Correlation(line['id'], line['created_at'], line['updated_at'],
                                                   line['distance'], str(int(float(line['from_node_code']))), line['from_node_id'],
                                                   line['from_node_name'], line['from_node_type'], line['risk_probability'],
                                                   line['time'], str(int(float(line['to_node_code']))),
                                                   line['to_node_id'], line['to_node_name'],
                                                   line['to_node_type'])
            correlation_controller.add(corr)
        print('\tLoad ma trận khoảng cách: done')
        return correlation_controller
