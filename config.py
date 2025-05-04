import numpy as np
import pandas as pd
from geopy.distance import geodesic

class Params:
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.get_map_size()
        self.generate_nodes()
        self.generate_transportation_cost()
        self.generate_capacity()
        self.generate_build_cost()
        self.generate_demand()
        self.generate_xi()

        self.B =self.parameters['Budget']
        self.epsilon = self.parameters['epsilon']



    def get_map_size(self):
        # determine the regional size according to a data file containing latitude and longitude
        path = self.parameters['data_path']
        data = pd.read_excel(path)
        self.min_lat = data['latitude'].min()
        self.min_lon = data['longitude'].min()
        self.max_lat = data['latitude'].max()
        self.max_lon = data['longitude'].max()

    def generate_nodes(self):
        self.I_num = self.parameters['I_num']
        self.J_num = self.parameters['J_num']
        self.I_set = range(1, self.I_num + 1)
        self.J_set = range(1, self.J_num + 1)
        self.J0 = range(self.J_num + 1)
        self.geo_shelter_info = {}
        self.geo_demand_info = {}
        for i in self.I_set:
            self.geo_demand_info[i] = [np.random.uniform(self.min_lon, self.max_lon), np.random.uniform(self.min_lat, self.max_lat)]
        for j in self.J_set:
            self.geo_shelter_info[j] = [np.random.uniform(self.min_lon, self.max_lon), np.random.uniform(self.min_lat, self.max_lat)]

    def generate_transportation_cost(self):
        # transportation cost
        self.distance = {}
        self.c = {}
        self.c_cor = self.parameters['c_cor']
        for i in self.I_set:
            for j in self.J_set:
                self.distance[i, j] = geodesic(self.geo_shelter_info[j], self.geo_demand_info[i]).km
                self.c[i, j] = self.distance[i, j] * self.c_cor

        self.punish_c_bar = max(list(self.c.values()))
        for i in self.I_set:
            self.c[i, 0] = self.punish_c_bar * self.parameters['punish_c']

    def generate_capacity(self):
        self.Cap = {}
        self.Cap_min = self.parameters['Cap_min']
        self.Cap_max = self.parameters['Cap_max']

        self.Cap_bar = {}
        self.expand_cor = self.parameters['expand_cor']

        for j in self.J_set:
            self.Cap[j] = np.random.randint(self.Cap_min, self.Cap_max + 1)
            self.Cap_bar[j] = self.Cap[j] * self.expand_cor

    def generate_build_cost(self):
        self.f = {}
        self.h = {}
        self.f_min = self.parameters['f_min']
        self.f_max = self.parameters['f_max']
        self.h_min = self.parameters['h_min']
        self.h_max = self.parameters['h_max']
        for j in self.J_set:
            self.f[j] = np.random.uniform(self.f_min, self.f_max) * self.Cap[j]
            self.h[j] = np.random.uniform(self.h_min, self.h_max) * self.Cap_bar[j]

    def generate_demand(self):
        self.demand = {}
        self.total_demand = self.parameters['total_demand']
        self.demand_min = self.parameters['demand_min']
        self.demand_max = self.parameters['demand_max']
        for i in self.I_set:
            self.demand[i] = np.random.uniform(self.demand_min, self.demand_max) * self.total_demand / self.I_num

    def generate_xi(self):
        self.N_num = self.parameters['N_num']
        self.N_set = range(1, self.N_num + 1)
        self.p = {j: np.random.uniform(0.08, 0.12) for j in self.J_set}
        self.xi = {(j, n): 1 if np.random.random() < self.p[j] else 0 for j in self.J_set for n in self.N_set}

        for n in self.N_set:
            self.xi[0, n] = 1