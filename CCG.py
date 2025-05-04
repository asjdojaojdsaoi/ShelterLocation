from gurobipy import *
from config import *
from affine_decision_rule import *
from RO import *

class MasterProblem:
    def __init__(self, parameters):
        self.I_num = parameters.I_num
        self.J_num = parameters.J_num
        self.N_num = parameters.N_num
        self.I_set = parameters.I_set
        self.J_set = parameters.J_set
        self.N_set = parameters.N_set
        self.J0 = parameters.J0
        self.f = parameters.f
        self.h = parameters.h
        self.Cap = parameters.Cap
        self.demand = parameters.demand
        self.Cap_bar = parameters.Cap_bar
        self.c = parameters.c
        self.N_num = parameters.N_num
        self.xi = parameters.xi
        self.epsilon = parameters.epsilon
        self.Budget = parameters.B

    def construtModel(self):
        self.m = Model()

        self.x = {j: self.m.addVar(vtype=GRB.BINARY) for j in self.J_set}
        self.y = {j: self.m.addVar(vtype=GRB.BINARY) for j in self.J_set}

        self.lambda_ = self.m.addVar(vtype=GRB.CONTINUOUS)

        self.s = {n: self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_{n}") for n in self.N_set}

        obj = quicksum(
            self.f[j] * self.x[j] + self.h[j] * self.y[j] for j in self.J_set) + self.lambda_ * self.epsilon + (1 / self.N_num) * quicksum(self.s[n] for n in self.N_set)

        self.m.setObjective(obj, GRB.MINIMIZE)

        self.m.addConstr(quicksum(self.f[j] * self.x[j] + self.h[j] * self.y[j] for j in self.J_set) <= self.Budget,
                         name='budget')
        self.m.addConstrs(self.x[j] >= self.y[j] for j in self.J_set)

    def solveMaster(self):
        self.m.optimize()

    def get_decision(self):
        x_val = {key: value.x for key, value in self.x.items()}
        y_val = {key: value.x for key, value in self.y.items()}
        lambda_val = self.lambda_.x
        return x_val, y_val, lambda_val

    def add_optimality_cut(self, xi_hat, tau_hat):
        z = self.m.addVars(self.I_set, self.J0, self.N_set, vtype=GRB.CONTINUOUS)

        for n in self.N_set:
            self.m.addConstr(self.s[n] >= quicksum(self.c[i, j] * z[i, j, n] for i in self.I_set for j in self.J0) - quicksum(self.lambda_ * tau_hat[j, n] for j in self.J0))

        for n in self.N_set:
            for j in self.J_set:
                self.m.addConstr(quicksum(z[i, j, n] for i in self.I_set) <= (self.Cap[j] * self.x[j] + self.Cap_bar[j] * self.y[j]) * xi_hat[j, n])

            for i in self.I_set:
                self.m.addConstr(quicksum([z[i, j, n] for j in self.J0]) >= self.demand[i])

class SubProblem:
    def __init__(self, parameters):
        self.I_num = parameters.I_num
        self.J_num = parameters.J_num
        self.N_num = parameters.N_num
        self.I_set = parameters.I_set
        self.J_set = parameters.J_set
        self.N_set = parameters.N_set
        self.J0 = parameters.J0
        self.f = parameters.f
        self.h = parameters.h
        self.Cap = parameters.Cap
        self.demand = parameters.demand
        self.Cap_bar = parameters.Cap_bar
        self.c = parameters.c
        self.N_num = parameters.N_num
        self.xi_hat = parameters.xi
        self.epsilon = parameters.epsilon
        self.Budget = parameters.B
        self.M = 999999999

        self.m = None

    def construtModel(self, x_val, y_val, lambda_hat, n):

        self.m = Model()

        self.xi = self.m.addVars(self.J0, vtype=GRB.BINARY)
        self.tau = self.m.addVars(self.J0, ub=1, vtype=GRB.CONTINUOUS)
        self.mu = self.m.addVars(self.J_set)
        self.nu = self.m.addVars(self.I_set)
        self.mu_prime = self.m.addVars(self.J_set)

        obj = quicksum(self.demand[i] * self.nu[i] for i in self.I_set)
        obj -= quicksum((self.Cap[j] * x_val[j] + self.Cap_bar[j] * y_val[j]) * self.mu_prime[j] for j in self.J_set)
        obj -= quicksum(lambda_hat * self.tau[j] for j in self.J0)

        self.m.setObjective(obj, GRB.MAXIMIZE)

        for i in self.I_set:
            for j in self.J0:
                if j == 0:
                    self.m.addConstr(self.c[i, j] - self.nu[i] >= 0)
                else:
                    self.m.addConstr(self.c[i, j] + self.mu[j] - self.nu[i] >= 0)

        for j in self.J_set:
            self.m.addConstr(self.mu_prime[j] >= self.mu[j] - self.M * (1 - self.xi[j]))
            self.m.addConstr(self.mu_prime[j] <= self.M * self.xi[j])
            self.m.addConstr(self.mu_prime[j] <= self.mu[j])

        self.support_set_constr = {}

        for j in self.J0:
            if j == 0:
                self.m.addConstr(self.xi[j] == 1)
            else:
                self.support_set_constr[j, 1] = self.m.addConstr(self.tau[j] - self.xi[j]>= -self.xi_hat[j, n])
                self.support_set_constr[j, 2] = self.m.addConstr(self.tau[j] + self.xi[j] >= self.xi_hat[j, n])

    def solve_Subproblem(self):
        self.m.optimize()

    def get_decision(self):
        xi_hat = {key: value.x for key, value in self.xi.items()}
        tau_hat = {key: value.x for key, value in self.tau.items()}
        return xi_hat, tau_hat

    def update_model(self, x_val, y_val, lambda_hat, n):
        obj = quicksum(self.demand[i] * self.nu[i] for i in self.I_set)
        obj -= quicksum((self.Cap[j] * x_val[j] + self.Cap_bar[j] * y_val[j]) * self.mu_prime[j] for j in self.J_set)
        obj -= quicksum(lambda_hat * self.tau[j] for j in self.J0)

        self.m.setObjective(obj, GRB.MAXIMIZE)

        for j in self.J0:
            if j > 0:
                self.support_set_constr[j, 1].rhs = -self.xi_hat[j, n]
                self.support_set_constr[j, 2].rhs = self.xi_hat[j, n]

if __name__ == '__main__':
    # self-defined
    data_path = None
    I_num = None
    J_num = None
    N_num = None
    c_cor = None
    Cap_min = None
    Cap_max = None
    expand_cor = None
    f_min = None
    f_max = None
    h_min = None
    h_max = None
    total_demand = None
    demand_min = None
    demand_max = None
    punish_c = None
    epsilon = None
    Budget = None

    params = Params(data_path=data_path, I_num=I_num, J_num=J_num, N_num=N_num, c_cor=c_cor, Cap_max=Cap_max,
                    Cap_min=Cap_min,
                    expand_cor=expand_cor, f_min=f_min, f_max=f_max, h_min=h_min, h_max=h_max, epsilon=epsilon,
                    total_demand=total_demand, demand_max=demand_max, demand_min=demand_min, punish_c=punish_c,
                    Budget=Budget)

    UB = 99999999
    LB = -99999999

    MP = MasterProblem(params)
    MP.construtModel()

    # initial
    ro = RobustOptimization(params)
    ro.constructRO()
    ro.solve_RO()
    x_val, y_val = ro.get_decision()
    lambda_val = 0

    SP = SubProblem(params)

    xi_hat, tau_hat, sp_val = {}, {}, {}

    for n in params.N_set:
        if SP.m is None:
            SP.construtModel(x_val, y_val, lambda_val, n)
        else:
            SP.update_model(x_val, y_val, lambda_val, n)
        SP.solve_Subproblem()
        xi_hat_n, tau_hat_n = SP.get_decision()
        sp_val[n] = SP.m.objVal
        for j in params.J0:
            xi_hat[j, n] = xi_hat_n[j]
            tau_hat[j, n] = tau_hat_n[j]

    MP.add_optimality_cut(xi_hat, tau_hat)

    MP.solveMaster()

    UB = min(UB, MP.m.objVal - (1 / params.N_num) * sum([MP.s[n].x for n in params.N_set]) + (1 / params.N_num) * sum([sp_val[n] for n in params.N_set]))
    LB = max(LB, MP.m.objVal)

    accuracy = 0.01
    x_val, y_val, lambda_val = MP.get_decision()

    while abs(UB - LB) / UB > accuracy:
        for n in params.N_set:
            SP.update_model(x_val, y_val, lambda_val, n)
            SP.solve_Subproblem()
            xi_hat_n, tau_hat_n = SP.get_decision()
            sp_val[n] = SP.m.objVal
            for j in params.J0:
                xi_hat[j, n] = xi_hat_n[j]
                tau_hat[j, n] = tau_hat_n[j]

        MP.add_optimality_cut(xi_hat, tau_hat)

        MP.solveMaster()

        UB = min(UB,
                 MP.m.objVal - (1 / params.N_num) * sum([MP.s[n].x for n in params.N_set]) + (1 / params.N_num) * sum(
                     [sp_val[n] for n in params.N_set]))
        LB = max(LB, MP.m.objVal)

        x_val, y_val, lambda_val = MP.get_decision()
        print('bound: ', [UB, LB])



