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

        self.o = self.m.addVar(vtype=GRB.CONTINUOUS)

        obj = quicksum(
            self.f[j] * self.x[j] + self.h[j] * self.y[j] for j in self.J_set) + self.o

        self.m.setObjective(obj, GRB.MINIMIZE)

        self.m.addConstr(quicksum(self.f[j] * self.x[j] + self.h[j] * self.y[j] for j in self.J_set) <= self.Budget,
                         name='budget')
        self.m.addConstrs(self.x[j] >= self.y[j] for j in self.J_set)

    def solveMaster(self):
        self.m.optimize()

    def get_decision(self):
        x_val = {key: value.x for key, value in self.x.items()}
        y_val = {key: value.x for key, value in self.y.items()}
        return x_val, y_val

    def add_valid_cut(self, xi_hat):
        b = self.m.addVars(self.I_set, self.J0, self.N_set, self.J0, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        for j in self.J_set:
            for n in self.N_set:
                lhs = quicksum(b[i, j, n, t] * xi_hat[t, n] for i in self.I_set for t in self.J0)
                rhs = (self.Cap[j] * self.x[j] + self.Cap_bar[j] * self.y[j]) * xi_hat[j, n]
                self.m.addConstr(lhs <= rhs)

        for i in self.I_set:
            for n in self.N_set:
                lhs = quicksum(b[i, j, n, t] * xi_hat[t, n] for j in self.J0 for t in self.J0)
                rhs = self.demand[i]
                self.m.addConstr(lhs >= rhs)

        for i in self.I_set:
            for j in self.J0:
                for n in self.N_set:
                    self.m.addConstr(quicksum(b[i, j, n, t] * xi_hat[t, n] for t in self.J0) >= 0)

    def add_optimality_cut(self, R_hat, P_hat):
        obj1 = quicksum(R_hat[i, n] * self.demand[i] for i in self.I_set for n in self.N_set)
        obj2 = quicksum(
            P_hat[j, n, t] * (self.Cap[j] * self.x[j] + self.Cap_bar[j] * self.y[j])
            for j in self.J_set for n in self.N_set for t in self.J0 if t == j
        )
        self.m.addConstr(self.o >= obj1 - obj2)



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

    def construtModel(self, x_val, y_val):

        self.m = Model()

        R = {(i, n): self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"R_{i}_{n}")
             for i in self.I_set for n in self.N_set}

        P = {(j, n, t): self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"P_{j}_{n}_{t}")
             for j in self.J_set for n in self.N_set for t in self.J0}

        Q = {(j, n): self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Q_{j}_{n}")
             for j in self.J_set for n in self.N_set}

        A = {(i, n, t): self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"A_{i}_{n}_{t}")
             for i in self.I_set for n in self.N_set for t in self.J0}

        H = {(i, j, n, t): self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"H_{i}_{j}_{n}_{t}")
             for i in self.I_set for j in self.J0 for n in self.N_set for t in self.J0}

        U = {(i, j, n): self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{i}_{j}_{n}")
             for i in self.I_set for j in self.J0 for n in self.N_set}

        # tau = {(j, n): self.m.addVar(vtype=GRB.BINARY, name=f"tau_{j}_{n}")
        #        for j in self.J0 for n in self.N_set}

        tau = {(j, n): self.m.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f"tau_{j}_{n}")
               for j in self.J0 for n in self.N_set}

        # xi = {(t, n): self.m.addVar(vtype=GRB.BINARY, name=f"xi_{t}_{n}")
        #       for t in self.J0 for n in self.N_set}

        xi = {(t, n): self.m.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f"xi_{t}_{n}")
              for t in self.J0 for n in self.N_set}

        obj1 = quicksum(R[i, n] * self.demand[i] for i in self.I_set for n in self.N_set)
        obj2 = quicksum(
            P[j, n, t] * (self.Cap[j] * x_val[j] + self.Cap_bar[j] * y_val[j])
            for j in self.J_set for n in self.N_set for t in self.J0 if t == j
        )
        self.m.setObjective(obj1 - obj2, GRB.MAXIMIZE)

        self.m.addConstr(self.epsilon - (1 / self.N_num) * quicksum(tau[j, n] for j in self.J0 for n in self.N_set) >= 0)


        for n in self.N_set:
            for j in self.J0:
                if j == 0:
                    self.m.addConstr(xi[j, n] == 1)
                self.m.addConstr(tau[j, n] >= xi[j, n] - self.xi_hat[j, n])
                self.m.addConstr(tau[j, n] >= self.xi_hat[j, n] - xi[j, n])
                self.m.addConstr(tau[j, n] <= 1)


        for i in self.I_set:
            for j in self.J0:
                for n in self.N_set:
                    for t in self.J0:
                        lhs = (1 / self.N_num) * self.c[i, j] * xi[t, n] + (P[j, n, t] if j in self.J_set else 0)
                        self.m.addConstr(lhs - A[i, n, t] - H[i, j, n, t] == 0)

        for j in self.J_set:
            for n in self.N_set:
                for t in self.J0:
                    self.m.addConstr(Q[j, n] - P[j, n, t] >= 0)
                self.m.addConstr(P[j, n, 0] - Q[j, n] >= 0)

        for i in self.I_set:
            for n in self.N_set:
                for t in self.J0:
                    self.m.addConstr(R[i, n] - A[i, n, t] >= 0)
                self.m.addConstr(A[i, n, 0] - R[i, n] >= 0)

        for i in self.I_set:
            for j in self.J0:
                for n in self.N_set:
                    for t in self.J0:
                        self.m.addConstr(U[i, j, n] - H[i, j, n, t] >= 0)
                    self.m.addConstr(H[i, j, n, 0] - U[i, j, n] >= 0)

        self.R = R
        self.P = P
        self.xi = xi

    def update_model(self, x_val, y_val):
        obj1 = quicksum(self.R[i, n] * self.demand[i] for i in self.I_set for n in self.N_set)
        obj2 = quicksum(
            self.P[j, n, t] * (self.Cap[j] * x_val[j] + self.Cap_bar[j] * y_val[j])
            for j in self.J_set for n in self.N_set for t in self.J0 if t == j
        )
        self.m.setObjective(obj1 - obj2, GRB.MAXIMIZE)

    def solveSub(self):
        self.m.optimize()

    def get_decision(self):
        R_hat = {key: value.x for key, value in self.R.items()}
        P_hat = {key: value.x for key, value in self.P.items()}

        return R_hat, P_hat

    def get_scenario(self):
        xi_hat = {key: value.x for key, value in self.xi.items()}
        return xi_hat

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

    SP = SubProblem(params)
    SP.construtModel(x_val, y_val)
    SP.solveSub()

    R_hat, P_hat = SP.get_decision()
    MP.add_optimality_cut(R_hat, P_hat)

    MP.solveMaster()

    UB = min(UB, MP.m.objVal - MP.o.x + SP.m.objVal)
    LB = max(LB, MP.m.objVal)

    accuracy = 0.01
    x_val, y_val = MP.get_decision()
    SP = SubProblem(params)
    SP.construtModel(x_val, y_val)

    while abs(UB - LB) / UB > accuracy:
        SP.solveSub()
        R_hat, P_hat = SP.get_decision()
        xi_hat = SP.get_scenario()
        UB = min(UB, MP.m.objVal - MP.o.x + SP.m.objVal)
        MP.add_optimality_cut(R_hat, P_hat)
        MP.add_valid_cut(xi_hat)

        MP.solveMaster()
        LB = max(LB, MP.m.objVal)
        x_val, y_val = MP.get_decision()

        SP.update_model(x_val, y_val)
        print([UB, LB])




