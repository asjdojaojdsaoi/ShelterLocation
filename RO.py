from gurobipy import *
from config import *

class RobustOptimization:
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

        self.R_set = [r for r in range(1, 2 * (self.J_num + 1) + 1)]
        self.U = {}
        self.v = {}
        for r in self.R_set:

            if r <= self.J_num + 1:
                self.v[r] = 1
            elif self.J_num + 1 < r < self.R_set[-1]:
                self.v[r] = 0
            else:
                self.v[r] = -1

            for j in self.J0:
                if r <= self.J_num + 1:
                    self.U[r, j] = 1
                else:
                    self.U[r, j] = -1

    def constructRO(self):

        model = Model()
        x = {j: model.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in self.J_set}
        y = {j: model.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in self.J_set}

        s = {n: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_{n}") for n in self.N_set}
        gamma = {(r, n): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"gamma_{r}_{n}")
                 for r in self.R_set for n in self.N_set}
        lambda_ = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"lambda_")

        b = {(i, j, j_prime): model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in self.I_set for j in self.J0 for j_prime in self.J0}

        beta_2 = {j: model.addVar() for j in self.J_set}
        beta_4 = {i: model.addVar() for i in self.I_set}
        beta_5 = {(i, j): model.addVar() for i in self.I_set for j in self.J0}

        alpha_2 = {(j, j_prime): model.addVar() for j in self.J_set for j_prime in self.J0}
        alpha_4 = {(i, j_prime): model.addVar() for i in self.I_set for j_prime in self.J0}
        alpha_5 = {(i, j, j_prime): model.addVar() for i in self.I_set for j in self.J0 for j_prime in self.J0}

        for j in self.J_set:
            for j_prime in self.J0:
                if j_prime == j:
                    if j_prime == 0:
                        model.addConstr(quicksum(b[i, j, j_prime] for i in self.I_set) - alpha_2[j, j_prime] + beta_2[j] -
                                         (self.Cap[j] * x[j] + self.Cap_bar[j] * y[j]) <= 0)
                    else:
                        model.addConstr(quicksum(b[i, j, j_prime] for i in self.I_set) - alpha_2[j, j_prime] -
                                         (self.Cap[j] * x[j] + self.Cap_bar[j] * y[j]) <= 0)
                else:
                    if j_prime == 0:
                        model.addConstr(quicksum(b[i, j, j_prime] for i in self.I_set) - alpha_2[j, j_prime] + beta_2[j] <= 0)
                    else:
                        model.addConstr(quicksum(b[i, j, j_prime] for i in self.I_set) - alpha_2[j, j_prime] <= 0)

            model.addConstr(quicksum(alpha_2[j, j_prime] for j_prime in self.J0) - beta_2[j] <= 0)


        for i in self.I_set:
            for j_prime in self.J0:
                if j_prime in self.J_set:
                    model.addConstr(quicksum(-b[i, j, j_prime] for j in self.J0) - alpha_4[i, j_prime] <= 0)
                else:
                    model.addConstr(quicksum(-b[i, j, j_prime] for j in self.J0) - alpha_4[i, j_prime] + beta_4[i] <= 0)

            model.addConstr(quicksum(alpha_4[i, j_prime] for j_prime in self.J0) - beta_4[i] <= -self.demand[i])

        for i in self.I_set:
            for j in self.J0:
                    for j_prime in self.J0:
                        if j_prime in self.J_set:
                            model.addConstr(-b[i, j, j_prime] - alpha_5[i, j, j_prime] <= 0)
                        else:
                            model.addConstr(-b[i, j, j_prime] - alpha_5[i, j, j_prime] + beta_5[i, j] <= 0)

                    model.addConstr(quicksum(alpha_5[i, j, j_prime] for j_prime in self.J0) - beta_5[i, j] <= 0)

        for n in self.N_set:
            rhs = LinExpr()
            rhs += quicksum(gamma[r, n] * (self.v[r] - quicksum(self.U[r, j] * self.xi[j, n] for j in self.J0)) for r in self.R_set)
            rhs += quicksum(self.c[i, j] * b[i, j, j_prime] * self.xi[j_prime, n] for i in self.I_set for j in self.J0 for j_prime in self.J0)
            model.addConstr(s[n] >= rhs, name=f"s_def_{n}")


        for j in self.J0:
            for n in self.N_set:
                lhs = quicksum(self.U[r, j] * gamma[r, n] for r in self.R_set) - quicksum(self.c[i, j] * b[i, j, j_prime] for i in self.I_set for j_prime in self.J0)
                model.addConstr(lhs <= lambda_, name=f"dual1_{j}_{n}")


        for j in self.J0:
            for n in self.N_set:
                lhs = quicksum(self.c[i, j] * b[i, j, j_prime] for i in self.I_set for j_prime in self.J0) - quicksum(self.U[r, j] * gamma[r, n] for r in self.R_set)
                model.addConstr(lhs <= lambda_, name=f"dual2_{j}_{n}")


        model.addConstr(
            quicksum(self.f[j] * x[j] + self.h[j] * y[j] for j in self.J_set) <= self.Budget,
            name="budget"
        )

        for j in self.J_set:
            model.addConstr(y[j] <= x[j], name=f"yx_link_{j}")

            # === 目标函数 ===
            model.setObjective(
                quicksum(self.f[j] * x[j] + self.h[j] * y[j] for j in self.J_set) +
                (lambda_ * self.epsilon + (1 / self.N_num) * quicksum(s[n] for n in self.N_set)),
                GRB.MINIMIZE
            )

        self.model = model
        self.x = x
        self.y = y

    def solve_RO(self):
        self.model.optimize()

    def get_decision(self):
        x_val = {key: value.x for key, value in self.x.items()}
        y_val = {key: value.x for key, value in self.y.items()}
        return x_val, y_val

