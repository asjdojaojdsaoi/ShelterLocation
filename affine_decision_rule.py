from gurobipy import *
from config import *

class RSLP:
    def __init__(self, parameters):
        self.I_num = parameters.I_num
        self.J_num = parameters.J_num
        self.N_num= parameters.N_num
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


    def construct_model(self):
        self.m = Model()

        # self.m.setParam('MIPFocus', 3)
        self.x = {j: self.m.addVar(vtype=GRB.BINARY) for j in self.J_set}
        self.y = {j: self.m.addVar(vtype=GRB.BINARY) for j in self.J_set}

        self.s = {n: self.m.addVar(vtype=GRB.CONTINUOUS, lb=-10, ub=GRB.INFINITY) for n in self.N_set}
        self.lambda_ = self.m.addVar(vtype=GRB.CONTINUOUS)

        self.b = {(i, j, n, j_prime): self.m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY) for i in self.I_set for j in self.J0  for n in self.N_set for j_prime in self.J0}

        # dual variables
        self.pi_p_1 = {(n, j_prime): self.m.addVar() for j_prime in self.J0 for n in self.N_set}
        self.pi_n_1 = {(n, j_prime): self.m.addVar() for j_prime in self.J0 for n in self.N_set}


        self.beta_1 = {n: self.m.addVar() for n in self.N_set}
        beta_2 = {(j, n): self.m.addVar() for j in self.J_set for n in self.N_set}

        beta_4 = {(i, n): self.m.addVar() for i in self.I_set for n in self.N_set}
        beta_5 = {(i, j, n): self.m.addVar() for i in self.I_set for j in self.J0 for n in self.N_set}

        self.alpha_1 = {(j_prime, n): self.m.addVar() for j_prime in self.J0 for n in self.N_set}
        alpha_2 = {(j, n, j_prime): self.m.addVar() for j in self.J_set for n in self.N_set for j_prime in self.J0}

        alpha_4 = {(i, n, j_prime): self.m.addVar() for i in self.I_set for j_prime in self.J0 for n in self.N_set}
        alpha_5 = {(i, j, n, j_prime): self.m.addVar() for i in self.I_set for j in self.J0 for n in self.N_set for j_prime in self.J0}

        obj = quicksum(self.f[j] * self.x[j] + self.h[j] * self.y[j] for j in self.J_set) + self.lambda_ * self.epsilon + quicksum((1 / self.N_num) * self.s[n] for n in self.N_set)

        self.m.setObjective(obj, GRB.MINIMIZE)

        self.m.addConstr(quicksum(self.f[j] * self.x[j] + self.h[j] * self.y[j] for j in self.J_set) <= self.Budget, name='budget')
        self.m.addConstrs(self.x[j] >= self.y[j] for j in self.J_set)


        for n in self.N_set:
            self.m.addConstr(self.s[n] >= quicksum(self.pi_p_1[n, j_prime] * self.xi[j_prime, n] - self.pi_n_1[n, j_prime] * self.xi[j_prime, n] for j_prime in self.J0) +
                             quicksum(self.alpha_1[j_prime, n] for j_prime in self.J0) - self.beta_1[n])

            for j_prime in self.J0:
                self.m.addConstr(-self.lambda_ + self.pi_n_1[n, j_prime] + self.pi_p_1[n, j_prime] <= 0)
                if j_prime in self.J_set:
                    self.m.addConstr(
                        quicksum(self.c[i, j] * self.b[i, j, n, j_prime] for i in self.I_set for j in self.J0) -
                        self.pi_p_1[n, j_prime] + self.pi_n_1[n, j_prime] - self.alpha_1[j_prime, n] <= 0, name='norm_{}_{}'.format(n, j_prime))
                else:
                    self.m.addConstr(
                        quicksum(self.c[i, j] * self.b[i, j, n, j_prime] for i in self.I_set for j in self.J0) -
                        self.pi_p_1[n, j_prime] + self.pi_n_1[n, j_prime] - self.alpha_1[j_prime, n] + self.beta_1[n] <= 0, name='norm_{}_{}'.format(n, j_prime))


        for n in self.N_set:
            for j in self.J_set:
                for j_prime in self.J0:
                    if j_prime == j:
                        if j_prime == 0:
                            self.m.addConstr(quicksum(self.b[i, j, n, j_prime] for i in self.I_set) - alpha_2[j, n, j_prime] + beta_2[j, n] -
                                             (self.Cap[j] * self.x[j] + self.Cap_bar[j] * self.y[j]) <= 0)
                        else:
                            self.m.addConstr(quicksum(self.b[i, j, n, j_prime] for i in self.I_set) - alpha_2[j, n, j_prime] -
                                             (self.Cap[j] * self.x[j] + self.Cap_bar[j] * self.y[j]) <= 0)
                    else:
                        if j_prime == 0:
                            self.m.addConstr(quicksum(self.b[i, j, n, j_prime] for i in self.I_set) - alpha_2[j, n, j_prime] + beta_2[j, n] <= 0)
                        else:
                            self.m.addConstr(quicksum(self.b[i, j, n, j_prime] for i in self.I_set) - alpha_2[j, n, j_prime] <= 0)

                self.m.addConstr(quicksum(alpha_2[j, n, j_prime] for j_prime in self.J0) - beta_2[j, n] <= 0)


        for n in self.N_set:
            for i in self.I_set:
                for j_prime in self.J0:
                    if j_prime in self.J_set:
                        self.m.addConstr(quicksum(-self.b[i, j, n, j_prime] for j in self.J0) - alpha_4[i, n, j_prime] <= 0)
                    else:
                        self.m.addConstr(quicksum(-self.b[i, j, n, j_prime] for j in self.J0) - alpha_4[i, n, j_prime] + beta_4[i, n] <= 0)

                self.m.addConstr(quicksum(alpha_4[i, n, j_prime] for j_prime in self.J0) - beta_4[i, n] <= -self.demand[i])

        for i in self.I_set:
            for j in self.J0:
                    for n in self.N_set:
                        for j_prime in self.J0:
                            if j_prime in self.J_set:
                                self.m.addConstr(-self.b[i, j, n, j_prime] - alpha_5[i, j, n, j_prime] <= 0)
                            else:
                                self.m.addConstr(-self.b[i, j, n, j_prime] - alpha_5[i, j, n, j_prime] + beta_5[i, j, n] <= 0)

                        self.m.addConstr(quicksum(alpha_5[i, j, n, j_prime] for j_prime in self.J0) - beta_5[i, j, n] <= 0)

    def solve_model(self):
        self.m.optimize()

    def report_result(self):
        print('---------------------------------Report---------------------------')
        print('****************general******************')
        print('total capacity: {}'.format(sum([self.Cap[j] for j in self.J_set])))
        print('selected capacity: {}'.format(sum([self.Cap[j] * self.x[j].x for j in self.J_set])))
        print('expanded capacity: {}'.format(sum([self.Cap_bar[j] * self.y[j].x for j in self.J_set])))
        print('total demand: {}'.format(sum([self.demand[i] for i in self.I_set])))
        print('****************cost******************')
        print('total Budget: {}'.format(self.Budget))
        self.total_build_cost = sum([self.x[key].x * self.f[key] for key in self.x])
        self.total_build_num = sum([self.x[key].x for key in self.x])
        self.total_expanded_cost = sum([self.y[key].x * self.h[key] for key in self.x])
        self.total_expanded_num = sum([self.y[key].x for key in self.x])
        self.objective_value = self.m.Objval
        print('total build cost: {}'.format(sum([self.x[key].x * self.f[key] for key in self.x])))
        print('total build num: {}'.format(sum([self.x[key].x for key in self.x])))
        print('total expanded cost: {}'.format(sum([self.y[key].x * self.h[key] for key in self.x])))
        print('total expanded num: {}'.format(sum([self.y[key].x for key in self.x])))
        print('scenario cost', [self.s[n] for n in self.N_set])
        print('objective value {}'.format(self.m.Objval))

