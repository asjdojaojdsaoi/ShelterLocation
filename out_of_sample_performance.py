from gurobipy import *
from config import *
from affine_decision_rule import *

def get_two_stage_costs(xi_hat, x_val, y_val, params):
    m = Model()

    z = m.addVars(params.I_set, params.J0, vtype=GRB.CONTINUOUS)

    for i in params.I_set:
        m.addConstr(quicksum(z[i, j] for j in params.J0) >= params.demand[i])

    for j in params.J_set:
        m.addConstr(quicksum(z[i, j] for i in params.I_set) <= (params.Cap[j] * x_val[j] + params.Cap_bar[j] * y_val[j]) * xi_hat[j])

    obj = quicksum(params.c[i, j] * z[i, j] for i in params.I_set for j in params.J0)
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()
    return m.objVal

if __name__ == '__main__':
    data_path = './data/geo.xlsx'
    I_num = 30
    J_num = 30
    N_num = 10
    c_cor = 0.1
    Cap_min = 200
    Cap_max = 4000
    expand_cor = 0.1
    f_min = 1
    f_max = 3
    h_min = 0.5
    h_max = 2.8
    total_demand = 30000
    demand_min = 0.8
    demand_max = 1.2
    punish_c = 3
    epsilon = 0.9
    Budget = 2 * 0.5 * (Cap_max + Cap_min) * 0.5 * (f_min + f_max) * 0.5 * J_num

    params = Params(data_path=data_path, I_num=I_num, J_num=J_num, N_num=N_num, c_cor=c_cor, Cap_max=Cap_max,
                    Cap_min=Cap_min,
                    expand_cor=expand_cor, f_min=f_min, f_max=f_max, h_min=h_min, h_max=h_max, epsilon=epsilon,
                    total_demand=total_demand, demand_max=demand_max, demand_min=demand_min, punish_c=punish_c,
                    Budget=Budget)

    rslp = RSLP(params)
    rslp.construct_model()
    rslp.solve_model()

    x_val, y_val = {key: value.x for key, value in rslp.x.items()}, {key: value.x for key, value in rslp.y.items()}

    num_samples = 1000
    loss_records = []
    for n in range(num_samples):
        xi_hat = {j: 0 if np.random.random() < params.p[j] else 1 for j in params.J_set}
        loss = get_two_stage_costs(xi_hat, x_val, y_val, params)
        loss_records.append(loss)