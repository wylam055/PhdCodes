from GraphClass import CausalGraph
import math
from itertools import combinations
import random
import numpy as np

##############################################################################################################

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

##############################################################################################################

def randomDAG(no_of_nodes, avg_deg, forward = False):
    total_no_of_edges = int(round(avg_deg * (no_of_nodes) / 2))
    assert total_no_of_edges <= nCr(no_of_nodes,2)

    range_of_nodes = range(no_of_nodes)
    possible_edges = list(combinations(range_of_nodes, 2))
    list_of_edges = random.sample(possible_edges, total_no_of_edges)
    if forward:
        return list_of_edges
    else:
        new_node_indices = random.sample(range_of_nodes, len(range_of_nodes))
        new_list_of_edges = [(new_node_indices[i], new_node_indices[j]) for (i, j) in list_of_edges]
    return new_list_of_edges

##############################################################################################################

def randomSEM(no_of_nodes, avg_deg, coefLow, coefHigh, sample_size, coefSymmetric = True):
    assert coefLow < coefHigh
    assert np.sign(coefLow) == np.sign(coefHigh)

    list_of_edges = randomDAG(no_of_nodes, avg_deg)
    coefMatrix = np.zeros((no_of_nodes,no_of_nodes))
    model = []

    for i in range(no_of_nodes):
        Xi = np.random.normal(0, 1, sample_size)
        if i == 0:
            model.append(Xi)
            continue
        else:
            parents = [j for (j, k) in list_of_edges if i == k]
            for parent in parents:
                if coefSymmetric:
                    coef = random.choice([random.uniform(coefLow, coefHigh), random.uniform(-coefHigh, -coefLow)])
                else:
                    coef = random.uniform(coefLow, coefHigh)
                coef = round(coef, 4)
                coefMatrix[i, parent] = coef
                Xi += coef * model[parent]
            model.append(Xi)

    cg = CausalGraph(no_of_nodes)
    cg.adjmat[cg.adjmat == 0] = -1
    for (i, j) in list_of_edges:
        cg.addDirectedEdge(i, j)
    cg.data = np.array(model).transpose()
    cg.coef_mat = coefMatrix
    return cg

##############################################################################################################

if __name__ == "__main__":
    cg = randomSEM(no_of_nodes=4, avg_deg=2, coefLow=0.2, coefHigh=0.7, sample_size=100, coefSymmetric = True)
    print(cg.adjmat)
    print(cg.data)
    print(cg.coef_mat)