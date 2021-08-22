from GraphClass import CausalGraph
import math
from itertools import combinations
import random
import numpy as np
from copy import deepcopy
##############################################################################################################

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

##############################################################################################################

def fancy_indexing(np_array, indices, columnsOnly = True):
    idx = np.empty_like(indices)
    idx[indices] = np.arange(len(indices))
    if columnsOnly:
        new_array = np_array[:, idx]
    else:
        new_array = np_array[idx, :]
        new_array = new_array[:, idx]
    return new_array

##############################################################################################################

def randomForwardDAG(no_of_nodes, avg_deg):
    total_no_of_edges = int(round(avg_deg * (no_of_nodes) / 2))
    assert total_no_of_edges <= nCr(no_of_nodes,2)

    range_of_nodes = range(no_of_nodes)
    possible_edges = list(combinations(range_of_nodes, 2))
    return random.sample(possible_edges, total_no_of_edges)

##############################################################################################################

def randomSEM(no_of_nodes, avg_deg, coefLow, coefHigh, sample_size, coefSymmetric = True, randomizeCol = True):
    assert coefLow < coefHigh
    assert np.sign(coefLow) == np.sign(coefHigh)

    list_of_edges = randomForwardDAG(no_of_nodes, avg_deg)
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
    cg.data = np.array(model).transpose()
    cg.coef_mat = coefMatrix
    add_edges = deepcopy(list_of_edges)
    if randomizeCol:
        node_indices = random.sample(range(no_of_nodes), no_of_nodes)
        add_edges = [(node_indices[i], node_indices[j]) for (i, j) in list_of_edges]
        cg.data = fancy_indexing(cg.data, node_indices, columnsOnly=True)
        cg.coef_mat = fancy_indexing(cg.coef_mat, node_indices, columnsOnly=False)
    for (i, j) in add_edges:
        cg.addDirectedEdge(i, j)
    return cg

##############################################################################################################

if __name__ == "__main__":
    cg = randomSEM(no_of_nodes=4, avg_deg=2, coefLow=0.2, coefHigh=0.7, sample_size=100, coefSymmetric = True, randomizeCol=True)
    print(cg.adjmat)
    print(cg.coef_mat)
    print(cg.findFullyDirected())