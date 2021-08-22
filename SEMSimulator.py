from GraphClass import CausalGraph
import math
from itertools import combinations
import random
import numpy as np
from copy import deepcopy
##############################################################################################################

def nCr(n,r):
    # Compute the standard (n choose r) operation
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

##############################################################################################################

def fancy_indexing(np_array, indices, columnsOnly = True):
    # A fancy but efficient way to rearrange the columns (and rows) of a numpy 2darray
    # Reference: https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
    idx = np.empty_like(indices)
    idx[indices] = np.arange(len(indices))
    new_array = np_array[:, idx]
    if not columnsOnly:
        new_array = new_array[idx, :]
    return new_array

##############################################################################################################

def randomForwardDAG(no_of_nodes, avg_deg):
    # Generate a list of directed edges (i, j) representing a random forward DAG
    total_no_of_edges = int(round(avg_deg * (no_of_nodes) / 2))
    assert total_no_of_edges <= nCr(no_of_nodes,2)
    range_of_nodes = range(no_of_nodes)
    possible_edges = list(combinations(range_of_nodes, 2))
    return random.sample(possible_edges, total_no_of_edges)

##############################################################################################################

def randomSEM(no_of_nodes, avg_deg, coefLow, coefHigh, sample_size, coefSymmetric = True, randomizeOrder = True):
    """ Generate a CausalGraph object by a structural equation model
    :param no_of_nodes: number of nodes
    :param avg_deg: average degree of the DAG
    :param coefLow: lower end of linear coefficient
    :param coefHigh: upper end of linear coefficient
    :param sample_size: sample size of the generated data
    :param coefSymmetric: the linear coefficients are symmetric over zero (i.e., +-[coefLow, coefHigh]) if True
    :param randomizeOrder: the order of the variables are randomized if True
    :return: a CausalGraph object cg
    """
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
    if randomizeOrder:
        node_indices = random.sample(range(no_of_nodes), no_of_nodes)
        add_edges = [(node_indices[i], node_indices[j]) for (i, j) in list_of_edges]
        cg.data = fancy_indexing(cg.data, node_indices, columnsOnly=True)
        cg.coef_mat = fancy_indexing(cg.coef_mat, node_indices, columnsOnly=False)
    for (i, j) in add_edges:
        cg.addDirectedEdge(i, j)
    return cg

##############################################################################################################

##############################################################################################################
### Example ##################################################################################################
##############################################################################################################

if __name__ == "__main__":
    cg = randomSEM(no_of_nodes=4, avg_deg=2, coefLow=0.2, coefHigh=0.7, sample_size=100, coefSymmetric = True, randomizeOrder=True)
    print("The adjacency matrix:\n", cg.adjmat, "\n")
    print("The list of directed edges x -> y as (x, y):\n", cg.findFullyDirected(), "\n")
    print("The coefficient matrix:\n", cg.coef_mat, "\n")
    print("The dataset:\n", cg.data, "\n")