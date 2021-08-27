from GraphClass import CausalGraph
import math
from itertools import combinations
import random
import numpy as np
from copy import deepcopy


##############################################################################################################

def nCr(n, r):
    # Compute the standard (n choose r) operation
    f = math.factorial
    return int(f(n) / f(r) / f(n - r))


##############################################################################################################

def fancy_indexing(np_array, indices, columnsOnly=True):
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
    assert total_no_of_edges <= nCr(no_of_nodes, 2)
    range_of_nodes = range(no_of_nodes)
    possible_edges = list(combinations(range_of_nodes, 2))
    return random.sample(possible_edges, total_no_of_edges)


##############################################################################################################

def randomSEM(no_of_nodes, avg_deg, coefLow, coefHigh, sample_size,
              coefSymmetric=True, varLow=1, varHigh=3, randomizeOrder=True):
    """ Generate a CausalGraph object by a structural equation model
    :param no_of_nodes: number of nodes
    :param avg_deg: average degree of the DAG
    :param coefLow: lower end of linear coefficient
    :param coefHigh: upper end of linear coefficient
    :param sample_size: sample size of the generated data
    :param coefSymmetric: the linear coefficients are symmetric over zero (i.e., +-[coefLow, coefHigh]) if True
    :param varLow: lower limit of the variance of the Gaussian noise term
    :param varHigh: upper limit of the variance of the Gaussian noise term
    :param randomizeOrder: the order of the variables are randomized if True
    :return: a CausalGraph object cg
    """
    assert coefLow < coefHigh
    assert np.sign(coefLow) == np.sign(coefHigh)

    list_of_edges = randomForwardDAG(no_of_nodes, avg_deg)
    coefMatrix = np.zeros((no_of_nodes, no_of_nodes))
    model = []

    for i in range(no_of_nodes):
        var_e = random.uniform(varLow, varHigh)
        Xi = np.random.normal(0, var_e, sample_size)
        if i == 0:
            model.append(Xi)
            continue
        else:
            parents = [j for (j, k) in list_of_edges if i == k]
            for parent in parents:
                coef = round(random.uniform(coefLow, coefHigh), 4)
                if coefSymmetric:
                    sign = 1 if random.random() < 0.5 else -1
                    coef = coef * sign
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
    cg.toNxGraph()
    cg.toNxSkeleton()
    cg.sample_size = sample_size
    return cg


##############################################################################################################

##############################################################################################################
### Example ##################################################################################################
##############################################################################################################

if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    no_of_nodes = 5
    avg_deg = 2
    coefLow = 0.2
    coefHigh = 1.2
    sample_size = 1000000
    cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=avg_deg, coefLow=coefLow, coefHigh=coefHigh,
                   sample_size=sample_size, coefSymmetric=True, randomizeOrder=True)
    print("Adjacency matrix")
    print("(where [x,y]=1 & [y,x]=0 represent x→y; [x,y]=[y,x]=0 represent x-y); non-adjacency as -1)")
    print(cg.adjmat, "\n")
    print("List of directed edges (where (x,y) represents x→y):\n", cg.findFullyDirected(), "\n")
    print("Coefficient matrix:\n", cg.coef_mat, "\n")
    BIC = cg.getBIC()
    print(f"BIC score: {BIC}")
    # np.savetxt("temp/test_data_" + str(no_of_nodes) + "_" + str(avg_deg) +"_" + str(sample_size) +
    #            ".csv", cg.data, delimiter=",")

##############################################################################################################
