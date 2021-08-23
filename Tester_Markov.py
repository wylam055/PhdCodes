#######################################################################################################################
from itertools import combinations
from Helper import powerset, listMinus
from networkx import d_separated
from SEMSimulator import randomSEM
import numpy as np
#######################################################################################################################

def dSepRelations(nx_graph):
    nodes = nx_graph.nodes
    edges = list(nx_graph.edges)
    set_of_adj = set([(i, j) for (i, j) in edges if i < j] + [(j, i) for (i, j) in edges if i > j])
    possible_adj = list(combinations(nodes, 2))
    nonadj = [(i, j) for (i, j) in possible_adj if (i, j) not in set_of_adj]
    for (i, j) in nonadj:
        remaining_nodes = listMinus(nodes, [i, j])
        cond_sets = powerset(remaining_nodes)
        for S in cond_sets:
            if d_separated(nx_graph, {i}, {j}, S):
                yield [i, j, S]

#######################################################################################################################

def CMCTester(cg, test_name, alpha):
    """Test CMC
    :param cg: CausalGraph object
    :param test_name: name of the independence test being used (string)
    :param alpha: desired significance level in (0, 1) (float)
    :return:
    1. CMC: True if CMC is satisfied, and False otherwise
    2. I_G_star: I(G*) if CMC is true, else []
    """
    cg.setTestName(test_name)
    cg.corr_mat = np.corrcoef(cg.data, rowvar=False) if test_name == "Fisher_Z" else []

    CMC = True
    I_G_star = []

    for (i, j, S) in dSepRelations(cg.nx_graph):
        if not CMC:
            break
        I_G_star.append([i, j, S])
        p = cg.ci_test(i, j, S)
        if p <= alpha:
            CMC = False
            break

    if CMC:
        return [True, I_G_star]
    else:
        return [False, []]

#######################################################################################################################

if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    no_of_nodes = 4
    avg_deg = 1
    coefLow = 0.2
    coefHigh = 0.7
    sample_size = 100
    number_of_runs = 50
    alpha = 0.01
    testName = "Fisher_Z"

    ######################
    ### Run simulation ###
    ######################
    CMC_sum = 0
    for i in range(number_of_runs):
        cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=avg_deg, coefLow=coefLow, coefHigh=coefHigh,
                       sample_size=sample_size, coefSymmetric = True, randomizeOrder = True)
        [CMC, I_G_star] = CMCTester(cg, testName, alpha)
        sym = '\u2713' if CMC else 'x'
        print(f"Run {i+1}: CMC {sym}")
        CMC_sum += CMC
    print("\n")
    print(f"CMC is satisfied in {CMC_sum} out of {number_of_runs} runs.")

#######################################################################################################################