#######################################################################################################################
from itertools import combinations
from Tester_Markov import CMCTester
from Helper import powerset, listMinus
from copy import deepcopy
from SEMSimulatorRandom import randomSEM
import numpy as np
#######################################################################################################################

def faithfulnessTester(cg, test_name, alpha, **kwargs):
    """Test CFC, adj_faithfulness, and ori_faithfulness
    :param cg: the true CausalGraph object
    :param test_name: name of the independence test being used (string)
    :param alpha: desired significance level in (0, 1) (float)
    :return:
    1. CFC: True if CFC is satisfied, and False otherwise
    2. resF: True if res-faithfulness is satisfied, and False otherwise
    3. adjF: True if adj-faithfulness is satisfied, and False otherwise
    4. oriF: True if ori-faithfulness is satisfied, and False otherwise
    5. triF: True if tri-faithfulness is satisfied, and False otherwise
    """
    cg.setTestName(test_name)
    cg.getCorrMatrix() if test_name == "Fisher_Z" else []

    if "CMC_result" in kwargs:
        CMC = kwargs["CMC_result"][0]
        I_G_star = kwargs["CMC_result"][1]
    else:
        [CMC, I_G_star] = CMCTester(cg, test_name, alpha)

    if not CMC:
        return [False, False, False, False, False]
    else:
        CI_facts = kwargs["CI_facts"] + I_G_star if "CI_facts" in kwargs else deepcopy(I_G_star)
        CD_facts = kwargs["CD_facts"] if "CD_facts" in kwargs else []

    range_of_nodes = cg.variables
    adj = [(i, j) for (i, j) in cg.findAdj() if i < j]
    UT = [(i, j, k) for (i, j, k) in cg.findUnshieldedTriples() if i < k]
    tri = [(i, j, k) for (i, j, k) in cg.findTriangles() if i < k]
    CFC = True
    triF = True
    adjF = True
    oriF = True

    for (i, j) in combinations(range_of_nodes, 2):

        remaining_nodes = listMinus(range_of_nodes, [i, j])
        cond_sets = powerset(remaining_nodes)

        # First, verify tri-faithfulness (weaker than adj-faithfulness):
        tri_ij = [(x, y, z) for (x, y, z) in tri if x == i and z == j]
        if len(tri_ij) != 0 and triF:
            for (x, y, z) in tri_ij:
                if cg.isCollider(x, y, z):
                    for S in [S_sets for S_sets in cond_sets if y in S_sets]:
                        if [i, j, S] in CI_facts:
                            triF = False
                            adjF = False
                            CFC = False
                        elif [i, j, S] in CD_facts:
                            continue
                        else:
                            p = cg.ci_test(i, j, S)
                            if p > alpha:
                                triF = False
                                adjF = False
                                CFC = False
                else:
                    for S in [S_sets for S_sets in cond_sets if y not in S_sets]:
                        if [i, j, S] in CI_facts:
                            triF = False
                            adjF = False
                            CFC = False
                        elif [i, j, S] in CD_facts:
                            continue
                        else:
                            p = cg.ci_test(i, j, S)
                            if p > alpha:
                                triF = False
                                adjF = False
                                CFC = False

        # Second, verify adj-faithfulness
        elif (i, j) in adj and adjF:
            for S in cond_sets:
                if [i, j, S] in CI_facts:
                    adjF = False
                    CFC = False
                elif [i, j, S] in CD_facts:
                    continue
                else:
                    p = cg.ci_test(i, j, S)
                    if p > alpha:
                        adjF = False
                        CFC = False

        else:
            # Third, verify ori-faithfulness
            UT_ij = [(x, y, z) for (x, y, z) in UT if x == i and z == j]
            if len(UT_ij) != 0 and oriF:
                for (x, y, z) in UT_ij:
                    if cg.isCollider(x, y, z):
                        for S in [S_sets for S_sets in cond_sets if y in S_sets]:
                            if [i, j, S] in CI_facts:
                                oriF = False
                                CFC = False
                            elif [i, j, S] in CD_facts:
                                continue
                            else:
                                p = cg.ci_test(i, j, S)
                                if p > alpha:
                                    oriF = False
                                    CFC = False
                    else:
                        for S in [S_sets for S_sets in cond_sets if y not in S_sets]:
                            if [i, j, S] in CI_facts:
                                oriF = False
                                CFC = False
                            elif [i, j, S] in CD_facts:
                                continue
                            else:
                                p = cg.ci_test(i, j, S)
                                if p > alpha:
                                    oriF = False
                                    CFC = False

            elif CFC:
                for S in cond_sets:
                    if not cg.isDSep(i, j, S):
                        if [i, j, S] in CI_facts:
                            CFC = False
                        elif [i, j, S] in CD_facts:
                            continue
                        else:
                            p = cg.ci_test(i, j, S)
                            if p > alpha:
                                CFC = False

    resF = adjF and oriF
    return [CFC, resF, adjF, oriF, triF]

#######################################################################################################################

if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    no_of_nodes = 4
    avg_deg = 2
    coefLow = 0.2
    coefHigh = 0.7
    sample_size = 100
    number_of_runs = 50
    alpha = 0.01
    testName = "Fisher_Z"

    ######################
    ### Run simulation ###
    ######################
    CMC_sum  = 0
    CFC_sum = 0
    resF_sum = 0
    adjF_sum = 0
    oriF_sum = 0
    triF_sum = 0
    for i in range(number_of_runs):
        cg = randomSEM(no_of_nodes=no_of_nodes, avg_deg=avg_deg, coefLow=coefLow, coefHigh=coefHigh,
                       sample_size=sample_size, coefSymmetric = True, randomizeOrder = True)
        CMC, I_G_star = CMCTester(cg, testName, alpha)
        [CFC, resF, adjF, oriF, triF] = faithfulnessTester(cg, testName, alpha, CMC_result = [CMC, I_G_star])
        CMC_sym = '\u2713' if CMC else 'x'
        CFC_sym = '\u2713' if CFC else 'x'
        resF_sym = '\u2713' if resF else 'x'
        adjF_sym = '\u2713' if adjF else 'x'
        oriF_sym = '\u2713' if oriF else 'x'
        triF_sym = '\u2713' if triF else 'x'
        CMC_sum += CMC
        CFC_sum += CFC
        resF_sum += resF
        adjF_sum += adjF
        oriF_sum += oriF
        triF_sum += triF
        print(f"Run {i+1}: CMC {CMC_sym}, CFC {CFC_sym}, ResF {resF_sym}, AdjF {adjF_sym}, OriF {oriF_sym}, TriF {triF_sym}")
    print("\n")
    print(f"CMC is satisfied in {CMC_sum} out of {number_of_runs} runs.")
    print(f"CFC is satisfied in {CFC_sum} out of {number_of_runs} runs.")
    print(f"Res-faithfulness is satisfied in {resF_sum} out of {number_of_runs} runs.")
    print(f"Adj-faithfulness is satisfied in {adjF_sum} out of {number_of_runs} runs.")
    print(f"Ori-faithfulness is satisfied in {oriF_sum} out of {number_of_runs} runs.")
    print(f"Tri-faithfulness is satisfied in {triF_sum} out of {number_of_runs} runs.")

#######################################################################################################################