#######################################################################################################################
import concurrent.futures

from Tester_Markov import CMCTester
from Tester_Faithfulness import faithfulnessTester
from Tester_Perm_based_razors import permutationBasedRazorsTester
from SEMSimulatorRandom import randomSEM
import numpy as np
import time


#######################################################################################################################

def razorsTester(cg, test_name, alpha):
    """
    Verify a list of causal razors
    :param cg: CausalGraph object
    :param test_name: name of the independence test being used (string)
    :param alpha: desired significance level in (0, 1) (float)
    :return:
    1. CMC: True if CMC is satisfied, and False otherwise
    2. SGS: True if SGS-minimality is satisfied, and False otherwise
    3. Pm: True if P-minimality is satisfied, and False otherwise
    4. Fr: True if frugality is satisfied, and False otherwise
    5. uFr: True if u-frugality is satisfied, and False otherwise
    6. adjF: True if adj-faithfulness is satisfied, and False otherwise
    7. oriF: True if ori-faithfulness is satisfied, and False otherwise
    8. triF: True if tri-faithfulness is satisfied, and False otherwise
    9. resF: True if both adj-faithfuless and ori-faithfulness are satisfied and False otherwise
    10. CFC: True if CFC is satisfied, and False otherwise
    """
    start = time.time()
    [CMC, I_G_star] = CMCTester(cg, test_name, alpha)

    if not CMC:
        end1 = time.time()
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, round(end1 - start, 2)]
    else:
        [SGS, Pm, Fr, uFr, CI_facts, CD_facts] = permutationBasedRazorsTester(cg, test_name, alpha,
                                                                              CMC_result=[CMC, I_G_star])
        [CFC, resF, adjF, oriF, triF] = faithfulnessTester(cg, test_name, alpha, CMC_result=[CMC, I_G_star],
                                                           CI_facts=CI_facts, CD_facts=CD_facts)
        end2 = time.time()

        return [int(CMC), int(SGS), int(Pm), int(Fr), int(uFr),
                int(adjF), int(oriF), int(triF), int(resF), int(CFC),
                round(end2 - start, 2)]


######################################################################################################################

def simulateAndTest(no_of_nodes, avg_deg, coefLow, coefHigh, sample_size, testName, alpha, coefSymmetric=True,
                    randomizeOrder=True):
    cg = randomSEM(no_of_nodes, avg_deg, coefLow, coefHigh, sample_size, coefSymmetric, randomizeOrder)
    return razorsTester(cg, testName, alpha)


if __name__ == "__main__":
    ##########################
    ### Simulation setting ###
    ##########################
    no_of_nodes = 6
    avg_deg = 2
    coefLow = 0.2
    coefHigh = 0.7
    sample_size = 10000
    number_of_runs = 50
    alpha = 0.01
    testName = "Fisher_Z"
    fileName = f"{no_of_nodes}_{avg_deg}_{sample_size}_LG_{coefLow}_{coefHigh}"
    print(f"Simulation setting: {fileName}\n")

    ######################
    ### Run simulation ###
    ######################
    CMC_sum = 0
    SGS_sum = 0
    Pm_sum = 0
    Fr_sum = 0
    uFr_sum = 0
    CFC_sum = 0
    resF_sum = 0
    adjF_sum = 0
    oriF_sum = 0
    triF_sum = 0
    elapsed_sum = 0
    output = []
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(simulateAndTest,
                                   no_of_nodes, avg_deg, coefLow, coefHigh, sample_size, testName, alpha)
                   for i in range(number_of_runs)]

        runs_iter = iter(range(1, number_of_runs + 1))

        for f in concurrent.futures.as_completed(results):
            [CMC, SGS, Pm, Fr, uFr, adjF, oriF, triF, resF, CFC, elapsed] = f.result()
            output.append([CMC, SGS, Pm, Fr, uFr, adjF, oriF, triF, resF, CFC, elapsed])
            CMC_sym = '\u2713' if CMC else 'x'
            CFC_sym = '\u2713' if CFC else 'x'
            resF_sym = '\u2713' if resF else 'x'
            adjF_sym = '\u2713' if adjF else 'x'
            oriF_sym = '\u2713' if oriF else 'x'
            triF_sym = '\u2713' if triF else 'x'
            SGS_sym = '\u2713' if SGS else 'x'
            Pm_sym = '\u2713' if Pm else 'x'
            Fr_sym = '\u2713' if Fr else 'x'
            uFr_sym = '\u2713' if uFr else 'x'
            CMC_sum += CMC
            CFC_sum += CFC
            resF_sum += resF
            adjF_sum += adjF
            oriF_sum += oriF
            triF_sum += triF
            SGS_sum += SGS
            Pm_sum += Pm
            Fr_sum += Fr
            uFr_sum += uFr
            elapsed_sum += elapsed
            print(f"Run {next(runs_iter)}: CMC {CMC_sym}, SGS {SGS_sym}, Pm {Pm_sym}, Fr {Fr_sym}, uFr {uFr_sym}, "
                  f"adjF {adjF_sym}, oriF {oriF_sym}, triF {triF_sym}, resF {resF_sym}, CFC {CFC_sym}; "
                  f"Elapsed time (sec): {elapsed}")
        print("\n")
        print(f"CMC is satisfied in {CMC_sum} out of {number_of_runs} runs.")
        print(f"SGS-minimality is satisfied in {SGS_sum} out of {number_of_runs} runs.")
        print(f"P-minimality is satisfied in {Pm_sum} out of {number_of_runs} runs.")
        print(f"frugality is satisfied in {Fr_sum} out of {number_of_runs} runs.")
        print(f"u-frugality is satisfied in {uFr_sum} out of {number_of_runs} runs.")
        print(f"Adj-faithfulness is satisfied in {adjF_sum} out of {number_of_runs} runs.")
        print(f"Ori-faithfulness is satisfied in {oriF_sum} out of {number_of_runs} runs.")
        print(f"Tri-faithfulness is satisfied in {triF_sum} out of {number_of_runs} runs.")
        print(f"Res-faithfulness is satisfied in {resF_sum} out of {number_of_runs} runs.")
        print(f"CFC is satisfied in {CFC_sum} out of {number_of_runs} runs.")
        print(f"Total elapsed time [w/o multiprocessing] (sec): {round(elapsed_sum, 2)}")
    end_time = time.time()
    print(f"Total elapsed time [with multiprocessing] (sec): {round(end_time - start_time, 2)}")
    np.savetxt("results/" + fileName + "_" + str(alpha) + ".csv", np.array(output), delimiter=",", fmt="%s")
######################################################################################################################
