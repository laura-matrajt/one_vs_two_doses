import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
# import seaborn as sns
# from matplotlib import pyplot as plt
import pickle
# from matplotlib.colors import ListedColormap
# # %matplotlib qt
# from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
import sys, getopt
import time
import multiprocessing as mp
import os
import pandas as pd
sys.path.insert(1, '../')



from twoDosesOptimizationFunctionsP import optimizationSimplexAndNM, evalGS, defineInitCond2D, pickBestSolForEachObjective2D
from twoDosesOptimizationFunctionsP import createProRataVac2D, createVectorHighRiskFirst2, createProRataVac2DAdultsOnly
from coronavirusMainFunctions_twoDoses import findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis
from saveLoadFunctions import saveResults, loadResults




def run_main():

    today = time.strftime("%d%b%Y", time.localtime())

    index = os.environ['SLURM_ARRAY_TASK_ID']

    myIndex = int(index) + 20

    VC = int(os.environ['SLURM_JOB_NAME'])
   

    myobjectives = ['totalInfections', 'totalSymptomaticInfections', 'deaths', 'hosp_peak', 'ICU_peak']

    mytime = time.localtime()
    myseed = np.abs(int((np.random.normal())*1000000))
    np.random.seed(myseed)

    options, args = getopt.getopt(sys.argv[1:], '',
    ['VE_S1=', 'VE_S2=', 'VE_P1=', 'VE_P2=', 'VE_I1=', 'VE_I2=', 'SD=', 'numDoses=', 'redA=', 'myprev=',
    'frac_rec='])

    for option, value in options:
        if option in ('--VE_S1'):
            VE_S1 = float(value)

        if option in ('--VE_S2'):
            VE_S2 = float(value)

        if option in ('--VE_P1'):
            VE_P1 = float(value)

        if option in ('--VE_P2'):
            VE_P2 = float(value)

        if option in ('--VE_I1'):
            VE_I1 = float(value)

        if option in ('--VE_I2'):
            VE_I2 = float(value)

        if option in ('--SD'):
            SDval = int(value)

        if option in ('--numDoses'):
            numDosesVal = int(value)

        if option in ('--redA'):
            redA = float(value)

        if option in ('--myprev'):
            myprev = float(value)

        if option in ('--frac_rec'):
            frac_rec = float(value)


    #load the data needed to run the model here;

    ################### Common parameters that will not vary from run to run:    ###################
    N = 7.615 * 10**(6) #Washington state population

    #load contact matrices
    myfilename = '../data/consistentMatricesUS_polymodMethod01Jun2020.pickle'
    mymats = loadResults(myfilename)
    mymatAll = mymats['all']

    if SDval == 0:
        mymatSD = mymats['home'] + 0.6*mymats['work'] + 0.2*mymats['otherloc'] + 0.1*mymats['school']
    elif SDval == 1:
        mymatSD = mymats['home'] + 0.6 * mymats['work'] + 0.5 * mymats['otherloc'] + 0.5 * mymats[
            'school']  # results in Reff = 1.7 with 10% recovered
    elif SDval == 2:
        mymatSD = mymatAll
    elif SDval == 3:
        mymatSD = mymats['home'] + 0.6 * mymats['work'] + 0.4 * mymats['otherloc'] + 0.1 * mymats[
            'school']  # results in Reff = 1.39 with 10% recovered
    else:
        print('SD option not allowed')
        sys.exit()





    # load fractions in each age and vaccine group:
    myfilename = '../data/populationUS16ageGroups03Jun2020.pickle'
    popUS16 = loadResults(myfilename)
    popUS16fracs = popUS16[1]

    myfilename = '../data/populationUSGroupsForOptimization03Jun2020.pickle'
    groupInfo = loadResults(myfilename)
    groupFracs = groupInfo['groupsFracs']
    fracOfTotalPopulationPerVaccineGroup = groupInfo['fracOfTotalPopulationPerVaccineGroup']
    [relativeFrac75_80, relativeFrac80andAbove] = groupInfo['split75andAbove']


    #Split the population in 16 age groups:
    totalPop16 = N * popUS16fracs
    #Split the population in 5 vaccine groups:
    totalPop5 = N*fracOfTotalPopulationPerVaccineGroup

    numAgeGroups = 16
    numVaccineGroups = 5

    #load disease severity parameters:
    myfilename = '../data/disease_severity_parametersFerguson.pickle'
    diseaseParams = loadResults(myfilename)
    hosp_rate_16 = diseaseParams['hosp_rate_16']
    icu_rate_16 = diseaseParams['icu_rate_16']


    #load mortality parameters
    myfilename = '../data/salje_IFR_from_hospitalized_cases.pickle'
    salje_mortality_rate_16 = loadResults(myfilename)
    mortality_rate_16 = salje_mortality_rate_16

    # this is just 1 - ICU rate useful to compute it in advance and pass it to the ODE
    oneMinusICUrate = np.ones(numAgeGroups) - icu_rate_16
    # this is just 1 - Hosp rate useful to compute it in advance and pass it to the ODE
    oneMinusHospRate = np.ones(numAgeGroups) - hosp_rate_16

    #time horizon for the intervention:
    tspan = 28*7#np.linspace(0, 365, 365 * 2)

    ########################################################################################################################
    ######################## Parameters that will change for sensitivity analysis ####################################
    # Model parameters

    #fraction of symptomatic people
    frac_asymptomatic = 0.4 * np.ones(16) #see notesTwoDoses.md for details
    frac_asymptomatic[0:4] = 0.75 #see notes for details
    frac_sym = (1 - frac_asymptomatic) * np.ones(16)  # fraction of infected that are symptomatic
    #fraction of symptomatic children
    # frac_sym[0:4] = 0.2
    oneMinusSymRate = np.ones(16) - frac_sym
    print(oneMinusSymRate)

    #transition rates:
    durI = 4  # duration of infectiousness after developing symptoms
    durP = 2  # duration of infectiousness before developing symptoms
    durA = durI + durP  # the duration of asymptomatic infections is equal to that of symptomatic infections
    gammaA = 1 / durA  # recovery rate for asymptomatic

    gammaI = 1 / durI  # recovery rate for symptomatic infections (not hospitalized)

    gammaP = 1 / durP  # transition rate fromm pre-symptomatic to symptomatic
    gammaE = 1 / 3  # transition rate from exposed to infectious

    #reduction/increase of infectiousness
    redH = 0.  # assume no transmission for hospitalized patients
    redP = 1.   #see notesTwoDoses.md for the sources to assume equal transmissibility


    #hospitalization duration based on age:
    # hospital stays
    gammaH = np.ones(16)
    gammaH[0:10] = 1 / 3
    gammaH[10:13] = 1 / 4
    gammaH[13:] = 1 / 6

    gammaICU = np.ones(16)
    gammaICU[0:10] = 1/10
    gammaICU[10:13] = 1/14
    gammaICU[13:] = 1/12


    red_sus = np.ones(16) #assume no reduction in susceptibility
    red_sus[0:3] = 0.56 #reduction in susceptibility: taken from Viner 2020
    # red_sus[3:13] = 1
    red_sus[13:16] = 2.7 #taken from Bi et al medarxiv 2021

    sigma_base = 1 / 3.8
    sigma = sigma_base * np.ones(16)


    # Disease severity
    R0 = 3

    # compute beta based on these parameters:
    beta = findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(mymatAll, frac_sym,
                            gammaA, gammaE,  gammaI, gammaP, hosp_rate_16, redA,  redP, red_sus, sigma, R0, totalPop16)



    # With this values for mymatSD and for 10% of the pop recovered, we get an effective R of 1.2
    paramsODE = [beta, mymatSD, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16, icu_rate_16, numAgeGroups,
     oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus,sigma, totalPop16,
                 VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2]


    currentInfections = myprev * (1 / 100) * N * popUS16fracs

    y0 = defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups,
                     oneMinusHospRate, oneMinusICUrate,
                     totalPop16)
    numSamples = 10000
    dim = 10 #dimension of the unit simplex



    fracCov = 0.1 * VC  # vaccination coverage of the total population
    # print(fracCov)

    numVaccinesAvailable = round(fracCov * N)  # number of total vaccines available assuming that coverage

    if numDosesVal == 0:
        numDosesPerWeek = numVaccinesAvailable
        numDoses2 = 'all'
    else:
        numDosesPerWeek = numDosesVal*1000
        numDoses2 = str(int(numDosesVal))

    extraParams = [mortality_rate_16, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
               totalPop16, totalPop5, tspan]


    #perform grid search in a fixed number of samples
    start = timer()
    myMat = evalGS(dim, numSamples, extraParams)
    end = timer()
    results = pickBestSolForEachObjective2D(myMat)
    print('grid search for' + str(int(numSamples)) + ' samples was done in ')
    print(end - start)
    print('coarse search done')

    # #####################################   NM SEARCH   ############################################################
    repairFunOpt = 0
    numOfBestSols = 25
    numOfBestSolsNM = 25
    proRataVec2D = createProRataVac2DAdultsOnly(numVaccinesAvailable,  totalPop5)
    highRiskFirstVec = createVectorHighRiskFirst2(numVaccinesAvailable, totalPop5)
    extraParamsNM = [extraParams, myIndex, repairFunOpt]


    resultsNM = optimizationSimplexAndNM(myMat, highRiskFirstVec, numOfBestSols, numOfBestSolsNM, proRataVec2D, extraParamsNM)
    end = timer()
    print('NM for' + str(int(numOfBestSols))+ ' solutions was done in ') 
    print(end - start)
    #print(resultsNM)

    #store the results:
    fullOutput = [results, resultsNM, R0, extraParams, myseed]

    if numDoses2 == 'all':

        myfilename = 'resultsOptimization2D/NM/SD/vaccination6/instantaneousVac/' + myobjectives[myIndex-21] + '/matrixResults_' + \
                     'VES1_' +  str(int(VE_S1*100)) +'_VES2_' +  str(int(VE_S2*100)) + \
                     '_VE_P1_' + str(int(VE_P1*100)) + '_VE_P2_' + str(int(VE_P2*100)) + \
                     '_VE_I1_' + str(int(VE_I1 * 100)) + '_VE_I2_' + str(int(VE_I2 * 100)) + \
                     '_frac_coverage_' + str(int(fracCov * 100)) + '_fracRec_' + \
                     str(int(frac_rec * 100)) + '_redA_' + str(int(redA*100)) + '_' + numDoses2 +\
                     '_dosesPerWeek_' + today + 'SD' + str(SDval) + '_myprev_' + str(int(myprev*100)) + '6m.pickle'
    else:
        myfilename = 'resultsOptimization2D/NM/SD/vaccination6/vaccineCampaign/' + myobjectives[myIndex - 21] + '/matrixResults_' + \
                     'VES1_' + str(int(VE_S1 * 100)) + '_VES2_' + str(int(VE_S2 * 100)) + \
                     '_VE_P1_' + str(int(VE_P1 * 100)) + '_VE_P2_' + str(int(VE_P2 * 100)) + \
                     '_VE_I1_' + str(int(VE_I1 * 100)) + '_VE_I2_' + str(int(VE_I2 * 100)) + \
                     '_frac_coverage_' + str(int(fracCov * 100)) + '_fracRec_' + \
                     str(int(frac_rec * 100)) + '_redA_' + str(int(redA*100)) + '_' + numDoses2 + \
                     '_dosesPerWeek_' + today + 'SD' + str(SDval) + '_myprev_' + str(int(myprev*100)) + '6m.pickle'

    myfile = open(myfilename, 'wb')
    print(myfilename)
    pickle.dump(fullOutput, myfile)
    myfile.close()





if __name__ == "__main__":
    run_main()

    #










