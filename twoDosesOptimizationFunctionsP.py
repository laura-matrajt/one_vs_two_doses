import numpy as np
import pickle
from scipy.integrate import odeint
from scipy.optimize import minimize
from timeit import default_timer as timer
import sys
import time
# import pyswarms as ps
sys.path.insert(1, '../coronavirus_optimization/')
from matplotlib import pyplot as plt
from coronavirusMainFunctions_twoDoses import splitVaccineAmongAgeGroups, uniformSampleSimplex
from coronavirusMainFunctions_twoDoses import coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis
from coronavirusMainFunctions_twoDoses import findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis
from saveLoadFunctions import saveResults, loadResults
from runVaccination import runVaccination2DPulse6


def computeVE_P(VE_S, VE_SP):
    VE_P = 1 - ((1 - VE_SP)/(1 - VE_S))
    return VE_P

def computeVE_SP(VE_P, VE_S):
    VE_SP = 1 - (1-VE_S)*(1- VE_P)
    return VE_SP


def findFractionSus2D(y, numAgeGroups):
    """
        finds the relative fraction of  the  susceptibles, among susceptibles, exposed, asymptomatic infected,  pre-symptomatic
        infected, and recovered asymptomatic in each age group. This is valid ONLY for equations
        oronavirusEqs_withHospitalizationsAndICU_withVaccine2D assuming NO ONE has been vaccinated before.
        :param y: the state vector at the time of vaccination, it is a 34*numAgeGroups by 1 vector.
        :return: the fraction of susceptibles in each age group
        """
    temp = np.reshape(y, (34, numAgeGroups))
    relativeTotal = (temp[0, :] +  # susceptibles
                     temp[1, :] +  # exposed
                     temp[2, :] +  # asymptomatic infected
                     temp[3, :] +  # pre-symptomatic infected
                     temp[8, :])  # recovered asymptomatic

    fractionSus = np.divide(temp[0, :], relativeTotal, out=np.zeros_like(temp[0, :]), where=relativeTotal!=0)
    return fractionSus


def fromFracVacsToNumberPeopleVaccinated2d(fracVacs, numVaccinesAvailable):
    '''

    :param fracVacs: a 2xnumVaccineGroups array that represents the fraction of vaccine to be given to each vaccine group with one dose
    (first row) and two doses (second row)
    :param numVaccinesAvailable: number of vaccines available total
    :param numVaccineGroups: number of groups to vaccinate in the population
    :param totalPopByVacineGroup: number of people per vaccine group.
    :return: a 2xnumVaccineGroups array that represents the fraction of the population in each vaccine group to vaccinate
    with one dose (first row) and with two doses (second row)
    '''
    numVaccines1d = np.floor(fracVacs[0,:]*numVaccinesAvailable)  #number of vaccines to be used with the 1d schedule
    numVaccines2d = np.floor(fracVacs[1,:]*numVaccinesAvailable)  #number of vaccines to be used with the 2d schedule

    numPeopleVaccinated1d = numVaccines1d
    numPeopleVaccinated2d = np.floor(0.5*numVaccines2d)
    return np.array([numPeopleVaccinated1d, numPeopleVaccinated2d])


def fromNumPeopleVaccinatedToFracVacs(numPeopleVac, numVaccinesAvailable):
    '''

    :param numPeopleVac: an array of size 2x5, the first row people vaccinated with 1d and the second people vaccinated
    with two doses so that numPeopleVac[0,:] + 2*[numPeopleVac[1,:] < numVaccines available
    :param numVaccinesAvailable:
    :return: a 2x5 vector of fracVacs
    '''

    temp1 = numPeopleVac[0,:]
    temp2 = numPeopleVac[1,:]

    fracVacs = np.array([temp1/numVaccinesAvailable, 2*temp2/numVaccinesAvailable])
    return fracVacs

def skimOffExcessVaccine2D(fracVacs, numVaccinesAvailable, numVaccineGroups, totalPopByVaccineGroup):
    '''
    This function removes the excess vaccine in each vaccination group so that we do not vaccinate more people than what
    there is in each vaccination group. It does it in the following way:
    We compare the number of people vaccinated with 2D and with 1D to the vaccination group population in steps:
    - if numVaccinated2D[i] > pop[i]
        if this is true, then set the numVaccinated1D[i] = 0 and skim off the extra vaccine from numVaccinated2D[i]. The
        rationale being that if that combination has all of the population vaccinated with 2 doses, then no one should be
        vaccinated with one dose.
    - if numVaccinated2D[i] < pop[i], now we want to check if numVaccinated1D[i] + numVaccinated2D[i] > pop[i]. That is,
    now we want to check if vaccinating people with one and two doses exceeds the actual population.
    If this is true, then again, we "prefer" to keep the vaccination with two doses, so we keep everyone vaccinated with
    two doses and remove excess vaccinated with one dose so that the sum numVaccinated1D[i] + numVaccinated2D[i] = pop[i]
    :param fracVacs: a 2xnumVaccineGroups array that represents the fraction of vaccine to be given to each vaccine group with one dose
    (first row) and two doses (second row)
    :param numVaccinesAvailable: number of vaccines available total
    :param numVaccineGroups: number of groups to vaccinate in the population
    :param totalPopByVacineGroup: number of people per vaccine group.
    :return: an array newarray of size 2*5 where newarray[0,:] = number of people vaccinated with 1 dose EXCLUSIVELY
    and newarray[1,:] = number of people vaccinated with 2 doses
    '''

    #calculate the number of people vaccinated both with one or tow doses per vaccine group:
    [numPeopleVaccinated1d, numPeopleVaccinated2d] = fromFracVacsToNumberPeopleVaccinated2d(fracVacs, numVaccinesAvailable)
    excessVaccine = np.zeros(numVaccineGroups)


    for ivals in range(numVaccineGroups):

        temp = ((numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals]) - totalPopByVaccineGroup[ivals])


        #case 1: number of people vaccinated with two doses > pop in that group
        if (numPeopleVaccinated2d[ivals] > totalPopByVaccineGroup[ivals]):
            ## JE: MOVED THIS LINE UP TO THE TOP OF THE IF STATEMENT...
            excessVaccine[ivals] = 2*(numPeopleVaccinated2d[ivals] - totalPopByVaccineGroup[ivals]) + \
                                   numPeopleVaccinated1d[ivals]
            numPeopleVaccinated2d[ivals] = totalPopByVaccineGroup[ivals]     #skim off the excess vaccine from the 2D group

            numPeopleVaccinated1d[ivals] = 0  # set to 0 the number of people getting 1D

        #case 2: the sum of the number of people vaccinated with one and two doses > pop in that group, in this case
        # keep all of the ones vaccinated with two doses and skim off the excess in the one dose group
        if ((numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals]) > totalPopByVaccineGroup[ivals]):
            excessVaccine[ivals] = numPeopleVaccinated1d[ivals] - (totalPopByVaccineGroup[ivals] - numPeopleVaccinated2d[ivals])

            excessVaccine[ivals] = (numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals] - totalPopByVaccineGroup[ivals] )
            numPeopleVaccinated1d[ivals] = totalPopByVaccineGroup[ivals] - numPeopleVaccinated2d[ivals]
        # else:


    newarray = np.array([numPeopleVaccinated1d, numPeopleVaccinated2d])
    return [newarray, excessVaccine]



def skimOffExcessVaccine2DBis(fracVacs, numVaccinesAvailable, numVaccineGroups, totalPopByVaccineGroup):
    '''
    This function removes the excess vaccine in each vaccination group so that we do not vaccinate more people than what
    there is in each vaccination group. It does it in the following way:

    '''

    #calculate the number of people vaccinated both with one or tow doses per vaccine group:
    [numPeopleVaccinated1d, numPeopleVaccinated2d] = fromFracVacsToNumberPeopleVaccinated2d(fracVacs, numVaccinesAvailable)
    # print([numPeopleVaccinated1d, numPeopleVaccinated2d])

    # excessVaccine = np.zeros(numVaccineGroups)

    newarray = np.zeros((2,5))
    EPS = 0.5
    totVac = numPeopleVaccinated1d + numPeopleVaccinated2d
    # print(totVac)
    for ivals in range(numVaccineGroups):
        if (totVac[ivals] - totalPopByVaccineGroup[ivals]) < EPS:
           newarray[0,ivals] = numPeopleVaccinated1d[ivals]
           newarray[1,ivals] = numPeopleVaccinated2d[ivals]
        else:
            f1 = numPeopleVaccinated1d[ivals]/totVac[ivals]
            f2 = numPeopleVaccinated2d[ivals]/totVac[ivals]

            newarray[0, ivals] = f1*totalPopByVaccineGroup[ivals]
            newarray[1, ivals] = f2*totalPopByVaccineGroup[ivals]
    # print(np.sum(newarray))
    excessVaccine = numVaccinesAvailable - np.sum(newarray[0,:]) - 2*np.sum(newarray[1,:])
    # print(excessVaccine)
    return [newarray, excessVaccine]



def objectiveFunction2D(fracVacs, extraParams):
    '''

    :param fracVacs:
    :param extraParams:
    :return:
    '''
    # print('fracvacs at optimizatoin ft', (fracVacs))
    [deathRate, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
     totalPopByAgeGroup, totalPopByVaccineGroup, tspan] = extraParams
    # print(numVaccinesAvailable)
    # print(tspan)
    [newarray, excessVaccine] = skimOffExcessVaccine2DBis(fracVacs, numVaccinesAvailable, numVaccineGroups, totalPopByVaccineGroup)


    numVaccinatedAgeGroup1d = splitVaccineAmongAgeGroups(newarray[0,:], groupFracs, totalPopByAgeGroup)
    numVaccinatedAgeGroup2d = splitVaccineAmongAgeGroups(newarray[1,:], groupFracs, totalPopByAgeGroup)


    [totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU] = \
        runVaccination2DPulse6(deathRate, y0, groupFracs, numAgeGroups,  numDosesPerWeek,
                          numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups, numVaccinesAvailable, paramsODE,
                          tspan)

    mytempList = [fracVacs[0, :].tolist(), fracVacs[1, :].tolist(), newarray[0, :].tolist(), newarray[1,:].tolist(), [np.sum(excessVaccine), totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU]]
    flat_list = [item for sublist in mytempList for item in sublist]
    return flat_list










def numVaccinatedPerVaccineGroup(numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d):
    '''
    gives the number of vaccinated people in each vaccine group starting with age groups.
    :param numVaccinatedAgeGroup1d:
    :param numVaccinatedAgeGroup2d:
    :return:
    '''
    numVaccinatedVaccineGroup1d, numVaccinatedVaccineGroup2d = np.zeros(5), np.zeros(5)
    ageGroupsInVaccineGroups = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12], [13, 14], [15]]
    for ivals in range(5):
        mylist = ageGroupsInVaccineGroups[ivals]
        # print(mylist)
        for jvals in range(len(mylist)):
            numVaccinatedVaccineGroup1d[ivals] += numVaccinatedAgeGroup1d[mylist[jvals]]
            numVaccinatedVaccineGroup2d[ivals] += numVaccinatedAgeGroup2d[mylist[jvals]]




    return([numVaccinatedVaccineGroup1d, numVaccinatedVaccineGroup2d])


def evalGS(dim, numSamples, extraParams):
    '''
    This function takes a sample of size numSamples of the unit simplex and evaluates all the points of the sample
    utilitizing the function objectiveFunction2D.
    :param numSamplePoints: number of poits to be evaluated from the grid search
    :param extraParams: all the other params needed to evaluated
    :return: a matrix where each row represents a feasible vector evaluated and all of the objective functions.
    the matrix will then have size numSamples * 26 with the following structure in each row:
    columns 0:5 = fracVacs[0, :] #proportion of available vaccine to be given with 1 dose
    columns 5:10 = fracVacs[1, :] #proportion of available vaccine to be given with 2 doses
    columns 10:15 =newarray[0, :] #number of people vaccinated with 1 dose
    columns 15:20 = newarray[1,:] #number of people vaccinated with 2 doses
    column 20 = excessVaccine
    columns 21:26 =  totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU
    '''

    mysample = uniformSampleSimplex(dim, numSamples)
    ndim = np.shape(mysample)[0]
    myMat = np.zeros((ndim, 26))
    for ivals in range(ndim):
        fracVacs = mysample[ivals]
        fracVacs = fracVacs.reshape(2,5)
        # print(fracVacs)
        # print(np.sum(fracVacs))

        temp = objectiveFunction2D(fracVacs, extraParams)
        myMat[ivals, :] = temp

    return myMat

###########################################################################################








###########################################################################################



###########################################################################################

def repairVector2D(fracVacs):
    '''
    Repairs a fracVacs vector so that if it sums up to more than 1 it sums up to exactly one.
    :param fracVacs:
    :return:
    '''
    fracVacs = fracVacs.reshape(10)
    fracVacs[fracVacs < 0] = 0  #change negative entries to 0

    vectorSum = np.sum(fracVacs)
    # print(vectorSum)
    if vectorSum > 1: #check if the sum of entries is bigger than 1, if it is, repair it so that it is exactly one
        newFracVacs = np.array([fracVacs[i]/vectorSum for i in range(len(fracVacs))])  #repair the vector
    else:
        newFracVacs = fracVacs          #or keep the original one
    return newFracVacs.reshape((2,5))


def repairVector2DBothSides(fracVacs):
    '''
    Repairs a fracVacs vector so that if it sums up to more than 1 it sums up to exactly one.
    :param fracVacs:
    :return:
    '''
    fracVacs = fracVacs.reshape(10)
    fracVacs[fracVacs < 0] = 0  #change negative entries to 0

    vectorSum = np.sum(fracVacs)
    # print(vectorSum)
    # if vectorSum > 1: #check if the sum of entries is bigger than 1, if it is, repair it so that it is exactly one
    newFracVacs = np.array([fracVacs[i]/vectorSum for i in range(len(fracVacs))])  #repair the vector
    # else:
    #     newFracVacs = fracVacs          #or keep the original one
    return newFracVacs.reshape((2,5))


def repairNumPeopleToVaccinate(numPeopleToVac, numVaccinesAvailable, totalPopByAgeGroup):
    '''
    radially increase the vaccine uptake radially.
    :param numPeopleToVac: array of the form:
     [numPeopleVaccinated1d,
     numPeopleVaccinated2d]
    :param numVaccinesAvailable:
    :param totalPopByAgeGroup:
    :return:
    '''
    numPeopleVaccinated1d= numPeopleToVac[0,:]
    numPeopleVaccinated2d = numPeopleToVac[1,:]
    temp = np.sum(numPeopleVaccinated1d + 2*numPeopleVaccinated2d)
    # print(temp)
    newPeoVac = np.zeros((2,5))
    alpha = (numVaccinesAvailable/temp)
    newNumPeopleVaccinated2d = alpha*numPeopleVaccinated2d
    excessVaccine = numVaccinesAvailable - (numPeopleVaccinated1d + 2*newNumPeopleVaccinated2d)

    alpha1d = np.min
    for ivals in range(5):
        newPeoVac[0, ivals] = np.minimum(totalPopByAgeGroup[ivals], alpha*numPeopleVaccinated1d[ivals])
        newPeoVac[1, ivals] = np.minimum(totalPopByAgeGroup[ivals], alpha * numPeopleVaccinated2d[ivals])
    return newPeoVac


def repairNumPeopleToVaccinate1Dose(numPeopleToVac, numVaccinesAvailable, totalPopByAgeGroup):
    '''
    radially increase the vaccine uptake in two steps. First in the second dose direction vector, and then in the one dose
    direction vector
    :param numPeopleToVac: array of the form:
     [numPeopleVaccinated1d,
     numPeopleVaccinated2d]
    :param numVaccinesAvailable:
    :param totalPopByAgeGroup:
    :return:
    '''
    numPeopleVaccinated1d= numPeopleToVac[0,:]
    numPeopleVaccinated2d = numPeopleToVac[1,:]
    temp = np.sum(numPeopleVaccinated1d)
    # print(temp)
    alpha = np.minimum(numVaccinesAvailable, numVaccinesAvailable/temp)
    newNumPeopleVaccinated2d = alpha*numPeopleVaccinated2d
    excessVaccine = numVaccinesAvailable - (numPeopleVaccinated1d + 2*newNumPeopleVaccinated2d)

    newPeoVac = np.zeros((5))
    for ivals in range(5):
        temp = np.maximum(alpha * numPeopleVaccinated1d[ivals], numPeopleVaccinated1d[ivals])
        newPeoVac[ivals] = np.minimum(totalPopByAgeGroup[ivals]-numPeopleVaccinated2d[ivals], temp)


    return np.array([newPeoVac, numPeopleVaccinated2d])


def objectiveFunction_NM2D(fracVacs, extraParamsNM):
    '''
    This is the objective function that will be passed to Nelder-Mead. It evaluates the function objectiveFunction2D
    for the decision variable fracVacs given. Depending on the variable myIndex, it returns the appropriate objective.
    Because NM will try decision variables that will NOT satisfy the constraints a priori, we need to first check that
    this particular fracVacs satisfies being between 0 and 1 and "repair" it if it is not.
    '''

    #output of objectiveFunction2D is as follows
    #fracVacs[0, :], fracVacs[1, :], newarray[0, :], newarray[1,:], np.sum(excessVaccine), totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU
    # index 0 to 4 = fracVacs[0, :]
    # index 5 to 9 = fracVacs[1, :]
    # index 10 to 14 = newarray[0, :]
    # index 15 to 19 = newarray[1,:]
    # index 20 = totalExcessvaccine
    # index 21 = totalInfections
    # index 22 = totalSymptomaticInfections
    # index 23 = totalDeaths
    # index 24 = maxHosp
    # index 25 = maxICU
    # print('entre aqui')
    [extraParams, myIndex, repairFunOpt] = extraParamsNM
    [deathRate, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
     totalPopByAgeGroup, totalPopByVaccineGroup, tspan] = extraParams
    # print(fracVacs)
    #repair the vector of fracVacs before passing it to NM:
    if repairFunOpt == 0:
        # newFracVacs = repairVector2DBothSides(fracVacs)
        newFracVacs = repairVector2D(fracVacs)
    # elif repairFunOpt == 1:
    #     newFracVacs = repairVector2D_quadratic_subproblem_INEQUALITY(fracVacs, numVaccinesAvailable, totalPopByVaccineGroup)
    # elif repairFunOpt == 2:
    #     newFracVacs = repairVector2D_quadratic_subproblem(fracVacs, numVaccinesAvailable,
    #                                                                  totalPopByVaccineGroup)

    else:
        print('wrong repair function value')
        sys.exit()
    # print(np.sum(newFracVacs))
    modelOutput = objectiveFunction2D(newFracVacs, extraParams)
    # print(float(modelOutput[myIndex]))
    return float(modelOutput[myIndex])



def objectiveFunction_PS2D(fracs, extraParamsPS):
    # output of objectiveFunction2D is as follows
    # fracVacs[0, :], fracVacs[1, :], newarray[0, :], newarray[1,:], np.sum(excessVaccine), totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU
    # index 0 to 4 = fracVacs[0, :]
    # index 5 to 9 = fracVacs[1, :]
    # index 10 to 14 = newarray[0, :]
    # index 15 to 19 = newarray[1,:]
    # index 20 = totalExcessvaccine
    # index 21 = totalInfections
    # index 22 = totalSymptomaticInfections
    # index 23 = totalDeaths
    # index 24 = maxHosp
    # index 25 = maxICU
    #extraParams =  [deathRate, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
    #  totalPopByAgeGroup, totalPopByVaccineGroup, tspan]
    [extraParams, myIndex, repairFunOpt] = extraParamsPS
    [deathRate, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
     totalPopByAgeGroup, totalPopByVaccineGroup, tspan] = extraParams
    d = fracs.shape[0]
    myOutput = np.zeros(d)
    for ivals in range(d):
        fracVacs = fracs[ivals].reshape(2, numVaccineGroups)
        if repairFunOpt == 0:
            # newFracVacs = repairVector2DBothSides(fracVacs)
            newFracVacs = repairVector2D(fracVacs)
        # elif repairFunOpt == 1:
        #
        #     newFracVacs = repairVector2D_quadratic_subproblem_INEQUALITY(fracVacs, numVaccinesAvailable,
        #                                                                  totalPopByVaccineGroup)
        else:
            print('wrong repair function value')
            sys.exit()

        modelOutput = objectiveFunction2D(newFracVacs, extraParams)
        myOutput[ivals] = modelOutput[myIndex]
    # print(myOutput)
    return myOutput



def createProRataVac2D(fracOfTotalPopulationPerVaccineGroup, numVaccinesAvailable,  totalPopByVaccineGroup):
    '''
    creates a proRata vector with one or two doses as follows:
    given a numVaccinesAvailable, it will first vaccinate everyone (or as many people as possible in the population) with
    single dose, then it will allocate the remainder vaccine to the vaccination groups in a pro-rata fashion
    :param numVaccinesAvailable: number of vaccine doses available
    :return: a vector of vaccine allocation as described above
    '''

    # check to see if there is enough vaccine to cover everyone, if there is not, then allocate the vaccine doses proportionally to the
    # population with one dose of vaccine
    if numVaccinesAvailable < np.sum(totalPopByVaccineGroup):
        proRata1D = fracOfTotalPopulationPerVaccineGroup
        proRata2D = np.zeros(len(totalPopByVaccineGroup))
    elif numVaccinesAvailable < 2*np.sum(totalPopByVaccineGroup): # check that there is enough vaccine for everyone to get
        #one dose but not enough for everyone to get two doses:
        #compute the extra number of doses
        excessVaccine = numVaccinesAvailable - np.sum(totalPopByVaccineGroup)
        # print(excessVaccine)
        #numVaccines1d = totalPop5 #everyone gets one dose
        numVaccines2d = np.floor(excessVaccine*fracOfTotalPopulationPerVaccineGroup) #number of people who will get an additional second dose
        # print(numVaccines2d)
        ## Since everyone already has 1 dose in this case, 
        ## the  number of people who get a second dose 
        ## equals numVaccines2d and equals the excessVaccine*fraction.
        numPeopleVaccinatedWith2D = numVaccines2d
        numPeopleVaccinatedWith1D = totalPopByVaccineGroup - numPeopleVaccinatedWith2D

        proRata1D = numPeopleVaccinatedWith1D/numVaccinesAvailable
        ## the proRata vector is _number of vaccines_ which is why
        ## the 2D is multiplied by two.
        proRata2D = 2*numPeopleVaccinatedWith2D/numVaccinesAvailable
        # print([proRata1D, proRata2D])

    else: #enough vaccine for everyone to get 2 doses, so set that up:
        proRata1D = np.zeros(len(totalPopByVaccineGroup))
        proRata2D = fracOfTotalPopulationPerVaccineGroup

    # print(np.sum(proRata1D + proRata2D))
    proRataVec = np.array([proRata1D, proRata2D])
    return proRataVec


def createProRataVac2DAdultsOnly(numVaccinesAvailable,  totalPopByVaccineGroup):
    '''
    only works with 5 vaccination groups! where children are all inteh first group
    creates a proRata vector with one or two doses as follows:
    given a numVaccinesAvailable, it will first vaccinate everyone (or as many people as possible in the population) with
    single dose, then it will allocate the remainder vaccine to the vaccination groups in a pro-rata fashion
    :param numVaccinesAvailable: number of vaccine doses available
    :return: a vector of vaccine allocation as described above
    '''

    proRataVec = np.zeros((2, 5))
    popAdults = totalPopByVaccineGroup[1:]
    totalPopAdults = np.sum(popAdults)
    relFracAdults = np.divide(popAdults, totalPopAdults)

    if numVaccinesAvailable > totalPopAdults:
        proRataVec[0, 1:] = relFracAdults

    else:
        temp = createProRataVac2D(relFracAdults, numVaccinesAvailable, popAdults)

    # print(temp)
        proRataVec[0, 1:] = temp[0]
        proRataVec[1, 1:] = temp[1]

    return proRataVec




def createVectorHighRiskFirst(numVaccinesAvailable,  totalPopByVaccineGroup):
    """
    This vector will allocate two doses of vaccine to the high-risk groups (>65 y0) if available, and then a single dose to
    anyone else going in decreasing order. If there is not enough vaccine to cover the high-risk groups with two doses
    it will try to allocate one dose to all of >65 and the remaining to the >75 with an additional dose.
    :param fracOfTotalPopulationPerVaccineGroup: vector representing the fraction of the population in each vaccination
    group
    :param numVaccinesAvailable: number of vaccines available
    :param totalPopByVaccineGroup: vector representing the number of people in each vaccination group
    :return: a vector representing the fraction of vaccine to be allocated to each vaccination group
    """
    totalPopByVaccineGroup = np.floor(totalPopByVaccineGroup)
    fracVacs1d = np.zeros(5)
    fracVacs2d = np.zeros(5)
    numVaccinesByGroup = np.zeros(3)
    EPS = 0.5

    #allocate two doses of vaccine to the oldest age-groups:

    #CASE 1: not enough vaccine for all the people> 65 to get one dose:
    if numVaccinesAvailable < (totalPopByVaccineGroup[3] + totalPopByVaccineGroup[4]):
        # subcases 1.1 not enough vaccine to cover even those >75 with one dose
        if numVaccinesAvailable < totalPopByVaccineGroup[4]:
            fracVacs1d[4] = 1 #if there is very little vaccine not enough to cover hte oldest age group, allocate all of the vaccine to them

        # subcases 1.2: enough vaccine to cover all of those >75 with one dose, but not all of the ones 65-75 yo
        else:
            fracVacs1d[4] = totalPopByVaccineGroup[4]/numVaccinesAvailable
            fracVacs1d[3] = (numVaccinesAvailable - totalPopByVaccineGroup[4])/numVaccinesAvailable

    #CASE 2: enough vaccien to cover >65 with one dose but not enough vaccine for all the people> 65 to get two doses:
    elif numVaccinesAvailable  < 2*(totalPopByVaccineGroup[3] + totalPopByVaccineGroup[4]):
        #subcase 2.1 enough vaccine to give two doses to people >75:
        if numVaccinesAvailable > 2*totalPopByVaccineGroup[4]:
            # print('entre aqui')
            fracVacs2d[4] = 2*totalPopByVaccineGroup[4]/numVaccinesAvailable
            excess = numVaccinesAvailable - 2*totalPopByVaccineGroup[4]

            #subcase 2.1.1: enough vaccine to give 2 doses to >75 and to give those 65-75 one dose and there is vaccine left:
            if excess > totalPop5[3]:
                fracVacs1d[3] = totalPop5[3]/numVaccinesAvailable #vaccinate 65-75 with one dose
                excess2 = excess - totalPop5[3]                   #compute the excess vaccine
                #allocate excess for second doses to the 65-75 yo
                fracVacs2d[3] = (excess2)/numVaccinesAvailable

            # subcase 2.1.2: enough vaccine to give 2 doses to >75 but not enough to get all of the 65-75 one dose
            else: #not enough to get all of the 65-75 one dose
                fracVacs1d[3] = excess/numVaccinesAvailable       #vaccinate as many as 65-75 yo with a single dose

    #CASE 3: enough vaccine to cover all the >65 with two doses and have left over vaccine
    else:
        fracVacs2d[3] = 2 * totalPopByVaccineGroup[3] / numVaccinesAvailable
        fracVacs2d[4] = 2 * totalPopByVaccineGroup[4] / numVaccinesAvailable

        excessVaccine = numVaccinesAvailable - 2*(np.sum(totalPopByVaccineGroup[3:5]))

        #vaccinate every group with a single dose in descending order
        # numVaccinesByGroup = np.zeros(3)
        vaccineSaturationByGroup = np.zeros(3)
        reallocationOrder = list(range(3 - 1, -1, -1))
        numUnvaccinatedByGroup = totalPop5[0:3]


        #allocates vaccine to each group 0-65 yo with one dose
        for i in reallocationOrder:
            # print(i)
            if vaccineSaturationByGroup[i] == 0:
                ##
                # excessVaccine - numUnvaccinatedByGroup[i] < EPS
                ## If there is less excess vaccine than
                ## remaining people in group to be vaccinated,
                ## put all excess vaccine in that group.
                if excessVaccine - numUnvaccinatedByGroup[i] < EPS:
                    # print("case 1")
                    numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                    excessVaccine = 0
                    numUnvaccinatedByGroup[i] = numUnvaccinatedByGroup[i] - excessVaccine
                    ## If excessVaccine equals numUnvaccinatedByGroup[i]
                    ## then additionally, that group is now saturated.
                    if abs(excessVaccine - numUnvaccinatedByGroup[i]) < EPS:
                        vaccineSaturationByGroup[i] = 1
                ## Otherwise, vaccinate everyone remaining in the group
                ## and reset the excess vaccine to be reallocated
                ## in the next group.
                else:
                    # print("case 2")
                    excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                    numVaccinesByGroup[i] = totalPopByVaccineGroup[i]
                    vaccineSaturationByGroup[i] = 1


        #################### check this when come back from vacation #########################
        #repeat this same process with the excess vaccine:
        vaccineSaturationByGroup = np.zeros(3)
        numUnvaccinatedByGroup = totalPop5[0:3]
        #allocates vaccine to each group 0-65 yo with one dose
        for i in reallocationOrder:
            # print(i)
            if vaccineSaturationByGroup[i] == 0:
                ##
                # excessVaccine - numUnvaccinatedByGroup[i] < EPS
                ## If there is less excess vaccine than
                ## remaining people in group to be vaccinated,
                ## put all excess vaccine in that group.
                if excessVaccine - numUnvaccinatedByGroup[i] < EPS:
                    # print("case 1")
                    numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                    excessVaccine = 0
                    ## If excessVaccine equals numUnvaccinatedByGroup[i]
                    ## then additionally, that group is now saturated.
                    if abs(excessVaccine - numUnvaccinatedByGroup[i]) < EPS:
                        vaccineSaturationByGroup[i] = 1
                ## Otherwise, vaccinate everyone remaining in the group
                ## and reset the excess vaccine to be reallocated
                ## in the next group.
                else:
                    # print("case 2")
                    excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                    numVaccinesByGroup[i] = numVaccinesByGroup[i] + totalPopByVaccineGroup[i]
                    vaccineSaturationByGroup[i] = 1

    # fracVacs1d[0:3] = np.divide(numVaccinesByGroup, numVaccinesAvailable)
    for ivals in range(3):

        if numVaccinesByGroup[ivals] - totalPopByVaccineGroup[ivals] < EPS:
            fracVacs1d[ivals] = numVaccinesByGroup[ivals]/numVaccinesAvailable
        else:
            if numVaccinesByGroup[ivals] - 2*totalPopByVaccineGroup[ivals] < EPS:
                fracVacs2d[ivals] = (numVaccinesByGroup[ivals])/numVaccinesAvailable
            else:
                fracVacs2d[ivals] = 2*totalPopByVaccineGroup[ivals]/numVaccinesAvailable
                fracVacs1d[ivals] = (numVaccinesByGroup[ivals] - 2*totalPopByVaccineGroup[ivals])/numVaccinesAvailable

    return np.array([fracVacs1d, fracVacs2d])

def createVectorHighRiskFirst2(numVaccinesAvailable,  totalPopByVaccineGroup):
    """
    This vector will allocate two doses of vaccine to the high-risk groups (>65 y0) if available, and then a single dose to
    anyone else going in decreasing order. If there is not enough vaccine to cover the high-risk groups with two doses
    it will try to allocate one dose to all of >65 and the remaining to the >75 with an additional dose.
    :param fracOfTotalPopulationPerVaccineGroup: vector representing the fraction of the population in each vaccination
    group
    :param numVaccinesAvailable: number of vaccines available
    :param totalPopByVaccineGroup: vector representing the number of people in each vaccination group
    :return: a vector representing the fraction of vaccine to be allocated to each vaccination group
    """
    totalPopByVaccineGroup = np.floor(totalPopByVaccineGroup)
    # print(totalPopByVaccineGroup)
    fracVacs1d = np.zeros(5)
    fracVacs2d = np.zeros(5)

    numVaccinesByGroup = np.zeros(3)
    EPS = 0.5
    reallocationOrder = list(range(3 - 1, -1, -1))
    # print(reallocationOrder)
    #allocate two doses of vaccine to the oldest age-groups:

    #CASE 1: not enough vaccine for all the people> 65 to get one dose:
    if numVaccinesAvailable < (totalPopByVaccineGroup[3] + totalPopByVaccineGroup[4]):
        # print('case1')
        # subcases 1.1 not enough vaccine to cover even those >75 with one dose
        if numVaccinesAvailable < totalPopByVaccineGroup[4]:
            fracVacs1d[4] = 1 #if there is very little vaccine not enough to cover hte oldest age group, allocate all of the vaccine to them

        # subcases 1.2: enough vaccine to cover all of those >75 with one dose, but not all of the ones 65-75 yo
        else:
            fracVacs1d[4] = totalPopByVaccineGroup[4]/numVaccinesAvailable
            fracVacs1d[3] = (numVaccinesAvailable - totalPopByVaccineGroup[4])/numVaccinesAvailable

    #CASE 2: enough vaccien to cover >65 with one dose but not enough vaccine for all the people> 65 to get two doses:
    elif numVaccinesAvailable  < 2*(totalPopByVaccineGroup[3] + totalPopByVaccineGroup[4]):
        # print('case2')
        #subcase 2.1 enough vaccine to give two doses to people >75:
        if numVaccinesAvailable > 2*totalPopByVaccineGroup[4]:
            # print('entre aqui')
            fracVacs2d[4] = 2*totalPopByVaccineGroup[4]/numVaccinesAvailable
            excess = numVaccinesAvailable - 2*totalPopByVaccineGroup[4]

            #subcase 2.1.1: enough vaccine to give 2 doses to >75 and to give those 65-75 one dose and there is vaccine left:
            if excess > totalPopByVaccineGroup[3]:
                fracVacs1d[3] = totalPopByVaccineGroup[3]/numVaccinesAvailable #vaccinate 65-75 with one dose
                excess2 = excess - totalPopByVaccineGroup[3]                   #compute the excess vaccine
                #allocate excess for second doses to the 65-75 yo
                fracVacs2d[3] = (excess2)/numVaccinesAvailable

            # subcase 2.1.2: enough vaccine to give 2 doses to >75 but not enough to get all of the 65-75 one dose
            else: #not enough to get all of the 65-75 one dose
                fracVacs1d[3] = excess/numVaccinesAvailable       #vaccinate as many as 65-75 yo with a single dose

    #CASE 3: enough vaccine to cover all the >65 with two doses and have left over vaccine
    else:
        # print('case3')

        fracVacs2d[3] = 2 * totalPopByVaccineGroup[3] / numVaccinesAvailable
        fracVacs2d[4] = 2 * totalPopByVaccineGroup[4] / numVaccinesAvailable

        excessVaccine = numVaccinesAvailable - 2*(np.sum(totalPopByVaccineGroup[3:5]))
        # print(excessVaccine)
        #subcase 3.1 enough vaccine to get all the other groups one dose:
        if excessVaccine - np.sum(totalPopByVaccineGroup[0:3]) > EPS:
            # print('case 3.1')
            numVaccinesByGroup = totalPopByVaccineGroup[0:3]
            excessVaccine = excessVaccine - np.sum(totalPopByVaccineGroup[0:3])

            #allocate the remainder vaccine with two doses in descending order.
            vaccineSaturationByGroup = np.zeros(3)
            numUnvaccinatedByGroup = totalPop5[0:3]

            for i in reallocationOrder:
                # print(i)
                if vaccineSaturationByGroup[i] == 0:
                    mydiff = excessVaccine - numUnvaccinatedByGroup[i]
                    # excessVaccine - numUnvaccinatedByGroup[i] < EPS
                    ## If there is less excess vaccine than
                    ## remaining people in group to be vaccinated,
                    ## put all excess vaccine in that group.
                    if mydiff < EPS:
                        numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                        excessVaccine = 0
                        numUnvaccinatedByGroup[i] = numUnvaccinatedByGroup[i] - excessVaccine
                    else: #there is more vaccine than people in that group
                        numVaccinesByGroup[i] = numVaccinesByGroup[i] + numUnvaccinatedByGroup[i]
                        excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                        numUnvaccinatedByGroup[i] = 0
                        vaccineSaturationByGroup[i] = 1
        #subcase 3.2 #no vaccine for everyone to have a single dose, allocate one dose vaccines to the vaccination groups in descending
        #order
        else:
            # print('case 3.2')
            # print('excess vaccine', excessVaccine)
            vaccineSaturationByGroup = np.zeros(3)
            numUnvaccinatedByGroup = np.copy(totalPopByVaccineGroup[0:3])
            for i in reallocationOrder:
                # print(i)
                if vaccineSaturationByGroup[i] == 0:
                    mydiff = excessVaccine - numUnvaccinatedByGroup[i]
                    # excessVaccine - numUnvaccinatedByGroup[i] < EPS
                    ## If there is less excess vaccine than
                    ## remaining people in group to be vaccinated,
                    ## put all excess vaccine in that group.
                    if mydiff < EPS:
                        # print('diff < eps')
                        numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                        excessVaccine = 0
                        numUnvaccinatedByGroup[i] = numUnvaccinatedByGroup[i] - excessVaccine
                        # print(totalPopByVaccineGroup)
                    else: #there is more vaccine than people in that group
                        # print('diff > eps')
                        numVaccinesByGroup[i] = numVaccinesByGroup[i] + numUnvaccinatedByGroup[i]
                        excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                        numUnvaccinatedByGroup[i] = 0
                        vaccineSaturationByGroup[i] = 1
            # print(np.divide(numVaccinesByGroup, numVaccinesAvailable))
            # print(np.sum(numVaccinesByGroup))
            # print(excessVaccine)

    for ivals in range(3):
        # print('entre aqui')
        mydiff = numVaccinesByGroup[ivals] - totalPopByVaccineGroup[ivals]
        if mydiff < EPS:
            fracVacs1d[ivals] = numVaccinesByGroup[ivals]/numVaccinesAvailable
        else:
            numPeopleVacWith1d = totalPopByVaccineGroup[ivals] - mydiff
            fracVacs1d[ivals] = numPeopleVacWith1d/numVaccinesAvailable
            fracVacs2d[ivals] = (totalPopByVaccineGroup[ivals] - numPeopleVacWith1d)/numVaccinesAvailable
    # print('aqui', totalPopByVaccineGroup)
    # print(fracVacs1d)
    # print(fracVacs2d)
    # print(np.sum(fracVacs1d + fracVacs2d))
    # print(np.sum(fracVacs1d*numVaccinesAvailable + fracVacs2d*numVaccinesAvailable)/numVaccinesAvailable)
    return np.array([fracVacs1d, fracVacs2d])

def createVectorPracticalStrategy(numVaccinesAvailable,  totalPopByVaccineGroup):
    """
    This vector will allocate two doses of vaccine to the high-risk groups (>65 y0) if available, and then a single dose to
    anyone else in adult age groups pro-rata.
    :param numVaccinesAvailable: number of vaccines available
    :param totalPopByVaccineGroup: vector representing the number of people in each vaccination group
    :return: a vector representing the fraction of vaccine to be allocated to each vaccination group
    """
    #check there is just enough vaccine or less to cover everyone < 65 with one dose and everyoen 65+ with two
    if numVaccinesAvailable > (np.sum(totalPopByVaccineGroup[0:3]) + 2*np.sum(totalPopByVaccineGroup[3:5])):
        print('too much vaccine for thsi function ')
        sys.exit()

    totalPopByVaccineGroup = np.floor(totalPopByVaccineGroup)
    # print(totalPopByVaccineGroup)
    fracVacs1d = np.zeros(5)
    fracVacs2d = np.zeros(5)

    numVaccinesByGroup = np.zeros(3)
    EPS = 0.5
    reallocationOrder = list(range(3 - 1, -1, -1))
    # print(reallocationOrder)
    #allocate two doses of vaccine to the oldest age-groups:

    #CASE 1: not enough vaccine for all the people> 65 to get one dose:
    if numVaccinesAvailable < 2*(totalPopByVaccineGroup[4]):
        # print('case1')
        fracVacs2d[4] = 1

    elif numVaccinesAvailable  < 2*(totalPopByVaccineGroup[3] + totalPopByVaccineGroup[4]):
        #enough vaccine to cover 75+ with two doses but not those 65-75
        fracVacs2d[4] = 2*totalPopByVaccineGroup[4]/numVaccinesAvailable
        excessVaccine = numVaccinesAvailable - 2*totalPopByVaccineGroup[4]

        #allocate the excess vaccine to the 65-75
        fracVacs2d[3] = excessVaccine/numVaccinesAvailable
    else:
        fracVacs2d[4] = 2 * totalPopByVaccineGroup[4] / numVaccinesAvailable
        fracVacs2d[3] = 2 * totalPopByVaccineGroup[3] / numVaccinesAvailable
        excessVaccine = numVaccinesAvailable - 2*(np.sum(totalPopByVaccineGroup[3:5]))
        relFrac = totalPopByVaccineGroup[1]/(totalPopByVaccineGroup[1] + totalPopByVaccineGroup[2])
        fracVacs1d[1] = relFrac*excessVaccine/numVaccinesAvailable
        fracVacs1d[2] = (1-relFrac)*excessVaccine/numVaccinesAvailable
    return np.array([fracVacs1d, fracVacs2d])

def createVectorPracticalStrategy(numVaccinesAvailable,  totalPopByVaccineGroup):
    """
    This vector will allocate two doses of vaccine to the high-risk groups (>65 y0) if available, and then a single dose to
    anyone else in adult age groups pro-rata.
    :param numVaccinesAvailable: number of vaccines available
    :param totalPopByVaccineGroup: vector representing the number of people in each vaccination group
    :return: a vector representing the fraction of vaccine to be allocated to each vaccination group
    """
    #check there is just enough vaccine or less to cover everyone < 65 with one dose and everyoen 65+ with two
    if numVaccinesAvailable > (np.sum(totalPopByVaccineGroup[0:3]) + 2*np.sum(totalPopByVaccineGroup[3:5])):
        print('too much vaccine for thsi function ')
        sys.exit()

    totalPopByVaccineGroup = np.floor(totalPopByVaccineGroup)
    # print(totalPopByVaccineGroup)
    fracVacs1d = np.zeros(5)
    fracVacs2d = np.zeros(5)

    numVaccinesByGroup = np.zeros(3)
    EPS = 0.5
    reallocationOrder = list(range(3 - 1, -1, -1))
    # print(reallocationOrder)
    #allocate two doses of vaccine to the oldest age-groups:

    #CASE 1: not enough vaccine for all the people> 65 to get one dose:
    if numVaccinesAvailable < 2*(totalPopByVaccineGroup[4]):
        # print('case1')
        fracVacs2d[4] = 1

    elif numVaccinesAvailable  < 2*(totalPopByVaccineGroup[3] + totalPopByVaccineGroup[4]):
        #enough vaccine to cover 75+ with two doses but not those 65-75
        fracVacs2d[4] = 2*totalPopByVaccineGroup[4]/numVaccinesAvailable
        excessVaccine = numVaccinesAvailable - 2*totalPopByVaccineGroup[4]

        #allocate the excess vaccine to the 65-75
        fracVacs2d[3] = excessVaccine/numVaccinesAvailable
    else:
        fracVacs2d[4] = 2 * totalPopByVaccineGroup[4] / numVaccinesAvailable
        fracVacs2d[3] = 2 * totalPopByVaccineGroup[3] / numVaccinesAvailable
        excessVaccine = numVaccinesAvailable - 2*(np.sum(totalPopByVaccineGroup[3:5]))
        relFrac = totalPopByVaccineGroup[1]/(totalPopByVaccineGroup[1] + totalPopByVaccineGroup[2])
        fracVacs1d[1] = relFrac*excessVaccine/numVaccinesAvailable
        fracVacs1d[2] = (1-relFrac)*excessVaccine/numVaccinesAvailable
    return np.array([fracVacs1d, fracVacs2d])



def createVectorHighRiskFirst2DosesOnly(numVaccinesAvailable, numVaccineGroups,  totalPopByVaccineGroup):
    """
    This vector will allocate two doses of vaccine to the high-risk groups (>65 y0) if available, and then
    anyone else going in decreasing order with two doses only.
    :param fracOfTotalPopulationPerVaccineGroup: vector representing the fraction of the population in each vaccination
    group
    :param numVaccinesAvailable: number of vaccines available
    :param totalPopByVaccineGroup: vector representing the number of people in each vaccination group
    :return: a vector representing the fraction of vaccine to be allocated to each vaccination group
    """
    ## below, we're doublin the population so that it's referring to doses now not pepole since
    ## each person gets two doses.
    totalPopByVaccineGroupTemp = np.floor(2*totalPopByVaccineGroup)
    # print(totalPopByVaccineGroupTemp)
    fracVacs1d = np.zeros(numVaccineGroups)
    fracVacs2d = np.zeros(numVaccineGroups)

    numVaccinesByGroup = np.zeros(numVaccineGroups)
    EPS = 0.5
    reallocationOrder = list(range(numVaccineGroups-1,-1,-1))
    # print(reallocationOrder)
    #allocate two doses of vaccine to the oldest age-groups:

    vaccineSaturationByGroup = np.zeros(numVaccineGroups)
    numUnvaccinatedByGroup = totalPopByVaccineGroupTemp
    excessVaccine = np.floor(numVaccinesAvailable)
    # print(excessVaccine)
    for i in reallocationOrder:
        # print(i)
        # print(excessVaccine)
        # print(excessVaccine - numUnvaccinatedByGroup[i])
        if vaccineSaturationByGroup[i] == 0:
            ##
            # excessVaccine - numUnvaccinatedByGroup[i] < EPS
            ## If there is less excess vaccine than
            ## remaining people in group to be vaccinated,
            ## put all excess vaccine in that group.
            if excessVaccine - numUnvaccinatedByGroup[i] < EPS:
                # print("case 1")
                numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                # print('numVaccinesByGroup[i]', numVaccinesByGroup[i])
                excessVaccine = 0
                ## If excessVaccine equals numUnvaccinatedByGroup[i]
                ## then additionally, that group is now saturated.
                if abs(excessVaccine - numUnvaccinatedByGroup[i]) < EPS:
                    vaccineSaturationByGroup[i] = 1
            ## Otherwise, vaccinate everyone remaining in the group
            ## and reset the excess vaccine to be reallocated
            ## in the next group.
            else:
                # print("case 2")
                excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                # print(excessVaccine)
                numVaccinesByGroup[i] = totalPopByVaccineGroupTemp[i]
                vaccineSaturationByGroup[i] = 1
    # print(numVaccinesByGroup)

    fracVacs2d = np.divide(numVaccinesByGroup, numVaccinesAvailable)
    # print(np.sum(fracVacs2d))
    return np.array([fracVacs1d, fracVacs2d])




def createVectorHighRiskFirst2DosesOnlyAtPopLevel(numVaccinesAvailable, numAgeGroups,  totalPopByAgeGroup):
    """
    This vector will allocate two doses of vaccine to the high-risk groups (>65 y0) if available, and then
    anyone else going in decreasing order with two doses only.
    :param fracOfTotalPopulationPerVaccineGroup: vector representing the fraction of the population in each vaccination
    group
    :param numVaccinesAvailable: number of vaccines available
    :param totalPopByVaccineGroup: vector representing the number of people in each vaccination group
    :return: a vector representing the fraction of vaccine to be allocated to each vaccination group
    """
    ## below, we're doublin the population so that it's referring to doses now not pepole since
    ## each person gets two doses.
    totalPopByVaccineGroupTemp = np.floor(2*totalPopByAgeGroup)
    # print(totalPopByVaccineGroupTemp)
    fracVacs1d = np.zeros(numAgeGroups)
    fracVacs2d = np.zeros(numAgeGroups)

    numVaccinesByGroup = np.zeros(numAgeGroups)
    EPS = 0.5
    reallocationOrder = list(range(numAgeGroups-1,-1,-1))
    # print(reallocationOrder)
    #allocate two doses of vaccine to the oldest age-groups:

    vaccineSaturationByGroup = np.zeros(numAgeGroups)
    numUnvaccinatedByGroup = totalPopByVaccineGroupTemp
    excessVaccine = np.floor(numVaccinesAvailable)
    # print(excessVaccine)
    for i in reallocationOrder:
        # print(i)
        # print(excessVaccine)
        # print(excessVaccine - numUnvaccinatedByGroup[i])
        if vaccineSaturationByGroup[i] == 0:
            ##
            # excessVaccine - numUnvaccinatedByGroup[i] < EPS
            ## If there is less excess vaccine than
            ## remaining people in group to be vaccinated,
            ## put all excess vaccine in that group.
            if excessVaccine - numUnvaccinatedByGroup[i] < EPS:
                # print("case 1")
                numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                # print('numVaccinesByGroup[i]', numVaccinesByGroup[i])
                excessVaccine = 0
                ## If excessVaccine equals numUnvaccinatedByGroup[i]
                ## then additionally, that group is now saturated.
                if abs(excessVaccine - numUnvaccinatedByGroup[i]) < EPS:
                    vaccineSaturationByGroup[i] = 1
            ## Otherwise, vaccinate everyone remaining in the group
            ## and reset the excess vaccine to be reallocated
            ## in the next group.
            else:
                # print("case 2")
                excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                # print(excessVaccine)
                numVaccinesByGroup[i] = totalPopByVaccineGroupTemp[i]
                vaccineSaturationByGroup[i] = 1
    # print(numVaccinesByGroup)

    # fracVacs2d = np.divide(numVaccinesByGroup, numVaccinesAvailable)
    # print(np.sum(fracVacs2d))
    return np.array([fracVacs1d, np.floor(0.5*numVaccinesByGroup)])


def optimizationSimplexAndNM(myMat, highRiskFirstVec, numOfBestSols, numOfBestSolsNM, proRataVec2D, extraParamsNM):
    '''
    runs the NM algorithm using numBestSols best solutions from the matrix myMat as starting points.
     It also tries out the proRataVector as part of the solutions.
    :param myMat: matrix with the vectors and solutions obtained previously by randomly sampling the simplex and evaluating
    the objective functions there
    :param numOfBestSols: number of best solutions to take from the matrix
    :param numOfBestSolsNM: number of best solutions to return from NM
    :param proRataVac2D: proRata vector that will depend on the number of doses available
    :param extraParamsNM: list-container containing all the other parameters to run the model
    :return: a matrix with the best solutions and corresponding best values
    '''
    [extraParams, myIndex, repairFunOpt] = extraParamsNM
    [deathRate, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
     totalPopByAgeGroup, totalPopByVaccineGroup, tspan] = extraParams
    #check that the number of best solutions is < than number of sols in the matrix
    numRows = np.shape(myMat)[0]
    if numRows < numOfBestSols:
        print('numBestSols larger than input matrix')
        sys.exit()



    # sort the matrix by the myIndex column.
    sortedMat = myMat[myMat[:, myIndex].argsort()]

    # select the first numOfBestSols solutions from the GS to be inputs as initial conditions for NM and add the pro-rata sol
    myBestDecisionVariables = np.vstack((highRiskFirstVec.reshape(np.size(sortedMat[0, 0:10])), proRataVec2D.reshape(np.size(sortedMat[0, 0:10])), sortedMat[:numOfBestSols, 0:10]))
    results = np.zeros(((numOfBestSols+2), 21))
    timermat = np.zeros(numOfBestSols+2)
    for ivals in range((numOfBestSols+2)):
        # print('number of iterations of NM ',ivals)
        start = timer()
        x0 = myBestDecisionVariables[ivals]
        # minimize(objectiveFunction_NM2D, x0, args=(extraParamsNM,), method='nelder-mead')
        res = minimize(objectiveFunction_NM2D, x0, args=(extraParamsNM,), method='nelder-mead')

        results[ivals, 0:10] = res.x
        if repairFunOpt == 0:
            temp  = repairVector2D(res.x)
            # print(np.shape(temp))
            tempPeo = skimOffExcessVaccine2DBis(temp, numVaccinesAvailable, numVaccineGroups, totalPopByVaccineGroup)[0]
            # print(np.shape(tempPeo))
            tempRes = fromNumPeopleVaccinatedToFracVacs(tempPeo, numVaccinesAvailable)


        else:
            print('wrong repair function value')
            sys.exit()
        results[ivals, 10:20] = tempRes.reshape(10)
        results[ivals, 20] = res.fun
        end = timer()
        timermat[ivals] = end - start

    # print(np.mean(timermat))
    sortedResults= results[results[:, 20].argsort()]
    return sortedResults[:numOfBestSolsNM, :]


def optimizationSimplexAndPS(myMat, highRiskFirstVec, numOfBestSols, numIterationsPS,  numRandomPoints, proRataVec2D, extraParamsPS):
    """
    Utilizes particle swarms to perform the optimization
    :param myMat:
    :param highRiskFirstVec:
    :param numOfBestSols:
    :param numOfBestSolsNM:
    :param proRataVec2D:
    :param extraParamsNM:
    :return:
    """
    # extraParams =  [deathRate, groupFracs, y0, numAgeGroups, numDosesPerWeek, numVaccinesAvailable, numVaccineGroups, paramsODE,
    #  totalPopByAgeGroup, totalPopByVaccineGroup, tspan]
    [extraParams, myIndex, repairFunOpt] =  extraParamsPS
    numVaccineGroups = extraParams[6]
    # create a list to store the results:
    results = []

    ## NUMBER OF ROWS IN OUTPUT.
    numRows = np.shape(myMat)[0]
    if numRows < numOfBestSols:
        print('numBestSols larger than input matrix')
        sys.exit()

   # sort the matrix by the myIndex column.
    sortedMat = myMat[myMat[:, myIndex].argsort()]


    #define parameters for PS optimization
    lb = [0] * numVaccineGroups*2
    ub = [1] * numVaccineGroups*2
    bounds = (lb, ub)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    kwargs = {"extraParamsPS": extraParamsPS}
    myBestDecisionVariables = np.vstack((highRiskFirstVec.reshape(np.size(sortedMat[0, 0:10])),
                                         proRataVec2D.reshape(np.size(sortedMat[0, 0:10])),
                                         sortedMat[:numOfBestSols, 0:10]))

    randomSample = uniformSampleSimplex(dim=numVaccineGroups*2, numSamples=numRandomPoints)
    flat1 = myBestDecisionVariables.flatten(order='C')
    flat2 = randomSample.flatten(order='C')
    flat3 = np.append(flat1, flat2)
    numCol = myBestDecisionVariables.shape[1]
    numRow = myBestDecisionVariables.shape[0] + randomSample.shape[0]
    initialSwarm = flat3.reshape(numRow, numCol)
    # print(initialSwarm)
    # print(type(numOfBestSols + 2 + numRandomPoints))
    # print(type(extraParams[5]))
    # print(np.shape(initialSwarm))
    # print(type(initialSwarm))
    optimizer = ps.single.GlobalBestPSO(n_particles= (numOfBestSols + 2 + numRandomPoints),
                                        dimensions=numVaccineGroups*2,
                                        options=options,
                                        bounds=bounds,
                                        init_pos=initialSwarm
                                        )
    output = optimizer.optimize(objectiveFunction_PS2D,
                                iters=numIterationsPS, **kwargs)
    return output


def defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups,
                     oneMinusHospRate, oneMinusICUrate,
                     totalPop16):
    '''
    returns a vector with the initial conditions prior to vaccination based on the fraction of recovered people and on the
    number of current infections
    :param frac_rec: vector with the number of recovered people in each age group. It will be distributed assuming the parameters
    with which we will run the model.
    :param currentInfections: Number of current infections, will be distributed according to the paramters of hte model
    :return:
    '''
    ##################################################################################################################
    S0 = (1 - frac_rec) * (totalPop16 - currentInfections)
    E0 = (1 / 3) * currentInfections  # np.zeros(numAgeGroups)
    sympCurrentInfections = (2/3) * currentInfections
    A0 = np.multiply(sympCurrentInfections, (np.ones(16) - frac_sym))
    I0 = np.multiply(oneMinusHospRate, frac_sym * sympCurrentInfections)
    H0 = np.multiply(oneMinusICUrate, np.multiply(hosp_rate_16, frac_sym*sympCurrentInfections))
    ICU0 = np.multiply(icu_rate_16, np.multiply(hosp_rate_16, frac_sym*sympCurrentInfections))
    # print('current infections',np.sum(E0 + A0 + I0 + H0 + ICU0))
    # print(np.sum(sympCurrentInfections + E0))
    # I0 = frac_sym * (2/3) * currentInfections
    #
    # A0 = (1 - frac_sym) * (2/3)*currentInfections
    P0 = np.zeros(numAgeGroups)
    # H0 = np.zeros(numAgeGroups)
    # ICU0 = np.zeros(numAgeGroups)
    Rec0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections), oneMinusHospRate)
    RecA0 = (1 - frac_sym) * frac_rec * (totalPop16 - currentInfections)
    RecH0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections),
                        np.multiply(hosp_rate_16, oneMinusICUrate))
    RecICU0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections),
                          np.multiply(hosp_rate_16, icu_rate_16))

    # Vaccinated 1d initial conditions
    V10 = np.zeros(numAgeGroups)
    E_V10, A_V10, P_V10, I_V10, H_V10, ICU_V10, RecV_10, RecAV_10, RecHV_10, RecICUV_10 = np.zeros(numAgeGroups), \
                                                                                np.zeros(numAgeGroups), np.zeros(
        numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), \
                                                                                np.zeros(numAgeGroups), np.zeros(
        numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups)

    # Vaccinated 2d initial conditions
    V20 = np.zeros(numAgeGroups)
    E_V20, A_V20, P_V20, I_V20, H_V20, ICU_V20, RecV_20, RecAV_20, RecHV_20, RecICUV_20 = np.zeros(numAgeGroups), \
                                                                                np.zeros(numAgeGroups), np.zeros(
        numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), \
                                                                                np.zeros(numAgeGroups), np.zeros(
        numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups)

    Cases0 = np.copy(I0) + np.copy(A0)
    # print(Cases0)
    # print(Cases0)

    y0 = np.array([S0, E0, A0, P0, I0, H0, ICU0, Rec0, RecA0, RecH0, RecICU0,
                   V10, E_V10, A_V10, P_V10, I_V10, H_V10, ICU_V10, RecV_10, RecAV_10, RecHV_10, RecICUV_10,
                   V20, E_V20, A_V20, P_V20, I_V20, H_V20, ICU_V20, RecV_20, RecAV_20, RecHV_20, RecICUV_20,
                   Cases0]).reshape((34 * numAgeGroups))

    return y0

def pickBestSolForEachObjective2D(myMat):
    """
     picks the optimal solution among all of the sols of myMat for each
     of the objectives we care about and returns them
     in a matrix
     :param myMat: is a matrix of size numSamples x 26. The objective functions are given here
     columns 21:26 =  totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU
     :param numOfBestSols:
     :return:
     """

    myobjectives = ['totalInfections', 'totalSymptomaticInfections',  'deaths', 'hosp_peak', 'ICU_peak']

    func_dict_columnVals = {'totalInfections': 21,
                            'totalSymptomaticInfections': 22,
                            'deaths': 23,
                            'hosp_peak': 24,
                            'ICU_peak': 25}
    # create a list to store the results:
    results = np.zeros((5, 21))
    for ivals in range(5):
        keyvals = myobjectives[ivals]
        # print(keyvals)
        columnIndex = func_dict_columnVals[keyvals]

        # sort the matrix by that column:
        sortedMat = myMat[myMat[:, columnIndex].argsort()]
        temp = np.around(np.hstack((sortedMat[0, 0:20], sortedMat[0, columnIndex])),
                         decimals=2)
        results[ivals, :] = temp
    return results

def fromFracVacsToFracVaccinatedInEachVaccineGroup(fracVacs, numVaccinesAvailable, totalPopByVaccineGroup):
    fracVacs = repairVector2D(fracVacs)
    temp = fromFracVacsToNumberPeopleVaccinated2d(fracVacs, numVaccinesAvailable)
    return np.array([np.divide(temp[0, :], totalPopByVaccineGroup), np.divide(temp[1, :], totalPopByVaccineGroup)])



