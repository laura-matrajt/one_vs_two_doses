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
from optimizationFunctionsCoronavirus4 import splitVaccineAmongAgeGroups, uniformSampleSimplex
from coronavirusMainFunctions import coronavirusEqs_withHospitalizationsAndICU_withVaccine2D, findBetaNewModel_eqs2_withHosp4
from coronavirusMainFunctions import coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis
from coronavirusMainFunctions import findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis
from saveLoadFunctions import saveResults, loadResults
# from cvxopt import matrix
# from cvxopt import solvers



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
        # calculate the number of people vaccinated both with one or tow doses per vaccine group:
        ## JE: SHOULDN'T THIS BE THE NUMBER OF *UNVACCINATED* IN EACH VACCINE GROUP?
        ##     TOTAL NUMBER VACCINATED IS (numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals])
        ##     SO TOTAL NUMBER UNVACCINATED IS WHAT "temp" is...
        ##          ( totalPopByVaccineGroup[ivals]  - (numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals]) )
        ##    AND THEN WE'D WANT TO RESET TO ZERO IF NEGATIVE....
        temp = ((numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals]) - totalPopByVaccineGroup[ivals])
        ## BUT...WHERE IS "temp" USED BELOW?? I DON'T THINK IT'S NECESSARY AT ALL.

        #case 1: number of people vaccinated with two doses > pop in that group
        if (numPeopleVaccinated2d[ivals] > totalPopByVaccineGroup[ivals]):
            ## JE: MOVED THIS LINE UP TO THE TOP OF THE IF STATEMENT...
            excessVaccine[ivals] = 2*(numPeopleVaccinated2d[ivals] - totalPopByVaccineGroup[ivals]) + \
                                   numPeopleVaccinated1d[ivals]
            numPeopleVaccinated2d[ivals] = totalPopByVaccineGroup[ivals]     #skim off the excess vaccine from the 2D group
            ## JE: THE PROBLEM WITH THE NEXT LINE IS THAT WE ALREADY REASSIGNED numPeopleVaccinated2d to equal
            ## totalPopByVaccineGroup so the term in the parentheses is zero.
            ## We should assign excessVaccine *before* reassigning numPeopleVaccinated2d
            # excessVaccine[ivals] = 2*(numPeopleVaccinated2d[ivals] - totalPopByVaccineGroup[ivals]) + numPeopleVaccinated1d[ivals]
            numPeopleVaccinated1d[ivals] = 0  # set to 0 the number of people getting 1D

        #case 2: the sum of the number of people vaccinated with one and two doses > pop in that group, in this case
        # keep all of the ones vaccinated with two doses and skim off the excess in the one dose group
        if ((numPeopleVaccinated1d[ivals] + numPeopleVaccinated2d[ivals]) > totalPopByVaccineGroup[ivals]):
            excessVaccine[ivals] = numPeopleVaccinated1d[ivals] - (totalPopByVaccineGroup[ivals] - numPeopleVaccinated2d[ivals])
            ## JE: CLEARER TO ME TO DEFINE IT THIS WAY, BELOW...
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

# def skimOffExcessVaccine2D_3(fracVacs, numVaccinesAvailable, numVaccineGroups, totalPopByVaccineGroup):
#     '''
#     This function removes the excess vaccine in each vaccination group so that we do not vaccinate more people than what
#     there is in each vaccination group. It does it in the following way:
#
#     '''
#
#     #calculate the number of people vaccinated both with one or tow doses per vaccine group:
#     [numPeopleVaccinated1d, numPeopleVaccinated2d] = fromFracVacsToNumberPeopleVaccinated2d(fracVacs, numVaccinesAvailable)
#     # excessVaccine = np.zeros(numVaccineGroups)
#
#     newarray = np.zeros((2,5))
#     EPS = 0.5
#     totVac = numPeopleVaccinated1d + numPeopleVaccinated2d
#     for ivals in range(numVaccineGroups):
#         if (totVac[ivals] - totalPop5[ivals]) < EPS:
#            newarray[0,ivals] = numPeopleVaccinated1d[ivals]
#            newarray[1,ivals] = numPeopleVaccinated2d[ivals]
#         else:
#             f1 = numPeopleVaccinated1d[ivals]/totVac[ivals]
#             f2 = numPeopleVaccinated2d[ivals]/totVac[ivals]
#
#             newarray[0, ivals] = f1*totalPop5[ivals]
#             newarray[1, ivals] = f2*totalPop5[ivals]
#
#     excessVaccine = numVaccinesAvailable - np.sum(newarray)
#     return [newarray, excessVaccine]




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
    # print(newarray)
    # print('sum people vac', np.sum(newarray))
    # print('newarray', newarray)
    # print(newarray[1,:])
    # print(np.sum(np.divide(newarray, numVaccinesAvailable)))

    # newarray = repairNumPeopleToVaccinate(newarray, numVaccinesAvailable, totalPopByAgeGroup)

    numVaccinatedAgeGroup1d = splitVaccineAmongAgeGroups(newarray[0,:], groupFracs, totalPopByAgeGroup)
    numVaccinatedAgeGroup2d = splitVaccineAmongAgeGroups(newarray[1,:], groupFracs, totalPopByAgeGroup)

    # print(numVaccinatedAgeGroup1d)
    # print(numVaccinatedAgeGroup2d)

    # print((np.sum(newarray[0,:]) + 2*np.sum(newarray[1,:]))/numVaccinesAvailable)
    # [totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU] = \
    #     runVaccination2DPulse4(deathRate, y0, numAgeGroups, numDosesPerWeek,
    #                           numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccinesAvailable, paramsODE, tspan)

    [totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU] = \
        runVaccination2DPulse6(deathRate, y0, groupFracs, numAgeGroups,  numDosesPerWeek,
                          numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups, numVaccinesAvailable, paramsODE,
                          tspan)
    # print(np.sum(excessVaccine))
    # print([fracVacs[0, :], fracVacs[1, :], newarray[0, :], newarray[1,:], excessVaccine, totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU])
    mytempList = [fracVacs[0, :].tolist(), fracVacs[1, :].tolist(), newarray[0, :].tolist(), newarray[1,:].tolist(), [np.sum(excessVaccine), totalInfections, totalSymptomaticInfections, totalDeaths, maxHosp, maxICU]]
    flat_list = [item for sublist in mytempList for item in sublist]
    return flat_list






def runVaccination2DPulse6(deathRate, y0, groupFracs, numAgeGroups,  numDosesPerWeek,
                          numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,numVaccinesAvailable, paramsODE,
                          tspan):
    '''
    The difference between this and runVaccination2DPulse5 is that here in the last week of vac
    campaign with one dose we will start the second dose vaccination.

    :param deathRate: 1x16 vector with the death rate AMONG ALL HOSPITALIZED CASES
    :param initCond: initial conditions just before vaccination
    :param numAgeGroups: number of age groups in the model
    :param numDosesPerWeek: number of doses that can be administered by week
    :param numVaccinatedAgeGroup1d: number of people to get vaccinated with 1dose per age group
    :param numVaccinatedAgeGroup2d: number of people to get vaccinated with 1dose per age group
    :param paramsODE: all the parameters needed to run the ODE
    :param tspan: time horizon for the optimization, assumed to be a multiple of 7 (so that it can easily be converted to
    number of weeks)
    :return:
    '''

    #test that the number of weeks in the simulation will be enough to vaccinate everyone:
    numWeeksToUseVaccine = numVaccinesAvailable/numDosesPerWeek

    intNumWeeksToUseVaccine = int(np.floor(numVaccinesAvailable/numDosesPerWeek))
    # print(intNumWeeksToUseVaccine)
    # print(tspan)
    # print('numWeeksToUseVaccine*7', numWeeksToUseVaccine * 7)
    if numWeeksToUseVaccine < 0:
        print('problem: number of weeks to use vaccine is negative')
        sys.exit()
    # print(numWeeksToUseVaccine*7)

    if numWeeksToUseVaccine*7 > tspan:
        print('numWeeksToUseVaccine*7', numWeeksToUseVaccine*7)
        print('Not enough time to use all the vaccine at this rate')
        sys.exit()

    #total number of people receiving  one dose only.
    totalPeopleVacWithOneDose = np.sum(numVaccinatedAgeGroup1d) #+ np.sum(numVaccinatedAgeGroup2d)
    # print('totalPeopleVacWithOneDose', totalPeopleVacWithOneDose)

    # compute the total number of people getting vaccinated with an additional second dose:
    totalPeopleVacWithTwoDoses = np.sum(numVaccinatedAgeGroup2d)
    # print(totalPeopleVacWithTwoDoses)

    #Prepare initial conditions
    initCond = np.copy(y0)
    # Compute the fraction of people susceptible in each age group relative
    # to other groups that could potentially receive vaccination.
    fractionSus = findFractionSus2D(initCond, numAgeGroups)
    # print(fractionSus)
    initCond = initCond.reshape((34, numAgeGroups))
    # print(initCond)
    numVaccinesGiven = 0
    vaccinationOrder = list(range(numVaccineGroups - 1, -1, -1))

    # we will need to select for each vaccination group the corresponding age-groups
    ageGroupsInVaccineGroupsIndices = [[0,4], [4,10], [10,13], [13,15], [15,16]]

    #start a counter for the number of weeks the model has run:
    actualWeeks = 0
    realVacGiven1D = np.zeros(5)
    realVacGiven = np.zeros(5)
    if intNumWeeksToUseVaccine > 1:
        numWeeksToVac1d = totalPeopleVacWithOneDose / numDosesPerWeek


        numWeeksToVac2d = totalPeopleVacWithTwoDoses / numDosesPerWeek


        if (numWeeksToVac1d * 7 + numWeeksToVac2d * 7) > tspan:
            print('sera este Not enough time to use all the vaccine at this rate')
            sys.exit()

        # define variables for bookeeping and glueing all the simulations together.
        numberOfWeeksSpan = np.rint(tspan / 7)  # length of the simulation in weeks
        numWeeksSinceBegSimulation = 0  # number of weeks that have elapsed since beginning of simulation
        timeSeries = np.empty([34 * numAgeGroups])  # timeSeries where we will stack all the simulations


        # start a counter for the number of doses that have been given to make sure we are using all the vaccine in the appropriate way:
        total1Dgiven = 0
        total2Dgiven = 0
        # realVacGiven = np.zeros(5)

        [numVaccinatedVaccineGroup1d, numVaccinatedVaccineGroup2d] = numVaccinatedPerVaccineGroup(numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d)

        numVaccinatedVaccineGroup_1round = np.copy(numVaccinatedVaccineGroup2d)
        numVaccinatedVaccineGroup_2round = np.copy(numVaccinatedVaccineGroup2d)
        numVaccinatedVaccineGroup1dTemp = numVaccinatedVaccineGroup1d

        #two dose campaign:
        if totalPeopleVacWithTwoDoses > 0: #if there are people to be vaccinated with two doses, we start with them first
            #we will go through each vaccine group and allocate two doses of vaccine starting with the oldest first.


            #loop through vaccination groups in decreasing order
            for mvals in vaccinationOrder:

                #check if that vaccination group has yet to receive vaccine in the first or second round
                if (numVaccinatedVaccineGroup_1round[mvals]>0) or (numVaccinatedVaccineGroup_2round[mvals]>0):

                    #select the indices for the age groups in that vaccination group:
                    ageIndex = ageGroupsInVaccineGroupsIndices[mvals]
                    mystart = (ageIndex[0])
                    myend = (ageIndex[1])
                    #calculate how many weeks we need to spend in this vaccination group:
                    numWeeks = np.floor(numVaccinatedVaccineGroup_1round[mvals]/numDosesPerWeek)
                    # print(mvals, numWeeks)
                    #calculate how to allocate the doses within that vaccination group (proportional to the size of each age group)
                    numDosesTemp = groupFracs[mvals]*numDosesPerWeek

                    ########### loop to do the first dose  ##############################################
                    #do the actual vaccination FIRST DOSE:
                    for weeks in range(int(numWeeks)):
                        actualWeeks += 1
                        newInitCond = np.copy(initCond)
                        peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                                numDosesTemp*fractionSus[mystart:myend])

                        newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                        newInitCond[11, mystart:myend] += peopleVac1

                        ############################### run the model here  ###############################
                        tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                        newInitCond = newInitCond.reshape(34 * numAgeGroups)
                        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspanWeek,
                                     args=(paramsODE,))

                        ############################ upadate initial conditions for next week here ###############################
                        initCond = out[-1, :].reshape(34, numAgeGroups)
                        fractionSus = findFractionSus2D(initCond, numAgeGroups)
                        numWeeksSinceBegSimulation += 1

                        # bookeeping
                        total2Dgiven += np.sum(numDosesTemp)
                        realVacGiven[mvals] += np.sum(peopleVac1)

                        # stack solutions in the time series:
                        timeSeries = np.vstack((timeSeries, out[:, :]))

                    #do the last week:
                    actualWeeks += 1 #update counter with the number of weeks
                    newInitCond = np.copy(initCond)
                    #compute how many doses we need to allocate in the last week:
                    lastWeekTemp = numVaccinatedVaccineGroup_1round[mvals] - numWeeks*numDosesPerWeek

                    #split doses within the age groups of that vaccination group:
                    numDosesLastWeekTemp = groupFracs[mvals]*lastWeekTemp
                    #transfer people from susceptible to vaccinated with one dose:
                    peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                            numDosesLastWeekTemp*fractionSus[mystart:myend])
                    newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                    newInitCond[11, mystart:myend] += peopleVac1

                    #start vaccinating with the second dose in this same week with the remaining doses for the week:
                    remainingDoses = numDosesPerWeek - lastWeekTemp #doses that we can still give that week that
                    #will be allocated to the second dose vaccination campaign

                    #we will allocate the minimum between the number of doses left and the number of doses to be given to
                    #that vaccination group in the second round
                    remTemp = np.minimum(numVaccinatedVaccineGroup_2round[mvals], remainingDoses)
                    #distribute remaining doses among the age groups in that vaccination group:
                    remTempAge = groupFracs[mvals]*remTemp
                    #then check to vaccinate the people in the vaccine group and not more!
                    peopleVac2 = np.minimum(newInitCond[11, mystart:myend], remTempAge)
                    #transfer vaccinated people in that vaccine group from vaccinated with 1 dose to vaccinated with 2.
                    newInitCond2 = np.copy(newInitCond)
                    newInitCond2[11, mystart:myend] = newInitCond[11, mystart:myend] - peopleVac2
                    newInitCond2[22, mystart:myend] += peopleVac2

                    #remove those vaccinated from our counter of number of people to be vaccinated in that vaccine group
                    numVaccinatedVaccineGroup_2round[mvals] = numVaccinatedVaccineGroup_2round[mvals] - remTemp

                    ############################### run the model here  ###############################
                    tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                    newInitCond2 = newInitCond2.reshape(34 * numAgeGroups)
                    out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond2, tspanWeek,
                                 args=(paramsODE,))

                    ############################ upadate initial conditions for next week here ###############################
                    initCond = out[-1, :].reshape(34, numAgeGroups)
                    fractionSus = findFractionSus2D(initCond, numAgeGroups)
                    numWeeksSinceBegSimulation += 1

                    # bookeeping
                    total2Dgiven += (lastWeekTemp + remTemp)
                    realVacGiven[mvals] += (np.sum(peopleVac1) + np.sum(peopleVac2))

                    # stack solutions in the time series:
                    timeSeries = np.vstack((timeSeries, out[:, :]))



                    ########### loop to do the second dose  ##############################################
                    #recompute how many weeks we need to vaccinate:
                    numWeeks2Temp = np.floor(np.divide((numVaccinatedVaccineGroup_2round[mvals]), numDosesPerWeek))

                    # do the actual vaccination SECOND DOSE:
                    for weeks in range(int(numWeeks2Temp)):
                        actualWeeks += 1
                        newInitCond = np.copy(initCond)
                        peopleVac2 = np.minimum(initCond[11, mystart:myend], numDosesTemp) #compute number of people to be vaccinated
                        newInitCond[11, mystart:myend] = initCond[11, mystart:myend] - peopleVac2 #move people from 1D to 2D
                        newInitCond[22, mystart:myend] += peopleVac2

                        ############################### run the model here  ###############################
                        tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                        newInitCond = newInitCond.reshape(34 * numAgeGroups)
                        if (np.any(newInitCond<0)):
                            print('help')
                            sys.exit()
                        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspanWeek,
                                     args=(paramsODE,))

                        ############################ upadate initial conditions for next week here ###############################
                        initCond = out[-1, :].reshape(34, numAgeGroups)
                        fractionSus = findFractionSus2D(initCond, numAgeGroups)
                        numWeeksSinceBegSimulation += 1

                        # bookeeping
                        total2Dgiven += np.sum(numDosesTemp)
                        realVacGiven[mvals] += np.sum(peopleVac2)

                        # stack solutions in the time series:
                        timeSeries = np.vstack((timeSeries, out[:, :]))

                    ############## Last week of the second-dose vaccination campaign: ###################
                    #two things happening here:
                    # 1) finish vaccinating people in that vaccination group with their second dose
                    #of vaccine
                    actualWeeks += 1
                    newInitCond = np.copy(initCond)
                    #compute how many doses of vaccine I have to give in the last week:
                    lastWeekTemp = numVaccinatedVaccineGroup_2round[mvals] - numWeeks2Temp * numDosesPerWeek

                    #distribute them across the age groups
                    numDosesLastWeekTemp = groupFracs[mvals] * lastWeekTemp
                    #actually move people from 1D to 2D:
                    peopleVac2 = np.minimum(initCond[11, mystart:myend], numDosesLastWeekTemp)
                    newInitCond[11, mystart:myend] = initCond[11, mystart:myend] - peopleVac2
                    newInitCond[22, mystart:myend] += peopleVac2

                    #2) if there is any vaccine left, start vaccinating those in the next vaccination group:
                    #compute the number of remaining doses for that week:
                    remainingDoses2 = np.maximum((numDosesPerWeek - lastWeekTemp), 0)

                    newInitCond3 = np.copy(newInitCond)
                    #give the remaining doses to the group that follow:
                    if mvals > 0:
                        tempVar = mvals-1
                        if numVaccinatedVaccineGroup_1round[tempVar] >0:

                            #get the indices of the previous group:
                            ageIndex = ageGroupsInVaccineGroupsIndices[tempVar]
                            mystart = (ageIndex[0])
                            myend = (ageIndex[1])

                            #check that the number of doses left is less or equal to the number of people that
                            #need to be vaccinated in that group
                            tempVaccine = np.minimum(numVaccinatedVaccineGroup_1round[tempVar], remainingDoses2)
                            #split remaining doses among age groups in that vaccine group
                            tempDosesExtra = groupFracs[tempVar]*tempVaccine
                            peopleVac1DoseExtra = np.minimum(newInitCond[0, mystart:myend], tempDosesExtra*fractionSus[mystart:myend])

                            #move people from unvaccinated to vaccinated:
                            newInitCond3[0, mystart:myend] = newInitCond[0, mystart:myend] - peopleVac1DoseExtra
                            newInitCond3[11, mystart:myend] += peopleVac1DoseExtra

                            #remove the people who have been vaccinated with this first dose from that vaccination group
                            numVaccinatedVaccineGroup_1round[tempVar] = numVaccinatedVaccineGroup_1round[tempVar] - tempVaccine

                            #bookeeping
                            total2Dgiven += np.sum(tempVaccine)
                            realVacGiven[tempVar] += np.sum(peopleVac1DoseExtra)
                            ############################### run the model here  ###############################
                    tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                    newInitCond3 = newInitCond3.reshape(34 * numAgeGroups)
                    if (np.any(newInitCond3 < 0)):
                        print('help')
                    out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond3, tspanWeek,
                                 args=(paramsODE,))

                    ############################ upadate initial conditions for next week here ###############################
                    initCond = out[-1, :].reshape(34, numAgeGroups)
                    fractionSus = findFractionSus2D(initCond, numAgeGroups)
                    numWeeksSinceBegSimulation += 1

                    # bookeeping
                    total2Dgiven += np.sum(lastWeekTemp)
                    realVacGiven[mvals] += np.sum(peopleVac2)


                    # stack solutions in the time series:
                    timeSeries = np.vstack((timeSeries, out[:, :]))

        ########################### end vaccination with two doses ##########################


        ######################### start one dose vaccination campaign   #######################
        if totalPeopleVacWithOneDose > 0:
            # print('entre aqui 1D')
            # realVacGiven1D = np.zeros(5)

            for mvals in vaccinationOrder:
                if numVaccinatedVaccineGroup1dTemp[mvals]>0:
                    # select the indices for the age groups in
                    # that vaccination group:
                    ageIndex = ageGroupsInVaccineGroupsIndices[mvals]
                    mystart = (ageIndex[0])
                    myend = (ageIndex[1])
                    # calculate how many weeks we need to spend in this vaccination group:
                    numWeeks1 = np.floor(numVaccinatedVaccineGroup1dTemp[mvals]/numDosesPerWeek)
                    #calculate how to allocate the doses within that vaccination group (proportional to the size of each age group)
                    numDoses1Temp = groupFracs[mvals]*numDosesPerWeek
                    # print(mvals, numVaccinatedVaccineGroup1dTemp[mvals]/numDosesPerWeek)
                    #do the actual vaccination ONE DOSE:
                    for weeks in range(int(numWeeks1)):
                        actualWeeks += 1
                        newInitCond = np.copy(initCond)
                        peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                                numDoses1Temp*fractionSus[mystart:myend])
                        newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                        newInitCond[11, mystart:myend] += peopleVac1

                        ############################### run the model here  ###############################
                        tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                        newInitCond = newInitCond.reshape(34 * numAgeGroups)
                        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspanWeek,
                                     args=(paramsODE,))

                        ############################ upadate initial conditions for next week here ###############################
                        initCond = out[-1, :].reshape(34, numAgeGroups)
                        fractionSus = findFractionSus2D(initCond, numAgeGroups)
                        numWeeksSinceBegSimulation += 1

                        # bookeeping
                        # print('np.sum(numDoses1Temp)', np.sum(numDoses1Temp))
                        total1Dgiven += np.sum(numDoses1Temp)
                        realVacGiven1D[mvals] += np.sum(peopleVac1)

                        # stack solutions in the time series:
                        timeSeries = np.vstack((timeSeries, out[:, :]))

                    ##########################        Last week: ########################################################
                    #1) vaccinate the remainder people in that vaccination group with one dose:
                    actualWeeks += 1
                    newInitCond = np.copy(initCond)
                    #compute how many doses we need to give that week:
                    lastWeekTemp = numVaccinatedVaccineGroup1d[mvals] - numDosesPerWeek*numWeeks1
                    numDosesLastWeekTemp = groupFracs[mvals] * lastWeekTemp     #split across age groups proportionally
                    #transfer people from susceptible to vaccinated with 1D
                    peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                            numDosesLastWeekTemp * fractionSus[mystart:myend])
                    newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                    newInitCond[11, mystart:myend] += peopleVac1

                    # 2) start vaccinating with one dose the next vaccination group:
                    #check if there are doses left for the last week:
                    remainingDoses1 = numDosesPerWeek - lastWeekTemp
                    newInitCond1 = np.copy(newInitCond)
                    #give the remaining doses to the next groups while vaccine is available and there is some vaccination
                    # group to vaccinate:
                    if mvals >0:
                        tempOrder = list(range(mvals - 1, -1, -1))

                        for tempVar in tempOrder:
                            if remainingDoses1 > 1:
                                if numVaccinatedVaccineGroup1dTemp[tempVar] >0: #check if the previous group needs to be vaccinated with one dose
                                    #get the indices of the previous group:
                                    ageIndex = ageGroupsInVaccineGroupsIndices[tempVar]
                                    mystart = (ageIndex[0])
                                    myend = (ageIndex[1])

                                    #check if the number of doses left is bigger than the number of doses that is
                                    #supposed to be given to that group and choose the minimum:
                                    tempVaccine1 = np.minimum(numVaccinatedVaccineGroup1dTemp[tempVar], remainingDoses1)
                                    #split across age groups:
                                    tempVaccine1bis = groupFracs[tempVar]*tempVaccine1
                                    #take the minimum between the vaccine available and the number of susceptibles in those age groups:
                                    peopleVac1DoseExtraBis = np.minimum(newInitCond[0, mystart:myend], tempVaccine1bis*fractionSus[mystart:myend])
                                    #move people from susceptibles to vaccinated:
                                    newInitCond1[0, mystart:myend] = newInitCond[0, mystart:myend] - peopleVac1DoseExtraBis
                                    newInitCond1[11, mystart:myend] += peopleVac1DoseExtraBis
                                    numVaccinatedVaccineGroup1dTemp[tempVar] = numVaccinatedVaccineGroup1dTemp[tempVar] - tempVaccine1
                                    remainingDoses1 = remainingDoses1 - tempVaccine1
                                    total1Dgiven += tempVaccine1
                                    realVacGiven1D[tempVar] += np.sum(peopleVac1DoseExtraBis)

                    ############################### run the model here  ###############################
                    tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                    newInitCond1 = newInitCond1.reshape(34 * numAgeGroups)
                    out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond1, tspanWeek,
                                 args=(paramsODE,))

                    ############################ upadate initial conditions for next week here ###############################
                    initCond = out[-1, :].reshape(34, numAgeGroups)
                    fractionSus = findFractionSus2D(initCond, numAgeGroups)
                    numWeeksSinceBegSimulation += 1

                    # stack solutions in the time series:
                    timeSeries = np.vstack((timeSeries, out[:, :]))

                    # bookeeping
                    total1Dgiven += (lastWeekTemp)
                    realVacGiven1D[mvals] += np.sum(peopleVac1)
        # print('actual weeks after 1 and 2 D', actualWeeks)
        # print('numWeeksSinceBegSimulation', numWeeksSinceBegSimulation)
        # print('total infections', np.sum(initCond[33 , :]))
        #run the model for the rest of the weeks:
        ############################### run the model here  ###############################
        numOfWeeksRemaining = ((numberOfWeeksSpan - numWeeksSinceBegSimulation))
        # print('numOfWeeksRemaining after 1D and 2d campaigns', numOfWeeksRemaining)
        if numOfWeeksRemaining > 0:
            numOfWeeksRemaining = int(numOfWeeksRemaining)
            tspan2 = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + numOfWeeksRemaining) * 7,
                                 numOfWeeksRemaining * 14)
            newInitCond = initCond.reshape(34 * numAgeGroups)
            out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspan2,
                         args=(paramsODE,))
            timeSeries = np.vstack((timeSeries, out[:, :]))
        timeSeries = timeSeries[1:, :]
        # tempX = out[-1,:].reshape(34, numAgeGroups)
        # print('total infections', np.sum(tempX[33, :]))
        numVaccinesGiven = total1Dgiven + total2Dgiven
        # print(numVaccinesGiven)
        # print(realVacGiven1D)
        # print(realVacGiven)

    else:
        # print('instantaneous vaccination')
        #instantaneous vaccination:
        initCond = initCond.reshape((34, numAgeGroups))
        # print(initCond)
        initCond2 = np.copy(initCond)
        # move first people vaccinated with two doses:
        peopleVac2 = np.minimum(initCond[0, :], numVaccinatedAgeGroup2d * fractionSus)
        initCond2[0, :] = initCond[0, :] - peopleVac2  # move people out of the susceptible class
        initCond2[22, :] += peopleVac2  # move those people to the vaccinated with two doses class

        # move the people vaccinated with one dose:
        initCond3 = np.copy(initCond2)
        peopleVac1 = np.minimum(initCond2[0, :], numVaccinatedAgeGroup1d * fractionSus)
        initCond3[0, :] = initCond2[0, :] - peopleVac1
        initCond3[11, :] += peopleVac1

        initCond3 = initCond3.reshape(34 * numAgeGroups)
        # print(np.shape(initCond3))
        tspan2 = np.linspace(0, tspan, tspan * 2)
        # run the ODEs
        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, initCond3, tspan2, args=(paramsODE,))
        timeSeries = np.copy(out)
        numVaccinesGiven = numVaccinatedAgeGroup1d + 2*numVaccinatedAgeGroup2d
        # print(numVaccinesGiven)

        ##### READ THE OUTPUTS:
        ############################# retrieve the outputs of interest:  ###############################################
    out2 = timeSeries[-1, :].reshape(34, numAgeGroups)
    # retrieve the objectives we are interested on:
    infections = out2[33, :]
    totalInfections = np.sum(infections)

    # this reads the recovered symptomatic groups. We need to substract from here the recovered that were already there before vaccination
    y0 = y0.reshape(34, numAgeGroups)
    totalSymptomaticRecoveredPreVaccination = np.sum(y0[7, :] + y0[9, :] + y0[10, :])
    totalSymptomaticInfections = np.sum(timeSeries[-1, (7 * numAgeGroups): (8 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (9 * numAgeGroups): (11 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (18 * numAgeGroups):(19 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (20 * numAgeGroups):(22 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (29 * numAgeGroups):(30 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (31 * numAgeGroups):(33 * numAgeGroups)]) - \
                                 totalSymptomaticRecoveredPreVaccination

    # compute the hospitalization curve, for vaccinated and unvaccinated groups
    hosp = np.sum(timeSeries[:, (5 * numAgeGroups):(6 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (16 * numAgeGroups): (17 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (27 * numAgeGroups): (28 * numAgeGroups)], 1)

    # compute the ICU hospitalizations for vaccinated and unvaccinated groups:
    icu = np.sum(timeSeries[:, (6 * numAgeGroups):(7 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (17 * numAgeGroups): (18 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (28 * numAgeGroups): (29 * numAgeGroups)], 1)

    maxHosp = np.max(hosp)
    maxICU = np.max(icu)

    # compute the total number of deaths by reading the recovered hospitalized from non-ICU and ICU
    totalHospitalizedRecoveredPriorVaccination = (y0[9, :] + y0[10, :])
    recHosp = (out2[9, :]) + out2[10, :] + out2[20, :] + out2[21, :] + out2[31, :] + out2[32, :] - \
              totalHospitalizedRecoveredPriorVaccination
    # print(recHosp)
    deaths = np.multiply(deathRate, recHosp)
    totalDeaths = np.sum(deaths)


    return [np.rint(totalInfections), np.rint(totalSymptomaticInfections), np.rint(totalDeaths), np.rint(maxHosp),
            np.rint(maxICU)]



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
    ## JE: THE ABOVE LINE I COULDN'T GET TO WORK. I UNDERSTAND WHAT IT'S SUPPOSED TO DO
    ## BUT IT DIDN'T DO IT ON MY COMPUTER.
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
    elif repairFunOpt == 1:
        newFracVacs = repairVector2D_quadratic_subproblem_INEQUALITY(fracVacs, numVaccinesAvailable, totalPopByVaccineGroup)
    elif repairFunOpt == 2:
        newFracVacs = repairVector2D_quadratic_subproblem(fracVacs, numVaccinesAvailable,
                                                                     totalPopByVaccineGroup)

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
        elif repairFunOpt == 1:

            newFracVacs = repairVector2D_quadratic_subproblem_INEQUALITY(fracVacs, numVaccinesAvailable,
                                                                         totalPopByVaccineGroup)
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
        # print(numPeopleVaccinatedWith1D)
        # print(numPeopleVaccinatedWith2D)
        # print(numPeopleVaccinatedWith1D + numPeopleVaccinatedWith2D)
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
        elif repairFunOpt == 1:
            tempRes =  repairVector2D_quadratic_subproblem_INEQUALITY(res.x, numVaccinesAvailable,
                                                                         totalPopByVaccineGroup)
        elif repairFunOpt == 2:
            tempRes =  repairVector2D_quadratic_subproblem(res.x, numVaccinesAvailable,
                                                              totalPopByVaccineGroup)

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



def runVaccination2DPulse6FullOutput(deathRate, y0, groupFracs, numAgeGroups,  numDosesPerWeek,
                          numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,numVaccinesAvailable, paramsODE,
                          tspan):
    '''
    The difference between this and runVaccination2DPulse5 is that here in the last week of vac
    campaign with one dose we will start the second dose vaccination.

    :param deathRate: 1x16 vector with the death rate AMONG ALL HOSPITALIZED CASES
    :param initCond: initial conditions just before vaccination
    :param numAgeGroups: number of age groups in the model
    :param numDosesPerWeek: number of doses that can be administered by week
    :param numVaccinatedAgeGroup1d: number of people to get vaccinated with 1dose per age group
    :param numVaccinatedAgeGroup2d: number of people to get vaccinated with 1dose per age group
    :param paramsODE: all the parameters needed to run the ODE
    :param tspan: time horizon for the optimization, assumed to be a multiple of 7 (so that it can easily be converted to
    number of weeks)
    :return:
    '''

    #test that the number of weeks in the simulation will be enough to vaccinate everyone:
    numWeeksToUseVaccine = numVaccinesAvailable/numDosesPerWeek

    intNumWeeksToUseVaccine = int(np.floor(numVaccinesAvailable/numDosesPerWeek))

    if numWeeksToUseVaccine < 0:
        print('problem: number of weeks to use vaccine is negative')
        sys.exit()
    # print(numWeeksToUseVaccine*7)

    if numWeeksToUseVaccine*7 > tspan:
        print('Not enough time to use all the vaccine at this rate')
        sys.exit()

    #total number of people receiving  one dose only.
    totalPeopleVacWithOneDose = np.sum(numVaccinatedAgeGroup1d) #+ np.sum(numVaccinatedAgeGroup2d)
    # print('totalPeopleVacWithOneDose', totalPeopleVacWithOneDose)

    # compute the total number of people getting vaccinated with an additional second dose:
    totalPeopleVacWithTwoDoses = np.sum(numVaccinatedAgeGroup2d)
    # print(totalPeopleVacWithTwoDoses)

    #Prepare initial conditions
    initCond = np.copy(y0)
    # Compute the fraction of people susceptible in each age group relative
    # to other groups that could potentially receive vaccination.
    fractionSus = findFractionSus2D(initCond, numAgeGroups)
    # print(fractionSus)
    initCond = initCond.reshape((34, numAgeGroups))
    # print(initCond)
    numVaccinesGiven = 0
    vaccinationOrder = list(range(numVaccineGroups - 1, -1, -1))

    # we will need to select for each vaccination group the corresponding age-groups
    ageGroupsInVaccineGroupsIndices = [[0,4], [4,10], [10,13], [13,15], [15,16]]

    #start a counter for the number of weeks the model has run:
    actualWeeks = 0
    realVacGiven1D = np.zeros(5)
    realVacGiven = np.zeros(5)
    if intNumWeeksToUseVaccine > 1:
        numWeeksToVac1d = totalPeopleVacWithOneDose / numDosesPerWeek


        numWeeksToVac2d = totalPeopleVacWithTwoDoses / numDosesPerWeek


        if (numWeeksToVac1d * 7 + numWeeksToVac2d * 7) > tspan:
            print('Not enough time to use all the vaccine at this rate')
            sys.exit()

        # define variables for bookeeping and glueing all the simulations together.
        numberOfWeeksSpan = np.rint(tspan / 7)  # length of the simulation in weeks
        numWeeksSinceBegSimulation = 0  # number of weeks that have elapsed since beginning of simulation
        timeSeries = np.empty([34 * numAgeGroups])  # timeSeries where we will stack all the simulations


        # start a counter for the number of doses that have been given to make sure we are using all the vaccine in the appropriate way:
        total1Dgiven = 0
        total2Dgiven = 0
        # realVacGiven = np.zeros(5)

        [numVaccinatedVaccineGroup1d, numVaccinatedVaccineGroup2d] = numVaccinatedPerVaccineGroup(numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d)

        numVaccinatedVaccineGroup_1round = np.copy(numVaccinatedVaccineGroup2d)
        numVaccinatedVaccineGroup_2round = np.copy(numVaccinatedVaccineGroup2d)
        numVaccinatedVaccineGroup1dTemp = numVaccinatedVaccineGroup1d

        #two dose campaign:
        if totalPeopleVacWithTwoDoses > 0: #if there are people to be vaccinated with two doses, we start with them first
            #we will go through each vaccine group and allocate two doses of vaccine starting with the oldest first.


            #loop through vaccination groups in decreasing order
            for mvals in vaccinationOrder:

                #check if that vaccination group has yet to receive vaccine in the first or second round
                if (numVaccinatedVaccineGroup_1round[mvals]>0) or (numVaccinatedVaccineGroup_2round[mvals]>0):
                    # print(mvals)
                    #select the indices for the age groups in that vaccination group:
                    ageIndex = ageGroupsInVaccineGroupsIndices[mvals]
                    mystart = (ageIndex[0])
                    myend = (ageIndex[1])
                    #calculate how many weeks we need to spend in this vaccination group:
                    numWeeks = np.floor(numVaccinatedVaccineGroup_1round[mvals]/numDosesPerWeek)
                    # print(numWeeks)
                    #calculate how to allocate the doses within that vaccination group (proportional to the size of each age group)
                    numDosesTemp = groupFracs[mvals]*numDosesPerWeek

                    ########### loop to do the first dose  ##############################################
                    #do the actual vaccination FIRST DOSE:
                    for weeks in range(int(numWeeks)):
                        actualWeeks += 1
                        newInitCond = np.copy(initCond)
                        peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                                numDosesTemp*fractionSus[mystart:myend])

                        newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                        newInitCond[11, mystart:myend] += peopleVac1

                        ############################### run the model here  ###############################
                        tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                        newInitCond = newInitCond.reshape(34 * numAgeGroups)
                        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspanWeek,
                                     args=(paramsODE,))

                        ############################ upadate initial conditions for next week here ###############################
                        initCond = out[-1, :].reshape(34, numAgeGroups)
                        fractionSus = findFractionSus2D(initCond, numAgeGroups)
                        numWeeksSinceBegSimulation += 1

                        # bookeeping
                        total2Dgiven += np.sum(numDosesTemp)
                        realVacGiven[mvals] += np.sum(peopleVac1)

                        # stack solutions in the time series:
                        timeSeries = np.vstack((timeSeries, out[:, :]))

                    #do the last week:
                    actualWeeks += 1 #update counter with the number of weeks
                    newInitCond = np.copy(initCond)
                    #compute how many doses we need to allocate in the last week:
                    lastWeekTemp = numVaccinatedVaccineGroup_1round[mvals] - numWeeks*numDosesPerWeek

                    #split doses within the age groups of that vaccination group:
                    numDosesLastWeekTemp = groupFracs[mvals]*lastWeekTemp
                    #transfer people from susceptible to vaccinated with one dose:
                    peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                            numDosesLastWeekTemp*fractionSus[mystart:myend])
                    newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                    newInitCond[11, mystart:myend] += peopleVac1

                    #start vaccinating with the second dose in this same week with the remaining doses for the week:
                    remainingDoses = numDosesPerWeek - lastWeekTemp #doses that we can still give that week that
                    #will be allocated to the second dose vaccination campaign

                    #we will allocate the minimum between the number of doses left and the number of doses to be given to
                    #that vaccination group in the second round
                    remTemp = np.minimum(numVaccinatedVaccineGroup_2round[mvals], remainingDoses)
                    #distribute remaining doses among the age groups in that vaccination group:
                    remTempAge = groupFracs[mvals]*remTemp
                    #then check to vaccinate the people in the vaccine group and not more!
                    peopleVac2 = np.minimum(newInitCond[11, mystart:myend], remTempAge)
                    #transfer vaccinated people in that vaccine group from vaccinated with 1 dose to vaccinated with 2.
                    newInitCond2 = np.copy(newInitCond)
                    newInitCond2[11, mystart:myend] = newInitCond[11, mystart:myend] - peopleVac2
                    newInitCond2[22, mystart:myend] += peopleVac2

                    #remove those vaccinated from our counter of number of people to be vaccinated in that vaccine group
                    numVaccinatedVaccineGroup_2round[mvals] = numVaccinatedVaccineGroup_2round[mvals] - remTemp

                    ############################### run the model here  ###############################
                    tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                    newInitCond2 = newInitCond2.reshape(34 * numAgeGroups)
                    out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond2, tspanWeek,
                                 args=(paramsODE,))

                    ############################ upadate initial conditions for next week here ###############################
                    initCond = out[-1, :].reshape(34, numAgeGroups)
                    fractionSus = findFractionSus2D(initCond, numAgeGroups)
                    numWeeksSinceBegSimulation += 1

                    # bookeeping
                    total2Dgiven += (lastWeekTemp + remTemp)
                    realVacGiven[mvals] += (np.sum(peopleVac1) + np.sum(peopleVac2))

                    # stack solutions in the time series:
                    timeSeries = np.vstack((timeSeries, out[:, :]))



                    ########### loop to do the second dose  ##############################################
                    #recompute how many weeks we need to vaccinate:
                    numWeeks2Temp = np.floor(np.divide((numVaccinatedVaccineGroup_2round[mvals]), numDosesPerWeek))

                    # do the actual vaccination SECOND DOSE:
                    for weeks in range(int(numWeeks2Temp)):
                        actualWeeks += 1
                        newInitCond = np.copy(initCond)
                        peopleVac2 = np.minimum(initCond[11, mystart:myend], numDosesTemp) #compute number of people to be vaccinated
                        newInitCond[11, mystart:myend] = initCond[11, mystart:myend] - peopleVac2 #move people from 1D to 2D
                        newInitCond[22, mystart:myend] += peopleVac2

                        ############################### run the model here  ###############################
                        tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                        newInitCond = newInitCond.reshape(34 * numAgeGroups)
                        if (np.any(newInitCond<0)):
                            if np.any(newInitCond[newInitCond<0]) < 1:
                                print('help entre aqui')
                                print(newInitCond[newInitCond<0])
                                sys.exit()
                        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspanWeek,
                                     args=(paramsODE,))

                        ############################ upadate initial conditions for next week here ###############################
                        initCond = out[-1, :].reshape(34, numAgeGroups)
                        fractionSus = findFractionSus2D(initCond, numAgeGroups)
                        numWeeksSinceBegSimulation += 1

                        # bookeeping
                        total2Dgiven += np.sum(numDosesTemp)
                        realVacGiven[mvals] += np.sum(peopleVac2)

                        # stack solutions in the time series:
                        timeSeries = np.vstack((timeSeries, out[:, :]))

                    ############## Last week of the second-dose vaccination campaign: ###################
                    #two things happening here:
                    # 1) finish vaccinating people in that vaccination group with their second dose
                    #of vaccine
                    actualWeeks += 1
                    newInitCond = np.copy(initCond)
                    #compute how many doses of vaccine I have to give in the last week:
                    lastWeekTemp = numVaccinatedVaccineGroup_2round[mvals] - numWeeks2Temp * numDosesPerWeek

                    #distribute them across the age groups
                    numDosesLastWeekTemp = groupFracs[mvals] * lastWeekTemp
                    #actually move people from 1D to 2D:
                    peopleVac2 = np.minimum(initCond[11, mystart:myend], numDosesLastWeekTemp)
                    newInitCond[11, mystart:myend] = initCond[11, mystart:myend] - peopleVac2
                    newInitCond[22, mystart:myend] += peopleVac2

                    #2) if there is any vaccine left, start vaccinating those in the next vaccination group:
                    #compute the number of remaining doses for that week:
                    remainingDoses2 = np.maximum((numDosesPerWeek - lastWeekTemp), 0)

                    newInitCond3 = np.copy(newInitCond)
                    #give the remaining doses to the group that follow:
                    if mvals > 0:
                        tempVar = mvals-1
                        if numVaccinatedVaccineGroup_1round[tempVar] >0:

                            #get the indices of the previous group:
                            ageIndex = ageGroupsInVaccineGroupsIndices[tempVar]
                            mystart = (ageIndex[0])
                            myend = (ageIndex[1])

                            #check that the number of doses left is less or equal to the number of people that
                            #need to be vaccinated in that group
                            tempVaccine = np.minimum(numVaccinatedVaccineGroup_1round[tempVar], remainingDoses2)
                            #split remaining doses among age groups in that vaccine group
                            tempDosesExtra = groupFracs[tempVar]*tempVaccine
                            peopleVac1DoseExtra = np.minimum(newInitCond[0, mystart:myend], tempDosesExtra*fractionSus[mystart:myend])

                            #move people from unvaccinated to vaccinated:
                            newInitCond3[0, mystart:myend] = newInitCond[0, mystart:myend] - peopleVac1DoseExtra
                            newInitCond3[11, mystart:myend] += peopleVac1DoseExtra

                            #remove the people who have been vaccinated with this first dose from that vaccination group
                            numVaccinatedVaccineGroup_1round[tempVar] = numVaccinatedVaccineGroup_1round[tempVar] - tempVaccine

                            #bookeeping
                            total2Dgiven += np.sum(tempVaccine)
                            realVacGiven[tempVar] += np.sum(peopleVac1DoseExtra)
                            ############################### run the model here  ###############################
                    tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                    newInitCond3 = newInitCond3.reshape(34 * numAgeGroups)
                    if (np.any(newInitCond3 < 0)):
                        print('help')
                    out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond3, tspanWeek,
                                 args=(paramsODE,))

                    ############################ upadate initial conditions for next week here ###############################
                    initCond = out[-1, :].reshape(34, numAgeGroups)
                    fractionSus = findFractionSus2D(initCond, numAgeGroups)
                    numWeeksSinceBegSimulation += 1

                    # bookeeping
                    total2Dgiven += np.sum(lastWeekTemp)
                    realVacGiven[mvals] += np.sum(peopleVac2)


                    # stack solutions in the time series:
                    timeSeries = np.vstack((timeSeries, out[:, :]))

        ########################### end vaccination with two doses ##########################


        ######################### start one dose vaccination campaign   #######################
        if totalPeopleVacWithOneDose > 0:
            # print('entre aqui 1D')
            # realVacGiven1D = np.zeros(5)

            for mvals in vaccinationOrder:
                if numVaccinatedVaccineGroup1dTemp[mvals]>0:
                    # select the indices for the age groups in
                    # that vaccination group:
                    ageIndex = ageGroupsInVaccineGroupsIndices[mvals]
                    mystart = (ageIndex[0])
                    myend = (ageIndex[1])
                    # calculate how many weeks we need to spend in this vaccination group:
                    numWeeks1 = np.floor(numVaccinatedVaccineGroup1dTemp[mvals]/numDosesPerWeek)
                    #calculate how to allocate the doses within that vaccination group (proportional to the size of each age group)
                    numDoses1Temp = groupFracs[mvals]*numDosesPerWeek
                    # print(numWeeks1)
                    #do the actual vaccination ONE DOSE:
                    for weeks in range(int(numWeeks1)):
                        actualWeeks += 1
                        newInitCond = np.copy(initCond)
                        peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                                numDoses1Temp*fractionSus[mystart:myend])
                        newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                        newInitCond[11, mystart:myend] += peopleVac1

                        ############################### run the model here  ###############################
                        tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                        newInitCond = newInitCond.reshape(34 * numAgeGroups)
                        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspanWeek,
                                     args=(paramsODE,))

                        ############################ upadate initial conditions for next week here ###############################
                        initCond = out[-1, :].reshape(34, numAgeGroups)
                        fractionSus = findFractionSus2D(initCond, numAgeGroups)
                        numWeeksSinceBegSimulation += 1

                        # bookeeping
                        # print('np.sum(numDoses1Temp)', np.sum(numDoses1Temp))
                        total1Dgiven += np.sum(numDoses1Temp)
                        realVacGiven1D[mvals] += np.sum(peopleVac1)

                        # stack solutions in the time series:
                        timeSeries = np.vstack((timeSeries, out[:, :]))

                    ##########################        Last week: ########################################################
                    #1) vaccinate the remainder people in that vaccination group with one dose:
                    actualWeeks += 1
                    newInitCond = np.copy(initCond)
                    #compute how many doses we need to give that week:
                    lastWeekTemp = numVaccinatedVaccineGroup1d[mvals] - numDosesPerWeek*numWeeks1
                    numDosesLastWeekTemp = groupFracs[mvals] * lastWeekTemp     #split across age groups proportionally
                    #transfer people from susceptible to vaccinated with 1D
                    peopleVac1 = np.minimum(initCond[0, mystart:myend],
                                            numDosesLastWeekTemp * fractionSus[mystart:myend])
                    newInitCond[0, mystart:myend] = initCond[0, mystart:myend] - peopleVac1
                    newInitCond[11, mystart:myend] += peopleVac1

                    # 2) start vaccinating with one dose the next vaccination group:
                    #check if there are doses left for the last week:
                    remainingDoses1 = numDosesPerWeek - lastWeekTemp
                    newInitCond1 = np.copy(newInitCond)
                    #give the remaining doses to the next groups while vaccine is available and there is some vaccination
                    # group to vaccinate:
                    if mvals >0:
                        tempOrder = list(range(mvals - 1, -1, -1))

                        for tempVar in tempOrder:
                            if remainingDoses1 > 1:
                                if numVaccinatedVaccineGroup1dTemp[tempVar] >0: #check if the previous group needs to be vaccinated with one dose
                                    #get the indices of the previous group:
                                    ageIndex = ageGroupsInVaccineGroupsIndices[tempVar]
                                    mystart = (ageIndex[0])
                                    myend = (ageIndex[1])

                                    #check if the number of doses left is bigger than the number of doses that is
                                    #supposed to be given to that group and choose the minimum:
                                    tempVaccine1 = np.minimum(numVaccinatedVaccineGroup1dTemp[tempVar], remainingDoses1)
                                    #split across age groups:
                                    tempVaccine1bis = groupFracs[tempVar]*tempVaccine1
                                    #take the minimum between the vaccine available and the number of susceptibles in those age groups:
                                    peopleVac1DoseExtraBis = np.minimum(newInitCond[0, mystart:myend], tempVaccine1bis*fractionSus[mystart:myend])
                                    #move people from susceptibles to vaccinated:
                                    newInitCond1[0, mystart:myend] = newInitCond[0, mystart:myend] - peopleVac1DoseExtraBis
                                    newInitCond1[11, mystart:myend] += peopleVac1DoseExtraBis
                                    numVaccinatedVaccineGroup1dTemp[tempVar] = numVaccinatedVaccineGroup1dTemp[tempVar] - tempVaccine1
                                    remainingDoses1 = remainingDoses1 - tempVaccine1
                                    total1Dgiven += tempVaccine1
                                    realVacGiven1D[tempVar] += np.sum(peopleVac1DoseExtraBis)

                    ############################### run the model here  ###############################
                    tspanWeek = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + 1) * 7, 14)
                    newInitCond1 = newInitCond1.reshape(34 * numAgeGroups)
                    out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond1, tspanWeek,
                                 args=(paramsODE,))

                    ############################ upadate initial conditions for next week here ###############################
                    initCond = out[-1, :].reshape(34, numAgeGroups)
                    fractionSus = findFractionSus2D(initCond, numAgeGroups)
                    numWeeksSinceBegSimulation += 1

                    # stack solutions in the time series:
                    timeSeries = np.vstack((timeSeries, out[:, :]))

                    # bookeeping
                    total1Dgiven += (lastWeekTemp)
                    realVacGiven1D[mvals] += np.sum(peopleVac1)
        # print('actual weeks after 1 and 2 D', actualWeeks)
        # print('numWeeksSinceBegSimulation', numWeeksSinceBegSimulation)
        # print('total infections', np.sum(initCond[33 , :]))
        #run the model for the rest of the weeks:
        ############################### run the model here  ###############################
        numOfWeeksRemaining = ((numberOfWeeksSpan - numWeeksSinceBegSimulation))
        # print('numOfWeeksRemaining after 1D and 2d campaigns', numOfWeeksRemaining)
        if numOfWeeksRemaining > 0:
            numOfWeeksRemaining = int(numOfWeeksRemaining)
            tspan2 = np.linspace(numWeeksSinceBegSimulation * 7, (numWeeksSinceBegSimulation + numOfWeeksRemaining) * 7,
                                 numOfWeeksRemaining * 14)
            newInitCond = initCond.reshape(34 * numAgeGroups)
            out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, newInitCond, tspan2,
                         args=(paramsODE,))
            timeSeries = np.vstack((timeSeries, out[:, :]))
        timeSeries = timeSeries[1:, :]
        # tempX = out[-1,:].reshape(34, numAgeGroups)
        # print('total infections', np.sum(tempX[33, :]))
        numVaccinesGiven = total1Dgiven + total2Dgiven
        # print(numVaccinesGiven)
        # print(realVacGiven1D)
        # print(realVacGiven)

    else:
        # print('instantaneous vaccination')
        #instantaneous vaccination:
        initCond = initCond.reshape((34, numAgeGroups))
        # print(initCond)
        initCond2 = np.copy(initCond)
        # move first people vaccinated with two doses:
        peopleVac2 = np.minimum(initCond[0, :], numVaccinatedAgeGroup2d * fractionSus)
        initCond2[0, :] = initCond[0, :] - peopleVac2  # move people out of the susceptible class
        initCond2[22, :] += peopleVac2  # move those people to the vaccinated with two doses class

        # move the people vaccinated with one dose:
        initCond3 = np.copy(initCond2)
        peopleVac1 = np.minimum(initCond2[0, :], numVaccinatedAgeGroup1d * fractionSus)
        initCond3[0, :] = initCond2[0, :] - peopleVac1
        initCond3[11, :] += peopleVac1

        initCond3 = initCond3.reshape(34 * numAgeGroups)
        # print(np.shape(initCond3))
        tspan2 = np.linspace(0, tspan, tspan * 2)
        # run the ODEs
        out = odeint(coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis, initCond3, tspan2, args=(paramsODE,))
        timeSeries = np.copy(out)
        numVaccinesGiven = numVaccinatedAgeGroup1d + 2*numVaccinatedAgeGroup2d
        # print(numVaccinesGiven)
    return timeSeries
    #     ##### READ THE OUTPUTS:
    #     ############################# retrieve the outputs of interest:  ###############################################
    # out2 = timeSeries[-1, :].reshape(34, numAgeGroups)
    # # retrieve the objectives we are interested on:
    #
    # #total infections
    # y0 = y0.reshape(34, numAgeGroups)
    # totalInfectedPreVaccination = np.sum(y0[2, :] + y0[3, :] + y0[4, :] + y0[5, :] + y0[6, :])
    # totalInfections = np.sum(timeSeries[:, (2 * numAgeGroups): (7 * numAgeGroups)], 1) + \
    #                              np.sum(timeSeries[:, (13 * numAgeGroups): (18 * numAgeGroups)], 1) + \
    #                              np.sum(timeSeries[:, (24 * numAgeGroups):(29 * numAgeGroups)], 1) - \
    #                              totalInfectedPreVaccination
    #
    # # this reads the symptomatic infected groups. We need to substract from here the recovered that were already there before vaccination
    #
    # totalSymptomaticInfectedPreVaccination = np.sum(y0[4, :] + y0[5, :] + y0[6, :])
    # totalSymptomaticInfections = np.sum(timeSeries[:, (4 * numAgeGroups): (7 * numAgeGroups)], 1) + \
    #                              np.sum(timeSeries[:, (15 * numAgeGroups): (18 * numAgeGroups)], 1) + \
    #                              np.sum(timeSeries[:, (26 * numAgeGroups):(29 * numAgeGroups)], 1) - \
    #                              totalSymptomaticInfectedPreVaccination
    #
    # # compute the hospitalization curve, for vaccinated and unvaccinated groups
    # hosp = np.sum(timeSeries[:, (5 * numAgeGroups):(6 * numAgeGroups)], 1) + \
    #        np.sum(timeSeries[:, (16 * numAgeGroups): (17 * numAgeGroups)], 1) + \
    #        np.sum(timeSeries[:, (27 * numAgeGroups): (28 * numAgeGroups)], 1)
    #
    # # compute the ICU hospitalizations for vaccinated and unvaccinated groups:
    # icu = np.sum(timeSeries[:, (6 * numAgeGroups):(7 * numAgeGroups)], 1) + \
    #       np.sum(timeSeries[:, (17 * numAgeGroups): (18 * numAgeGroups)], 1) + \
    #       np.sum(timeSeries[:, (28 * numAgeGroups): (29 * numAgeGroups)], 1)
    #
    # return [totalInfections, totalSymptomaticInfections, hosp, icu]


def run_model(allOtherParams, fracVacs, numDosesPerWeek, numVaccinesAvailable, SDval, VE_I1, VE_I2, VE_P1, VE_P2, VE_S1,
                  VE_S2):
    # print('run model',SDval)
    [currentInfections, frac_rec, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP,
     groupFracs, hosp_rate_16, icu_rate_16, mortality_rate_16,
     myContactMats, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate,
     redA, redH, redP, red_sus, R0, sigma, totalPop16, totalPop5] = allOtherParams

    myContactMatAll = myContactMats['all']
    # print('run model2', SDval)
    #JE added to test error and to get a skeleton plot going
    mymatSD = np.zeros((16,16))
    if SDval == 0:
        # print('entre aqui', SDval)
        mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.2 * myContactMats['otherloc'] + 0.1 * \
                  myContactMats['school']
    elif SDval == 1:
        # print('entre aqui', SDval)
        mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.5 * myContactMats['otherloc'] + 0.5 * \
                  myContactMats[
                      'school']  # results in Reff = 1.7 with 10% recovered
    elif SDval == 2:
        # print('entre aqui', SDval)
        mymatSD = myContactMatAll
    elif SDval == 3:
        # print('entre aqui', SDval)
        mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.4 * myContactMats['otherloc'] + 0.1 * \
                  myContactMats[
                      'school']  # results in Reff = 1.39 with 10% recovered

    else:
        print('problem')

    numVaccineGroups = 5
    tspan = 28 * 7
    # compute beta based on these parameters:
    beta = findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(myContactMatAll, frac_sym,
                                                                                    gammaA, gammaE, gammaI, gammaP,
                                                                                    hosp_rate_16, redA, redP, red_sus,
                                                                                    sigma, R0, totalPop16)

    # With this values for mymatSD and for 10% of the pop recovered, we get an effective R of 1.2
    paramsODE = [beta, mymatSD, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16, icu_rate_16,
                 numAgeGroups,
                 oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus, sigma, totalPop16,
                 VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2]

    # fraction of the population assumed to be immune at the beginning of vaccination


    y0 = defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups,
                          oneMinusHospRate, oneMinusICUrate,
                          totalPop16)
    fracVacs = repairVector2D(fracVacs)
    [newarray, excessVaccine] = skimOffExcessVaccine2DBis(fracVacs, numVaccinesAvailable, numVaccineGroups,
                                                          totalPop5)

    numVaccinatedAgeGroup1d = splitVaccineAmongAgeGroups(newarray[0, :], groupFracs, totalPop16)
    numVaccinatedAgeGroup2d = splitVaccineAmongAgeGroups(newarray[1, :], groupFracs, totalPop16)

    timeSeries = runVaccination2DPulse6FullOutput(mortality_rate_16, y0, groupFracs, numAgeGroups, numDosesPerWeek,
                                                  numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,
                                                  numVaccinesAvailable, paramsODE,
                                                  tspan)

    ##### READ THE OUTPUTS:
    ############################# retrieve the outputs of interest:  ###############################################
    # out2 = timeSeries[-1, :].reshape(34, numAgeGroups)
    # retrieve the objectives we are interested on:

    # total infections
    y0 = y0.reshape(34, numAgeGroups)
    totalInfectedPreVaccination = np.sum(y0[2, :] + y0[3, :] + y0[4, :] + y0[5, :] + y0[6, :])
    totalInfections = np.sum(timeSeries[:, (2 * numAgeGroups): (7 * numAgeGroups)], 1) + \
                      np.sum(timeSeries[:, (13 * numAgeGroups): (18 * numAgeGroups)], 1) + \
                      np.sum(timeSeries[:, (24 * numAgeGroups):(29 * numAgeGroups)], 1) #- \
                      #totalInfectedPreVaccination

    # this reads the symptomatic infected groups. We need to substract from here the recovered that were already there before vaccination

    totalSymptomaticInfectedPreVaccination = np.sum(y0[4, :] + y0[5, :] + y0[6, :])
    totalSymptomaticInfections = np.sum(timeSeries[:, (4 * numAgeGroups): (7 * numAgeGroups)], 1) + \
                                 np.sum(timeSeries[:, (15 * numAgeGroups): (18 * numAgeGroups)], 1) + \
                                 np.sum(timeSeries[:, (26 * numAgeGroups):(29 * numAgeGroups)], 1) #- \
                                 #totalSymptomaticInfectedPreVaccination

    # compute the hospitalization curve, for vaccinated and unvaccinated groups
    hosp = np.sum(timeSeries[:, (5 * numAgeGroups):(6 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (16 * numAgeGroups): (17 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (27 * numAgeGroups): (28 * numAgeGroups)], 1)

    # compute the ICU hospitalizations for vaccinated and unvaccinated groups:
    icu = np.sum(timeSeries[:, (6 * numAgeGroups):(7 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (17 * numAgeGroups): (18 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (28 * numAgeGroups): (29 * numAgeGroups)], 1)

    totalRecoveredPreVaccination = np.sum(y0[7, :] + y0[8, :] + y0[9, :] + y0[10, :])
    totalRecovered = np.sum(timeSeries[:, (7 * numAgeGroups): (11 * numAgeGroups)], 1) + \
                     np.sum(timeSeries[:, (18 * numAgeGroups):(22 * numAgeGroups)], 1) + \
                     np.sum(timeSeries[:, (29 * numAgeGroups):(33 * numAgeGroups)], 1) #- \
                     #totalRecoveredPreVaccination

    totalSymptomaticRecoveredPreVaccination = np.sum(y0[7, :] + y0[9, :] + y0[10, :])
    totalSymptomaticRecovered = np.sum(timeSeries[:, (7 * numAgeGroups): (8 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (9 * numAgeGroups): (11 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (18 * numAgeGroups):(19 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (20 * numAgeGroups):(22 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (29 * numAgeGroups):(30 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (31 * numAgeGroups):(33 * numAgeGroups)], 1) #- \
                                #totalSymptomaticRecoveredPreVaccination
    # print(hosp[0])
    # print([np.max(hosp), np.max(icu)])
    return [totalInfections, totalSymptomaticInfections, hosp, icu, totalRecovered, totalSymptomaticRecovered]


def run_model2(allOtherParams, fracVacs, numDosesPerWeek, numVaccinesAvailable, SDval, VE_I1, VE_I2, VE_P1, VE_P2, VE_S1,
                  VE_S2):
    # print('run model',SDval)
    # print('VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2', [VE_I1, VE_I2, VE_P1, VE_P2, VE_S1,
    #               VE_S2])
    [currentInfections, frac_rec, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP,
     groupFracs, hosp_rate_16, icu_rate_16, mortality_rate_16,
     myContactMats, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate,
     redA, redH, redP, red_sus, R0, sigma, totalPop16, totalPop5] = allOtherParams

    myContactMatAll = myContactMats['all']
    # print('run model2', SDval)
    #JE added to test error and to get a skeleton plot going
    mymatSD = np.zeros((16,16))
    if SDval == 0:
        # print('entre aqui', SDval)
        mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.2 * myContactMats['otherloc'] + 0.1 * \
                  myContactMats['school']
    elif SDval == 1:
        # print('entre aqui', SDval)
        mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.5 * myContactMats['otherloc'] + 0.5 * \
                  myContactMats[
                      'school']  # results in Reff = 1.7 with 10% recovered
    elif SDval == 2:
        # print('entre aqui', SDval)
        mymatSD = myContactMatAll
    elif SDval == 3:
        # print('entre aqui', SDval)
        mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.4 * myContactMats['otherloc'] + 0.1 * \
                  myContactMats[
                      'school']  # results in Reff = 1.39 with 10% recovered

    else:
        print('problem')

    numVaccineGroups = 5
    tspan = 28 * 7
    # compute beta based on these parameters:
    beta = findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(myContactMatAll, frac_sym,
                                                                                    gammaA, gammaE, gammaI, gammaP,
                                                                                    hosp_rate_16, redA, redP, red_sus,
                                                                                    sigma, R0, totalPop16)

    # With this values for mymatSD and for 10% of the pop recovered, we get an effective R of 1.2
    paramsODE = [beta, mymatSD, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16, icu_rate_16,
                 numAgeGroups,
                 oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus, sigma, totalPop16,
                 VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2]

    # fraction of the population assumed to be immune at the beginning of vaccination


    y0 = defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups,
                          oneMinusHospRate, oneMinusICUrate,
                          totalPop16)
    fracVacs = repairVector2D(fracVacs)
    [newarray, excessVaccine] = skimOffExcessVaccine2DBis(fracVacs, numVaccinesAvailable, numVaccineGroups,
                                                          totalPop5)

    # print(newarray)
    numVaccinatedAgeGroup1d = splitVaccineAmongAgeGroups(newarray[0, :], groupFracs, totalPop16)
    numVaccinatedAgeGroup2d = splitVaccineAmongAgeGroups(newarray[1, :], groupFracs, totalPop16)

    # print(numVaccinatedAgeGroup1d)
    # print(numVaccinatedAgeGroup2d)

    timeSeries = runVaccination2DPulse6FullOutput(mortality_rate_16, y0, groupFracs, numAgeGroups, numDosesPerWeek,
                                                  numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,
                                                  numVaccinesAvailable, paramsODE,
                                                  tspan)

    ##### READ THE OUTPUTS:
    ############################# retrieve the outputs of interest:  ###############################################
    out2 = timeSeries[-1, :].reshape(34, numAgeGroups)
    # retrieve the objectives we are interested on:
    infections = out2[33, :]
    totalCumInfections = np.sum(infections)

    # this reads the recovered symptomatic groups. We need to substract from here the recovered that were already there before vaccination
    y0 = y0.reshape(34, numAgeGroups)
    totalSymptomaticRecoveredPreVaccination = np.sum(y0[7, :] + y0[9, :] + y0[10, :])
    totalCumSymptomaticInfections = np.sum(timeSeries[-1, (7 * numAgeGroups): (8 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (9 * numAgeGroups): (11 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (18 * numAgeGroups):(19 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (20 * numAgeGroups):(22 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (29 * numAgeGroups):(30 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (31 * numAgeGroups):(33 * numAgeGroups)]) - \
                                 totalSymptomaticRecoveredPreVaccination

    # total infections
    y0 = y0.reshape(34, numAgeGroups)
    totalInfectedPreVaccination = np.sum(y0[2, :] + y0[3, :] + y0[4, :] + y0[5, :] + y0[6, :])
    totalInfections = np.sum(timeSeries[:, (2 * numAgeGroups): (7 * numAgeGroups)], 1) + \
                      np.sum(timeSeries[:, (13 * numAgeGroups): (18 * numAgeGroups)], 1) + \
                      np.sum(timeSeries[:, (24 * numAgeGroups):(29 * numAgeGroups)], 1) #- \
                      # totalInfectedPreVaccination

    # this reads the symptomatic infected groups. We need to substract from here the recovered that were already there before vaccination

    totalSymptomaticInfectedPreVaccination = np.sum(y0[4, :] + y0[5, :] + y0[6, :])
    totalSymptomaticInfections = np.sum(timeSeries[:, (4 * numAgeGroups): (7 * numAgeGroups)], 1) + \
                                 np.sum(timeSeries[:, (15 * numAgeGroups): (18 * numAgeGroups)], 1) + \
                                 np.sum(timeSeries[:, (26 * numAgeGroups):(29 * numAgeGroups)], 1) #- \
                                 # totalSymptomaticInfectedPreVaccination

    # compute the hospitalization curve, for vaccinated and unvaccinated groups
    hosp = np.sum(timeSeries[:, (5 * numAgeGroups):(6 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (16 * numAgeGroups): (17 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (27 * numAgeGroups): (28 * numAgeGroups)], 1)

    # compute the ICU hospitalizations for vaccinated and unvaccinated groups:
    icu = np.sum(timeSeries[:, (6 * numAgeGroups):(7 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (17 * numAgeGroups): (18 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (28 * numAgeGroups): (29 * numAgeGroups)], 1)

    maxHosp = np.max(hosp)
    maxICU = np.max(icu)

    totalRecoveredPreVaccination = np.sum(y0[7, :] + y0[8, :] + y0[9, :] + y0[10, :])
    totalRecovered = np.sum(timeSeries[:, (7 * numAgeGroups): (11 * numAgeGroups)], 1) + \
                     np.sum(timeSeries[:, (18 * numAgeGroups):(22 * numAgeGroups)], 1) + \
                     np.sum(timeSeries[:, (29 * numAgeGroups):(33 * numAgeGroups)], 1) #- \
                     # totalRecoveredPreVaccination

    totalSymptomaticRecoveredPreVaccination = np.sum(y0[7, :] + y0[9, :] + y0[10, :])
    totalSymptomaticRecovered = np.sum(timeSeries[:, (7 * numAgeGroups): (8 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (9 * numAgeGroups): (11 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (18 * numAgeGroups):(19 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (20 * numAgeGroups):(22 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (29 * numAgeGroups):(30 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (31 * numAgeGroups):(33 * numAgeGroups)], 1) #- \
                                # totalSymptomaticRecoveredPreVaccination
    # print([np.max(hosp), np.max(icu)])
    totalHospitalizedRecoveredPriorVaccination = (y0[9, :] + y0[10, :])
    recHosp = (out2[9, :]) + out2[10, :] + out2[20, :] + out2[21, :] + out2[31, :] + out2[32, :] - \
              totalHospitalizedRecoveredPriorVaccination
    # print(recHosp)
    deaths = np.multiply(mortality_rate_16, recHosp)
    totalDeaths = np.sum(deaths)


    return [totalInfections, totalSymptomaticInfections, totalDeaths, hosp, icu, totalRecovered, totalSymptomaticRecovered,
            totalCumInfections, totalCumSymptomaticInfections, totalDeaths, maxHosp, maxICU]



def run_model3(allOtherParams, fracVacs, numDosesPerWeek, numVaccinesAvailable, SDcoeffs, VE_I1, VE_I2, VE_P1, VE_P2, VE_S1,
                  VE_S2, y0):
    # print('run model',SDval)
    # print('VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2', [VE_I1, VE_I2, VE_P1, VE_P2, VE_S1,
    #               VE_S2])
    [currentInfections, frac_rec, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP,
     groupFracs, hosp_rate_16, icu_rate_16, mortality_rate_16,
     myContactMats, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate,
     redA, redH, redP, red_sus, R0, sigma, totalPop16, totalPop5] = allOtherParams

    myContactMatAll = myContactMats['all']
    # print('run model2', SDval)
    #JE added to test error and to get a skeleton plot going
    mymatSD = np.zeros((16,16))
    # if SDval == 0:
        # print('entre aqui', SDval)
    mymatSD = SDcoeffs[0] * myContactMats['home'] + SDcoeffs[1] * myContactMats['work'] + \
              SDcoeffs[2] * myContactMats['otherloc'] + SDcoeffs[3] * \
              myContactMats['school']
    # elif SDval == 1:
    #     # print('entre aqui', SDval)
    #     mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.5 * myContactMats['otherloc'] + 0.5 * \
    #     SDval == 2:
    #     # print('entre aqui', SDval)
    #     mymatSD = myContactMatAll
    # elif SDval == 3:
    #     # print('entre aqui', SDval)
    #     mymatSD = myContactMats['home'] + 0.6 * myContactMats['work'] + 0.4 * myContactMats['otherloc'] + 0.1 * \
    #               myContactMats[
    #                   'school']  # results in Reff = 1.39 with 10% recovered
    # elif SDval == 4:
    #     mymatSD = myContactMats['home'] + 0.3 * myContactMats['work'] + 0.4 * myContactMats['otherloc'] + 0.7 * \
    #               myContactMats[
    #                   'school']
    # else:
    #     print('problem')

    numVaccineGroups = 5
    tspan = 28 * 7
    # compute beta based on these parameters:
    beta = findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(myContactMatAll, frac_sym,
                                                                                    gammaA, gammaE, gammaI, gammaP,
                                                                                    hosp_rate_16, redA, redP, red_sus,
                                                                                    sigma, R0, totalPop16)

    # With this values for mymatSD and for 10% of the pop recovered, we get an effective R of 1.2
    paramsODE = [beta, mymatSD, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16, icu_rate_16,
                 numAgeGroups,
                 oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus, sigma, totalPop16,
                 VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2]

    # fraction of the population assumed to be immune at the beginning of vaccination



    fracVacs = repairVector2D(fracVacs)
    [newarray, excessVaccine] = skimOffExcessVaccine2DBis(fracVacs, numVaccinesAvailable, numVaccineGroups,
                                                          totalPop5)

    # print(newarray)
    numVaccinatedAgeGroup1d = splitVaccineAmongAgeGroups(newarray[0, :], groupFracs, totalPop16)
    numVaccinatedAgeGroup2d = splitVaccineAmongAgeGroups(newarray[1, :], groupFracs, totalPop16)

    # print(numVaccinatedAgeGroup1d)
    # print(numVaccinatedAgeGroup2d)

    timeSeries = runVaccination2DPulse6FullOutput(mortality_rate_16, y0, groupFracs, numAgeGroups, numDosesPerWeek,
                                                  numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,
                                                  numVaccinesAvailable, paramsODE,
                                                  tspan)

    ##### READ THE OUTPUTS:
    ############################# retrieve the outputs of interest:  ###############################################
    out2 = timeSeries[-1, :].reshape(34, numAgeGroups)
    # retrieve the objectives we are interested on:
    infections = out2[33, :]
    totalCumInfections = np.sum(infections)

    # this reads the recovered symptomatic groups. We need to substract from here the recovered that were already there before vaccination
    y0 = y0.reshape(34, numAgeGroups)
    totalSymptomaticRecoveredPreVaccination = np.sum(y0[7, :] + y0[9, :] + y0[10, :])
    totalCumSymptomaticInfections = np.sum(timeSeries[-1, (7 * numAgeGroups): (8 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (9 * numAgeGroups): (11 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (18 * numAgeGroups):(19 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (20 * numAgeGroups):(22 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (29 * numAgeGroups):(30 * numAgeGroups)]) + \
                                 np.sum(timeSeries[-1, (31 * numAgeGroups):(33 * numAgeGroups)]) - \
                                 totalSymptomaticRecoveredPreVaccination

    # total infections
    y0 = y0.reshape(34, numAgeGroups)
    totalInfectedPreVaccination = np.sum(y0[2, :] + y0[3, :] + y0[4, :] + y0[5, :] + y0[6, :])
    totalInfections = np.sum(timeSeries[:, (2 * numAgeGroups): (7 * numAgeGroups)], 1) + \
                      np.sum(timeSeries[:, (13 * numAgeGroups): (18 * numAgeGroups)], 1) + \
                      np.sum(timeSeries[:, (24 * numAgeGroups):(29 * numAgeGroups)], 1) #- \
                      # totalInfectedPreVaccination

    # this reads the symptomatic infected groups. We need to substract from here the recovered that were already there before vaccination

    totalSymptomaticInfectedPreVaccination = np.sum(y0[4, :] + y0[5, :] + y0[6, :])
    totalSymptomaticInfections = np.sum(timeSeries[:, (4 * numAgeGroups): (7 * numAgeGroups)], 1) + \
                                 np.sum(timeSeries[:, (15 * numAgeGroups): (18 * numAgeGroups)], 1) + \
                                 np.sum(timeSeries[:, (26 * numAgeGroups):(29 * numAgeGroups)], 1) #- \
                                 # totalSymptomaticInfectedPreVaccination

    # compute the hospitalization curve, for vaccinated and unvaccinated groups
    hosp = np.sum(timeSeries[:, (5 * numAgeGroups):(6 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (16 * numAgeGroups): (17 * numAgeGroups)], 1) + \
           np.sum(timeSeries[:, (27 * numAgeGroups): (28 * numAgeGroups)], 1)

    # compute the ICU hospitalizations for vaccinated and unvaccinated groups:
    icu = np.sum(timeSeries[:, (6 * numAgeGroups):(7 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (17 * numAgeGroups): (18 * numAgeGroups)], 1) + \
          np.sum(timeSeries[:, (28 * numAgeGroups): (29 * numAgeGroups)], 1)

    maxHosp = np.max(hosp)
    maxICU = np.max(icu)

    totalRecoveredPreVaccination = np.sum(y0[7, :] + y0[8, :] + y0[9, :] + y0[10, :])
    totalRecovered = np.sum(timeSeries[:, (7 * numAgeGroups): (11 * numAgeGroups)], 1) + \
                     np.sum(timeSeries[:, (18 * numAgeGroups):(22 * numAgeGroups)], 1) + \
                     np.sum(timeSeries[:, (29 * numAgeGroups):(33 * numAgeGroups)], 1) #- \
                     # totalRecoveredPreVaccination

    totalSymptomaticRecoveredPreVaccination = np.sum(y0[7, :] + y0[9, :] + y0[10, :])
    totalSymptomaticRecovered = np.sum(timeSeries[:, (7 * numAgeGroups): (8 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (9 * numAgeGroups): (11 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (18 * numAgeGroups):(19 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (20 * numAgeGroups):(22 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (29 * numAgeGroups):(30 * numAgeGroups)], 1) + \
                                np.sum(timeSeries[:, (31 * numAgeGroups):(33 * numAgeGroups)], 1) #- \
                                # totalSymptomaticRecoveredPreVaccination
    # print([np.max(hosp), np.max(icu)])
    totalHospitalizedRecoveredPriorVaccination = (y0[9, :] + y0[10, :])
    recHosp = (out2[9, :]) + out2[10, :] + out2[20, :] + out2[21, :] + out2[31, :] + out2[32, :] - \
              totalHospitalizedRecoveredPriorVaccination
    # print(recHosp)
    deaths = np.multiply(mortality_rate_16, recHosp)
    totalDeaths = np.sum(deaths)


    return [totalInfections, totalSymptomaticInfections, totalDeaths, hosp, icu, totalRecovered, totalSymptomaticRecovered,
            totalCumInfections, totalCumSymptomaticInfections, totalDeaths, maxHosp, maxICU]





if __name__ == '__main__':
    # np.set_printoptions(precision=2, suppress=True)
    N = 7.615 * 10**(6) #Washington state population

    # load contact matrices
    myfilename = '../data/consistentMatricesUS_polymodMethod01Jun2020.pickle'
    mymats = loadResults(myfilename)
    mymatAll = mymats['all']


    myseed = np.abs(111111)
    np.random.seed(myseed)

    # load fractions in each age and vaccine group:
    myfilename = '../data/populationUS16ageGroups03Jun2020.pickle'
    popUS16 = loadResults(myfilename)
    popUS16fracs = popUS16[1]

    myfilename = '../data/populationUSGroupsForOptimization03Jun2020.pickle'
    groupInfo = loadResults(myfilename)
    groupFracs = groupInfo['groupsFracs']
    fracOfTotalPopulationPerVaccineGroup = groupInfo['fracOfTotalPopulationPerVaccineGroup']
    [relativeFrac75_80, relativeFrac80andAbove] = groupInfo['split75andAbove']

    # Split the population in 16 age groups:
    totalPop16 = N * popUS16fracs
    # Split the population in 5 vaccine groups:
    totalPop5 = N * fracOfTotalPopulationPerVaccineGroup

    numAgeGroups = 16
    numVaccineGroups = 5

    # load disease severity parameters:
    myfilename = '../data/disease_severity_parametersFerguson.pickle'
    diseaseParams = loadResults(myfilename)
    hosp_rate_16 = diseaseParams['hosp_rate_16']
    icu_rate_16 = diseaseParams['icu_rate_16']

    # load mortality parameters
    myfilename = '../data/salje_IFR_from_hospitalized_cases.pickle'
    salje_mortality_rate_16 = loadResults(myfilename)
    mortality_rate_16 = salje_mortality_rate_16

    # this is just 1 - ICU rate useful to compute it in advance and pass it to the ODE
    oneMinusICUrate = np.ones(numAgeGroups) - icu_rate_16
    # this is just 1 - Hosp rate useful to compute it in advance and pass it to the ODE
    oneMinusHospRate = np.ones(numAgeGroups) - hosp_rate_16

    # time horizon for the intervention:
    tspan = 28 * 7  # np.linspace(0, 365, 365 * 2)
    tspan2 = 32 * 7
    ########################################################################################################################
    ######################## Parameters that will change for sensitivity analysis ####################################
    # Model parameters

    # fraction of symptomatic people
    frac_asymptomatic = 0.4 * np.ones(16)  # see notesTwoDoses.md for details
    frac_asymptomatic[0:4] = 0.75  # see notes for details
    frac_sym = (1 - frac_asymptomatic) * np.ones(16)  # fraction of infected that are symptomatic
    # fraction of symptomatic children
    # frac_sym[0:4] = 0.2
    oneMinusSymRate = np.ones(16) - frac_sym
    print(oneMinusSymRate)

    # transition rates:
    durI = 4  # duration of infectiousness after developing symptoms
    durP = 2  # duration of infectiousness before developing symptoms
    durA = durI + durP  # the duration of asymptomatic infections is equal to that of symptomatic infections
    gammaA = 1 / durA  # recovery rate for asymptomatic

    gammaI = 1 / durI  # recovery rate for symptomatic infections (not hospitalized)

    gammaP = 1 / durP  # transition rate fromm pre-symptomatic to symptomatic
    gammaE = 1 / 3  # transition rate from exposed to infectious

    # reduction/increase of infectiousness
    redA = 0.75  # reduction of infectiousness for asymptomatic infections, CDC recommendation
    redH = 0.  # assume no transmission for hospitalized patients
    redP = 1.  # see notesTwoDoses.md for the sources to assume equal transmissibility

    # hospitalization duration based on age:
    # hospital stays
    gammaH = np.ones(16)
    gammaH[0:10] = 1 / 3
    gammaH[10:13] = 1 / 4
    gammaH[13:] = 1 / 6

    gammaICU = np.ones(16)
    gammaICU[0:10] = 1 / 10
    gammaICU[10:13] = 1 / 14
    gammaICU[13:] = 1 / 12

    red_sus = np.ones(16)  # assume no reduction in susceptibility
    red_sus[0:3] = 0.56  # reduction in susceptibility: taken from Viner 2020
    # red_sus[3:13] = 1
    red_sus[13:16] = 2.7  # taken from Bi et al medarxiv 2021

    sigma_base = 1 / 3.8
    sigma = sigma_base * np.ones(16)

    # Disease severity
    R0 = 3
    frac_rec = 0.2
    currentInfections = 0.3 * (1 / 100) * N * popUS16fracs

    # compute beta based on these parameters:
    beta = findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(mymatAll, frac_sym,
                                                                                    gammaA, gammaE, gammaI, gammaP,
                                                                                    hosp_rate_16, redA, redP, red_sus,
                                                                                    sigma, R0, totalPop16)

    myprev = 0.1
    currentInfections = myprev * (1 / 100) * N * popUS16fracs
    print(np.sum(currentInfections))
    y0 = defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups,
                          oneMinusHospRate, oneMinusICUrate,
                          totalPop16)
    proRataVec = createProRataVac2DAdultsOnly(0.8*N, totalPop5)
    highRisk = createVectorHighRiskFirst2DosesOnly(0.5*N, 5, totalPop5)
    print(proRataVec)
    print(fromFracVacsToFracVaccinatedInEachVaccineGroup(proRataVec, 0.5*N, totalPop5))
    print(fromFracVacsToFracVaccinatedInEachVaccineGroup(highRisk, 0.5 * N, totalPop5))
    fracVacs = np.array([[0., 0.3, 0.1, 0.1, 0], [0, 0, 0.1, 0.2, 0.2]])
    print(fromFracVacsToFracVaccinatedInEachVaccineGroup(repairVector2D(fracVacs), 0.5 * N, totalPop5))
    # VEmat = [[0.18, 0.9, 0.0, 0.0, 0, 0],
    #          [0.45, 0.9, 0.0, 0.0, 0, 0],
    #          [0.72, 0.9, 0.0, 0.0, 0, 0],
    #          [0.1, 0.66, 0.09, 0.7, 0, 0],
    #          [0.26, 0.66, 0.26, 0.7, 0, 0],
    #          [0.44, 0.66, 0.5, 0.7, 0, 0],
    #          [0.0, 0.0, 0.18, 0.9, 0, 0],
    #          [0.0, 0.0, 0.45, 0.9, 0, 0],
    #          [0.0, 0.0, 0.72, 0.9, 0, 0]]
    #
    # # parameters for the Social Distancing
    # SD0coeffs = [0.6, 0.2, 0.1]
    # SD1coeffs = [0.6, 0.5, 0.5]
    # SD2coeffs = [1, 1, 1]
    # SD3coeffs = [0.6, 0.4, 0.1]
    #
    # SDmat = [SD0coeffs, SD1coeffs, SD2coeffs, SD3coeffs]
    #
    # for sivals in range(1):
    #     print('******** si vals', sivals)
    #     SDcoeffs = SDmat[sivals]
    #     mymatSD = mymats['home'] + SDcoeffs[0] * mymats['work'] + SDcoeffs[1] * mymats['otherloc'] + SDcoeffs[2] * mymats[
    #         'school']
    #     # plt.figure(sivals)
    #     for vevals in range(4,5):
    #         print('******** ve vals', vevals)
    #         [VE_P1, VE_P2, VE_S1, VE_S2, VE_I1, VE_I2] = VEmat[vevals]
    #         print([VE_P1, VE_P2, VE_S1, VE_S2, VE_I1, VE_I2])
    #         # With this values for mymatSD and for 10% of the pop recovered, we get an effective R of 1.2
    #         paramsODE = [beta, mymatSD, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16, icu_rate_16,
    #                      numAgeGroups,
    #                      oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus, sigma, totalPop16,
    #                      VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2]
    #
    #         paramsODEBaseline = [beta, mymatSD, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16,
    #                      icu_rate_16,
    #                      numAgeGroups,
    #                      oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus, sigma,
    #                      totalPop16,
    #                      0,0,0,0,0,0]
    #         # #fraction of the population assumed to be immune at the beginning of vaccination
    #         # frac_rec = 0.1
    #         #
    #         # #number of current infections:
    #         # currentInfections = 0.3 * (1 / 100) * N * popUS16fracs#1000*popUS16fracs
    #         ##################################################################################################################
    #         S0 = (1 - frac_rec) * (totalPop16 - currentInfections)
    #         # print(S0)
    #         I0 = frac_sym * currentInfections
    #         E0 = np.zeros(numAgeGroups)
    #         A0 = (1 - frac_sym) * currentInfections
    #         P0 = np.zeros(numAgeGroups)
    #         H0 = np.zeros(numAgeGroups)
    #         ICU0 = np.zeros(numAgeGroups)
    #         Rec0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections), oneMinusHospRate)
    #         RecA0 = (1 - frac_sym) * frac_rec * (totalPop16 - currentInfections)
    #         RecH0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections),
    #                             np.multiply(hosp_rate_16, oneMinusICUrate))
    #         RecICU0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections),
    #                               np.multiply(hosp_rate_16, icu_rate_16))
    #
    #         # print(np.sum(Rec0 + RecA0 + RecH0 + RecICU0)/N)
    #         # Vaccinated 1d initial conditions
    #         V10 = np.zeros(numAgeGroups)
    #         E_V10, A_V10, P_V10, I_V10, H_V10, ICU_V10, RecV_10, RecAV_10, RecHV_10, RecICUV_10 = np.zeros(numAgeGroups), \
    #                                                                                     np.zeros(numAgeGroups), np.zeros(
    #             numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), \
    #                                                                                     np.zeros(numAgeGroups), np.zeros(
    #             numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups)
    #
    #         # Vaccinated 2d initial conditions
    #         V20 = np.zeros(numAgeGroups)
    #         E_V20, A_V20, P_V20, I_V20, H_V20, ICU_V20, RecV_20, RecAV_20, RecHV_20, RecICUV_20 = np.zeros(numAgeGroups), \
    #                                                                                     np.zeros(numAgeGroups), np.zeros(
    #             numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), \
    #                                                                                     np.zeros(numAgeGroups), np.zeros(
    #             numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups)
    #
    #         Cases0 = np.copy(I0) + np.copy(A0)
    #         # print(Cases0)
    #         # print(Cases0)
    #
    #         y0 = np.array([S0, E0, A0, P0, I0, H0, ICU0, Rec0, RecA0, RecH0, RecICU0,
    #                        V10, E_V10, A_V10, P_V10, I_V10, H_V10, ICU_V10, RecV_10, RecAV_10, RecHV_10, RecICUV_10,
    #                        V20, E_V20, A_V20, P_V20, I_V20, H_V20, ICU_V20, RecV_20, RecAV_20, RecHV_20, RecICUV_20,
    #                        Cases0]).reshape((34 * numAgeGroups))
    #
    #
    #         y0bis = defineInitCond2D(currentInfections, frac_rec, frac_sym, hosp_rate_16, icu_rate_16, numAgeGroups, oneMinusHospRate, oneMinusICUrate,
    #                          totalPop16)
    #
    #         # baseline =  (runVaccination2DPulse5(salje_mortality_rate_16, y0, groupFracs, numAgeGroups, 300000,
    #         #                                    np.zeros(16), np.zeros(16), numVaccineGroups,10,
    #         #                                    paramsODE,
    #         #                                    tspan))
    #         #
    #         # # baseline2 = (runVaccination2DPulse5(salje_mortality_rate_16, y0, groupFracs, numAgeGroups, 300000,
    #         # #                                    np.zeros(16), np.zeros(16), numVaccineGroups, 10,
    #         # #                                    paramsODE,
    #         # #                                    tspan2))
    #         # print('baseline', baseline)
    #         # # print('baseline2', baseline2)
    #         # numSamples = 10
    #         # dim = 10
    #         # mysample = uniformSampleSimplex(dim, numSamples)
    #         # res = np.zeros(11)
    #
    #
    #         for jvals in range(2,3):
    #             print('******** jvals', jvals)
    #             numVaccinesAvailable = np.rint(0.1 * jvals * N)
    #             # print(numVaccinesAvailable)
    #             numDosesPerWeek = 300000
    #             # proRataVec = createProRataVac2D(fracOfTotalPopulationPerVaccineGroup, numVaccinesAvailable,  totalPop5)
    #             # # highRisk = (createVectorHighRiskFirst2DosesOnly(numVaccinesAvailable, numVaccineGroups, totalPop5))
    #             # print(proRataVec)
    #             # print(fromFracVacsToFracVaccinatedInEachVaccineGroup(highRisk, numVaccinesAvailable, totalPop5))
    #             # temp = createVectorPracticalStrategy(numVaccinesAvailable, totalPop5)
    #             # print(temp)
    #             #
    #             # temp2 = createVectorHighRiskFirst2(numVaccinesAvailable, totalPop5)
    #             # print(temp2)
    #             #
    #             # temp3 = createVectorHighRiskFirst2DosesOnly(numVaccinesAvailable, numVaccineGroups, totalPop5)
    #             # print(temp3)
    #             # tempPep = fromFracVacsToFracVaccinatedInEachVaccineGroup(temp, numVaccinesAvailable, totalPop5)
    #             # print(tempPep)
    #             # popAdults = totalPop5[1:]
    #             # totalPopAdults = np.sum(popAdults)
    #             # relFracAdults = np.divide(popAdults, totalPopAdults)
    #             # # print(relFracAdults)
    #             # # x = proRataAdultsOnly = createProRataVac2D(relFracAdults, numVaccinesAvailable,  popAdults)
    #             # # print(x[0])
    #             # # vacperadultGroup = x[0]*numVaccinesAvailable
    #             # # print(vacperadultGroup)
    #             # # print(np.divide(vacperadultGroup, totalPop5[1:]))
    #             #
    #             proRataVec2 = createProRataVac2DAdultsOnly(numVaccinesAvailable, totalPop5)
    #             print(proRataVec2)
    #             print(fromFracVacsToFracVaccinatedInEachVaccineGroup(proRataVec2, numVaccinesAvailable, totalPop5))
    #
    #    #          fracVacs1 = np.array([[3.29573005e-02, 0.00000000e+00, 2.89355889e-03, 0.00000000e+00,
    #    # 6.14854463e-04], [2.00377859e-03, 1.03642713e-02, 3.36772796e-02,
    #    # 6.26483386e+00, 1.17559295e+01]])
    #    #          for fracVacs in [highRisk, fracVacs1]:
    #    #              print(fromFracVacsToFracVaccinatedInEachVaccineGroup(fracVacs, numVaccinesAvailable, totalPop5))
    #    #          [newarray, excessVaccine] = skimOffExcessVaccine2DBis(proRataVec2, numVaccinesAvailable, numVaccineGroups,
    #    #                                                            totalPop5)
    #    #
    #    #          numVaccinatedAgeGroup1d = splitVaccineAmongAgeGroups(newarray[0, :], groupFracs, totalPop16)
    #    #          numVaccinatedAgeGroup2d = splitVaccineAmongAgeGroups(newarray[1, :], groupFracs, totalPop16)
    #    #          print(numVaccinatedAgeGroup1d)
    #    #          tempBas = (
    #    #          runVaccination2DPulse6(salje_mortality_rate_16, y0, groupFracs, numAgeGroups, numDosesPerWeek,
    #    #                                         numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,
    #    #                                         numVaccinesAvailable,
    #    #                                         paramsODE,
    #    #                                         tspan))
    #    #          print('tempBas', tempBas)
    #    #
    #    #              temp = (runVaccination2DPulse6(salje_mortality_rate_16, y0, groupFracs, numAgeGroups, numDosesPerWeek,
    #    #                                         numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups, numVaccinesAvailable,
    #    #                                         paramsODE,
    #    #                                         tspan))
    #    #              print('temp', temp)
    #    #              print('averted', np.divide((np.array(tempBas) - np.array(temp)),np.array(tempBas)))
    #             # print(np.array(tempBas)-np.array(temp))
    #                 # for kvals in range(5):
    #                 #     plt.subplot(1,5,kvals+1)
    #                 #     plt.plot(jvals, temp[kvals], 'o')
    #
    #             # temp = (runVaccination2DPulse6(salje_mortality_rate_16, y0, groupFracs, numAgeGroups, numVaccinesAvailable,
    #             #                            numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,
    #             #                            numVaccinesAvailable,
    #             #                            paramsODE,
    #             #                            tspan))
    #             # print(temp)
    #
    #             # tempBas = (runVaccination2DPulse6(salje_mortality_rate_16, y0, groupFracs, numAgeGroups, numDosesPerWeek,
    #             #                            numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccineGroups,
    #             #                            numVaccinesAvailable,
    #             #                            paramsODEBaseline,
    #             #                            tspan2))
    #             # print(tempBas)
    #             # print((np.array(baseline) - np.array(temp)))
    #
    #             # tempBas = (runVaccination2DPulse5(salje_mortality_rate_16, y0,  numAgeGroups, numDosesPerWeek,
    #             #                                numVaccinatedAgeGroup1d, numVaccinatedAgeGroup2d, numVaccinesAvailable,
    #             #                                paramsODE,
    #             #                                tspan))
    #             # print('tempBas',tempBas)
    #         #     propAverted = np.divide((np.array(baseline) - np.array(temp)), np.array(baseline))
    #         #
    #         #     res[ivals] = propAverted[myobj]
    #         # print(res)
    #         # plt.subplot(3, 3, vevals +1)
    #         # plt.plot(range(0,11), res)
    #     # # # x = (objectiveFunction2D(fracVacs, extraParams))
    #     # # # print(x[-5:])
    #
    # # plt.show()



