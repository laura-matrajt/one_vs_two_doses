import numpy as np
import sys
from scipy.integrate import odeint
sys.path.insert(1, '../coronavirus_optimization/')
from coronavirusMainFunctions_twoDoses import coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis

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


    # compute the total number of people getting vaccinated with an additional second dose:
    totalPeopleVacWithTwoDoses = np.sum(numVaccinatedAgeGroup2d)
    # print(totalPeopleVacWithTwoDoses)

    #Prepare initial conditions
    initCond = np.copy(y0)
    # Compute the fraction of people susceptible in each age group relative
    # to other groups that could potentially receive vaccination.
    fractionSus = findFractionSus2D(initCond, numAgeGroups)

    initCond = initCond.reshape((34, numAgeGroups))

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
        numVaccinesGiven = total1Dgiven + total2Dgiven


    else:
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