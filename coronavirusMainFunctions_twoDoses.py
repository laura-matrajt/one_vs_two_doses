import numpy as np
from scipy.integrate import odeint


def findBetaModel_coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(C, frac_sym, gammaA, gammaE,  gammaI, gammaP, hosp_rate, redA,  redP, red_sus, sigma, R0, totalPop):
    #compute the value of beta for model described in coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis
    #here, frac_sym is a vector not a scalar and there is a different beta for each age group representing a different
    #susceptibility
    # compute the eignevalues of F*V^(-1) assuming the infected states are 5, namely: E, A, P, I, H
    [n1, n2] = np.shape(C)

    # create F
    N = np.sum(totalPop)
    Z = np.zeros((n1, n1))
    C1 = np.zeros((n1, n1))
    for ivals in range(n1):
        for jvals in range(n1):
            C1[ivals, jvals] = red_sus[ivals] * C[ivals, jvals] * totalPop[ivals]/totalPop[jvals]


    #create F by concatenating different matrices:
    F1 = np.concatenate((Z, redA*C1, redP*C1, C1), 1)
    F2 = np.zeros((3*n1, 4*n1))

    F = np.concatenate((F1, F2), 0)
    # print(np.shape(F))
    # print(np.shape(F1))
    # print(np.shape(F2))
    #create V
    VgammaE = np.diag(gammaE * np.ones(n1))
    VgammaA = np.diag(gammaA * np.ones(n1))
    VgammaP = np.diag(gammaP * np.ones(n1))



    Vsub1 = np.diag(-(np.ones(n1)-frac_sym) * gammaE)
    Vsub2 = np.diag(-(frac_sym) * gammaE)

    Vsub3 = np.diag((np.ones(n1) - hosp_rate) * gammaI + np.multiply(sigma, hosp_rate))

    # print(V)

    V1 = np.concatenate((VgammaE, Z, Z, Z), 1)
    V2 = np.concatenate((Vsub1, VgammaA, Z, Z), 1)
    V3 = np.concatenate((Vsub2, Z, VgammaP, Z), 1)
    V4 = np.concatenate((Z, Z, -VgammaP, Vsub3), 1)


    V = np.concatenate((V1, V2, V3, V4), 0)
    # print(np.shape(V))

    myProd = np.dot(F, np.linalg.inv(V))
    # print(myProd)
    myEig = np.linalg.eig(myProd)
    # print(myEig)
    largestEig = np.max(myEig[0])
    if largestEig.imag == 0.0:

        beta = R0 / largestEig.real
        # print('beta', beta)
        return beta
    else:
        print(largestEig)
        raise Exception('largest eigenvalue is not real')


def coronavirusEqs_withHospitalizationsAndICU_withVaccine2DBis(y, t, params):
    """

    USING A MATRIX OF CONTACTS THAT IS NOT SYMMETRIC and vaccination
    :param y: vector of the current state of the system
    :param t: time
    :param params: all the params to run the ODE, defined below
    beta: rate of infection given contact
    C: contact matrix across age groups

    gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP: transition rates out of the asymptomatic, exposed,
    infectioussymptomatic hospitalized, infectioussymptomatic non-hospitalized, infectioussymptomatic hospitalized in the ICU,
    infectiouspre-symptomatic
     presymptomatic classes
     hospRate, ICUrate: rates of hospitalization in each age group
    numGroups: number of groups in the simualation
    redA,  redP: reduction in the infectiousness for asymptomatic and pre-symptomatic
    totalPop: a vector of size 1x numGroups with the population in each of the age groups.
    VE: vaccine efficacy
    :return:
    """
    [beta, C, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hospRate, ICUrate,  numGroups,
     oneMinusHospRate, oneMinusICUrate, oneMinusSympRate, redA, redH, redP, red_sus, sigma, totalPop, VE_I1, VE_I2, VE_P1, VE_P2, VE_S1, VE_S2] = params


    theta1 = 1 - VE_S1
    theta2 = 1 - VE_S2
    phi1 = 1 - VE_P1
    phi2 = 1 - VE_P2

    psi1 = 1 - VE_I1
    psi2 = 1 - VE_I2

    #beta: infection rate
    #C: matrix of contact rates, c_ij represents the contact rate between group i and group j
    temp = np.reshape(y, (34, numGroups))


    S = temp[0, :]  #susceptibles
    E = temp[1, :]  #Exposed
    A = temp[2, :]  #Asymptomatic infected
    P = temp[3, :]  # Pre-symptomatic infected
    I = temp[4, :]  #Symptomatic infe11,12cted
    H = temp[5, :]  #Hospitalized non ICU
    ICU = temp[6, :]  #Hospitalized ICU
    R = temp[7, :]  #Recovered symptomatic
    RA = temp[8, :] #Recovered Asymptomatic
    RH = temp[9, :] #recovered hospitalized
    RC = temp[10, :] #recovered hospitalized ICU

    S_1V = temp[11, :]  # susceptibles vaccinated
    E_1V = temp[12, :]  # Exposed vaccinated
    A_1V = temp[13, :]  # Asymptomatic infected vaccinated
    P_1V = temp[14, :]  # Pre-symptomatic infected vaccinated
    I_1V = temp[15, :]  # Symptomatic infected vaccinated
    H_1V = temp[16, :]  # Hospitalizes symptomatic infected vaccinated
    ICU_1V = temp[17, :]  # Hospitalized ICU vaccinated
    R_1V = temp[18, :]  # Recovered symptomatic vaccinated
    RA_1V = temp[19, :]  # Recovered Asymptomatic vaccinated
    RH_1V =temp[20,:] #Recovered Hospitalized vaccinated
    RC_1V = temp[21, :]  # recovered hospitalized ICU

    S_2V = temp[22,:]
    E_2V = temp[23,:]
    A_2V = temp[24,:]
    P_2V = temp[25,:]
    I_2V = temp[26,:]
    H_2V = temp[27,:]
    ICU_2V = temp[28,:]
    R__2V = temp[29,:]
    RA_2V = temp[30,:]
    RH_2V = temp[31,:]
    RC_2V = temp[32,:]


    totalInf = temp[33, :]   #total cummulative infections

    Cnew = np.multiply(C, red_sus[:, np.newaxis])  #
    mylambda = np.dot(Cnew, beta * np.divide((redA * (A + psi1*A_1V + psi2*A_2V) +
                                           redP * (P + psi1*P_1V + psi2*P_2V) +
                                           redH * (H + psi1*H_1V + psi2*H_2V) +
                                           (I + psi1*I_1V + psi2*I_2V)), totalPop))  # force of infection

    dS = - np.multiply(mylambda, S)
    dE = np.multiply(mylambda, S) - gammaE * E
    dA = gammaE * np.multiply(E, oneMinusSympRate) - gammaA * A
    dP = gammaE * np.multiply(E, frac_sym) - gammaP * P
    dI = gammaP * P - np.multiply(oneMinusHospRate, gammaI * I) - np.multiply(hospRate, np.multiply(sigma, I))
    dH = np.multiply(oneMinusICUrate, np.multiply(hospRate, np.multiply(sigma, I))) - gammaH * H
    dICU = np.multiply(ICUrate, np.multiply(hospRate, np.multiply(sigma, I))) - gammaICU * ICU
    dR = np.multiply(gammaI * I, oneMinusHospRate)
    dRA = gammaA * A
    dRH = gammaH * H
    dRICU = gammaICU * ICU


    # vaccinated with one dose equations
    dS_1V = - theta1 * np.multiply(mylambda, S_1V)
    dE_1V = theta1 * np.multiply(mylambda, S_1V) - gammaE * E_1V
    dA_1V = (np.ones(numGroups) - phi1*frac_sym) * gammaE * E_1V - gammaA * A_1V
    dP_1V = phi1*frac_sym * gammaE * E_1V - gammaP * P_1V
    dI_1V = gammaP * P_1V - np.multiply(oneMinusHospRate, gammaI * I_1V) - np.multiply(hospRate, np.multiply(sigma, I_1V))
    dH_1V =  np.multiply(oneMinusICUrate, np.multiply(hospRate, np.multiply(sigma, I_1V))) - gammaH * H_1V
    dICU_1V = np.multiply(ICUrate, np.multiply(hospRate, np.multiply(sigma, I_1V))) - gammaICU * ICU_1V
    dR_1V = np.multiply(oneMinusHospRate, gammaI * I_1V)
    dRA_1V = gammaA * A_1V
    dRH_1V = gammaH * H_1V
    dRICU_1V = gammaICU * ICU_1V

    # vaccinated with two doses equations
    dS_2V = - theta2 * np.multiply(mylambda, S_2V)
    dE_2V = theta2 * np.multiply(mylambda, S_2V) - gammaE * E_2V
    dA_2V = (np.ones(numGroups) - phi2*frac_sym) * gammaE * E_2V - gammaA * A_2V
    dP_2V = phi2*frac_sym * gammaE * E_2V - gammaP * P_2V
    dI_2V = gammaP * P_2V - np.multiply(oneMinusHospRate, gammaI * I_2V) - np.multiply(hospRate, np.multiply(sigma, I_2V))
    dH_2V = np.multiply(oneMinusICUrate, np.multiply(hospRate, np.multiply(sigma, I_2V))) - gammaH * H_2V
    dICU_2V = np.multiply(ICUrate, np.multiply(hospRate, np.multiply(sigma, I_2V)))- gammaICU * ICU_2V
    dR_2V = np.multiply(oneMinusHospRate, gammaI * I_2V)
    dRA_2V = gammaA * A_2V
    dRH_2V = gammaH * H_2V
    dRICU_2V = gammaICU * ICU_2V

    dtotalInf = np.multiply(mylambda, S) + theta1 * np.multiply(mylambda, S_1V) + theta2 * np.multiply(mylambda, S_2V)

    dydt = np.array([dS, dE, dA, dP, dI, dH, dICU, dR, dRA, dRH, dRICU,
                     dS_1V, dE_1V, dA_1V, dP_1V, dI_1V, dH_1V, dICU_1V, dR_1V, dRA_1V, dRH_1V, dRICU_1V,
                     dS_2V, dE_2V, dA_2V, dP_2V, dI_2V, dH_2V, dICU_2V, dR_2V, dRA_2V, dRH_2V, dRICU_2V, dtotalInf]).reshape((numGroups * 34))
    return dydt



def splitVaccineAmongAgeGroups(vacVector, fracPerAgeGroup, totalPop):
    """
    This function will split vaccine across age groups for a single vaccination group
    :param vacVector: a vector of size 1*5 with the number of vaccines for each vaccination group
    fracPerAgeGroup: a list with 5 entries, each entry has the number of age groups in that vaccination group.
    :return: a vector of size 1X16 with the number of vaccines to be given to each age group
    """
    #group 1: 0-20:  this includes 4 age groups: 0-5, 5-10, 10-15, 15-20
    #group 2: 20-50: this includes 6 age groups: 20-25, 25-30, 30-35, 35-40, 40-45, 45-50
    #group 3: 50-65: this includes 3 age groups: 50-55, 55-60, 60-65
    #group 4: 65-75  this includes 2 age groups: 65-70, 70-75
    #group 5: 75+    this includes 1 age group:  75+
    mylist = np.zeros(16)

    mylist[0: 4] = vacVector[0]*fracPerAgeGroup[0]
    mylist[4: 10] = vacVector[1]*fracPerAgeGroup[1]
    mylist[10:13] = vacVector[2]*fracPerAgeGroup[2]
    mylist[13:15] = vacVector[3]*fracPerAgeGroup[3]
    mylist[15] = vacVector[4]*fracPerAgeGroup[4]

    mylist2 = np.minimum(np.floor(mylist), np.floor(totalPop))

    return mylist2


def uniformSampleSimplex(dim, numSamples):
    """
    This functions samples from the N-1 unit simplex (embedded in N-dimensional space)
    i.e. the polytope whose entries are nonnegative and sum to one.
    Currently no errors are thrown if the sample space or ambient space
    dimension are insufficient. But, sample size should be at least 1 and
    ambient space dimension should be at least 2.
    """
    ambient_space_dimension = dim
    sample_size = numSamples
    random_sample = np.zeros((sample_size, ambient_space_dimension))

    for ROW in range(0, sample_size):
        # print(ROW)
        U = np.random.uniform(low=0.0, high=1.0, size=ambient_space_dimension)
        # print(U)
        E = -np.log(U)
        # print(E)
        S = np.sum(E)
        # print(S)
        X = E / S
        # print(X)
        random_sample[ROW,] = X

    return random_sample