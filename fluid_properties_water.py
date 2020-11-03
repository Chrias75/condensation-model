# #####################################################################################################################
# PROJECT: fluid tool box
# MODULE: fluid properies water
# DESCRIPTION: fluid properties of water as a function of temperature
#
# VERSION: 0.2
# LAST UPDATE: 22/09/2015 by andreas.westhoff@dlr.de
# #####################################################################################################################

import numpy as np


def enthalpy_evaporation(temperature):
    """
    FUNCTION:
        enthalpy_evaporation

    DESCRIPTION:
        enthalpy of evaporation as a function of temperature

    INPUT:
        __temperature__             : temperature in Celsius


    RETURNS: enthalpy [J/Kg]
    """
    __temp = temperature + 273.15
    return 1918.46 * (__temp * (__temp - 33.91) ** (-1)) ** 2 * 1000


def dynamic_viscosity(temperature):
    """
    FUNCTION:
        dynamic_viscosity

    DESCRIPTION:
        dynamic viscosity as a function of the temperature

    INPUT:
        temperature                 : temperature [C]


    RETURNS:
        dynamic viscosity           : dynamic viscosity [N s / m^2]
    """
    __temp = temperature + 273.15
    # http://www.engineeringtoolbox.com/
    p1 = [3.24515737e-11, -4.45178793e-08, 2.29372345e-05, -5.26671091e-03, 4.55634344e-01]
    __dynamic_viscosity__ = 0
    for n in range(len(p1)):
        __dynamic_viscosity__ += p1[n] * __temp ** (len(p1)-n-1)
    return __dynamic_viscosity__


def heat_capacity(temperature):
    """
    FUNCTION:
        dynamic_viscosity

    DESCRIPTION:
        dynamic viscosity as a function of the temperature

    INPUT:
        temperature                 : temperature [C]


    RETURNS:
        dynamic viscosity           : dynamic viscosity [N s / m^2]
    """
    __temp = temperature + 273.15
    __p2__ = [3.34656282e-06, -4.46672225e-03, 2.23957021e+00, -4.99435662e+02, 4.59421964e+04]
    __heat_capacity__ = 0
    for n in range(len(__p2__)):
        __heat_capacity__ += __p2__[n] * __temp ** (len(__p2__)-n-1)
    return __heat_capacity__


def thermal_conductivity(temperature):
    """
    FUNCTION: thermal conductivity

    DESCRIPTION: theraml conductivity of water as a function of the temperature

    INPUT:
        temperature                 : temperature [C]

    RETURNS: thermal_conductivity [W/mK]
    """
    __temp = temperature + 273.15
    __c__ = [5.85664336e-10, -7.63692502e-07, 3.62572178e-04, -7.29371838e-02, 5.73565105e+00]
    return np.polyval(__c__, __temp)


def density(temperature):
    """
    FUNCTION: density

    DESCRIPTION: density of water as a function of the temperature

    INPUT:
        temperature                 : temperature [C]

    RETURNS: density [kg/m^3]
    """
    __temp = temperature + 273.15
    __c__ = [-1.34324009e-10, 1.89316803e-07, -1.02599899e-04, 2.46788875e-02, -1.19662999e+00]
    return np.polyval(__c__, __temp)


