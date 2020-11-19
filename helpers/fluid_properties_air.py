# #####################################################################################################################
# PROJECT: fluid tool box
# MODULE: fluid properies moist air
# DESCRIPTION: fluid properties of moist air as a function of air pressure, temperature and vapour pressure
#
# VERSION: 0.4
# LAST UPDATE: 23/09/2015 by andreas.westhoff@dlr.de
# #####################################################################################################################

from numpy import exp, log, log10
import helpers.fluid_properties_water as fpw

# #####################################################################################################################
# CONSTANTS
# #####################################################################################################################
MOLES_MASS_AIR = 28.96  # g/mol
MOLES_MASS_VAPOUR = 18.01528  # g/mol
GAS_CONSTANT = 8.3144598  # J/mol*K

# #####################################################################################################################
# MOIST AIR PROPERTIES / HUMIDITY
# #####################################################################################################################


def __interaction_parameters__(__dyn_viscosity_dry_air__, __dyn_viscosity_vapour__):
    """
    FUNCTION: __interaction_parameters__

    DESCRIPTION: interaction parameters based on thermophysical and transport properties of humid air
        at temperature range between 0 and 100 C, P.T. Tsilingiris, Energy Conversion & Management (2008)

    INPUT:
        __temperature__                 : temperature [K]

    RETURNS: interaction_parameters [-]
    """
    __mol_av__ = MOLES_MASS_AIR * MOLES_MASS_VAPOUR ** (-1)
    __mol_va__ = __mol_av__ ** (-1)
    __dv_av__ = __dyn_viscosity_dry_air__ * __dyn_viscosity_vapour__ ** (-1)
    __dv_va__ = __dv_av__ ** (-1)
    __ip_av__ = 2 ** 0.5 * 0.25 * (1 + __mol_av__) ** (-0.5) * (1 + __dv_av__ ** 0.5 * __mol_va__ ** 0.25) ** 2
    __ip_va__ = 2 ** 0.5 * 0.25 * (1 + __mol_va__) ** (-0.5) * (1 + __dv_va__ ** 0.5 * __mol_av__ ** 0.25) ** 2
    return __ip_av__, __ip_va__


def __moles_fraction_mixture__(vapour_pressure, pressure, temperature, matter='water'):
    """
    FUNCTION: molar_fraction

    DESCRIPTION: vapour molar and total moles fraction bvased on thermophysical and transport properties of humid air
        at temperature range between 0 and 100 C, P.T. Tsilingiris, Energy Conversion & Management (2008)

    INPUT:
        vapour_pressure             : vapour pressure [Pa]
        pressure                    : total pressure [Pa]
        temperature                 : temperature [K]
        matter                      : state of matter (water or ice)

    RETURNS:
        __moles_fraction_saturation__   : molar fraction of saturation vapour moles and total moles
        __moles_fraction_vapour__       : molar fraction of vapour moles and total moles
    """
    __mfc1__ = [3.53624e-4, 2.93228e-5, 2.61474e-7, 8.57538e-9]
    __mfc2__ = [-1.07588e1, 6.32529e-2, -2.53591e4, 6.33784e-7]
    __zeta1__ = 0
    __zeta2__ = 0
    __temp = temperature + 273.15
    __saturation_vapour_pressure__ = temperature2saturation_vapour_pressure(temperature, matter)
    __pressure_ratio__ = __saturation_vapour_pressure__ * pressure ** (-1)
    for i in range(4):
        __zeta1__ += __mfc1__[i] * __temp ** i
        __zeta2__ += __mfc2__[i] * __temp ** i
    __zeta2__ = exp(__zeta2__)
    __factor__ = exp(__zeta1__ * (1 - __pressure_ratio__) + __zeta2__ * (__pressure_ratio__ - 1))
    __moles_fraction_saturation__ = __factor__ * __pressure_ratio__
    __dewpoint = vapour_pressure2dewpoint(vapour_pressure)
    __relative_humidity__ = relativehumidity(temperature, __dewpoint)
    __moles_fraction_vapour__ = __moles_fraction_saturation__ * __relative_humidity__
    return __moles_fraction_saturation__, __moles_fraction_vapour__


def moist_air_dynamic_viscosity(temperature, pressure, vapour_pressure, matter='water'):
    """
    FUNCTION: moist_air_dynamic_viscosity

    DESCRIPTION: dynamic viscosity of moist air as a function of temperature, pressure and vapour pressure
        at temperature range between 0 and 100 C, P.T. Tsilingiris, Energy Conversion & Management (2008)

    INPUT:
        vapour_pressure             : vapour pressure [Pa]
        pressure            : total pressure [Pa]
        temperature                 : temperature [K]
        matter                      : state of matter (water or ice), default = water

    RETURNS:
        __molar_fraction_saturation__   : molar fraction of saturation vapour moler and total moles
        __moles
    """
    __dyn_visc_a__ = dry_air_dynamic_viscosity(temperature, dry_air_density(temperature, pressure))
    __dyn_visc_v__ = fpw.dynamic_viscosity(temperature)
    __inpa_av__, __inpa_va__ = __interaction_parameters__(__dyn_visc_a__, __dyn_visc_v__)
    __mofr_s__, __mofr_v__ = __moles_fraction_mixture__(vapour_pressure, pressure, temperature, matter)
    __moist_air_dyn_visc_1__ = (1 - __mofr_v__) * __dyn_visc_a__ * ((1 - __mofr_v__) + __mofr_v__ * __inpa_av__) ** (-1)
    __moist_air_dyn_visc_2__ = __mofr_v__ * __dyn_visc_v__ * (__mofr_v__ + (1 - __mofr_v__) * __inpa_va__) ** (-1)
    return __moist_air_dyn_visc_1__ + __moist_air_dyn_visc_2__


def moist_air_heat_capacity(temperature, pressure, vapour_pressure, matter='water'):
    """
    FUNCTION: moist_air_heat_capacity

    DESCRIPTION: heat capacity of moist air as a function of  temperature, pressure and vapour pressure

    INPUT:
        density_vapour                  : density of vapour in kg(air+vapour) / m^3
        density_dry_air                 : density of moist air in kg(air) / m^3

    RETURNS: heat capacity of moist air in J/kg*K
    """
    __mf_sat__, __mf_vap__ = __moles_fraction_mixture__(vapour_pressure, pressure, temperature, matter)
    __mf_air__ = 1 - __mf_vap__
    __m_mol_mass__ = MOLES_MASS_AIR * __mf_air__ + MOLES_MASS_VAPOUR * __mf_vap__
    __cpa__ = dry_air_heat_capacity(temperature)
    __cpv__ = fpw.heat_capacity(temperature)
    return (__cpa__ * __mf_air__ * MOLES_MASS_AIR + __cpv__ * __mf_vap__ * MOLES_MASS_VAPOUR) * __m_mol_mass__ ** (-1)


def moist_air_thermal_conductivity(temperature, pressure, vapour_pressure, matter='water'):
    """
    FUNCTION: moist_air_thermal_conductivity

    DESCRIPTION: thermal_conductivity of moist air as a function of temperature, pressure and vapour pressure
        at temperature range between 0 and 100 C, P.T. Tsilingiris, Energy Conversion & Management (2008)

    INPUT:
        vapour_pressure             : vapour pressure [Pa]
        pressure                    : total pressure [Pa]
        temperature                 : temperature [K]
        matter                      : state of matter (water or ice), default = water

    RETURNS: thermal conductivity [W/mK]
    """
    __dyn_visc_a__ = dry_air_dynamic_viscosity(temperature, dry_air_density(temperature, pressure))
    __dyn_visc_v__ = fpw.dynamic_viscosity(temperature)
    __ip_av__, __ip_va__ = __interaction_parameters__(__dyn_visc_a__, __dyn_visc_v__)
    __k_a__ = dry_air_thermal_conductivity(temperature)
    __k_v__ = fpw.thermal_conductivity(temperature)
    __mf_s__, __mf_v__ = __moles_fraction_mixture__(vapour_pressure, pressure, temperature, matter)
    __k1__ = (1 - __mf_v__) * __k_a__ * ((1 - __mf_v__) + __mf_v__ * __ip_av__) ** (-1)
    __k2__ = __mf_v__ * __k_v__ * ((1 - __mf_v__) * __ip_va__ + __mf_v__) ** (-1)
    return __k1__ + __k2__


def moist_air_thermal_diffusivity(temperature, pressure, vapour_pressure, matter='water'):
    """
    FUNCTION: thermal_diffusivity

    DESCRIPTION: thermal_diffusivityof moist air as a function of temperature, pressure and vapour pressure
        at temperature range between 0 and 100 C, P.T. Tsilingiris, Energy Conversion & Management (2008)

    INPUT:
        vapour_pressure             : vapour pressure [Pa]
        pressure                    : total pressure [Pa]
        temperature                 : temperature [K]
        matter                      : state of matter (water or ice), default = water

    RETURNS: thermal diffusivity [m^2/s]
    """
    __k = moist_air_thermal_conductivity(temperature, pressure, vapour_pressure, matter)
    __cp = moist_air_heat_capacity(temperature, pressure, vapour_pressure, matter)
    __density = moist_air_density(pressure, vapour_pressure, temperature)
    return __k * (__cp * __density) ** (-1)


def temperature2saturation_vapour_pressure(temperature, matter='water'):
    """
    FUNCTION:
        temperature2saturation_vapour_pressure

    DESCRIPTION:
        saturation vapour pressure over water based on SO90w  equation (Sonntag and Heinze 1990)
        saturation vapour pressure over ice based on equation of Murphy and Koop (2005)

    INPUT:
        __temperature__             : temperature in Celsius
        __state_of_matter__         : water or ice

    RETURNS:                        saturation vapour pressure in Pa
    """
    __temp = temperature + 273.15
    if matter == 'water':
        __svc1__ = [-6096.9385,  16.635794,  -2.711193E-2,  1.673952E-5,  2.433502]
        return exp(__svc1__[0] * __temp ** (-1) + __svc1__[1] + __svc1__[2] * __temp +
                   __svc1__[3] * __temp ** 2 + __svc1__[4] * log(__temp))*100
    elif matter == 'ice':
        __svc2__ = [9.550426,  -5723.265,  3.53068,  -0.00728332]
        return exp(__svc2__[0] + __svc2__[1] * __temp ** (-1) +
                   __svc2__[2] * log(__temp) + __svc2__[3] * __temp) * 100
    else:
        return 0


def dewpoint2vapour_pressure(temperature, dewpoint, matter='water'):
    """
    FUNCTION:             dewpoint2vapour_pressure

    DESCRIPTION:        vapour pressure as a function of dewpoint and temperature

    INPUT:
        temperature                 : temperature [C]
        dew_point                   : dew point [C]
        state_of_matter             : 'water' or 'ice'

    RETURNS: vapour_pressure [Pa]
    """
    __tem_C = temperature
    __dp_C = dewpoint
    __svp = temperature2saturation_vapour_pressure(temperature, matter)
    if matter == 'water':
        __c1 = [6.1221, 18.67,  257.14,  234.5]
        __p1 = (__c1[1] - __tem_C / __c1[3]) * (__tem_C / (__c1[2] + __tem_C))
        # print __p1
        return exp((__dp_C * (__c1[1] - __p1) - __c1[2] * __p1) / (__dp_C + __c1[2])) * __svp
    elif matter == 'ice':
        __c2 = [6.1121, 18.678,  257.14,  234.5]
        __p2 = (__c2[1] - __tem_C / __c2[3]) * (__tem_C / (__c2[2] + __tem_C))
        return exp((__dp_C * (__c2[1] - __p2) - __c2[2] * __p2) / (__dp_C + __c2[2])) * __svp
    else:
        return 0


def vapour_pressure2vapour_density(temperature, vapour_pressure):
    """
    FUNCTION:             vapour_pressure2vapour_density

    DESCRIPTION:        converts vapour pressure to vapour density

    INPUT:
        __temperature__                 : temperature in Celsius
        __vapour_pressure__             : vapour pressure in Pa

    RETURNS:
       vapour_density                : vapour density kg(vapour) / m^3
    """
    __temp = temperature + 273.15
#    print __temp
    return vapour_pressure * (461.5 * __temp) ** (-1)


def moist_air_density(pressure, vapour_pressure, temperature):
    """
    FUNCTION:           density_moist_air

    DESCRIPTION:        calculation of the density of moist air and the fraction of dry air

    INPUT:
        pressure                    : ambient air pressure in Pa
        vapour_pressure             : vapour pressure in Pa
        temperature                 : temperature in Celsius

    RETURNS: density of moist air in kg(air+vapour) / m^3
    """
    __temp = temperature + 273.15
    __vapour_density__ = vapour_pressure2vapour_density(temperature, vapour_pressure)
    __density_dry_air__ = (pressure - vapour_pressure) * (287.1 * __temp) ** (-1)
    return __density_dry_air__ + __vapour_density__


def density2absolute_humidity(density_vapour, density_dry_air):
    """
    FUNCTION:   density2absolute_humidity

    DESCRIPTION: calculation of absolute humidity defined as the ratio vapour mass and air mass per volume

    INPUT:
        density_vapour                  : density of vapour in kg(vapour) / m^3
        density_dry_air                 : density of dry air in kg(air) / m^3

    RETURNS: absolute humidity ratio of mass(vapour) to mass(dry air)
    """
    return density_vapour * density_dry_air ** (-1)


def vapour_density2vapour_pressure(temperature, vapour_density):
    """
    FUNCTION: vapour_density2vapour_pressure

    DESCRIPTION: vapour pressure as a function of vapour density

    INPUT:
        temperature                 : temperature [C]
        vapour_density              : density of vapour [kg/m^3]

    RETURNS: vapour pressure [Pa]
    """
    __temp = temperature + 273.15
    return vapour_density * 461.5 * __temp


def relativehumidity(temperature, dewpoint, matter='water'):
    """
    FUNCTION: relativehumidity

    DESCRIPTION: vapour pressure as a function of dewpoint and temperature

    INPUT:
        vapour_pressure                 : vapour pressure [Pa]
        saturation_vapour_pressure      : saturation vapour pressure [Pa]

    RETURNS: relative humidity [-]
    """
    __temp = temperature + 273.15
    if matter == 'water':
        __svc1__ = [-6096.9385,  16.635794,  -2.711193E-2,  1.673952E-5,  2.433502]
        p_vap_s = exp(__svc1__[0] * __temp ** (-1) + __svc1__[1] + __svc1__[2] * __temp +
                   __svc1__[3] * __temp ** 2 + __svc1__[4] * log(__temp))*100
    elif matter == 'ice':
        __svc2__ = [9.550426,  -5723.265,  3.53068,  -0.00728332]
        p_vap_s = exp(__svc2__[0] + __svc2__[1] * __temp ** (-1) +
                   __svc2__[2] * log(__temp) + __svc2__[3] * __temp) * 100
    else:
        p_vap_s = 0
    __tem_C = temperature
    __dp_C = dewpoint
    if matter == 'water':
        __c1 = [6.1221, 18.67,  257.14,  234.5]
        __p1 = (__c1[1] - __tem_C / __c1[3]) * (__tem_C / (__c1[2] + __tem_C))
        p_vap = exp((__dp_C * (__c1[1] - __p1) - __c1[2] * __p1) / (__dp_C + __c1[2])) * p_vap_s
    elif matter == 'ice':
        __c2 = [6.1121, 18.678,  257.14,  234.5]
        __p2 = (__c2[1] - __tem_C / __c2[3]) * (__tem_C / (__c2[2] + __tem_C))
        p_vap = exp((__dp_C * (__c2[1] - __p2) - __c2[2] * __p2) / (__dp_C + __c2[2])) * p_vap_s
    else:
        p_vap = 0
    return p_vap * p_vap_s ** (-1)


def diffusion_coefficient(temperature):
    """
    FUNCTION: diffusion

    DESCRIPTION: diffusion of water air mixture as a function of the temperature

    INPUT:
        temperature                 : temperature [C]

    RETURNS: density [kg/m^3]
    """
    __temp = temperature + 273.15
    __c__ = [-2.775e-6, 4.479e-8, 1.656e-10]
    return __c__[0] + __c__[1] * __temp + __c__[2] * __temp ** 2


def vapour_pressure2dewpoint(vapour_pressure):
    a = 7.5
    b = 237.3
    v = log10(vapour_pressure * 100 ** (-1) * 6.1078 ** (-1))
    dewpoint = (b * v) * (a - v) ** (-1)
    # print dewpoint
    return dewpoint

# #####################################################################################################################
# DRY AIR PROPERTIES
# #####################################################################################################################


def real_gas_factor(temperature, pressure):
    """
    FUNCTION: real_gas_factor

    DESCRIPTION: calculates the real gas factor as a function of pressure and temperature

    INPUT:
        temperature                            : temperature [C]
        pressure                               : pressure [bar]

    RETURNS:
        compressibility                         : real gas factor [-]
    """
    __temp = temperature + 273.15
    __c1__ = [-9.5378E-3,  5.1986E-5, -7.0621E-8,  0]
    __c2__ = [3.1753E-5,  -1.7155E-7, 2.4630E-10,  0]
    __c3__ = [6.3764E-7,  -6.4678E-9, 2.1880E-11,  -2.4691E-14]
    __pg__ = pressure * 1E-5 - 1.01325
    __pa__ = 1.01325 + __pg__
    __z1__ = 0
    __z2__ = 0
    __z3__ = 0
    for i in range(4):
        __z3__ += __c3__[i] * __temp ** i
        if i <= 2:
            __z1__ += __c1__[i] * __temp ** i
            __z2__ += __c2__[i] * __temp ** i
    return 1 + __z1__ * (__pa__-1) ** 1 + __z2__ * (__pa__-1) ** 2 + __z3__ * (__pa__ - 1) ** 3


def dry_air_thermal_conductivity(temperature):
    """
    FUNCTION: dry_air_thermal_conductivity

    DESCRIPTION: thermal conductivity of dry air as a function of temperature at ambient pressure

    INPUT:
        temperature             : temperature [C]

    RETURNS: thermal conductivity [W/m K]
    """
    __temp = temperature + 273.15
    __c__ = [-2.93500854e-12, 4.11177923e-09, -2.12665783e-06, 5.52396055e-04, -3.53798775e-02]
    __thermal_conductivity__ = 0
    for i in range(5):
        __thermal_conductivity__ += __c__[i] * __temp ** (4-i)
    return __thermal_conductivity__


def dry_air_density(temperature, pressure):
    """
    FUNCTION: dry_air_density

    DESCRIPTION: density of dry air as a function of pressure and temperature

    INPUT:
        __temperature__             : temperature [C]
        __pressure__                : pressure [bar]

    RETURNS: density of dry air [kg/m^3]
    """
    __temp = temperature + 273.15
    __rgf__ = real_gas_factor(__temp, pressure)
    return pressure * (__rgf__ * 287.1 * __temp) ** (-1)


def dry_air_dynamic_viscosity(temperature, density):
    """
    FUNCTION: dry_air_dynamic_viscosity

    DESCRIPTION: dynamic viscosity of dry air as a function of temperature

    INPUT:
        __temperature__             : temperature [C]
        __density__                 : density [kg/m^3]

    RETURNS: dynamic viscosity kg / m s
    """
    __temp = temperature + 273.15
    __c__ = [0.000, 1.021E-8, 5.969E-11]
    __mu1__ = 1.458E-6 * __temp ** 1.5 * (110.4 + __temp) ** (-1)
    __mu2__ = 0
    for i in range(3):
        __mu2__ += __c__[i] * density ** i
    return __mu1__ + __mu2__


def dry_air_heat_capacity(temperature):
    """
    FUNCTION: dry_air_heat_capacity

    DESCRIPTION: dry air heat capacity as a function of temperature

    INPUT:
        temperature                 : temperature [C]

    RETURNS: heat_capacity [W/Kg K]
    """
    __temp = temperature + 273.15
    __c__ = [2.99897346e-04, -1.43427702e-01, 1.02180709e+03]
    __heat_capacity__ = 0
    for i in range(3):
        __heat_capacity__ += __c__[i] * __temp ** (2-i)
    return __heat_capacity__


def dry_air_thermal_diffusivity(temperature, pressure):
    """
    FUNCTION: dry_air_heat_capacity

    DESCRIPTION: dry air heat capacity as a function of temperature

    INPUT:
        temperature                 : temperature [C]
        pressure                    : Pascal [Pa]

    RETURNS: heat_capacity [W/Kg K]
    """
    __k__ = dry_air_thermal_conductivity(temperature)
    __cp__ = dry_air_heat_capacity(temperature)
    __density__ = dry_air_density(temperature, pressure)
    return __k__ * (__cp__ * __density__) ** (-1)
