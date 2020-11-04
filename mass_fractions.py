import fluid_properties_air as fpa


def mass_fraction_interface(pressure, temp_interface):
    """noncondensable gas mass fraction at the gas-liquid interface with the assumption of saturation following the
       Gibbs-Dalton Law as formulated by
       G. Caruso, D. Di Vitale Maio, Heat and mass transfer analogy applied to condensation in the presence of
       noncondensable gases inside inclined tubes, Int. J. Heat Mass Transf. 68 (2014) 401–414,
       doi: 10.1016/j.ijheatmasstransfer.2013.09.049"""
    p_sat = fpa.temperature2saturation_vapour_pressure(temp_interface)
    M_v = fpa.MOLES_MASS_VAPOUR
    M_air = fpa.MOLES_MASS_AIR
    return (pressure - p_sat) / (pressure - (1 - M_v / M_air) * p_sat)


def mass_fraction_bulk(mass_flow_ncg, mass_flow_water):
    """noncondensable gas mass fraction in the bulk flow"""
    return mass_flow_ncg / (mass_flow_water + mass_flow_ncg)
