import fluid_properties_air as fpa
import fluid_properties_water as fpw
import numpy as np
from scipy.integrate import quad


def Nu_lam(reynolds, prandtl, dia_hydr, length):
    """ mean laminar Nusselt number for constant temperature difference taken from Heat Transfer in Pipe Flow Chapter
        of the VDI Heat Atlas by Gnielinski, V. (2010) p. 693 ff doi: 10.1007/978-3-540-77877-6
    """
    Nu_mt1 = 3.66
    Nu_mt2 = 1.615 * (reynolds * prandtl * dia_hydr / length) ** (1 / 3)
    Nu_mt3 = (2 / (1 + 22 * prandtl)) ** (1 / 6) * (reynolds * prandtl * dia_hydr / length) ** (1 / 2)
    return (Nu_mt1 ** 3 + 0.7 ** 3 + (Nu_mt2 - 0.7) ** 3 + Nu_mt3 ** 3) ** (1 / 3)


def Nu_turb(reynolds, prandtl, dia_hydr, length):
    """ mean turbulent Nusselt number taken from Heat Transfer in Pipe Flow Chapter of the VDI Heat Atlas by
        Gnielinski, V. (2010) p. 693 ff doi: 10.1007/978-3-540-77877-6
    """
    xi = (1.8 * np.log10(reynolds) - 1.5) ** (-2)
    return (xi / 8 * reynolds * prandtl) / (1 + 12.7 * np.sqrt(xi / 8) * (prandtl ** (2 / 3) - 1)) * \
           (1 + (dia_hydr / length) ** (2 / 3))


def Nu_sen(reynolds, prandtl, dia_hydr, length):
    """ sensible Nusselt number for entire range of Reynolds numbers taken from Heat Transfer in Pipe Flow Chapter
        of the VDI Heat Atlas by Gnielinski, V. (2010) p. 693 ff doi: 10.1007/978-3-540-77877-6
        INPUT:
            Reynolds number; Prandtl number; hydraulic diameter [m]; length of the heat transfer surface [m]
        RETURN:
            sensible Nusselt number for pipe flow equivalent
    """
    if reynolds < 2300:
        return Nu_lam(reynolds, prandtl, dia_hydr, length)
    elif reynolds > 10000:
        return Nu_turb(reynolds, prandtl, dia_hydr, length)
    else:
        gamma = (reynolds - 2300) / (10000 - 2300)
        return (1 - gamma) * Nu_lam(2300, prandtl, dia_hydr, length) + gamma * Nu_turb(10000, prandtl, dia_hydr, length)


def Nu_lat(sherwood, prandtl, schmidt, jakob, b_i):
    """ latent Nusselt number according to
        G. Caruso, D. Di Vitale Maio, Heat and mass transfer analogy applied to condensation in the presence of
        noncondensable gases inside inclined tubes, Int. J. Heat Mass Transf. 68 (2014) 401–414,
        doi: 10.1016/j.ijheatmasstransfer.2013.09.049
    """
    return sherwood * prandtl * b_i * (schmidt * jakob) ** (-1)


def mass_fraction_interface(pressure, temp_interface):
    """ noncondensable gas mass fraction at the gas-liquid interface with the assumption of saturation following the
        Gibbs-Dalton Law as formulated by
        G. Caruso, D. Di Vitale Maio, Heat and mass transfer analogy applied to condensation in the presence of
        noncondensable gases inside inclined tubes, Int. J. Heat Mass Transf. 68 (2014) 401–414,
        doi: 10.1016/j.ijheatmasstransfer.2013.09.049
    """
    p_sat = fpa.temperature2saturation_vapour_pressure(temp_interface)
    M_v = fpa.MOLES_MASS_VAPOUR
    M_air = fpa.MOLES_MASS_AIR
    return (pressure - p_sat) / (pressure - (1 - M_v / M_air) * p_sat)


def mass_fraction_bulk(mass_flow_ncg, mass_flow_water):
    """ noncondensable gas mass fraction in the bulk flow """
    return mass_flow_ncg / (mass_flow_water + mass_flow_ncg)


def driving_force_mt(t_int, pressure, mf_bulk):
    """ driving force for the mass transfer to the condensate interface
        F. Eimann, S. Zheng, C. Philipp, T. Fieback, U. Gross, Convective dropwise condensation out of humid air inside
        a horizontal channel – experimental investigation of the condensate heat transfer resistance, Int. J. Heat Mass
        Transf. 127 (2018) 448–464, doi: 10.1016/j.ijheatmasstransfer.2018.08.015
    """
    return np.log(mass_fraction_interface(pressure, t_int) / mf_bulk)


def corr_suction_ht(sher, nuss, pran, schm, rh, t_int, t, p):
    """ suction correction for the heat transfer
        H.J.H. Brouwers, Effect of fog formation on turbulent vapor condensation with noncondensable gases,
        J. Heat Transf. 118 (1) (1996) 243–245,
        doi: 10.1115/1.2824052
    """
    p_v = rh * fpa.temperature2saturation_vapour_pressure(t)
    c_p = fpa.moist_air_heat_capacity(t, p, p_v)
    c_pv = fpw.heat_capacity(t)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p, t_int)
    r_t = c_pv / c_p * (sher * pran) / (schm * nuss) * np.log((1 - x_vb) / (1 - x_vi))
    return -1 * r_t / (np.exp(-r_t) - 1)


def corr_fog_ht(sher, nuss, pran, schm, rh, t_int, t, p):
    """ fog correction for the heat transfer
        H.J.H. Brouwers, Effect of fog formation on turbulent vapor condensation with noncondensable gases,
        J. Heat Transf. 118 (1) (1996) 243–245,
        doi: 10.1115/1.2824052
    """
    p_v = rh * fpa.temperature2saturation_vapour_pressure(t)
    c_p = fpa.moist_air_heat_capacity(t, p, p_v)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p, t_int)
    lmbda = fpw.enthalpy_evaporation(t)
    # c_p_alt = c_p_mixture(x_vb, t)
    # lmbda_mol = lmbda * fpa.MOLES_MASS_VAPOUR
    # print('1_mol: ', lmbda_mol / c_p_alt)
    # print('1: ', lmbda / c_p)
    # print('2: ', pran / schm)
    # print('3: ', (x_vb - x_vi) / (t - t_int))
    # print('4: ', sher / nuss)
    # print('5: ', saturation_line_slope(t_int))
    # print('6: ', (lmbda / c_p * pran / schm * 2 * (x_vb - x_vi) / (t - t_int) * sher / nuss))
    # print('7: ', (lmbda / c_p * pran / schm * saturation_line_slope(t_int)))
    # print('8: ', ((lmbda / c_p * pran / schm * 2 * (x_vb - x_vi) / (t - t_int) * sher / nuss) ** -1))
    # print('9: ', ((lmbda / c_p * pran / schm * saturation_line_slope(t_int)) ** -1))
    return (1 + lmbda / c_p * pran / schm * (x_vb - x_vi) / (t - t_int) * sher / nuss) / \
           (1 + lmbda / c_p * pran / schm * saturation_line_slope(t_int, p))


def corr_suction_mt(rh, t_int, t, p):
    """ suction correction for the mass transfer
        H.J.H. Brouwers, Effect of fog formation on turbulent vapor condensation with noncondensable gases,
        J. Heat Transf. 118 (1) (1996) 243–245,
        doi: 10.1115/1.2824052
    """
    p_v = rh * fpa.temperature2saturation_vapour_pressure(t)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p, t_int)
    r_w = (x_vb - x_vi) / (1 - x_vi)
    return np.log(1 - r_w) / (-r_w)


def corr_fog_mt(sher, nuss, pran, schm, rh, t_int, t, p):
    """ fog correction for the mass transfer
        H.J.H. Brouwers, Effect of fog formation on turbulent vapor condensation with noncondensable gases,
        J. Heat Transf. 118 (1) (1996) 243–245,
        doi: 10.1115/1.2824052
    """
    p_v = rh * fpa.temperature2saturation_vapour_pressure(t)
    c_p = fpa.moist_air_heat_capacity(t, p, p_v)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p, t_int)
    lmbda = fpw.enthalpy_evaporation(t)
    # c_p_alt = c_p_mixture(x_vb, t)
    # lmbda_mol = lmbda * fpa.MOLES_MASS_VAPOUR
    return (1. + (lmbda / c_p * pran / schm * (x_vb - x_vi) / (t - t_int) * sher / nuss) ** -1) / \
           (1. + (lmbda / c_p * pran / schm * saturation_line_slope(t_int, p)) ** -1)


def cos_theta(phi, theta_max, theta_min):
    """ function to describe the contact angle in dependence of the azimutal angle
        A .I. El Sherbini, A .M. Jacobi, Liquid drops on vertical and inclined surfaces: i. An experimental study of drop
        geometry, J. Colloid Interface Sci. 273 (2) (2004) 556–565, doi: 10.1016/j.jcis.2003.12.067
    """
    theta_max_pi = np.deg2rad(theta_max)
    theta_min_pi = np.deg2rad(theta_min)
    return ((2. * (np.cos(theta_max_pi) - np.cos(theta_min_pi)) * phi ** 3 / np.pi ** 3) -
            (3. * (np.cos(theta_max_pi) - np.cos(theta_min_pi)) * phi ** 2 / np.pi ** 2) +
            np.cos(theta_max_pi))


def minimum_contact_angle(Bo, theta_adv):
    """ Calculate minimum contact angle as function of the Bond number
        A .I. El Sherbini, A .M. Jacobi, Liquid drops on vertical and inclined surfaces: i. An experimental study of drop
        geometry, J. Colloid Interface Sci. 273 (2) (2004) 556–565, doi: 10.1016/j.jcis.2003.12.067
        theta_adv (deg/rad)
        returns (deg/rad)
    """
    return theta_adv * (0.01 * Bo ** 2 - 0.155 * Bo + 0.97)


def cos_theta_integrate(phi, theta_max, theta_min, d, aspect_ratio):
    return zeta(phi, d, aspect_ratio) * cos_theta(phi, theta_max, theta_min) * np.cos(phi)


def zeta(phi, r_cl, aspect_ratio):
    """ local droplet radius as a function of the azimutal angle
        A.D. Sommers, J. Ying, K.F. Eid, Predicting the onset of condensate droplet departure from a vertical surface
        due to air flow-Applications to topographically-modified, micro-grooved surfaces, Exp. Therm. Fluid Sci. 40
        (2012) 38–49, doi: 10.1016/j.expthermflusci.2012.01.031
    """
    return ((abs(np.cos(phi)) / r_cl) ** 3 + (abs(aspect_ratio * np.sin(phi)) / r_cl) ** 3) ** (-1. / 3.)


def f_grav(r_d, density_cond):
    """ gravitational force of a spherical drop on a vertical wall
        A.D. Sommers, J. Ying, K.F. Eid, Predicting the onset of condensate droplet departure from a vertical surface
        due to air flow-Applications to topographically-modified, micro-grooved surfaces, Exp. Therm. Fluid Sci. 40
        (2012) 38–49, doi: 10.1016/j.expthermflusci.2012.01.031
    """
    return density_cond * 9.81 * 2. / 3. * np.pi * r_d ** 3


def f_grav_vert(r_d, density_cond, theta_max, theta_min):
    """ gravitational force of a spherical cap on a vertical wall, using the mean contact angle """
    return density_cond * 9.81 * np.pi / 3. * r_d ** 3 * (2. + np.cos(np.deg2rad((theta_max + theta_min) / 2.))) * \
        (1. - np.cos(np.deg2rad((theta_max + theta_min) / 2.))) ** 2


def f_surf_tens(r_d, gamma, theta_max, theta_min, aspect_ratio):
    """ surface tension force as a function of the droplet radius
        A.D. Sommers, J. Ying, K.F. Eid, Predicting the onset of condensate droplet departure from a vertical surface
        due to air flow-Applications to topographically-modified, micro-grooved surfaces, Exp. Therm. Fluid Sci. 40
        (2012) 38–49, doi: 10.1016/j.expthermflusci.2012.01.031
    """
    r_cl = r_d * np.sin(np.deg2rad((theta_max + theta_min) / 2))
    return gamma * quad(cos_theta_integrate, 0., 2. * np.pi, args=(theta_max, theta_min, r_cl, aspect_ratio,))[0]


def Bo(rho, d, surf_tens, g=9.81):
    """ Calculates Bond number using the droplet diameter
        A .I. El Sherbini, A .M. Jacobi, Liquid drops on vertical and inclined surfaces: i. An experimental study of drop
        geometry, J. Colloid Interface Sci. 273 (2) (2004) 556–565, doi: 10.1016/j.jcis.2003.12.067
    """
    return (rho * g * d ** 2) / surf_tens


def aspect_ratio(Bo):
    """ aspect ratio of the droplets major to minor half widths as a function of Bond number
        A .I. El Sherbini, A .M. Jacobi, Liquid drops on vertical and inclined surfaces: i. An experimental study of drop
        geometry, J. Colloid Interface Sci. 273 (2) (2004) 556–565, doi: 10.1016/j.jcis.2003.12.067
    """
    return 1. + 0.096 * Bo


def f_drag(r_d, density_air, v, coef_drag, theta_max, theta_min):
    """ drag force with a droplets side profile as projection area (horizontal flow)
        A.D. Sommers, J. Ying, K.F. Eid, Predicting the onset of condensate droplet departure from a vertical surface
        due to air flow-Applications to topographically-modified, micro-grooved surfaces, Exp. Therm. Fluid Sci. 40
        (2012) 38–49, doi: 10.1016/j.expthermflusci.2012.01.031
    """
    theta_max_pi = np.deg2rad(theta_max)
    theta_min_pi = np.deg2rad(theta_min)
    beta = np.pi - theta_max_pi
    l_f = np.sin(theta_max_pi) * (1. - np.cos(theta_min_pi)) / (np.sin(theta_min_pi) * (1. - np.cos(theta_max_pi)))
    l_1 = 2. * r_d * np.sin((theta_max_pi + theta_min_pi) / 2.) * l_f / (1. + l_f)
    l_2 = 2. * r_d * np.sin((theta_max_pi + theta_min_pi) / 2.) / (1. + l_f)
    if theta_max > 90. > theta_min:
        return 0.5 * density_air * v ** 2 * coef_drag * \
               (np.pi * l_1 ** 2 / np.sin(beta) ** 2 * theta_min_pi / (2. * np.pi) +
                1. / 2. * l_1 ** 2 / np.tan(beta) +
                np.pi * l_2 ** 2 / np.sin(theta_min_pi) ** 2 * theta_min_pi / (2. * np.pi) -
                1. / 2. * l_2 ** 2 / np.tan(theta_min_pi))
    else:
        return 0.5 * density_air * v ** 2 * coef_drag * \
               (np.pi * l_1 ** 2 / np.sin(theta_max_pi) ** 2 * theta_min_pi / (2. * np.pi) +
                1. / 2. * l_1 ** 2 / np.tan(theta_max_pi) +
                np.pi * l_2 ** 2 / np.sin(theta_min_pi) ** 2 * theta_min_pi / (2. * np.pi) -
                1. / 2. * l_2 ** 2 / np.tan(theta_min_pi))


def f_drag_vert(r_d, density_air, v, coef_drag, theta_max, theta_min):
    """ drag force with a droplets top down profile as projection area (vertical flow) """
    a_proj = r_d ** 2 / 2. * (np.pi / 180 * (theta_max + theta_min) - np.sin(np.deg2rad((theta_max + theta_min))))
    return 0.5 * density_air * v ** 2 * coef_drag * a_proj


def correction_factor(r_m):
    """ correlations of parameters are added from Eimann, F., Zheng, S., Philipp, C., Omranpoor, A. H., & Gross, U.
        (2020). Dropwise condensation of humid air - Experimental investigation and modelling of the convective heat
        transfer. Int. Journal of Heat and Mass Transfer, 154
        https://doi.org/10.1016/j.ijheatmasstransfer.2020.119734
        C   :    correction factor for droplet/film. correlation of predicted and measured droplet maximum radius for
                 hemispherical droplets (0.2 mm < r_max < 1.61 mm)
        r_m :    droplet radius in m

    """
    r_d = r_m * 1000
    if r_d <= 1.3:
        __c = 0.28 * r_d + 1.155
    else:
        __c = 1.451 * r_d - 0.274
    return __c


def saturation_line_slope(t, p):
    """ slope of the saturation line for a given temperature und pressure """
    return (fpa.temperature2saturation_vapour_pressure(t + 1e-4) -
            fpa.temperature2saturation_vapour_pressure(t - 1e-4)) / (p * 2e-4)


def saturation_line_brouwers(t, p):
    """ slope of the saturation line as given in
        H.J.H. Brouwers, Effect of fog formation on turbulent vapor condensation with noncondensable gases,
        J. Heat Transf. 118 (1) (1996) 243–245, doi: 10.1115/1.2824052
        P_v(T) given by Reid et al, The Properties of Gases and Liquids, pp 629, 632,  1977
    """
    return (np.exp(11.6834 - 3816.44 / (227.02 + (t + 1e-4))) - np.exp(11.6834 - 3816.44 / (227.02 + (t - 1e-4)))) / \
           ((p * 1e-5) * 2e-4)


def c_p_mixture(x, t):
    """ gives the molar specific heat capacity of humid air with a water mass fraction x
        H.J.H. Brouwers, Effect of fog formation on turbulent vapor condensation with noncondensable gases,
        J. Heat Transf. 118 (1) (1996) 243–245, doi: 10.1115/1.2824052
    """
    c_pg = fpa.dry_air_heat_capacity(t) * fpa.MOLES_MASS_AIR
    c_pv = fpw.heat_capacity(t) * fpa.MOLES_MASS_VAPOUR
    return (1 - x) * c_pg + x * c_pv


def c_drag(r_d, theta_max, theta_min, rey, d_hyd):
    """ drag coefficient for a wide range of droplet reynolds numbers as assembled by
        A.D. Sommers, J. Ying, K.F. Eid, Predicting the onset of condensate droplet departure from a vertical surface
        due to air flow-Applications to topographically-modified, micro-grooved surfaces, Exp. Therm. Fluid Sci. 40
        (2012) 38–49, doi: 10.1016/j.expthermflusci.2012.01.031
        reynolds < 20 and 20 < reynolds < 400:
        R.A.M. Al-Hayes, R.H.S. Winterton, Bubble diameter on detachment in flowing liquids, Int. J. Heat Mass
        Transf. 24 (2) (1981) 223–230
        reynolds > 200:
        K.V. Beard, H.R. Pruppacher, Determination of the terminal velocity and drag of small water drops by means of
        a wind tunnel, J. Atmos. Sci. 26 (1969) doi: 10.1175/1520-0469(1969)026 < 1066:adottv > 2.0.co;2
        Turnover from fixed value 1.22 to the Beard & Pruppacher correlation at ~80 despite given validity range chosen
        for continuity
    """
    h_d = r_d * (1 - np.cos(np.deg2rad((theta_max + theta_min) / 2)))
    h_d = r_d
    re_drop = rey * h_d / d_hyd
    # print("re_drop", re_drop)
    if re_drop < 20.:
        return 24 / re_drop
    elif 20. < re_drop < 80.:
        return 1.22
    else:
        return 0.28 + (6 / np.sqrt(re_drop)) + (21 / re_drop)


def log_mean(x, y):
    """ logarithmic mean value of two given value x and y """
    x = np.array(x)
    y = np.array(y)
    __logmean = (np.maximum(x, y) - np.minimum(x, y)) / (np.log((np.maximum(x, y)) / (np.minimum(x, y))))
    return __logmean
