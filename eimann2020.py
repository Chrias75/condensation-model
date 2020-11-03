import numpy as np
import configparser
from modules import fluid_properties_air as fpa
from modules import fluid_properties_water as fpw
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
from scipy.integrate import quad

re, pr, sc, t_in, t_out, t_w, t_mean, t_dp_in, t_dp_out, rH, m_air, m_water, m_cond = [], [], [], [], [], [], [], [], [], [], [], [], []
b, h, l, p_standard = 0., 0., 0., 0.
####################################################################################
# Input
####################################################################################
# switch = 'config'
switch = 'dat'
data = '/home/brue_ch/Auswertungen/rH_variable/Profil/data_rH_2000_27_5.dat'
result_filename = '/home/brue_ch/Auswertungen/rH_variable/Profil/eimann_r-mod_2000_27_5.dat'
if switch == 'config':
    config = configparser.ConfigParser()
    config.read('model_config.cfg')
    cfg_numbers = config['dimless_numbers']
    re = np.array([cfg_numbers.getfloat('reynolds')])
    pr = np.array([cfg_numbers.getfloat('prandtl')])
    sc = np.array([cfg_numbers.getfloat('schmidt')])

    cfg_temp = config['temperatures']
    t_in = np.array([cfg_temp.getfloat('t_in')])
    t_out = np.array([cfg_temp.getfloat('t_out')])
    t_w = np.array([cfg_temp.getfloat('t_w')])
    t_mean = np.array([cfg_temp.getfloat('t_mean')])
    t_dp_in = np.array([cfg_temp.getfloat('t_dp_in')])
    t_dp_out = np.array([cfg_temp.getfloat('t_dp_out')])
    rH = np.array([cfg_temp.getfloat('rH')])

    cfg_dim = config['dimensions']
    b = cfg_dim.getfloat('width')
    h = cfg_dim.getfloat('height')
    l = cfg_dim.getfloat('length')

    cfg_oth = config['other']
    m_air = np.array([cfg_oth.getfloat('massflow_moistair')])
    m_water = np.array([cfg_oth.getfloat('massflow_water')])
    m_cond = np.array([cfg_oth.getfloat('massflow_condensate')])
    p_standard = cfg_oth.getfloat('pressure')
elif switch == 'dat':
    re = np.loadtxt(data, skiprows=1, usecols=2)
    pr = np.loadtxt(data, skiprows=1, usecols=4)

    t_in = np.loadtxt(data, skiprows=1, usecols=28)
    t_out = np.loadtxt(data, skiprows=1, usecols=30)
    t_w = np.loadtxt(data, skiprows=1, usecols=26)
    t_mean = np.loadtxt(data, skiprows=1, usecols=32)
    t_dp_in = np.loadtxt(data, skiprows=1, usecols=34)
    t_dp_out = np.loadtxt(data, skiprows=1, usecols=36)
    rH = np.loadtxt(data, skiprows=1, usecols=45)

    m_air = np.loadtxt(data, skiprows=1, usecols=50)
    m_water = np.loadtxt(data, skiprows=1, usecols=40)
    m_cond = np.loadtxt(data, skiprows=1, usecols=18)

    config = configparser.ConfigParser()
    config.read('model_config.cfg')
    cfg_dim = config['dimensions']
    b = cfg_dim.getfloat('width')
    h = cfg_dim.getfloat('height')
    l = cfg_dim.getfloat('length')
    p_standard = config['other'].getfloat('pressure')
    sc = (fpa.moist_air_dynamic_viscosity(t_mean, p_standard, fpa.dewpoint2vapour_pressure(t_mean, t_dp_in))) / \
         (fpa.diffusion_coefficient(t_mean) * fpa.moist_air_density(p_standard, fpa.dewpoint2vapour_pressure(t_mean, t_dp_in), t_mean))
else:
    print('switch not set')
    exit(0)
####################################################################################
# Functions
####################################################################################


def Nu_lam(reynolds, prandtl, dia_hydr, length):
    """mean laminar Nusselt number for constant temperature difference taken from Heat Transfer in Pipe Flow Chapter
       of the VDI Heat Atlas by Gnielinski, V. (2010) p. 693 ff doi: 10.1007/978-3-540-77877-6 """
    Nu_mt1 = 3.66
    Nu_mt2 = 1.615 * (reynolds * prandtl * dia_hydr / length) ** (1/3)
    Nu_mt3 = (2 / (1 + 22 * prandtl)) ** (1/6) * (reynolds * prandtl * dia_hydr / length) ** (1/2)
    return (Nu_mt1 ** 3 + 0.7 ** 3 + (Nu_mt2 - 0.7) ** 3 + Nu_mt3 ** 3) ** (1/3)


def Nu_turb(reynolds, prandtl, dia_hydr, length):
    """mean turbulent Nusselt number taken from Heat Transfer in Pipe Flow Chapter of the VDI Heat Atlas by
       Gnielinski, V. (2010) p. 693 ff doi: 10.1007/978-3-540-77877-6 """
    xi = (1.8 * np.log10(reynolds) - 1.5) ** (-2)
    return (xi / 8 * reynolds * prandtl) / (1 + 12.7 * np.sqrt(xi / 8) * (prandtl ** (2/3) - 1)) * \
           (1 + (dia_hydr / length) ** (2/3))


def Nu_sen(reynolds, prandtl, dia_hydr, length):
    """sensible Nusselt number for entire range of Reynolds numbers taken from Heat Transfer in Pipe Flow Chapter
       of the VDI Heat Atlas by Gnielinski, V. (2010) p. 693 ff doi: 10.1007/978-3-540-77877-6 """
    if reynolds < 2300:
        return Nu_lam(reynolds, prandtl, dia_hydr, length)
    elif reynolds > 10000:
        return Nu_turb(reynolds, prandtl, dia_hydr, length)
    else:
        gamma = (reynolds - 2300) / (10000 - 2300)
        return (1 - gamma) * Nu_lam(2300, prandtl, dia_hydr, length) + gamma * Nu_turb(10000, prandtl, dia_hydr, length)


def mass_fraction_interface(pressure, Temp_interface):
    """noncondensable gas mass fraction at the gas-liquid interface with the assumption of saturation following the
       Gibbs-Dalton Law as formulated by
       G. Caruso, D. Di Vitale Maio, Heat and mass transfer analogy applied to condensation in the presence of
       noncondensable gases inside inclined tubes, Int. J. Heat Mass Transf. 68 (2014) 401–414,
       doi: 10.1016/j.ijheatmasstransfer.2013.09.049"""
    p_sat = fpa.temperature2saturation_vapour_pressure(Temp_interface)
    M_v = fpa.MOLES_MASS_VAPOUR
    M_air = fpa.MOLES_MASS_AIR
    return (pressure - p_sat) / (pressure - (1 - M_v / M_air) * p_sat)


def mass_fraction_bulk(mass_flow_ncg, mass_flow_water):
    """noncondensable gas mass fraction in the bulk flow"""
    return mass_flow_ncg / (mass_flow_water + mass_flow_ncg)


def Nu_lat(sherwood, prandtl, schmidt, jakob, B_i):
    return sherwood * prandtl * B_i * (schmidt * jakob) ** (-1)


def corr_suction_ht(sher, nuss, pran, schm, dp, t_int, t):
    p_v = fpa.dewpoint2vapour_pressure(t, dp)
    c_p = fpa.moist_air_heat_capacity(t, p_standard, p_v)
    c_pv = fpw.heat_capacity(t)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p_standard, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p_standard, t_int)
    r_t = c_pv / c_p * (sher * pran) / (schm * nuss) * np.log((1 - x_vb) / (1 - x_vi))
    # print('1: ', c_pv / c_p)
    # print('2: ', pran / schm)
    # print('3: ', sher / nuss)
    # print('4: ', (1 - x_vb) / (1 - x_vi))
    # print('5: ', r_t)
    # print('6: ', np.exp(r_t) - 1)
    return r_t / np.exp(r_t) - 1


def corr_fog_ht(sher, nuss, pran, schm, dp, t_int, t):
    p_v = fpa.dewpoint2vapour_pressure(t, dp)
    c_p = fpa.moist_air_heat_capacity(t, p_standard, p_v)
    # print('c_p: ', c_p)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p_standard, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p_standard, t_int)
    lmbda = fpw.enthalpy_evaporation(t)
    # print('lambda: ', lmbda)
    # print('1: ', lmbda / c_p)
    # print('2: ', pran / schm)
    # print('3: ', (x_vb - x_vi) / (t - t_int))
    # print('4: ', sher / nuss)
    # print('5: ', saturation_line_slope(t_int))
    # print('6: ', (1 + lmbda / c_p * pran / schm * (x_vb - x_vi) / (t - t_int) * sher / nuss))
    # print('7: ', (1 + lmbda / c_p * pran / schm * saturation_line_slope(t_int)))
    return (1 + lmbda / c_p * pran / schm * (x_vb - x_vi) / (t - t_int) * sher / nuss) / \
           (1 + lmbda / c_p * pran / schm * saturation_line_slope(t_int))


def corr_suction_mt(dp, t_int, t):
    p_v = fpa.dewpoint2vapour_pressure(t, dp)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p_standard, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p_standard, t_int)
    r_w = (x_vb - x_vi) / (1 - x_vi)
    return np.log(1 - r_w) / (-r_w)


def corr_fog_mt(sher, nuss, pran, schm, dp, t_int, t):
    p_v = fpa.dewpoint2vapour_pressure(t, dp)
    c_p = fpa.moist_air_heat_capacity(t, p_standard, p_v)
    x_vbs, x_vb = fpa.__moles_fraction_mixture__(p_v, p_standard, t)
    x_vis, x_vi = fpa.__moles_fraction_mixture__(p_v, p_standard, t_int)
    lmbda = fpw.enthalpy_evaporation(t)
    return (1 + (lmbda / c_p * pran / schm * (x_vb - x_vi) / (t - t_int) * sher / nuss) ** -1) / \
           (1 + (lmbda / c_p * pran / schm * saturation_line_slope(t_int)) ** -1)


def cos_theta(phi, theta_max, theta_min):
    return ((2 * (np.cos(theta_max) - np.cos(theta_min)) * phi ** 3 / np.pi ** 3) -
            (3 * (np.cos(theta_max) - np.cos(theta_min)) * phi ** 2 / np.pi ** 2) + np.cos(theta_max)) * np.cos(phi)


def zeta(phi, d, beta):
    return ((abs(np.cos(phi)) / d) ** 3 + (abs(beta * np.sin(phi)) / d) ** 3) ** (-1 / 3)


def f_grav(r_d, density):
    return density * 9.81 * 2 / 3 * np.pi * r_d ** 3


def f_surf_tens(r_d, gamma, theta_max, theta_min):
    return r_d * gamma * quad(cos_theta, 0., 2. * np.pi, args=(theta_max, theta_min,))[0]


def f_drag(r_d, density, u, C_d, theta_max, theta_min, beta):
    l_f = np.sin(theta_max) * (1 - np.cos(theta_min)) / (np.sin(theta_min) * (1 - np.cos(theta_max)))
    return r_d ** 2 * 0.5 * density * u ** 2 * C_d / (1 + l_f) ** 2 * \
        (l_f ** 2 * theta_min / np.sin(beta) ** 2 + l_f ** 2 / np.tan(beta) + theta_min / np.sin(theta_min) ** 2 -
         1 / np.tan(theta_min))


def correction_factor(r_m):
    r_d = r_m * 1000
    if r_d <= 1.3:
        __c = 0.28 * r_d + 1.155
    else:
        __c = 1.451 * r_d - 0.274
    return __c


def saturation_line_slope(t):
    return (fpa.temperature2saturation_vapour_pressure(t + 1e-4) / p_standard -
            fpa.temperature2saturation_vapour_pressure(t - 1e-4) / p_standard) / 2e-4

####################################################################################
# Droplet Force Balance
####################################################################################


# r_max = 0.009
d_h = (4 * b * h) / (2 * b + 2 * h)
print(d_h / h)
# gravitational force
g = 9.81
rho_c = fpw.density(t_w)
rho_b = fpa.moist_air_density(p_standard, fpa.dewpoint2vapour_pressure(t_mean, t_dp_in), t_mean)

# surface tension force
theta_r = 40.65 / (180 * np.pi)
theta_a = 63.3 / (180 * np.pi)
gamma = PropsSI('SURFACE_TENSION', 'T', t_mean + 273.15, 'Q', 1, 'Water')

r_max = np.full(re.shape, 0.0015)
bo = rho_c * g * (2 * r_max) ** 2 / gamma
# print(bo)
beta = 1 + 0.096 * bo
theta_m = theta_a * (0.01 * bo ** 2 - 0.155 * bo + 0.97)
re_d = re * r_max / d_h
c_d = 0.28 + (6 / np.sqrt(re_d)) + (21 / re_d)

u = re * HAPropsSI('mu', 'T', t_mean + 273.15, 'P', p_standard, 'R', rH) / \
    (d_h * fpa.moist_air_density(p_standard, fpa.dewpoint2vapour_pressure(t_mean, t_dp_in), t_mean))
f_g = np.zeros(re.shape)
f_s = np.zeros(re.shape)
f_d = np.zeros(re.shape)

epsilon_1 = np.ones(re.shape)
for i, item in enumerate(re):
    # print(i)
    # print('Re: ', item)
    while abs(epsilon_1[i]) > 0.0000001:
        bo[i] = rho_c[i] * g * (2 * r_max[i]) ** 2 / gamma[i]
        # print(bo)
        beta[i] = 1 + 0.096 * bo[i]
        theta_m = theta_a * (0.01 * bo[i] ** 2 - 0.155 * bo[i] + 0.97)
        re_d[i] = re[i] * r_max[i] / d_h
        c_d[i] = 0.28 + (6 / np.sqrt(re_d[i])) + (21 / re_d[i])
        f_g[i] = f_grav(r_max[i], rho_c[i])
        f_s[i] = f_surf_tens(r_max[i], gamma[i], theta_a, theta_m)
        # print(f_s[i])
        f_d[i] = f_drag(r_max[i], rho_b[i], u[i], c_d[i], theta_a, theta_m, beta[i])
        epsilon_1[i] = f_g[i] + f_d[i] + f_s[i]
        # print('eps: ', epsilon_1)
        if epsilon_1[i] < 0.:
            r_max[i] += 0.00001
        else:
            r_max[i] -= 0.00001
print(r_max)
# print('eps: ', epsilon_1)
print('f_g: ', f_g)
print('f_s: ', f_s)
print('f_d: ', f_d)
r_max = np.full(re.shape, 0.0009)
####################################################################################
# Iteration
####################################################################################

# initial Interface Temperature
T_i_start = t_w
m_ncg = m_air - m_water
jakob = fpa.moist_air_heat_capacity(t_mean, p_standard, fpa.dewpoint2vapour_pressure(t_mean, t_dp_in)) * \
        (t_mean - T_i_start) / fpw.enthalpy_evaporation(t_mean)
# print('Ja: ', jakob)
"""correlations of parameters are added from Eimann, F., Zheng, S., Philipp, C., Omranpoor, A. H., & Gross, U.
   (2020). Dropwise condensation of humid air - Experimental investigation and modelling of the convective heat
   transfer. Int. Journal of Heat and Mass Transfer, 154
   https://doi.org/10.1016/j.ijheatmasstransfer.2020.119734
   C   :    correction factor for droplet/film. correlation of predicted and measured droplet maximum radius for
            hemispherical droplets (0.2 mm < r_max < 1.61 mm)
   h_d :    drop heat coefficient correlation (for 2000 < Re < 21400) based on thermographic measurements
   
"""


C = np.vectorize(correction_factor)(r_max)
print('C: ', C)
# Sherwood is gained analogously to Nusselt number with Schmidt instead of Prandtl number
Sh = np.vectorize(Nu_sen)(re, sc, d_h, l)
print('Sh: ', Sh)
print('Sc: ', sc)
print('Pr: ', pr)
# print(re.shape)

# h_d, Nu_g, B_i, h_g, h_t, q_t, t_i = 0., 0., 0., 0., 0., 0., 0.
h_d, Nu_g, B_i, h_g, h_t, q_t, t_i, sh_corr = np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape), \
                                              np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape), \
                                              np.zeros(re.shape), np.zeros(re.shape)

T_i = np.array(T_i_start)
epsilon_2 = np.ones(re.shape)
for i, item in enumerate(re):
    # print(i)
    # print('Re: ', item)
    while abs(epsilon_2[i]) > 0.05:
        B_i[i] = np.log(mass_fraction_interface(p_standard, T_i[i]) / mass_fraction_bulk(m_ncg[i], m_water[i]))
        h_d[i] = (0.347 * re[i]) / (1 + 1.75 * np.exp(-330.5 * B_i[i]))
        sh_corr[i] = Sh[i] * corr_fog_mt(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], t_dp_in[i], T_i[i], t_mean[i]) * \
            corr_suction_mt(t_dp_in[i], T_i[i], t_mean[i])
        Nu_g[i] = C[i] * (Nu_sen(re[i], pr[i], d_h, l) *
                          corr_suction_ht(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], t_dp_in[i], T_i[i], t_mean[i]) *
                          corr_fog_ht(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], t_dp_in[i], T_i[i], t_mean[i])
                          + Nu_lat(sh_corr[i], pr[i], sc[i], jakob[i], B_i[i]))
        h_g[i] = Nu_g[i] * fpa.moist_air_thermal_conductivity(T_i[i], p_standard, fpa.dewpoint2vapour_pressure(T_i[i], t_dp_in[i])) / d_h
        h_t[i] = (1 / h_d[i] + 1 / h_g[i]) ** (-1)
        q_t[i] = h_t[i] * (t_mean[i] - t_w[i])
        t_i[i] = t_w[i] + q_t[i] / h_d[i]
        # print('t_i: ', t_i)
        epsilon_2[i] = T_i[i] - t_i[i]
        # print('eps: ', epsilon)
        if t_i[i] > T_i[i]:
            T_i[i] += 0.01
        else:
            T_i[i] -= 0.01

sigma_s_h = corr_suction_ht(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, t_dp_in, t_i, t_mean)
sigma_s_m = corr_suction_mt(t_dp_in, t_i, t_mean)
sigma_f_h = corr_fog_ht(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, t_dp_in, t_i, t_mean)
sigma_f_m = corr_fog_mt(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, t_dp_in, t_i, t_mean)
p_v_in = fpa.dewpoint2vapour_pressure(t_mean, t_dp_in)
x_bs, x_b = fpa.__moles_fraction_mixture__(p_v_in, p_standard, t_mean)
x_is, x_i = fpa.__moles_fraction_mixture__(p_v_in, p_standard, t_i)
print('dp/dt: ', saturation_line_slope(t_i))
print('tangency: ', sigma_f_m / sigma_f_h * Sh / np.vectorize(Nu_sen)(re, pr, d_h, l) * (x_b - x_i) / (t_mean - t_i))
print('suction_ht:', sigma_s_h)
print('fog_ht', sigma_f_h)
print('suction_mt', sigma_s_m)
print('fog_mt', sigma_f_m)
print('t_i: ', t_i)
print('B_i: ', B_i)
nu_pt = Nu_lat(Sh, pr, sc, jakob, B_i)
print('Nu_lat: ', nu_pt)
q_lat = nu_pt * fpa.moist_air_thermal_conductivity(t_i, p_standard, p_v_in) * (t_mean - t_i) / d_h
print('q_lat: ', q_lat)
print('Nu_g: ', Nu_g)
print('h_d: ', h_d)
print('h_g: ', h_g)
print('h_t: ', h_t)
print('q_t: ', q_t)

wdata = [[re], [pr], [nu_pt], [Nu_g], [Sh], [h_t], [h_d], [h_g], [q_t], [t_in], [t_out], [t_i], [t_dp_in], [t_dp_out], [rH], [r_max]]
wdata = np.reshape(wdata, (16, re.shape[0]))
wdata = wdata.T
__f = open(result_filename, 'w')
__f.write('Re Pr Nu_lat Nu_g Sh h_t h_d h_g q_t T_in T_out T_i T_dp_in T_dp_out rH r_max\n')
for __l in wdata:
    __l = str(__l)
    __l = __l.replace('[', '')
    __l = __l.replace(']', '')
    __l = __l.replace('\n', '')
    __f.write(__l + '\n')
__f.close()
