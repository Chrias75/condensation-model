import read_config
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
from functions import *
from scipy.optimize import minimize
import logging

####################################################################################
# Input
####################################################################################

config_file = 'experiment_config.cfg'
# config_file = 'model_config.cfg'
data = '/home/brue_ch/Auswertungen/rH_variable/Profil/data_rH_2000_31_5.dat'
result_filename = '/home/brue_ch/Auswertungen/rH_variable/Profil/eimann_2000_31_5_h_d.dat'
re, pr, sc, t_in, t_out, t_w, t_mean, t_dp_in, t_dp_out, rH, mf_int, mf_bulk, b, h, l, p_standard, \
 theta_a, theta_r, flow_direction = read_config.read(config_file, data_file=data, switch='dat')

####################################################################################
# Logger Setup
####################################################################################

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
format_stream = logging.Formatter('%(levelname)s - %(message)s')

fh = logging.FileHandler('%s.log' % config_file[:-4])
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format_stream)
logger.addHandler(ch)

####################################################################################
# Force balance parameter initialisation
####################################################################################

d_h = (4 * b * h) / (2 * b + 2 * h)
# surface tension
surf_tens = PropsSI('SURFACE_TENSION', 'T', t_mean + 273.15, 'Q', 1, 'Water')
# water and air density
rho_c = fpw.density(t_w) * 1000
rho_b = fpa.moist_air_density(p_standard, rH * fpa.temperature2saturation_vapour_pressure(t_in), t_mean)
# initial value for critical radius
r_max = np.full(re.shape, 0.0016)
# initial values for Bond number, droplet aspect ratio, minimum contact angle and drag coefficient
bo = Bo(rho_c, r_max, surf_tens)
beta = aspect_ratio(bo)
theta_m_0 = minimum_contact_angle(bo, theta_a)
c_d = np.vectorize(c_drag)(r_max, theta_a, theta_m_0, re, d_h)
# bulk velocity based on reynolds number
u = re * HAPropsSI('mu', 'T', t_mean + 273.15, 'P', p_standard, 'R', rH) / \
    (d_h * fpa.moist_air_density(p_standard, rH * fpa.temperature2saturation_vapour_pressure(t_in), t_mean))
# variable initialisation for forces and iteration
f_g, f_s, f_d = np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape)
epsilon_1 = np.ones(re.shape)

# a lot of debugging info
logger.debug('flow direction: {a}'.format(a=flow_direction))
logger.debug('hydraulic diameter: {a}'.format(a=d_h))
logger.debug('duct height: {a}'.format(a=h))
logger.debug('diameter ratio: {a}'.format(a=d_h / h))
logger.debug('rho_water: {a}'.format(a=rho_c))
logger.debug('rho_air: {a}'.format(a=rho_b))
logger.debug('surface tension: {a}'.format(a=surf_tens))
logger.debug('Bo_0: {a}'.format(a=bo))
logger.debug('beta_0: {a}'.format(a=beta))
logger.debug('theta_max: {a}'.format(a=theta_a))
logger.debug('theta_min(Bo_0): {a}'.format(a=theta_m_0))
logger.debug('Re {a}'.format(a=re))
logger.debug('Re_d(Bo_0): {a}'.format(a=re * r_max * (1 - np.cos(np.deg2rad((theta_a + theta_m_0) / 2))) / d_h))
logger.debug('C_d(Bo_0): {a}'.format(a=c_d))
logger.debug('bulk velocity: {a}'.format(a=u))

####################################################################################
# Droplet Force Balance
####################################################################################

for i, item in enumerate(re):

    def func(r):
        try:
            bo[i] = Bo(rho_c[i], 2 * r, surf_tens[i])
        except IndexError:
            bo[i] = Bo(rho_c[i], 2 * r, np.array([surf_tens])[i])
        beta[i] = aspect_ratio(bo[i])
        theta_m = minimum_contact_angle(bo[i], theta_a)
        c_d[i] = c_drag(r, theta_a, theta_m, re[i], d_h)
        # a: scaling factor for the deformation of the droplet due to flow. arbitrary value based on observation
        a = 1.
        if flow_direction == 'horizontal':
            f_g[i] = f_grav(r, rho_c[i])
            try:
                f_s[i] = a * f_surf_tens(r, surf_tens[i], theta_a, theta_m, beta[i])
            except TypeError:
                f_s[i] = a * f_surf_tens(r, np.array([surf_tens])[i], theta_a, theta_m, beta[i])
            except IndexError:
                f_s[i] = a * f_surf_tens(r, np.array([surf_tens])[i], theta_a, theta_m, beta[i])
            f_d[i] = f_drag(r, rho_b[i], u[i], c_d[i], theta_a, theta_m)
            epsilon_1[i] = f_g[i] ** 2 + f_d[i] ** 2 - f_s[i] ** 2

        elif flow_direction == 'vertical':
            f_g[i] = f_grav_vert(r, rho_c[i], theta_a, theta_m)
            try:
                f_s[i] = a * f_surf_tens(r, surf_tens[i], theta_a, theta_m, beta[i])
            except TypeError:
                f_s[i] = a * f_surf_tens(r, np.array([surf_tens])[i], theta_a, theta_m, beta[i])
            except IndexError:
                f_s[i] = a * f_surf_tens(r, np.array([surf_tens])[i], theta_a, theta_m, beta[i])

            f_d[i] = f_drag_vert(r, rho_b[i], u[i], c_d[i], theta_a, theta_m)
            epsilon_1[i] = f_g[i] + f_d[i] - abs(f_s[i])

        else:
            logger.critical('no flow direction given')
            exit(0)

        return epsilon_1[i]

    res = minimize(func, 2e-3, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
    # print(f"r_max:\t {res.x[0]*1000.:0.3f}mm")
    r_max[i] = res.x[0]

logger.info('force balance results:')
logger.info('r_max force balance: {a}'.format(a=r_max))
logger.info('Bo(r_max): {a}'.format(a=bo))
logger.info('minimum contact angle: {a}'.format(a=minimum_contact_angle(bo, theta_a)))
logger.info('f_g: {a}'.format(a=f_g))
logger.info('f_s: {a}'.format(a=f_s))
logger.info('f_d: {a}'.format(a=f_d))

# overwriting the critical radius, if desired
r_max = np.full(re.shape, 0.0007)
logger.info('r_max overwritten manually; r_max: {a}'.format(a=r_max))
# exit(0)

####################################################################################
# heat transfer parameter initialisation
####################################################################################

# wall temperature serves as initial interface temperature
T_i_start = t_w
# simplification, as jakob is using wall temperature and is assumed constant
jakob = jakob(t_mean, T_i_start, t_in, p_standard, rH)
# Eimanns correction factor for total heat transfer based on critical radius
C = np.vectorize(correction_factor)(r_max)

# Sherwood is gained analogously to Nusselt number with Schmidt instead of Prandtl number
Sh = np.vectorize(Nu_sen)(re, sc, d_h, l)
# variable initialisation for heat transfer iteration
h_d, Nu_g, B_i, h_g, h_t, q_t, t_i, sh_corr, nu_corr = np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape), \
                                                       np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape), \
                                                       np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape)
T_i = np.array(T_i_start)
epsilon_2 = np.ones(re.shape)

# more logging
logger.debug('Ja: {a}'.format(a=jakob))
logger.info('Correction Factor(r_max): {a}'.format(a=C))
logger.debug('Nu_0: {a}'.format(a=np.vectorize(Nu_sen)(re, pr, d_h, l)))
logger.debug('Sh: {a}'.format(a=Sh))
logger.debug('Sc: {a}'.format(a=sc))
logger.debug('Pr: {a}'.format(a=pr))
logger.debug('t_mean: {a}'.format(a=t_mean))
logger.debug('rh: {a}'.format(a=rH))
logger.debug('T_i: {a}'.format(a=T_i))

####################################################################################
# calculating the heat transfer with unknown heat flux density
####################################################################################

for i, item in enumerate(re):
    logger.info('Re: {a}'.format(a=item))
    while abs(epsilon_2[i]) > 0.05:
        B_i[i] = driving_force_mt(t_i[i], p_standard, mf_bulk[i])
        h_d[i] = (0.347 * re[i]) / (1 + 1.75 * np.exp(-330.5 * B_i[i]))
        # sh_corr[i] = Sh[i] * \
        #     corr_fog_mt(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], rH[i], T_i[i], t_mean[i]) * \
        #     corr_suction_mt(rH[i], T_i[i], t_mean[i])
        # nu_corr[i] = Nu_sen(re[i], pr[i], d_h, l) * \
        #     corr_suction_ht(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], rH[i], T_i[i], t_mean[i]) * \
        #     corr_fog_ht(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], rH[i], T_i[i], t_mean[i])
        sh_corr[i] = Sh[i] * corr_suction_mt(rH[i], T_i[i], t_mean[i], p_standard)
        nu_corr[i] = Nu_sen(re[i], pr[i], d_h, l) * \
            corr_suction_ht(Sh[i], Nu_sen(re[i], pr[i], d_h, l), pr[i], sc[i], rH[i], T_i[i], t_mean[i], p_standard)
        Nu_g[i] = C[i] * (nu_corr[i] + Nu_lat(sh_corr[i], pr[i], sc[i], jakob[i], B_i[i]))
        h_g[i] = Nu_g[i] * \
            fpa.moist_air_thermal_conductivity(T_i[i], p_standard,
                                               rH[i] * fpa.temperature2saturation_vapour_pressure(t_in[i])) / d_h
        h_t[i] = (1 / h_d[i] + 1 / h_g[i]) ** (-1)
        q_t[i] = h_t[i] * (t_mean[i] - t_w[i])
        t_i[i] = t_w[i] + q_t[i] / h_d[i]
        epsilon_2[i] = T_i[i] - t_i[i]
        if t_i[i] > T_i[i]:
            T_i[i] += 0.01
        else:
            T_i[i] -= 0.01
    logger.info('interface temperature: {a}'.format(a=t_i[i]))

p_v_in = rH * fpa.temperature2saturation_vapour_pressure(t_in)
p_v_mean = rH * fpa.temperature2saturation_vapour_pressure(t_mean)
x_bs, x_b = fpa.__moles_fraction_mixture__(p_v_in, p_standard, t_mean)
x_is, x_i = fpa.__moles_fraction_mixture__(p_v_in, p_standard, t_i)

# final determination of the model nusselt numbers for latent and total heat transfer
nu_lat = Nu_lat(Sh, pr, sc, jakob, B_i)
q_lat = nu_lat * fpa.moist_air_thermal_conductivity(t_i, p_standard, p_v_in) * (t_mean - t_i) / d_h
nu_t = h_t * d_h / fpa.moist_air_thermal_conductivity(t_mean, p_standard, p_v_mean)

# correction factors for suction and fog debugging
sigma_s_h = corr_suction_ht(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, rH, t_i, t_mean, p_standard)
sigma_s_m = corr_suction_mt(rH, t_i, t_mean, p_standard)
sigma_f_h = corr_fog_ht(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, rH, t_i, t_mean, p_standard)
sigma_f_m = corr_fog_mt(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, rH, t_i, t_mean, p_standard)

# debugging info
logger.debug('dp/dt: {a}'.format(a=saturation_line_slope(t_i, p_standard)))
logger.debug('dp/dt(Brouwers): {a}'.format(a=saturation_line_brouwers(t_i, p_standard)))
logger.debug('tangency: {a}'.format(a=sigma_f_m / sigma_f_h * Sh / np.vectorize(Nu_sen)(re, pr, d_h, l) *
                                    (x_b - x_i) / (t_mean - t_i)))
logger.debug('suction_ht: {a}'.format(a=sigma_s_h))
logger.debug('fog_ht: {a}'.format(a=sigma_f_h))
logger.debug('suction_mt: {a}'.format(a=sigma_s_m))
logger.debug('fog_mt: {a}'.format(a=sigma_f_m))
logger.debug('B_i: {a}'.format(a=B_i))

# output info
logger.info('Nu_lat: {a}'.format(a=nu_lat))
logger.info('q_lat: {a}'.format(a=q_lat))
logger.debug('Nu_g: {a}'.format(a=Nu_g))
logger.info('h_d: {a}'.format(a=h_d))
logger.debug('h_g: {a}'.format(a=h_g))
logger.info('h_t: {a}'.format(a=h_t))
logger.info('Nu_t: {a}'.format(a=nu_t))
logger.info('q_t: {a}'.format(a=q_t))

# writing the data into result file

wdata = [[re], [pr], [nu_lat], [Nu_g], [nu_t], [Sh], [h_t], [h_d], [h_g], [q_t], [t_in], [t_out], [t_i], [t_dp_in],
         [t_dp_out], [rH], [r_max]]
wdata = np.reshape(wdata, (17, re.shape[0]))
wdata = wdata.T
__f = open(result_filename, 'w')
__f.write('Re Pr Nu_lat Nu_g Nu_t Sh h_t h_d h_g q_t T_in T_out T_i T_dp_in T_dp_out rH r_max\n')
for __l in wdata:
    __l = str(__l)
    __l = __l.replace('[', '')
    __l = __l.replace(']', '')
    __l = __l.replace('\n', '')
    __f.write(__l + '\n')
__f.close()
