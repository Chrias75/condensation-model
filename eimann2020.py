import read_config
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
import matplotlib.pyplot as plt
from functions import *
from scipy.optimize import minimize

####################################################################################
# Input
####################################################################################

config_file = 'experiment_config.cfg'
config_file = 'model_config.cfg'
data = '/home/brue_ch/Auswertungen/rH_variable/Profil/data_rH_variable_all.dat'
result_filename = '/home/brue_ch/Auswertungen/rH_variable/Profil/eimann_modell.dat'
re, pr, sc, t_in, t_out, t_w, t_mean, t_dp_in, t_dp_out, rH, mf_int, mf_bulk, b, h, l, p_standard, \
    theta_a, theta_r, flow_direction = read_config.read(config_file, data_file=None, switch='config')

####################################################################################
# Droplet Force Balance
####################################################################################


d_h = (4 * b * h) / (2 * b + 2 * h)
print(d_h / h)
print(flow_direction)
# gravitational force
g = 9.81
rho_c = fpw.density(t_w) * 1000
print('rho_w: ', rho_c)
rho_b = fpa.moist_air_density(p_standard, rH * fpa.temperature2saturation_vapour_pressure(t_in), t_mean)
print('rho_a: ', rho_b)
# surface tension
surf_tens = PropsSI('SURFACE_TENSION', 'T', t_mean + 273.15, 'Q', 1, 'Water')
print('gamma: ', surf_tens)
r_max = np.full(re.shape, 0.0016)

bo = Bo(rho_c, r_max, surf_tens)
beta = aspect_ratio(bo)
half_d = beta * r_max
print('Bo_0: ', bo)

print('d: ', half_d)
print('beta: ', beta)
print('theta_max: ', theta_a)
theta_m = minimum_contact_angle(bo, theta_a)
print('theta_min: ', theta_m)
print('Re_d: ', re * r_max * (1 - np.cos(np.deg2rad((theta_a + theta_m) / 2))) / d_h)
c_d = np.vectorize(c_drag)(r_max, theta_a, theta_m, re, d_h)
print('C_d: ', c_d)

u = re * HAPropsSI('mu', 'T', t_mean + 273.15, 'P', p_standard, 'R', rH) / \
    (d_h * fpa.moist_air_density(p_standard, rH * fpa.temperature2saturation_vapour_pressure(t_in), t_mean))
print('u: ', u)
f_g = np.zeros(re.shape)
f_s = np.zeros(re.shape)
f_d = np.zeros(re.shape)

# ar_test = np.linspace(0.00000001, 5.0e-3, 100)
# plt.figure(1)
# plt.title('horizontal')
# plt.plot(ar_test, f_drag(ar_test, rho_b, u, c_drag(ar_test, theta_a, theta_m, re, d_h), theta_a, theta_m), label='F_d')
# plt.plot(ar_test, f_grav(ar_test, rho_c), label='F_g')
# plt.plot(ar_test, np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta), label='F_s')
# plt.legend(loc=2)
# plt.figure(2)
# plt.title('vertical')
# plt.plot(ar_test, f_drag_vert(ar_test, rho_b, u, c_drag(ar_test, theta_a, theta_m, re, d_h), theta_a, theta_m), label='F_d')
# plt.plot(ar_test, np.vectorize(f_grav_vert)(ar_test, rho_c, theta_a, theta_m), label='F_g')
# plt.plot(ar_test, np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta), label='F_s')
# plt.legend(loc=2)
# plt.figure(3)
# plt.plot(ar_test, (f_drag(ar_test, rho_b, u, c_drag(ar_test, theta_a, theta_m, re, d_h), theta_a, theta_m) ** 2
#                    + f_grav(ar_test, rho_c) ** 2
#                    - np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta) ** 2), label='sum F')
# # plt.plot(ar_test, (f_drag(ar_test, rho_b, u, c_drag(ar_test, re, d_h), theta_a, theta_m)
# #                    + f_grav(ar_test, rho_c)
# #                    - abs(np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta))), label='sum F')
# plt.legend(loc=2)
# plt.figure(4)
# plt.plot(ar_test, (f_drag(ar_test, rho_b, u, c_drag(ar_test, theta_a, theta_m, re, d_h), theta_a, theta_m)
#                    + f_grav(ar_test, rho_c)
#                    - abs(np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta))), label='sum F hori')
# plt.plot(ar_test, (f_drag_vert(ar_test, rho_b, u, c_drag(ar_test, theta_a, theta_m, re, d_h), theta_a, theta_m)
#                    + np.vectorize(f_grav_vert)(ar_test, rho_c, theta_a, theta_m)
#                    - abs(np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta))), label='sum F vert')
# plt.legend(loc=2)
# plt.show()
# exit(0)
epsilon_1 = np.ones(re.shape)

for i, item in enumerate(re):

    def func(r):
        try:
            bo[i] = rho_c[i] * g * (2 * r) ** 2 / (surf_tens[i])
        except IndexError:
            bo[i] = rho_c[i] * g * (2 * r) ** 2 / (np.array([surf_tens])[i])
        # print(bo)
        beta[i] = 1 + 0.096 * bo[i]
        theta_m = theta_a * (0.01 * bo[i] ** 2 - 0.155 * bo[i] + 0.97)
        c_d[i] = c_drag(r, theta_a, theta_m, re[i], d_h)
        a = 1.
        if flow_direction.strip() == 'horizontal':
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
            epsilon_1[i] = f_g[i] + f_d[i] + f_s[i]

        else:
            print('no flow direction given')
            exit(0)

        return epsilon_1[i]

    res = minimize(func, 2e-3, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    print(f"r_max:\t {res.x[0]*1000.:0.3f}mm")
    r_max[i] = res.x[0]

print('r_max: ', r_max)
print('Bo: ', bo)
# print('eps: ', epsilon_1)
print('f_g: ', f_g)
print('f_s: ', f_s)
print('f_d: ', f_d)
# r_max = np.full(re.shape, 0.0012)
exit(0)
####################################################################################
# Iteration
####################################################################################

# initial Interface Temperature
T_i_start = t_w
jakob = fpa.moist_air_heat_capacity(t_mean, p_standard, rH * fpa.temperature2saturation_vapour_pressure(t_in)) * \
        (t_mean - T_i_start) / fpw.enthalpy_evaporation(t_mean)
print('Ja: ', jakob)


C = np.vectorize(correction_factor)(r_max)
print('C: ', C)
# Sherwood is gained analogously to Nusselt number with Schmidt instead of Prandtl number
Sh = np.vectorize(Nu_sen)(re, sc, d_h, l)
Nu_0 = np.vectorize(Nu_sen)(re, pr, d_h, l)
print('Nu_0: ', Nu_0)
print('Sh: ', Sh)
print('Sc: ', sc)
print('Pr: ', pr)
# print(re.shape)

# h_d, Nu_g, B_i, h_g, h_t, q_t, t_i = 0., 0., 0., 0., 0., 0., 0.
h_d, Nu_g, B_i, h_g, h_t, q_t, t_i, sh_corr, nu_corr = np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape), \
                                                       np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape), \
                                                       np.zeros(re.shape), np.zeros(re.shape), np.zeros(re.shape)
print('t_mean: ', t_mean)
print('rh: ', rH)
T_i = np.array(T_i_start)
print('T_i: ', T_i)
epsilon_2 = np.ones(re.shape)
for i, item in enumerate(re):
    # print(i)
    # print('Re: ', item)
    while abs(epsilon_2[i]) > 0.05:
        B_i[i] = np.log(mass_fraction_interface(p_standard, T_i[i]) / mf_bulk[i])
        # print('mf_int: ', mf.mass_fraction_interface(p_standard, T_i[i]))
        # print('mf_bulk: ', mf_bulk[i])
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
        # print('t_i: ', t_i)
        epsilon_2[i] = T_i[i] - t_i[i]
        # print('eps: ', epsilon)
        if t_i[i] > T_i[i]:
            T_i[i] += 0.01
        else:
            T_i[i] -= 0.01

sigma_s_h = corr_suction_ht(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, rH, t_i, t_mean, p_standard)
sigma_s_m = corr_suction_mt(rH, t_i, t_mean, p_standard)
sigma_f_h = corr_fog_ht(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, rH, t_i, t_mean, p_standard)
sigma_f_m = corr_fog_mt(Sh, np.vectorize(Nu_sen)(re, pr, d_h, l), pr, sc, rH, t_i, t_mean, p_standard)
p_v_in = rH * fpa.temperature2saturation_vapour_pressure(t_in)
p_v_mean = rH * fpa.temperature2saturation_vapour_pressure(t_mean)
x_bs, x_b = fpa.__moles_fraction_mixture__(p_v_in, p_standard, t_mean)
x_is, x_i = fpa.__moles_fraction_mixture__(p_v_in, p_standard, t_i)
# print('test: ', c_p_mixture(0.804, 94.))
print('dp/dt: ', saturation_line_slope(t_i, p_standard))
# print('dp/dt(Brouwers): ', saturation_line_bruowers(t_i))
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
nu_t = h_t * d_h / fpa.moist_air_thermal_conductivity(t_mean, p_standard, p_v_mean)
print('Nu_t: ', nu_t)
print('q_t: ', q_t)

wdata = [[re], [pr], [nu_pt], [Nu_g], [nu_t], [Sh], [h_t], [h_d], [h_g], [q_t], [t_in], [t_out], [t_i], [t_dp_in],
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
