import numpy as np
import fluid_properties_air as fpa
import fluid_properties_water as fpw

#####
# Input variables
#####

re = 22110.
t_b = 40.
t_w = 22.4
rh = 0.442
p_st = 101325
b = 0.012
h = 0.032
d_h = (4 * b * h) / (2 * b + 2 * h)
########
# calculations
#########

p_v = rh * fpa.temperature2saturation_vapour_pressure(t_b)
rho_v = fpa.vapour_pressure2vapour_density(t_b, p_v)
ah = fpa.density2absolute_humidity(rho_v, fpa.dry_air_density(t_b, p_st))

u = re * fpa.moist_air_dynamic_viscosity(t_b, p_st, p_v) / (d_h * fpa.moist_air_density(p_st, p_v, t_b))
m_air = u * b * h * fpa.moist_air_density(p_st, p_v, t_b)
m_water = ah * m_air

print('m_air: ', m_air)
print('m_water', m_water)

