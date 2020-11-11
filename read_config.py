import configparser
import numpy as np
import fluid_properties_air as fpa
import functions as mf


def log_mean(x, y):
    x = np.array(x)
    y = np.array(y)
    __logmean = (np.maximum(x, y) - np.minimum(x, y)) / (np.log((np.maximum(x, y)) / (np.minimum(x, y))))
    return __logmean


def read(config_file, data_file=None, switch='config'):
    if switch == 'config':
        config = configparser.ConfigParser()
        config.read(config_file)
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
        rh = np.array([cfg_temp.getfloat('rH')])
    
        cfg_dim = config['dimensions']
        b = cfg_dim.getfloat('width')
        h = cfg_dim.getfloat('height')
        l = cfg_dim.getfloat('length')

        cfg_oth = config['other']
        p_standard = cfg_oth.getfloat('pressure')
        theta_a = cfg_oth.getfloat('ascending_contact_angle') / (180 * np.pi)
        theta_r = cfg_oth.getfloat('receding_contact_angle') / (180 * np.pi)

        try:
            cfg_mf = config['mass_flow']
            m_air = np.array([cfg_mf.getfloat('massflow_moistair')])
            m_water = np.array([cfg_mf.getfloat('massflow_water')])
            m_cond = np.array([cfg_mf.getfloat('massflow_condensate')])
            mf_int = mf.mass_fraction_interface(p_standard, t_w)
            mf_bulk = mf.mass_fraction_bulk(m_air - m_water, m_water)
        except KeyError:
            print('using mass fraction')
            mf_int, mf_bulk = None, None
        if mf_int is None:
            try:
                cfg_mf = config['mass_fraction']
                mf_int = np.array(cfg_mf.getfloat('mass_fraction_interface'))
                mf_bulk = np.array(cfg_mf.getfloat('mass_fraction_bulk'))
            except KeyError:
                print('no mass flow or mass fractions were given.')
                mf_int, mf_bulk = [], []

        return re, pr, sc, t_in, t_out, t_w, t_mean, t_dp_in, t_dp_out, \
            rh, mf_int, mf_bulk, b, h, l, p_standard, theta_a, theta_r
    elif switch == 'dat' and data_file is not None:
        config = configparser.ConfigParser()
        config.read(config_file)
        cfg_dim = config['dimensions']
        b = cfg_dim.getfloat('width')
        h = cfg_dim.getfloat('height')
        l = cfg_dim.getfloat('length')
        p_standard = config['other'].getfloat('pressure')
        theta_a = config['other'].getfloat('ascending_contact_angle') / (180 * np.pi)
        theta_r = config['other'].getfloat('receding_contact_angle') / (180 * np.pi)
        re = np.loadtxt(data_file, skiprows=1, usecols=2)
        pr = np.loadtxt(data_file, skiprows=1, usecols=4)
    
        t_in = np.loadtxt(data_file, skiprows=1, usecols=28)
        t_out = np.loadtxt(data_file, skiprows=1, usecols=30)
        t_w = np.loadtxt(data_file, skiprows=1, usecols=26)
        t_mean = np.loadtxt(data_file, skiprows=1, usecols=32)
        t_dp_in = np.loadtxt(data_file, skiprows=1, usecols=34)
        t_dp_out = np.loadtxt(data_file, skiprows=1, usecols=36)
        # rh = np.loadtxt(data_file, skiprows=1, usecols=45)
        rh = fpa.relativehumidity(t_mean, log_mean(t_dp_in, t_dp_out))
    
        m_air = np.loadtxt(data_file, skiprows=1, usecols=50)
        m_water = np.loadtxt(data_file, skiprows=1, usecols=40)
        m_cond = np.loadtxt(data_file, skiprows=1, usecols=18)
        mf_int = mf.mass_fraction_interface(p_standard, t_w)
        mf_bulk = mf.mass_fraction_bulk(m_air - m_water, m_water)

        sc = (fpa.moist_air_dynamic_viscosity(t_mean, p_standard,
                                              rh * fpa.temperature2saturation_vapour_pressure(t_in))) / \
             (fpa.diffusion_coefficient(t_mean) *
              fpa.moist_air_density(p_standard, rh * fpa.temperature2saturation_vapour_pressure(t_in), t_mean))
        return re, pr, sc, t_in, t_out, t_w, t_mean, t_dp_in, t_dp_out, \
            rh, mf_int, mf_bulk, b, h, l, p_standard, theta_a, theta_r
    else:
        print('switch not set or data file missing')
        exit(0)
