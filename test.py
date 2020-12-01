import unittest
import numpy as np
from functions import f_surf_tens
from functions import cos_theta_integrate
from functions import Bo
from functions import aspect_ratio as ar
from functions import minimum_contact_angle
from functions import droplet_major_radius
from functions import zeta
from functions import cos_theta
from functions import f_surf_tens

from CoolProp.CoolProp import PropsSI


class TryTesting(unittest.TestCase):
    def test_surf_tension(self):
        sig_literature = 72.75E-3 # N/m  @ 20°C
        sig_propsi = PropsSI('SURFACE_TENSION', 'T', 20 + 273.15, 'Q', 1, 'Water')
        self.assertAlmostEqual(sig_literature, sig_propsi, 3)


    def test_cos_theta(self):

        # Trivial case 1: theta_min = theta_max
        theta_min = 85
        theta_max = 85
        for phi in np.linspace(0,np.pi,10):
            self.assertAlmostEqual(cos_theta(phi, theta_max, theta_min), np.cos(np.deg2rad(theta_max)), 8)

        # Trivial case 2: phi = 0
        theta_min, theta_max = np.random.random(2) * np.pi * 2
        self.assertAlmostEqual(cos_theta(0, theta_max, theta_min), np.cos(np.deg2rad(theta_max)), 4)


    def test_surf_force(self):
        # Trivial case: No surface force (theta_min == theta_max)
        gamma = 1.
        aspect_ratio = 1.
        r_d = 1
        theta_max = 105.
        theta_min = theta_max
        self.assertAlmostEqual(f_surf_tens(r_d, gamma, theta_max, theta_min, aspect_ratio), 0, 8)
        theta_max = 125.
        theta_min = theta_max
        self.assertAlmostEqual(f_surf_tens(r_d, gamma, theta_max, theta_min, aspect_ratio), 0, 8)
        theta_min = 85.
        self.assertAlmostEqual(f_surf_tens(r_d, gamma, theta_max, theta_min, aspect_ratio),
                               (-48. / 3.141592653589732 ** 3 * gamma * droplet_major_radius(r_d, theta_max, theta_min) *
                                (np.cos(np.deg2rad(theta_min)) - np.cos(np.deg2rad(theta_max)))), 8)


    def test_zeta(self):
        r_cl = 3E-3 # Major axis of eliptical projected area

        # Check symmetric case
        aspect_ratio = 1.
        phi = 3.141592653589732 / 2
        zeta_prog_1 = zeta(phi, r_cl, aspect_ratio)
        phi = 0.
        zeta_prog_2 = zeta(phi, r_cl, aspect_ratio)
        self.assertAlmostEqual(zeta_prog_1, zeta_prog_2, 8)
        self.assertAlmostEqual(zeta_prog_1, r_cl, 8)
        self.assertAlmostEqual(zeta_prog_2, r_cl, 8)


    def test_contact_line_radius(self):
        theta = 85 # °
        rd = 3E-3 # m

        r_cl_hand = 2.988E-3 # m
        r_cl_prog = droplet_major_radius(rd, theta, theta)

        self.assertAlmostEqual(r_cl_hand, r_cl_prog, 3)


    def test_bo(self):
        r = 3E-3 # m
        rho = 1000 # kg/m3
        sig = 72.75E-3 # N/m  @ 20°C

        Bo_prog = Bo(rho, 2*r,  sig)
        Bo_hand = 4.854
        self.assertAlmostEqual(Bo_prog, Bo_hand, 3)

    def aspect_ratio(self):
        Bo = 4.854

        ar_hand = 1.466
        ar_prog = ar(Bo)
        self.assertAlmostEqual(ar_prog, ar_hand, 3)

    def test_minimum_contact_angle(self):
        Bo = 4.854
        theta_adv = 85 # deg

        theta_min_prog = minimum_contact_angle(Bo, theta_adv)
        theta_min_hand = 38.526 # deg
        self.assertAlmostEqual(theta_min_prog, theta_min_hand, 3)

#np.vectorize(f_surf_tens)(ar_test, surf_tens, theta_a, theta_m, beta)
if __name__ == "__main__":
    unittest.main()
