import unittest
from functions import f_surf_tens
from functions import cos_theta_integrate
from functions import Bo
from functions import aspect_ratio as ar
from functions import minimum_contact_angle

from CoolProp.CoolProp import PropsSI

class TryTesting(unittest.TestCase):
    def test_surf_tension(self):
        sig_literature = 72.75E-3 # N/m  @ 20°C
        sig_propsi = PropsSI('SURFACE_TENSION', 'T', 20 + 273.15, 'Q', 1, 'Water')
        self.assertAlmostEqual(sig_literature, sig_propsi, 3)


    def test_always_fails(self):
        #theta_a =
        pass
        #f_surf_tens(ar_test, surf_tens, theta_a, theta_m, beta)

    def test_cos_theta_integrate(self):
        pass

    def test_zeta(self):
        D =  # half droplet width

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
