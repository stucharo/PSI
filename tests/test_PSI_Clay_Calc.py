from math import pi

import src.PSI_Clay_Calc as psi

def test_A():
    OD = 1
    ID = 0.5

    actual = psi.A(OD, ID)
    expected = pi * (OD**2 - ID**2) / 4

    assert actual == expected