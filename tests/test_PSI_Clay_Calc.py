from math import pi

import src.PSI_Clay_Calc as psi


def test_A():
    OD = 1
    ID = 0.5

    actual = psi.A(OD, ID)
    expected = pi * (OD ** 2 - ID ** 2) / 4

    assert actual == expected


def test_A_no_ID():
    OD = 1

    actual = psi.A(OD)
    expected = pi * OD ** 2 / 4

    assert actual == expected


def test_B_z_less_than_D_over_2():
    D = 1
    z = 0.1

    actual = psi.B(D, z)
    expected = 2 * (D * z - z ** 2) ** 0.5

    assert actual == expected


def test_B_z_greater_than_or_equal_to_D_over_2():
    D = 1
    z = 0.6

    actual = psi.B(D, z)
    expected = D

    assert actual == expected
