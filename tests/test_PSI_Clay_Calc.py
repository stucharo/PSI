from math import pi

import pytest
import numpy as np

import src.PSI_Clay_Calc as psi


def test_A():
    OD = 1
    ID = 0.5

    actual = psi.A(OD, ID)
    expected = pi * (OD**2 - ID**2) / 4

    assert actual == expected


def test_A_no_ID():
    OD = 1

    actual = psi.A(OD)
    expected = pi * OD**2 / 4

    assert actual == expected


def test_B_z_less_than_D_over_2():
    D = 1
    z = 0.1

    actual = psi.B(D, z)
    expected = 0.6

    assert actual == expected


def test_B_z_greater_than_or_equal_to_D_over_2():
    D = 1
    z = 0.6

    actual = psi.B(D, z)
    expected = D

    assert actual == expected


def test__cylinder_weight():

    OD = 0.1683
    ID = 0.1429
    rho_steel = 7850

    actual = psi._cylinder_weight(OD, ID, rho_steel)
    expected = 477.918

    assert actual == pytest.approx(expected)


def test_W_sub():

    OD = 0.1683
    ID = 0.1429
    rho_steel = 7850
    rho_conc = 2400
    rho_coat = 900
    rho_sw = 1025
    t_coat = 0.003
    t_conc = 0.04
    rho_cont = 150

    actual = psi.W_sub(
        OD, ID, rho_steel, rho_conc, rho_coat, rho_sw, t_coat, t_conc, rho_cont
    )
    expected = 0.6390397  # kN * m^-1

    assert actual == pytest.approx(expected)


def test_beta():

    D = np.array([1, 1, 1])
    z = np.array([0.1, 0.6, 0.5])

    actual = psi.beta(z, D)
    expected = np.array([0.64350111, pi / 2, pi / 2])

    np.testing.assert_array_almost_equal(actual, expected)


def test_lat_br():

    D = 1
    z = 0.1
    S_u = 4000
    Q_v = 1e5
    gamma_dash = 2500 * 9.80665
    W_case = 5000

    actual = psi.lat_br(z, D, Q_v, S_u, gamma_dash, W_case)
    expected = 3.0246474

    assert actual == pytest.approx(expected)


def test__Abm_z_less_than_D_over_2():

    D = np.array([1, 1])
    z = np.array([0.1, 0.1])

    actual = psi._Abm_z_less_than_D_over_2(D, z)
    expected = np.array([0.040875, 0.040875])

    np.testing.assert_array_almost_equal(actual, expected)


def test__Abm_z_greater_than_D_over_2():

    D = np.array([1, 1])
    z = np.array([0.6, 0.6])

    actual = psi._Abm_z_greater_than_D_over_2(D, z)
    expected = np.array([0.492699, 0.492699])

    np.testing.assert_array_almost_equal(actual, expected)


def test_Abm():

    D = np.array([1, 1])
    z = np.array([0.1, 0.6])

    actual = psi.Abm(D, z)
    expected = np.array([0.040875, 0.492699])

    np.testing.assert_array_almost_equal(actual, expected)


def test_Q_v():

    gamma_dash = np.array([400, 500, 600, 1000]) * 9.80665
    D = np.array([1, 0.8, 1.2, 0.1])
    z = np.array([0.1, 0.5, 0.12, 0.01])
    S_u = np.array([1000, 2000, 3000, 4000])

    actual = psi.Q_v(gamma_dash, D, z, S_u)
    expected = np.array([3614.55767342, 10972.65014397, 12666.0736242, 1355.6319235])

    np.testing.assert_array_almost_equal(actual, expected)


def test_OCR():

    W_hydro = np.array([500, 600, 700])
    W_case = np.array([700, 600, 500])

    actual = psi.OCR(W_hydro, W_case)
    expected = np.array([1, 1, 700 / 500])

    np.testing.assert_array_almost_equal(actual, expected)
