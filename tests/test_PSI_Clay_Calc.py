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


def test_wedge_factor():

    D = np.array([1, 1, 1])
    z = np.array([0.1, 0.6, 0.5])

    actual = psi.wedge_factor(z, D)
    expected = np.array([1.06808973, 1.27323954, 1.27323954])

    np.testing.assert_array_almost_equal(actual, expected)


def test_ax_res():

    axbr = np.array([300, 500, 700])
    E_res = np.array([0.25, 0.45, 0.65])

    actual = psi.ax_res(axbr, E_res)
    expected = axbr * E_res

    np.testing.assert_array_almost_equal(actual, expected)


def test_OCR():

    W_hydro = np.array([500, 600, 700])
    W_case = np.array([700, 600, 500])

    actual = psi.OCR(W_hydro, W_case)
    expected = np.array([1, 1, 700 / 500])

    np.testing.assert_array_almost_equal(actual, expected)


def test_k_lay():

    gamma_dash = np.array([400, 600, 800]) * 9.80665
    D = np.array([1, 0.3, 0.9])
    z = np.array([0.1, 0.2, 0.01])
    EI = np.array([1e6, 5e5, 2e6])
    S_ur = np.array([1000, 4000, 10000])
    T_0 = np.array([2.5e5, 1e4, 5e5])

    actual = psi.k_lay(gamma_dash, D, z, EI, S_ur, T_0)
    expected = np.array([1, 2.05226296, 1.27630148])

    np.testing.assert_array_almost_equal(actual, expected)


def test_mv_norm():

    S_u_mean = 0.22
    S_u_std = 0.13
    S_ur_mean = 0.11
    S_ur_std = 0.03
    gamma_mean = 3.92
    gamma_std = 0.53
    corr = 1
    n = 10

    means = np.array([S_u_mean, S_ur_mean, gamma_mean])
    std_devs = np.array([S_u_std, S_ur_std, gamma_std])

    S_u, S_ur, gamma = psi.mv_norm(means, std_devs, corr, n)

    # Check arrays are all 'n' elements long
    assert S_u.size == S_ur.size == gamma.size == n

    # test that the index of the smallest value is the same in all arrays
    np.testing.assert_array_almost_equal(
        np.where(S_u == S_u.min()), np.where(S_ur == S_ur.min())
    )
    np.testing.assert_array_almost_equal(
        np.where(S_ur == S_ur.min()), np.where(gamma == gamma.min())
    )

    # test that the index of the largestvalue is the same in all arrays
    np.testing.assert_array_almost_equal(
        np.where(S_u == S_u.max()), np.where(S_ur == S_ur.max())
    )
    np.testing.assert_array_almost_equal(
        np.where(S_ur == S_ur.max()), np.where(gamma == gamma.max())
    )

    # Check that the correllation coefficients are all equal
    # Note: this is only valid where 'corr = 1'. For any other value of
    # 'corr', these tests will ned to be updated to calculate the actual
    # corrcoef. It might be best t factor the `covs` calulation into a
    # seperate function in this case.
    np.testing.assert_array_almost_equal(
        np.corrcoef(S_u, S_ur), np.corrcoef(S_u, gamma), decimal=3
    )
    np.testing.assert_array_almost_equal(
        np.corrcoef(S_ur, gamma), np.corrcoef(S_ur, S_u), decimal=3
    )
    np.testing.assert_array_almost_equal(
        np.corrcoef(gamma, S_u), np.corrcoef(gamma, S_ur), decimal=3
    )


def test_get_soil_dist():

    S_u_mean = 0.22
    S_u_std = 0.13
    S_ur_mean = 0.11
    S_ur_std = 0.03
    gamma_mean = 3.92
    gamma_std = 0.53
    corr = 1
    n = 10_000

    S_u = {"mean": S_u_mean, "std_dev": S_u_std, "min": 0}
    S_ur = {"mean": S_ur_mean, "std_dev": S_ur_std, "min": 0}
    gamma = {"mean": gamma_mean, "std_dev": gamma_std, "min": 0}

    S_u_dist, S_ur_dist, gamma_dist = psi.get_soil_dist(S_u, S_ur, gamma, corr, n)

    assert S_u_dist.size == S_ur_dist.size == gamma_dist.size == n

    assert np.min(S_u_dist) >= S_u["min"]
    assert np.min(S_ur_dist) >= S_ur["min"]
    assert np.min(gamma_dist) >= gamma["min"]
