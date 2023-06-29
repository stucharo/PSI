# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:06:57 2023

@author: Laston
"""
import pathlib

import numpy as np
import pandas as pd
import time
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

input_file = pathlib.Path(__file__).parent.resolve() / "INPUT_FILE.csv"

idf = pd.read_csv(input_file)
idf["D_std"] = idf["D_Coeff_of_Var"] * idf["D_mean"]
idf["WT_std"] = idf["WT_Coeff_of_Var"] * idf["WT_mean"]
g = 9.81
rho_cont = np.array(idf.loc[0, "rho_cont"])
rho_steel = np.array(idf.loc[0, "rho_steel"])
rho_conc = np.array(idf.loc[0, "rho_conc"])
rho_coat = np.array(idf.loc[0, "rho_coat"])
rho_sw = np.array(idf.loc[0, "rho_sw"])
rho_cont_hydro = np.array(idf.loc[0, "rho_cont_hydro"])
t_coat = np.array(idf.loc[0, "t_coat"])
t_conc = np.array(idf.loc[0, "t_conc"])
S_u_grad = np.array(idf.loc[0, "S_u_grad"])
S_u_std_grad = np.array(idf.loc[0, "S_u_std_grad"])
S_ur_grad = np.array(idf.loc[0, "S_ur_grad"])
S_ur_mean = np.array(idf.loc[0, "S_ur_mean"])
S_ur_std = np.array(idf.loc[0, "S_ur_std"])
S_u_mean = np.array(idf.loc[0, "S_u_mean"])
S_u_std = np.array(idf.loc[0, "S_u_std"])
S_ur_std_grad = np.array(idf.loc[0, "S_ur_std_grad"])
EI = np.array(idf.loc[0, "EI"])
T_0_mean = np.array(idf.loc[0, "T_0_mean"])
T_0_std = np.array(idf.loc[0, "T_0_std"])
alpha = np.array(idf["alpha"])
g_rate = np.array(idf["g_rate"])
E_res = np.array(idf["E_res"])
idf["n"] = 1000000

# Vertical contact force to embed pipe in clay
def Q_v(gamma_dash, D, z, S_u_mudline, S_u_gradient):
    """
    Calculate the vertical force, Qv, required to penetrate the pipe to the embedment, z,
    assuming linear increase in shear strength with depth. This reflects the Model 2 approach
    presented in Eq. 4.8 of DNV-RP-F114 (2021).

    Parameters
    ----------
    gamma_dash : float | np.ndarray
        Soil submerged unit weight (N/m^3)
    D : float | np.ndarray
        pipe outside diameter including coating (m)
    z : float | np.ndarray
        pipe embedment (m)
    S_u_mudline : float | np.ndarray
        soil undrained shear strength at the mudline (Pa)
    S_u_gradient : float | np.ndarray
        soil undrained shear strength gradient (Pa * m^-1)

    Returns
    -------
    Q_v : float | np.ndarray
        Vertical force required to achieve penetration `z` (N*m^-1)
    """
    a = np.minimum(6 * (z / D) ** 0.25, 3.4 * (10 * z / D) ** 0.5)
    _S_u = S_u(z, S_u_mudline, S_u_gradient)
    return (a + (1.5 * gamma_dash * Abm(D, z) / D / _S_u)) * D * _S_u


def S_u(z, S_u_mudline, S_u_gradient):
    """
    Determine the soil shear strength at a given depth `z`.

    Parameters
    ----------
    z : float | np.ndarray
        pipe penetration (m)
    S_u_mudline : float | np.ndarray
        soil undrained shear strength at the mudline (Pa)
    S_u_gradient : float | np.ndarray
        soil undrained shear strength gradient (Pa * m^-1)

    Returns
    -------
    S_u : float | np.ndarray
        soil undrained shear strength at depth `z` (Pa)
    """
    return S_u_mudline + S_u_gradient * z


def W_sub(
    OD, ID, rho_steel, rho_conc, rho_coat, rho_sw, t_coat, t_conc, rho_cont, g=9.80665
):
    """
    Calculate submerged weight of pipe.

    Although the units of inputs are provided in kg-m-s units, the return value is provided
    in kN.

    TODO: understand why we need to change units halfway through, why can't we keep the whole
    model in base units?

    PARAMETERS
    ----------
    OD : float | np.ndarry
        Steel outer diameter (m)
    ID : float | nd.ndarray
        Steel innder diameter (m)
    rho_steel : float | np.ndarry
        Steel density (kg*m^-3)
    rho_conc : float | np.ndarry
        Concrete density (kg*m^-3)
    rho_coat : float | np.ndarry
        Coating density (kg*m^-3)
    rho_sw : float | np.ndarry
        Seawater density (kg*m^-3)
    t_coat : float | np.ndarry
        Thickness of coating layer (m)
    t_conc : float | np.ndarry
        Thickness of concrete wight coating layer (m)
    rho_cont : float | np.ndarry
        Contents density (kg*m^-3)
    g : float (optional)
        Gravitational acceleration (m*s^-2)

    Returns
    -------
    W_sub : float | np.ndarry
        Submerged weight of pipe (kN*m^-1)
    """
    # contents weight
    W_cont = _cylinder_weight(ID, 0, rho_cont)
    # steel weight
    W_steel = _cylinder_weight(OD, ID, rho_steel)
    # Corrosion coating weight
    W_coat = _cylinder_weight(OD + 2 * t_coat, OD, rho_coat)
    # Concrete weight coating weight
    W_conc = _cylinder_weight(OD + 2 * t_coat + 2 * t_conc, OD + 2 * t_coat, rho_conc)
    # Bouyancy
    B = _cylinder_weight(OD + 2 * t_coat + 2 * t_conc, 0, rho_sw)
    # Combined and convert to kN
    return (W_cont + W_steel + W_conc + W_coat - B) / 1000


def _cylinder_weight(OD, ID, rho, g=9.80665):
    return A(OD, ID) * rho * g


def A(OD, ID=0):
    """
    Calculate area of a circle.

    Returns the area of a ring or solid circle of ID = 0.

    Parameters
    ----------
    OD : float
    ID : float, optional
        The inner diameter of a ring (m)

    Returns
    -------
    area : float
        The area of the circle or ring (m^2)
    """
    return np.pi * (OD**2 - ID**2) / 4


def Abm(D, z):
    """
    Calculates the penetrated cross-sectional area of the pipe (Abm) using Eq. 4.7 in DNV-RP-F114 (2021)

    Parameters
    ----------
    D : np.ndarray
        overall diameter of pipe (m)
    z : np.ndarray
        pipe penetration (m)

    Results
    -------
    Abm : np.ndarray
        penetrated cross-sectional area of the pipe (m)
    """
    # TODO: find a more efficient way to solve this that doesn't involve calculating both arrays
    return np.where(
        z < D / 2, _Abm_z_less_than_D_over_2(D, z), _Abm_z_greater_than_D_over_2(D, z)
    )


def _Abm_z_less_than_D_over_2(D, z):
    """
    Private helper function for Abm where z < D / 2.

    Parameters
    ----------
    D : np.ndarray
        overall diameter of pipe (m)
    z : np.ndarray
        pipe penetration (m)

    Results
    -------
    Abm : np.ndarray
        penetrated cross-sectional area of the pipe when z < D / 2 (m)
    """
    _B = B(D, z)
    asin_B_D = np.arcsin(_B / D)
    return asin_B_D * D**2 / 4 - _B * D / 4 * np.cos(asin_B_D)


def _Abm_z_greater_than_D_over_2(D, z):
    """
    Private helper function for Abm where z >= D / 2.

    Parameters
    ----------
    D : np.ndarray
        overall diameter of pipe (m)
    z : np.ndarray
        pipe penetration (m)

    Results
    -------
    Abm : np.ndarray
        penetrated cross-sectional area of the pipe when z >= D / 2 (m)
    """
    return np.pi * D**2 / 8 + D * (z - D / 2)


def B(D, z):
    """
    Calculate the pipe-soil contact width.

    Calculates the pipe-soil contact width as a functions of z using Eq. 4.3 in DNV-RP-F104 (2021)

    Parameters
    ----------
    D : np.ndarray
        Overall diameter of pipe (m)
    z : np.ndarray
        Penetration (m)

    Returns
    -------
    B : np.ndarray
        Pipe-soil contact width (m)
    """
    return np.where(z < D / 2, 2 * (((z * D) - (z**2)) ** 0.5), D)


def k_lay(gamma_dash, D, z, EI, S_u_mudline, S_u_gradient, T_0):
    """
    Calculates the touchdown lay factor.

    Combines Eq. 4.13 and Eq. 4.14 from DNV-RP-F114 (2021).

    Note that `I` should be calculated from the actual OD and wt for each simulation.

    TODO: calculate I as an array based on OD and wt.
    TODO: check if there should be a distribution of E

    Parameters
    ----------
    gamma_dash : float | np.ndarray
        Soil submerged unit weight at the pipe invert depth (N*m^-1)
    D : float | np.ndarray
        Pipe overall diameter (m)
    z : float | np.ndarray
        pipe penetration (m)
    EI : float | np.ndarray
        Pipe bending stiffness (N*m^2)
    S_u_mudline : float | np.ndarray
        soil undrained shear strength at the mudline (Pa)
    S_u_gradient : float | np.ndarray
        soil undrained shear strength gradient (Pa * m^-1)
    T_o : float | np.ndarray
        Bottom lay tension (N)

    Returns
    -------
    k_lay : float | np.ndarray
        Touchdown lay factpr
    """
    _Q_V = Q_v(gamma_dash, D, z, S_u_mudline, S_u_gradient)
    return np.maximum(1, 0.6 + 0.4 * ((_Q_V * EI) / (T_0**2 * z)) ** 0.25)


def z_inst_numba(compare, z, S_ur, S_ur_grad, S_ur_stds_away):
    for i in range(len(z)):
        if compare[i] != "True":
            z[i] += 0.001
            S_ur[i] = (S_ur_mean + (S_ur_grad * z[i])) + (
                (S_ur_std + (S_ur_std_grad * z[i])) * S_ur_stds_away[i]
            )
    return z, S_ur


def solve_z_inst(df, gamma, D, z, S_ur, T_0, S_ur_stds_away):
    while (df["compare"] == "False").any():
        compare = np.array(df["compare"].values)
        z, S_ur = z_inst_numba(compare, z, S_ur, S_ur_grad, S_ur_stds_away)
        df["S_ur"] = S_ur
        df["k_lay2"] = k_lay(gamma, D, z, EI, S_ur, T_0)
        df["k_lay1"] = Q_v(gamma, D, z, S_ur) / df["W_inst"]
        df.loc[df["k_lay1"] < df["k_lay2"], "compare"] = "False"
        df.loc[df["k_lay1"] >= df["k_lay2"], "compare"] = "True"
        df["z_inst"] = z
    return df


def z_hydro_numba(compare, z, S_u_hydro, S_u_grad, S_u_stds_away):
    for i in range(len(compare)):
        if compare[i] != "True":
            z[i] += 0.001
            S_u_hydro[i] = (S_u_mean + (S_u_grad * z[i])) + (
                (S_u_std + (S_u_std_grad * z[i])) * S_u_stds_away[i]
            )
    return z, S_u_hydro


def solve_z_hydro(df, gamma, D, z, S_u_hydro, S_u_stds_away):
    while (df["compare"] == "False").any():
        compare = np.array(df["compare"].values)
        z, S_u_hydro = z_hydro_numba(compare, z, S_u_hydro, S_u_grad, S_u_stds_away)
        df["Q_v_hydro"] = Q_v(gamma, D, z, S_u_hydro)
        df.loc[df["Q_v_hydro"] < df["W_hydro"], "compare"] = "False"
        df.loc[df["Q_v_hydro"] >= df["W_hydro"], "compare"] = "True"
        df["z_hydro"] = z
        df["S_u_hydro"] = S_u_hydro
        # print(df)
    return df


def z_op_numba(compare, z, S_u_op, S_u_grad, S_u_stds_away):
    for i in range(len(compare)):
        if compare[i] != "True":
            z[i] += 0.001
            S_u_op[i] = (S_u_mean + (S_u_grad * z[i])) + (
                (S_u_std + (S_u_std_grad * z[i])) * S_u_stds_away[i]
            )
    return z, S_u_op


def solve_z_op(df, gamma, D, z, S_u_op, S_u_stds_away):
    while (df["compare"] == "False").any():
        compare = np.array(df["compare"].values)
        z, S_u_op = z_op_numba(compare, z, S_u_op, S_u_grad, S_u_stds_away)
        df["Q_v_op"] = Q_v(gamma, D, z, S_u_op)
        df.loc[df["Q_v_op"] < df["W_op"], "compare"] = "False"
        df.loc[df["Q_v_op"] >= df["W_op"], "compare"] = "True"
        df["z_op"] = z
        df["S_u_op"] = S_u_op
    return df


def ax_br(alpha, csr, Q_v, m, g_rate, z, D, W_hydro, W_case):
    return (
        alpha * csr * (OCR(W_hydro, W_case) ** m) * wedge_factor(z, D) * g_rate
    ) / Q_v


def wedge_factor(z, D):
    """
    Calculates the wedging factor for the beta calc.

    Parameters
    ----------
    z : float | np.ndarray
        Pipe penetration (m)
    D : float | np.ndarray
        Pipe diameter (m)

    Returns
    -------
    wedge_factor : float | np.ndarray
        Pipe diameter (-)
    """
    _beta = beta(z, D)
    return (2 * np.sin(_beta)) / (_beta + (np.sin(_beta) * np.cos(_beta)))


def beta(z, D):
    """
    Calculates beta for use when calculating the wedging factor

    Beta is the angle depicted in Fig. 4-12 of DNV-RP-F114 (2021)

    Parameters
    ----------
    z : float | np.ndarray
        pipe penetration (m)
    D: float | np.ndarray
        overall diameter (m)

    Returns
    -------
    beta : float | np.ndarray
        angle beta from Fig. 4-12 of DNV-RP-F114 (2021)
    """
    return np.where(z <= D / 2, np.arccos(1 - (2 * z) / D), np.pi / 2)


def OCR(W_hydro, W_case):
    """
    Calculate the overconsolidation ratio between the preloading (e.g. water filled)
    case and any other case. This is described as gamma_pre in DNV-RP-F114 (2021)

    TODO: consider renaming this to gamma_pre to reflect DNV-RP-F114 nomenclature.

    Parameters
    ----------
    W_hydro : float | np.ndarry
        Flooded pipe weight (N*m^-3)
    W_case : float | np.ndarry
        Pipe weight in current condition (N*m^-3)

    Returns
    -------
    OCR : float | np.ndarry
        Overconsolidation ratio (-)
    """
    return np.where(W_hydro > W_case, W_hydro / W_case, 1)


def ax_res(axbr, E_res):
    """
    Calculate axial resistance based on Eq. 4.17 in DNV-RP-F114.

    Parameters
    ----------
    axbr : float | np.ndarray
        Axial breakout friction factor (-)
    E_res : float | np.ndarray
        Residual friction factor (-)

    Returns
    -------
    ax_res : float | np.ndarray
        Axial resistance (-)
    """
    return axbr * E_res


def lat_br(z, D, Q_v, S_u, gamma_dash, W_case):
    """
    Calculates the lateral breakout friction factor using Model 2 defined by Eq 4.22
    of DNV-RP-F114 (2012).

    This rearranges the equations as it's written to return the breakout force and then
    divides by the submerged weight to calculate the friction factor.

    Parameters
    ----------
    z : float | np.ndarray
        Pipe pemetration (m)
    D : float | np.ndarray
        Pipe overall diameter (m)
    Q_v : float | np.ndarray
        Static vertical pipe-soil force for the condition considered, e.g. operation (N*m^-1)
    S_u : float | np.ndarray
        Soil undrained shear strength at the pipe invert depth (N*m^-1)
    gamma_dash : float | np.ndarray
        Soil submerged unit weight at the pipe invert depth (N*m^-1)
    W_case : float | np.ndarray
        Pipe unit weight for the condition considered, e.g. operation (m)

    Returns
    -------
    lat_br : float | np.ndarray
        Lateral breakout friction factor (-)
    """
    return (
        (
            (1.7 * ((z / D) ** 0.61))
            + (0.23 * (Q_v / (S_u * D)) ** 0.83)
            + (0.6 * (gamma_dash * D / S_u) * (z / D) ** 2)
        )
        * S_u
        * D
        / W_case
    )


def lat_res(z, D, W_case):
    return (0.32 + (0.8 * (z / D) ** 0.8)) / W_case


def create_fig(title, data, bins=100):
    plt.hist(data, bins=bins)
    plt.title(f"{title}")
    plt.savefig(f"{title}.png")
    plt.clf()


def get_soil_dist(S_u, S_ur, gamma, corr, n):
    """
    Generates correlated distributions of S_u, S_ur and gamma, of size n and with invalid
    values discarded.

    S_u, S_ur and gamma are provided as dicts of the form:
    {
        "mean": float,
        "std_dev": float,
        "min": float,
    }

    This provides all information neccessary to generate correlated arrays with invalid
    values removed i.e. values < 0 and where S_ur > S_u.

    Parameters
    ----------
    S_u : dict
        Dict of soil undrained shear strength  (N*m^-1) (as described above)
    S_ur : dict
        Dict of soil remoulded undrained shear strength  (N*m^-1) (as described above)
    gamma : dict
        Dict of soil submerged weight  (N*m^-3) (as described above)
    corr : float
        Correlation factor for distributions (-)
    n : int
        Number of samples to draw in each distribution

    Returns
    -------
    soil_dists : tuple of np.ndarrays
        Tuple containing arrays of length `n` for `S_u`, `S_ur`, and `gamma` with
        invalid values removed
    """

    # generate correlation arrays
    means = np.array([S_u["mean"], S_ur["mean"], gamma["mean"]])
    std_devs = np.array([S_u["std_dev"], S_ur["std_dev"], gamma["std_dev"]])

    # create empty arrays to hold valid values
    # note that a value of 0 is invalid for any of these arrays
    trunc_S_u = np.zeros(n)
    trunc_S_ur = np.zeros(n)
    trunc_gamma = np.zeros(n)

    while True:
        # check if any zeros are left in the distributions. These are invalid so
        # must be replaced. If no zeros are left then exit the while loop.
        zeros = np.argwhere(trunc_S_u == 0)
        if zeros.size == 0:
            break

        # the first element rpresents the first index that must be replaced
        first_zero = zeros[0, 0]

        # generated multivariate norm arrays of length equal to the number of
        # zeros left in the `trunc_` arrays.
        S_us, S_urs, gammas = mv_norm(means, std_devs, corr, n - first_zero)

        # find all indices where all constraints are satifies across all arrays
        idx = (
            (S_us > S_u["min"])                 # S_u greater than minimum
            * (S_urs > S_ur["min"])             # S_ur greater than minimum
            * (gammas > gamma["min"])           # gamma greater than minimum
            * (S_us > S_urs)                    # S_u greater S_ur
        )

        # remove invalid indices
        S_us = np.delete(S_us, ~idx)
        S_urs = np.delete(S_urs, ~idx)
        gammas = np.delete(gammas, ~idx)

        # add valid values to truncated arrays
        end_idx = S_us.size + first_zero
        trunc_S_u[first_zero:end_idx] = S_us
        trunc_S_ur[first_zero:end_idx] = S_urs
        trunc_gamma[first_zero:end_idx] = gammas

    return trunc_S_u, trunc_S_ur, trunc_gamma


def mv_norm(means: np.ndarray, std_devs: np.ndarray, corr: float, n: int):
    """
    Generates a multi-variate normal distribution with a given correlation between
    distributions.

    Parameters
    ----------
    means : np,ndarray
        Array of means for each ditribution in same order as std_devs
    std_devs : np.ndarray
        Array of std_devs for each ditribution in same order as `means`
    corr : float
        Correlation factor. In this case the same correlation is applied to all
        distributions.
    n : int
        Number of samples to draw in each distribution

    Returns
    -------
    mv_norms : tuple of np.ndarrays
        A tuple containing a normal distribution for each mean and std_dev combination
    """

    # create array of covs
    _c, c_ = np.meshgrid(std_devs, std_devs)
    i = np.identity(std_devs.size)
    covs = _c * c_ * ((~i.astype("bool")).astype("int8") * corr + i)

    # draw n samples for each mean
    mv_norms = np.random.multivariate_normal(means, covs, n)

    # unpack np.ndarry into tuple of arrays for each distribution
    return (mv_norms[:, i] for i in range(means.size))


def mc(idf):
    corr = 1
    covs1 = np.array(
        (
            [
                idf["S_u_std"] ** 2,
                idf["S_u_std"] * idf["S_ur_std"] * corr,
                idf["S_u_std"] * idf["gamma_std"] * corr,
            ]
        )
    )
    covs2 = np.array(
        [
            idf["S_u_std"] * idf["S_ur_std"] * corr,
            idf["S_ur_std"] ** 2,
            idf["S_ur_std"] * idf["gamma_std"] * corr,
        ]
    )
    covs3 = np.array(
        (
            [
                idf["S_u_std"] * idf["gamma_std"] * corr,
                idf["S_ur_std"] * idf["gamma_std"] * corr,
                idf["gamma_std"] ** 2,
            ]
        )
    )
    covs = np.concatenate((covs1, covs2, covs3), axis=1)
    m = np.random.multivariate_normal(
        np.ravel([idf["S_u_mean"], idf["S_ur_mean"], idf["gamma_mean"]]), covs, idf["n"]
    )
    gamma = m[:, 2]
    gamma = np.array(
        truncnorm.rvs(
            (0.05 - np.array(idf["gamma_mean"])) / np.array(idf["gamma_std"]),
            (10 - np.array(idf["gamma_mean"])) / np.array(idf["gamma_std"]),
            np.array(idf["gamma_mean"]),
            np.array(idf["gamma_std"]),
            idf["n"],
        )
    )
    S_u = m[:, 0]
    S_u = np.array(
        truncnorm.rvs(
            (0.05 - np.array(idf["S_u_mean"])) / np.array(idf["S_u_std"]),
            (10 - np.array(idf["S_u_mean"])) / np.array(idf["S_u_std"]),
            np.array(idf["S_u_mean"]),
            np.array(idf["S_u_std"]),
            idf["n"],
        )
    )
    S_ur = m[:, 1]
    S_ur = np.array(
        truncnorm.rvs(
            (0.05 - np.array(idf["S_ur_mean"])) / np.array(idf["S_ur_std"]),
            (10 - np.array(idf["S_ur_mean"])) / np.array(idf["S_ur_std"]),
            np.array(idf["S_ur_mean"]),
            np.array(idf["S_ur_std"]),
            idf["n"],
        )
    )
    D_min = idf.loc[0, "D_mean"] * 0.9925
    D_max = idf.loc[0, "D_mean"] * 1.0075
    D = np.array(
        truncnorm.rvs(
            (D_min - idf["D_mean"]) / idf["D_std"],
            (D_max - idf["D_mean"]) / idf["D_std"],
            idf["D_mean"],
            idf["D_std"],
            idf["n"],
        )
    )
    WT_max = idf.loc[0, "WT_mean"] * 1.125
    WT_min = idf.loc[0, "WT_mean"] * 0.90
    WT = np.array(
        truncnorm.rvs(
            (WT_min - idf["WT_mean"]) / idf["WT_std"],
            (WT_max - idf["WT_mean"]) / idf["WT_std"],
            idf["WT_mean"],
            idf["WT_std"],
            idf["n"],
        )
    )
    rho_op_f = np.random.normal(idf["rho_cont_mean"], idf["rho_cont_std"], idf["n"])
    csr = np.random.normal(idf["csr_mean"], idf["csr_std"], idf["n"])
    m = np.random.normal(idf["ocr_mean"], idf["ocr_std"], idf["n"])
    T_0 = np.random.normal(idf["T_0_mean"], idf["T_0_std"], idf["n"])
    df = pd.DataFrame()
    ID = D - (2 * WT)
    S_ur_stds_away = (S_ur - np.array(idf["S_ur_mean"])) / np.array(idf["S_ur_std"])
    S_u_stds_away = (S_u - np.array(idf["S_u_mean"])) / np.array(idf["S_u_std"])
    df["W_inst"] = W_sub(
        D, ID, rho_steel, rho_conc, rho_coat, rho_sw, t_coat, t_conc, rho_cont=0
    )
    df["W_hydro"] = W_sub(
        D, ID, rho_steel, rho_conc, rho_coat, rho_sw, t_coat, t_conc, rho_cont_hydro
    )
    df["W_op"] = W_sub(
        D, ID, rho_steel, rho_conc, rho_coat, rho_sw, t_coat, t_conc, rho_op_f
    )

    z = (D / D) * 0.001

    df["k_lay2"] = k_lay(gamma, D, z, EI, S_ur, T_0)
    df["k_lay1"] = Q_v(gamma, D, z, S_ur) / df["W_inst"]
    df.loc[df["k_lay1"] < df["k_lay2"], "compare"] = "False"
    df.loc[df["k_lay1"] >= df["k_lay2"], "compare"] = "True"
    df = solve_z_inst(df, gamma, D, z, S_ur, T_0, S_ur_stds_away)
    z = (D / D) * 0.001
    df["Q_v_inst"] = Q_v(gamma, D, np.array(df["z_inst"]), np.array(df["S_ur"]))
    df["axbr_inst"] = ax_br(
        alpha,
        csr,
        np.array(df["Q_v_inst"]),
        m,
        g_rate,
        np.array(df["z_inst"]),
        D,
        np.array(df["W_inst"]),
        np.array(df["W_inst"]),
    )
    df["axres_inst"] = ax_res(np.array(df["axbr_inst"]), E_res)
    df["latbr_inst"] = lat_br(
        np.array(df["z_inst"]),
        D,
        np.array(df["Q_v_inst"]),
        np.array(df["S_ur"]),
        gamma,
        np.array(df["W_inst"]),
    )
    df["latres_inst"] = lat_res(np.array(df["z_inst"]), D, np.array(df["W_inst"]))
    df = df.drop(["compare"], axis=1)

    S_u_hydro = S_u
    df["Q_v_hydro"] = Q_v(gamma, D, z, S_u_hydro)
    df.loc[df["Q_v_hydro"] < df["W_hydro"], "compare"] = "False"
    df.loc[df["Q_v_hydro"] >= df["W_hydro"], "compare"] = "True"
    df = solve_z_hydro(df, gamma, D, z, S_u_hydro, S_u_stds_away)
    if df["z_inst"].mean() > df["z_hydro"].mean():
        df["z_hydro"] = df["z_inst"]
    df["axbr_hydro"] = ax_br(
        alpha,
        csr,
        np.array(df["Q_v_hydro"]),
        m,
        g_rate,
        np.array(df["z_hydro"]),
        D,
        np.array(df["W_hydro"]),
        np.array(df["W_hydro"]),
    )
    df["axres_hydro"] = ax_res(np.array(df["axbr_hydro"]), E_res)
    df["latbr_hydro"] = lat_br(
        np.array(df["z_hydro"]),
        D,
        np.array(df["Q_v_hydro"]),
        np.array(df["S_u_hydro"]),
        gamma,
        np.array(df["W_hydro"]),
    )
    df["latres_hydro"] = lat_res(np.array(df["z_hydro"]), D, np.array(df["W_hydro"]))
    z = (D / D) * 0.001
    df = df.drop(["compare"], axis=1)

    S_u_op = S_u
    df["Q_v_op"] = Q_v(gamma, D, z, S_u_op)
    df.loc[df["Q_v_op"] < df["W_op"], "compare"] = "False"
    df.loc[df["Q_v_op"] >= df["W_op"], "compare"] = "True"
    df = solve_z_op(df, gamma, D, z, S_u_op, S_u_stds_away)
    if df["z_hydro"].mean() > df["z_op"].mean():
        df["z_op"] = df["z_hydro"]
    df["axbr_op"] = ax_br(
        alpha,
        csr,
        np.array(df["Q_v_op"]),
        m,
        g_rate,
        np.array(df["z_op"]),
        D,
        np.array(df["W_hydro"]),
        np.array(df["W_op"]),
    )
    df["axres_op"] = ax_res(np.array(df["axbr_op"]), E_res)
    df["latbr_op"] = lat_br(
        np.array(df["z_op"]),
        D,
        np.array(df["Q_v_op"]),
        np.array(df["S_u_op"]),
        gamma,
        np.array(df["W_op"]),
    )
    df["latres_op"] = lat_res(np.array(df["z_op"]), D, np.array(df["W_op"]))
    df = df.drop(["compare"], axis=1)

    df["gamma"] = gamma
    df["WT"] = WT
    df["T_0"] = T_0
    df["D"] = D
    df["S_ur"] = S_ur
    df["S_u"] = S_u

    idf["W_steel_avg"] = W_sub(
        idf["D_mean"], idf["D_mean"] - (2 * idf["WT_mean"]), rho_steel, 0, 0, 0, 0, 0, 0
    )
    df["W_steel"] = W_sub(D, ID, rho_steel, 0, 0, 0, 0, 0, 0)
    W_steel_min = (np.array(idf.loc[0, "W_steel_avg"] * 0.965)) * (D / D)
    W_steel_max = (np.array(idf.loc[0, "W_steel_avg"] * 1.1)) * (D / D)
    df["W_steel_min"] = W_steel_min
    df["W_steel_max"] = W_steel_max
    df.loc[
        (df["W_steel"] >= df["W_steel_min"]) & (df["W_steel"] <= df["W_steel_max"]),
        "compare",
    ] = True
    df = df[df["compare"] == True]
    df = df.drop(["compare", "W_steel", "W_steel_min", "W_steel_max"], axis=1)

    plots = {
        "Wall Thickness": df["WT"],
        "Bottom Tension": df["T_0"],
        "Gamma": df["gamma"],
        "S_ur": df["S_ur"],
        "S_u": df["S_u"],
        "Pipe Diameter": df["D"],
        "Embedment(install)": df["z_inst"],
        "Axial Breakout(install)": df["axbr_inst"],
        "Axial Residual(install)": df["axres_inst"],
        "Lateral Breakout(install)": df["latbr_inst"],
        "Lateral Residual(install)": df["latres_inst"],
        "Embedment(hydro)": df["z_hydro"],
        "Axial Breakout(hydro)": df["axbr_hydro"],
        "Axial Residual(hydro)": df["axres_hydro"],
        "Lateral Breakout(hydro)": df["latbr_hydro"],
        "Lateral Residual(hydro)": df["latres_hydro"],
        "Embedment(op)": df["z_op"],
        "Axial Breakout(op)": df["axbr_op"],
        "Axial Residual(op)": df["axres_op"],
        "Lateral Breakout(op)": df["latbr_op"],
        "Lateral Residual(op)": df["latres_op"],
    }

    [create_fig(title, data) for title, data in plots.items()]

    return df


if __name__ == "__main__":

    start_time = time.time()
    odf = mc(idf)

    pd.set_option("display.max_columns", None)
    print(odf.describe(percentiles=[0.025, 0.05, 0.5, 0.95, 0.975]))

    end_time = time.time()

    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time} minutes")
