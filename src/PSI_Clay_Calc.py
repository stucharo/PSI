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
def Q_v(gamma, D, z, S_u):
    a = np.min(((6 * (z / D) ** 0.25), 3.4 * (10 * z / D) ** 0.5))
    return (a + (1.5 * (gamma * Abm(D, z) / D * S_u))) * D * S_u


def W_sub(D, ID, rho_steel, rho_conc, rho_coat, rho_sw, t_coat, t_conc, rho_cont):
    W_cont = A(ID, 0) * rho_cont * g
    W_steel = A(D, ID) * rho_steel * g
    W_coat = A(D + 2 * t_coat, D) * rho_coat * g
    W_conc = A(D + 2 * t_coat + 2 * t_conc, D + 2 * t_coat) * rho_conc * g
    B = A(D + 2 * t_coat + 2 * t_conc, 0) * rho_sw * g
    return (W_cont + W_steel + W_conc + W_coat - B) / 1000


def A(OD, ID=0):
    return np.pi * (OD**2 - ID**2) / 4


def Abm(D, z):
    for i in range(len(z)):
        if z[i] < D[i] / 2:
            return (
                np.arcsin(B(D[i], z[i]) / D[i]) * ((D[i] ** 2) / 4)
                - (B(D[i], z[i]) * D[i] * np.cos(np.arcsin(B(D[i], z[i]) / D[i]))) / 4
            )
        else:
            return (np.pi * (D[i] ** 2) / 8) + D * (z[i] - (D[i] / 2))


def B(D, z):
    return 2 * (((z * D) - (z**2)) ** 0.5)


def k_lay(gamma, D, z, S_ur, T_0):
    _Q_V = Q_v(gamma, D, z, S_ur)
    return 0.6 + 0.4 * ((_Q_V * EI) / ((T_0**2) * z)) ** 0.25


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
        df["k_lay2"] = k_lay(gamma, D, z, S_ur, T_0)
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
    return (2 * np.sin(beta(z, D))) / (
        beta(z, D) + (np.sin(beta(z, D)) * np.cos(beta(z, D)))
    )


def beta(z, D):
    for i in range(len(z)):
        return np.min((np.pi / 2, np.arccos(1 - (2 * z[i]) / D[i])))


def OCR(W_hydro, W_case):
    OCR = W_hydro / W_case
    for i in range(len(W_hydro)):
        if OCR[i] > 1:
            return OCR
        else:
            return 1


def ax_res(axbr, E_res):
    return axbr * E_res


def lat_br(z, D, Q_v, S_u, gamma, W_case):
    return (
        (1.7 * ((z / D) ** 0.61))
        + (0.23 * (Q_v / (S_u * D)) ** 0.83)
        + (0.6 * (gamma * D / S_u) * (z / D) ** 2) * S_u * D
    ) / W_case


def lat_res(z, D, W_case):
    return (0.32 + (0.8 * (z / D) ** 0.8)) / W_case


def create_fig(data, title, bins=100):
    plt.hist(data, bins=bins)
    plt.title(f"{title}")
    plt.savefig(f"{title}.png")
    plt.clf()


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
            (0.001 - np.array(idf["gamma_mean"])) / np.array(idf["gamma_std"]),
            (10 - np.array(idf["gamma_mean"])) / np.array(idf["gamma_std"]),
            np.array(idf["gamma_mean"]),
            np.array(idf["gamma_std"]),
            idf["n"],
        )
    )
    S_u = m[:, 0]
    S_u = np.array(
        truncnorm.rvs(
            (0.001 - np.array(idf["S_u_mean"])) / np.array(idf["S_u_std"]),
            (10 - np.array(idf["S_u_mean"])) / np.array(idf["S_u_std"]),
            np.array(idf["S_u_mean"]),
            np.array(idf["S_u_std"]),
            idf["n"],
        )
    )
    S_ur = m[:, 1]
    S_ur = np.array(
        truncnorm.rvs(
            (0.001 - np.array(idf["S_ur_mean"])) / np.array(idf["S_ur_std"]),
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

    df["k_lay2"] = k_lay(gamma, D, z, S_ur, T_0)
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
