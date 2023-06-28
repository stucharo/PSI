import numpy as np

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


def mv_norm(means: np.ndarray, std_devs: np.ndarray, corr: float, n: int):

    _c, c_ = np.meshgrid(std_devs, std_devs)
    i = np.identity(std_devs.size)
    covs = _c * c_ * ((~i.astype("bool")).astype("int8") * corr + i)

    return np.random.multivariate_normal(means, covs, n)


mv_norms = mv_norm(means, std_devs, corr, n)
print(mv_norms, end="\n\n")

S_u = mv_norms[:, 0]
S_ur = mv_norms[:, 1]
gamma = mv_norms[:, 2]

print(f"Minimum value in S_u is at index: {np.where(S_u == S_u.min())[0]}")
print(f"Minimum value in S_ur is at index: {np.where(S_ur == S_ur.min())[0]}")
print(f"Minimum value in gamma is at index: {np.where(gamma == gamma.min())[0]}")
print(f"Maximum value in S_u is at index: {np.where(S_u == S_u.max())[0]}")
print(f"Maximum value in S_ur is at index: {np.where(S_ur == S_ur.max())[0]}")
print(f"Maximum value in gamma is at index: {np.where(gamma == gamma.max())[0]}")
print("\n")
print(f"{np.corrcoef(S_u, S_ur) = }")
print(f"{np.corrcoef(S_u, gamma) = }")
print(f"{np.corrcoef(S_ur, gamma) = }")
