from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt

OD_mean = 0.1683
OD_std = 0.01
OD_min = OD_mean * 0.95
OD_max = OD_mean * 1.1

wt_mean = 0.0127
wt_std = 0.001
wt_min = wt_mean * 0.95
wt_max = wt_mean * 1.1

W_min = 460
W_max = 520

n = 10_000_000

def get_truncnorm(mean, std, min, max, size):
    a, b = (min - mean) / std, (max - mean) / std
    return truncnorm.rvs(a, b, mean, std, size=size)

def get_A(OD, wt):
    return np.pi * (OD**2 - (OD-2*wt)**2) / 4

def get_W(OD, wt):
    rho = 7850
    g = 9.80665
    A = get_A(OD, wt)
    return A * rho * g

trunc_OD = np.empty(n)
trunc_wt = np.empty(n)
trunc_W = np.empty(n)

while True:
    zeros = np.argwhere(trunc_W==0)
    if zeros.size == 0:
        break
    
    first_zero = zeros[0,0]
    
    ODs = get_truncnorm(OD_mean, OD_std, OD_min, OD_max, n - first_zero)
    wts = get_truncnorm(wt_mean, wt_std, wt_min, wt_max, n - first_zero)
    Ws = get_W(ODs, wts)
    
    idx = (Ws>=W_min) * (Ws<=W_max)
    
    ODs = np.delete(ODs, ~idx)
    wts = np.delete(wts, ~idx)
    Ws = np.delete(Ws, ~idx)
    
    end_idx = Ws.size + first_zero
    print(f"Number of valid weights: {end_idx}")
    
    trunc_OD[first_zero:end_idx] = ODs
    trunc_wt[first_zero:end_idx] = wts
    trunc_W[first_zero:end_idx] = Ws

plt.clf()
plt.hist(trunc_W, density=True, bins=1000)
plt.title("Weights")
plt.show()

plt.clf()
plt.hist(trunc_OD, density=True, bins=1000)
plt.title("ODs")
plt.show()

plt.clf()
plt.hist(trunc_wt, density=True, bins=1000)
plt.title("Wall Thicknesses")
plt.show()