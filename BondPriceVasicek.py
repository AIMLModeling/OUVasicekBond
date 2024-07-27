import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
np.random.seed(seed=42)

# Parameters and Time Vector Initialization
N = 20000  # time steps
paths = 5000  # number of paths
T = 5
T_vec, dt = np.linspace(0, T, N, retstep=True)

# Vasicek Model Parameters
kappa = 3                # mean reversion coefficient
theta = 0.5              # long term mean
sigma = 0.5              # volatility coefficient
std_asy = np.sqrt(sigma**2 / (2 * kappa))  # asymptotic standard deviation

# Simulation of Ornstein-Uhlenbeck (OU) Paths
X0 = 2
X = np.zeros((N, paths))
X[0, :] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
for t in range(0, N - 1):
    X[t + 1, :] = theta + np.exp(-kappa * dt) * (X[t, :] - theta) + std_dt * W[t, :]
    
# Vasicek Bond Pricing
B = 1 / kappa * (1 - np.exp(-kappa * T))
A = np.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 / (4 * kappa) * B**2)
P = A * np.exp(-B * X0)
print("Vasicek bond price: ", P)

# Monte Carlo Simulation for Bond Pricing
disc_factor = np.exp(-X.mean(axis=0) * T)
P_MC = np.mean(disc_factor)
st_err = ss.sem(disc_factor)
print(f"Vasicek bond price by MC: {P_MC} with std error: {st_err}")

# PDE Method for Bond Pricing
Nspace = 6000  # M space steps
Ntime = 6000  # N time steps
r_max = 3  # A2
r_min = -0.8  # A1
r, dr = np.linspace(r_min, r_max, Nspace, retstep=True)  # space discretization
T_array, Dt = np.linspace(0, T, Ntime, retstep=True)  # time discretization
Payoff = 1  # Bond payoff

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms
V[:, -1] = Payoff  # terminal conditions
V[-1, :] = np.exp(-r[-1] * (T - T_array))  # lateral boundary condition
V[0, :] = np.exp(-r[0] * (T - T_array))  # lateral boundary condition

# Calculate the coefficients for the tri-diagonal matrix (`D`)
sig2 = sigma * sigma
drr = dr * dr
max_part = np.maximum(kappa * (theta - r[1:-1]), 0)  # upwind positive part
min_part = np.minimum(kappa * (theta - r[1:-1]), 0)  # upwind negative part

a = min_part * (Dt / dr) - 0.5 * (Dt / drr) * sig2
b = 1 + Dt * r[1:-1] + (Dt / drr) * sig2 + Dt / dr * (max_part - min_part)
c = -max_part * (Dt / dr) - 0.5 * (Dt / drr) * sig2

a0 = a[0]
cM = c[-1]  # boundary terms
aa = a[1:]
cc = c[:-1]  # upper and lower diagonals
D = sparse.diags([aa, b, cc], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()  # matrix D

# Perform backward computation using the PDE method to solve for bond prices
for n in range(Ntime - 2, -1, -1):
    offset[0] = a0 * V[0, n]
    offset[-1] = cM * V[-1, n]
    V[1:-1, n] = spsolve(D, (V[1:-1, n + 1] - offset))

# Bond Price Calculation and Visualization
# finds the bond price with initial value X0
Price = np.interp(X0, r, V[:, 0])
print("The Vasicek Bond price by PDE is: ", Price)
fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

ax1.text(X0, 0.06, "Bond Price")
ax1.plot([X0, X0], [0, Price], "k--")
ax1.plot([-0.5, X0], [Price, Price], "k--")
ax1.plot(r, V[:, 0], color="red", label="Initial Bond Price Curve")
ax1.set_xlim(-0.4, 2.5)
ax1.set_ylim(0.025, 0.12)
ax1.set_xlabel("Interest Rate")
ax1.set_ylabel("Bond Price")
ax1.legend(loc="upper right")
ax1.set_title("Vasicek bond price at t=0")

X_plt, Y_plt = np.meshgrid(T_array, r[700:-200])  # consider [700:-200] to remove lateral boundary effects
ax2.plot_surface(Y_plt, X_plt, V[700:-200])
ax2.set_title("Vasicek bond price surface")
ax2.set_xlabel("Interest Rate")
ax2.set_ylabel("Time")
ax2.set_zlabel("Bond Price")
plt.show()
