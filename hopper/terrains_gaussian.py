import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)

np.random.seed(0)

M = 10
mu_nom = 0.10
num_mu_features = 30
intensities = np.random.uniform(0, 1, (M, num_mu_features))
intensities = np.sqrt(2 / num_mu_features) * intensities
intensities = 0.025 * intensities
thetas = np.random.uniform(0, np.pi, (M, num_mu_features))
taus = np.random.uniform(0, 2*np.pi, (M, num_mu_features))

def friction_at_px(position_x, 
    intensity_vec, theta_vec, tau_vec):
    fns = intensity_vec * np.cos(theta_vec * position_x + tau_vec)
    mu = mu_nom + np.sum(fns)
    return mu

positions = np.linspace(-0.5, 2, 100)
plt.figure()
for i in range(M):
    mus_i = np.zeros(len(positions))
    for t in range(len(positions)):
        mus_i[t] = friction_at_px(positions[t], 
            intensities[i], thetas[i], taus[i])
    plt.plot(positions, mus_i)
plt.xlabel(r'$p_{x}(q)$', 
    fontsize=24)
plt.ylabel(r'$\mu(\cdot, \omega)$', 
    fontsize=24, rotation=0, labelpad=30)
plt.ylim((0.05, 0.15))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yticks(
    [0.05, 0.10, 0.15], 
    [0.05, 0.10, 0.15])
plt.grid()
plt.show()