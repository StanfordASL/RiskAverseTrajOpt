import numpy as np
from scipy.stats import chi2
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.patches as patches

import drone_params

def p_th_quantile_cdf_normal(probability):
    return norm.ppf(probability)

def p_th_quantile_chi_squared(probability, n_dofs):
    return chi2.ppf(probability, n_dofs)

def plot_ellipse(ax, mu, Q, additional_radius=0., color='blue', alpha=0.1):
    # Based on
    # http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(Q)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    # Eigenvalues give length of ellipse along each eigenvector
    w, h =  2. * (np.sqrt(vals) + additional_radius)
    ellipse = patches.Ellipse(mu, w, h, theta, color=color, alpha=alpha)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ax.add_artist(ellipse) 
    
def plot_gaussian_confidence_ellipse(
    ax, mu, Sigma, 
    probability=0.9):
    n_dofs = mu.shape[0]
    chi_squared_quantile_val = chi2.ppf(probability, n_dofs)
    Q = p_th_quantile_chi_squared(probability, n_dofs) * Sigma
    plot_ellipse(ax, mu, Q)

# Load drone model parameters
OSQP_POLISH = drone_params.OSQP_POLISH
OSQP_TOL = drone_params.OSQP_TOL
n_x = drone_params.n_x
n_u = drone_params.n_u
S = drone_params.S
# M = drone_params.M
T = drone_params.T
dt = drone_params.dt
R = drone_params.R
feedback_gain = drone_params.feedback_gain
u_max = drone_params.u_max
mass_nom = drone_params.mass_nom
mass_delta = drone_params.mass_delta
beta = drone_params.beta
drag_coefficient = drone_params.drag_coefficient
obs_positions = drone_params.obs_positions
obs_radii = drone_params.obs_radii
obs_radii_deltas = drone_params.obs_radii_deltas
n_obs = drone_params.n_obs
x_init = drone_params.x_init
x_final = drone_params.x_final

def sample_uncertain_parameters(method='saa', M=100, S=S, dt=dt):
    if method == 'saa':
        # mass of the system
        masses = np.random.uniform(
            mass_nom - mass_delta,
            mass_nom + mass_delta, M)
        # semi axes of the obstacles
        obs_Qs = np.zeros((M, n_obs, 3, 3))
        for obs_i in range(n_obs):
            for dim in range(3):
                obs_delta_r = np.random.uniform(
                    -obs_radii_deltas,
                    obs_radii_deltas, M)
                for i in range(M):
                    length = obs_radii[obs_i] + obs_delta_r[i]
                    obs_Qs[i, obs_i, dim, dim] = 1. / length**2
    if method == 'baseline':
        masses = np.random.uniform(
            mass_nom - 0 * mass_delta,
            mass_nom + 0 * mass_delta, M)
        obs_Qs = np.zeros((M, n_obs, 3, 3))
        for obs_i in range(n_obs):
            for dim in range(3):
                length = obs_radii[obs_i]
                obs_Qs[:, obs_i, dim, dim] = 1. / length**2
    # Brownian motion increments
    DWs = np.zeros((M, S, n_x))
    for i in range(M):
        for t in range(S):
            DWs[i, t, :] = np.sqrt(dt) * np.random.randn(n_x)
    if method == 'baseline':
        DWs = 0 * DWs
    return DWs, masses, obs_Qs