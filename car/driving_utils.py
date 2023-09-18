import numpy as np
from scipy.stats import chi2
from jax.scipy.stats import norm
import matplotlib.patches as patches

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