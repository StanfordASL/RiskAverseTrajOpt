import jax.numpy as jnp
# optimizer parameters
OSQP_POLISH = True
OSQP_TOL = 1e-3
# state-control dimensions
n_x = 6 # (px, py, pz, vx, vy, vz) - number of state variables
n_u = 3 # (ux, uy, uz) - number of control variables\
# time problem constants
S = 20 # number of control switches
M = 50 # number of samples
T = 50.0 # max final time horizon in sec
dt = T / S
R = jnp.eye(n_u) # cost term
feedback_gain = jnp.zeros((n_u, n_x))
feedback_gain = feedback_gain.at[:, :3].set(
    0.05 * jnp.eye(n_u))
feedback_gain = feedback_gain.at[:, 3:].set(
    0.25 * jnp.eye(n_u))
feedback_gain = -feedback_gain
# constants
u_max = 10
mass_nom = 32.0
mass_delta = 3
beta = 1e-2 # diffusion term magnitude
drag_coefficient = 0.2 
# obstacle - mean parameters
# obstacles are represented as ellipsoids:
# p \in obstacle
#      <=>
# (p-obs_position).T @ Q @ (p-obs_position) <= 1
# where Q = [1 / length_x**2, 0]
#           [0, 1 / length_y**2]
obs_positions = jnp.array([
    [-1.4, -0.1, 0],
    [-0.7, 0.3, 0],
    [-0.3, 0.25, 0]])
obs_radii = jnp.array([
    0.3,
    0.2,
    0.2])
obs_radii_deltas = 0.025
n_obs = obs_positions.shape[0] # number of obstacles
# initial and final conditions
x_init = jnp.array([-1.9, 0.05, 0.2, 0, 0, 0])
x_final = jnp.zeros(n_x)