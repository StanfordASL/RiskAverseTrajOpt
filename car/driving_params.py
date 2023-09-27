import jax.numpy as jnp
# optimizer parameters
OSQP_POLISH = True
OSQP_TOL = 3e-4
# state-control dimensions
n_x = 8 # number of state variables
        # (px_e, py_e, v_e, phi_e, 
        #  px_ped, py_ped, vx_ped, vy_ped)
n_u = 2 # (a, omega) - number of control variables
# time problem constants
S = 20 # number of control switches
M = 50 # number of samples
T = 10.0 # max final time horizon in sec
dt = T / S # discretization time
R = jnp.diag(jnp.array([1.0, 1. / 3.0])) # control penalization
# constants
u_max = 100
omega_speed_nom = 0.1
omega_speed_del = 0.075 
omega_repulsive_nom = 0.05
omega_repulsive_del = 0.045
# minimal separation distance
ego_width = 2.695 # length of Smart car
ego_height = 1.663 # width of Smart car
ped_radius = 0.5
min_separation_distance = (ped_radius + 
    jnp.sqrt(ego_width**2 + ego_height**2))
# initial conditions
speed_ped_des = 1.3 # desired speed of pedestrian
speed_ego_init = 4 # initial car speed
position_ego_init = jnp.array([-20., 0.])
position_ped_init = jnp.array([0., -6.])
velocity_ego_init = jnp.array([speed_ego_init, 0.])
velocity_ped_init = jnp.array([0., speed_ped_des])
position_ego_goal = jnp.array([20., 0.1])
velocity_ego_goal = jnp.array([4.1, 0.])
state_init = jnp.concatenate((position_ego_init,
    velocity_ego_init,
    position_ped_init, 
    velocity_ped_init), axis=-1)
variance_ped_initial_state = jnp.diag(
    jnp.array([1e-1, 1e-1, 1e-4, 1e-4])**2)