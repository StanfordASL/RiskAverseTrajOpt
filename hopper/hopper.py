# The dynamics and numerical values are from
# https://github.com/dojo-sim/ContactImplicitMPC.jl

import numpy as np
import osqp
import ipyopt
import scipy.sparse as sp
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rc
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)
import jax.numpy as jnp
from jax import jacrev, grad, hessian, jit, vmap
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

# first, run the baseline.
# then, run the sample average approximation approach.
B_compute_solution_baseline = True
B_compute_solution_saa = False
if B_compute_solution_baseline and B_compute_solution_saa:
    raise ValueError("First run the baseline, then the SAA, but not both.")
B_plot_all_results = True
B_validate_monte_carlo = True
alphas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
alphas_to_plot = [0.05, 0.2, 0.5, 0.75]
np.random.seed(1)
print("---------------------------------------")
print("[hopper.py] >>> Running with parameters:")
print("B_compute_solution_saa      =", B_compute_solution_saa)
print("B_compute_solution_baseline =", B_compute_solution_baseline)
print("B_plot_all_results          =", B_plot_all_results)
print("B_validate_monte_carlo      =", B_validate_monte_carlo)
print("alphas =", alphas)
print("---------------------------------------")

# time problem constants
S = 30 # number of control switches
M = 30 # number of samples
T = 2.0 # max final time horizon in sec
dt = T / S
time_jump = 10
time_land = 20
# state-control dimensions
n_x = 8 # number of state variables
        # (px, pz, phi, r, 
        #  px_dot, pz_dot, phi_dot, r_dot)
n_u = 4 # number of control variables with contact forces lambda
        # (tau, force, 
        #  contact_force_x, contact_force_z)
# number of optimization variables
num_vars = (S+1)*n_x + S*n_u + M + 2
# constants
u_max = 1000
mass_body = 3.0 # body mass (mb)
mass_leg = 0.3  # leg mass (ml)
inertia_body = 0.75 # body inertia (Jb)
inertia_leg = 0.075 # leg inertia (Jl)
gravity = 9.81
max_contact_force = 1000.0 # we assume the leg cannot break
# friction_coefficient
mu_nom = 0.10
num_mu_features = 30
intensities = np.random.uniform(0, 1, (M, num_mu_features))
intensities = np.sqrt(2 / num_mu_features) * intensities
intensities = 0.025 * intensities
thetas = np.random.uniform(0, np.pi, (M, num_mu_features))
taus = np.random.uniform(0, 2*np.pi, (M, num_mu_features))
@jit
def friction_at_px(
    position_x,
    intensity_vec, theta_vec, tau_vec):
    fns = intensity_vec * jnp.cos(theta_vec * position_x + tau_vec)
    mu = mu_nom + jnp.sum(fns)
    return mu
# initial conditions
state_initial = jnp.array([
    1e-6, 1.0, -1e-6, 1.0,
    0., 0., 0., 0.]) + 2e-7
state_final = state_initial
state_final = jnp.array([
    0.15, 1., -1e-6, 1.,
    0., 0., 0., 0.]) + 2e-7
class Model:
    def __init__(self, M, method='baseline', alpha=0.1):
        print("Initializing Model with")
        print("> method =", method)
        print("> alpha  =", alpha)
        self.method = method
        self.alpha = alpha
        if method=='baseline':
            self.intensities = 0 * intensities 
            self.thetas = 0 * thetas 
            self.taus = 0 * taus 
        if method=='saa':
            self.intensities = intensities 
            self.thetas = thetas 
            self.taus = taus

    def convert_z_to_variables(self, z):
        xs_vec = z[:((S+1)*n_x)]
        us_vec = z[((S+1)*n_x):(((S+1)*n_x)+S*n_u)]
        ys_risk = z[(((S+1)*n_x)+S*n_u):-2]
        slack_var = z[-2]
        t_risk = z[-1]
        return xs_vec, us_vec, ys_risk, slack_var, t_risk

    def convert_z_to_xs_us_mats(self, z):
        xs_vec, us_vec, _, _, _ = self.convert_z_to_variables(z)
        xs_mat = self.convert_xs_vec_to_xs_mat(xs_vec)
        us_mat = self.convert_us_vec_to_us_mat(us_vec)
        return xs_mat, us_mat

    def convert_xs_vec_to_xs_mat(self, xs_vec):
        xs_mat = jnp.reshape(xs_vec, (n_x, S+1), 'F')
        xs_mat = xs_mat.T # (S+1, n_x)
        xs_mat = jnp.array(xs_mat)
        return xs_mat

    def convert_us_vec_to_us_mat(self, us_vec):
        us_mat = jnp.reshape(us_vec, (n_u, S), 'F')
        us_mat = us_mat.T # (S, n_u)
        us_mat = jnp.array(us_mat)
        return us_mat

    def convert_us_mat_to_us_jaxvec(self, us_mat):
        us_vec = jnp.reshape(us_mat, (S*n_u), 'C')
        return us_vec

    def initial_guess(self):
        Zp = np.zeros((S+1)*n_x + S*n_u + M + 2)
        for t in range(time_land):
            idx_x = t*n_x
            Zp[idx_x:(idx_x+n_x)] = state_initial
        for t in range(time_land, S+1):
            idx_x = t*n_x
            Zp[idx_x:(idx_x+n_x)] = state_final
        u_initial_guess = 1e-5
        nominal_force = (mass_body + mass_leg) * gravity
        for t in range(S):
            idx_u = (S+1)*n_x + t*n_u
            Zp[idx_u+0] = 0
            Zp[idx_u+1] = 0
        for t in range(0, time_jump):
            idx_u = (S+1)*n_x + t*n_u
            Zp[idx_u+1] = nominal_force
            Zp[idx_u+2] = 0
            Zp[idx_u+3] = nominal_force
        for t in range(time_jump, time_land):
            idx_u = (S+1)*n_x + t*n_u
            Zp[idx_u+2] = 0#1e-6)
            Zp[idx_u+3] = 0
        for t in range(time_land, S):
            idx_u = (S+1)*n_x + t*n_u
            Zp[idx_u+1] = nominal_force
            Zp[idx_u+2] = 0
            Zp[idx_u+3] = nominal_force
        return np.array(Zp)

    @partial(jit, static_argnums=(0,))
    def end_effector_position(self, x):
        p_ee = jnp.array([
            x[0] + x[3] * jnp.sin(x[2]),
            x[1] - x[3] * jnp.cos(x[2])])
        return p_ee

    @partial(jit, static_argnums=(0,))
    def jacobian_end_effector_position(self, x):
        J_ee = jnp.array([
            [1., 0., x[3] * jnp.cos(x[2]), jnp.sin(x[2])],
            [0., 1., x[3] * jnp.sin(x[2]), -jnp.cos(x[2])]
            ])
        return J_ee

    @partial(jit, static_argnums=(0,))
    def M_inertia_matrix(self, x):
        M_vec = jnp.array([
            mass_body + mass_leg,
            mass_body + mass_leg,
            inertia_body + inertia_leg,
            mass_leg])
        M_mat = jnp.diag(M_vec)
        return M_mat

    @partial(jit, static_argnums=(0,))
    def M_inertia_matrix_inverse(self, x):
        Minv_vec = jnp.array([
            1. / (mass_body + mass_leg),
            1. / (mass_body + mass_leg),
            1. / (inertia_body + inertia_leg),
            1. / mass_leg])
        Minv_mat = jnp.diag(Minv_vec)
        return Minv_mat

    @partial(jit, static_argnums=(0,))
    def C_Coriolis_conservative_forces(self, x):
        C_vec = jnp.array([0.0,
            (mass_body + mass_leg) * gravity,
            0.0,
            0.0])
        return C_vec

    @partial(jit, static_argnums=(0,))
    def B_control_matrix(self, x):
        B_mat = jnp.array([
            [0., 0., 1., 0.],
            [-jnp.sin(x[2]), jnp.cos(x[2]), 0., 1.]
            ]).T
        return B_mat

    @partial(jit, static_argnums=(0,))
    def b(self, x, u):
        q, q_dot = x[:4], x[4:]
        u_robot, contact_forces = u[:2], u[2:]
        # acceleration
        M_inv = self.M_inertia_matrix_inverse(x)
        C = self.C_Coriolis_conservative_forces(x)
        B = self.B_control_matrix(x)
        J = self.jacobian_end_effector_position(x)
        q_ddot = M_inv @ (
            -C +
            B @ u_robot +
            J.T @ contact_forces)
        b_vec = jnp.concatenate((q_dot, q_ddot), axis=-1)
        return b_vec


    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### costs and constraints        
    def dynamics_constraints(self, Z):
        def dynamic_constraint(x, u, xn):
            # eq = xn - (x + self.b(x, u) * dt)
            # return eq
            k1 = self.b(x,             u)
            k2 = self.b(x + 0.5*dt*k1, u)
            k3 = self.b(x + 0.5*dt*k2, u)
            k4 = self.b(x + dt*k3,     u)
            eq_rk4 = xn - (x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4) * dt)
            return eq_rk4
        xs, us = self.convert_z_to_xs_us_mats(Z) # (S+1,n_x), (S,n_u)
        Xs = xs[:-1, :]
        Us = us
        Xns = xs[1:, :]
        gs = vmap(jit(dynamic_constraint))(Xs, Us, Xns)
        return gs.flatten()

    def initial_constraints(self, Z):
        xs, us = self.convert_z_to_xs_us_mats(Z)
        eq = xs[0, :] - state_initial
        return eq

    def final_constraints(self, Z):
        xs, us = self.convert_z_to_xs_us_mats(Z)
        eq = xs[-1, :] - state_final
        return (eq[4:6])

    def contact_constraints(self, Z):
        xs, us = self.convert_z_to_xs_us_mats(Z)
        ee_positions = vmap(self.end_effector_position)(xs)
        ee_pos_z = ee_positions[:, 1]
        eqs = jnp.concatenate([
            ee_pos_z[:time_jump],
            ee_pos_z[time_land:]])
        return eqs

    def leg_over_ground_constraints(self, Z):
        # leg_z >= 0 => -leg_z <= 0
        # => g(z) <= 0
        xs, us = self.convert_z_to_xs_us_mats(Z)
        ee_positions = vmap(self.end_effector_position)(xs)
        ee_pos_z = ee_positions[:, 1]
        gs = -ee_pos_z[time_jump:time_land]
        return gs

    def no_slip_constraints(self, Z):
        # contact is fixed in x constraint
        xs, us = self.convert_z_to_xs_us_mats(Z)
        # no slip => gamma = 0
        def no_slip_con(x):
            # J_T @ qdot = 0
            J_T = self.jacobian_end_effector_position(x)[0, :]
            q_dot = x[4:]
            constraint = J_T @ q_dot
            return constraint
        cons = vmap(no_slip_con)(xs)
        eqs = jnp.concatenate([
            cons[:time_jump],
            cons[time_land:]])
        return eqs

    def slip_risk_constraints(self, Z):
        variables = self.convert_z_to_variables(Z)
        ys, t_risk = variables[2], variables[4]
        slack_var = variables[3]
        xs_mat, us_mat = self.convert_z_to_xs_us_mats(Z)
        ee_positions_x = vmap(self.end_effector_position)(xs_mat)[:, 0]
        ee_positions_x = jnp.concatenate([
            ee_positions_x[:time_jump],
            ee_positions_x[time_land:-1]], axis=0)
        contact_forces = jnp.concatenate([
            us_mat[:time_jump, 2:],
            us_mat[time_land:, 2:]], axis=0)
        num_contacts = contact_forces.shape[0]

        # no slip cvar constraint
        def no_slip_con(
            position_x, contact_force, 
            intensity_vec, theta_vec, tau_vec):
            friction_coeff = friction_at_px(
                position_x, 
                intensity_vec, theta_vec, tau_vec)
            fx, fz = contact_force
            ineq = fx - friction_coeff * fz
            return ineq
        def no_slip_cons(
            positions_x, contact_forces, 
            intensity_vec, theta_vec, tau_vec):
            num_forces = contact_forces.shape[0]
            intensity_mat = jnp.repeat(
                intensity_vec[jnp.newaxis, :], num_forces, axis=0)
            theta_mat = jnp.repeat(
                theta_vec[jnp.newaxis, :], num_forces, axis=0)
            tau_mat = jnp.repeat(
                tau_vec[jnp.newaxis, :], num_forces, axis=0)
            cons = vmap(no_slip_con)(
                positions_x, contact_forces, 
                intensity_mat, theta_mat, tau_mat)
            return cons # (T)

        if self.method == 'baseline':
            gs = jnp.zeros(M*num_contacts)
            for i in range(M):
                # gij(u) <= 0 for all ij
                constraints = no_slip_cons(
                    ee_positions_x, contact_forces,
                    self.intensities[i], self.thetas[i], self.taus[i])
                idx = i*num_contacts
                gs = gs.at[idx:(idx+num_contacts)].set(
                    constraints - slack_var)

        elif self.method == 'saa':
            gs = jnp.zeros(1 + M + M*num_contacts + 1)

            # (N*alpha)t + sum_{i=1}^M yi <= 0
            gs = gs.at[0].set((M*self.alpha)*t_risk + jnp.sum(ys))

            # -yi <= 0 for all i
            gs = gs.at[1:(1+M)].set(-ys)

            for i in range(M):
                # gi(u) - t - yi <= 0 for all i
                constraints = no_slip_cons(
                    ee_positions_x, contact_forces,
                    self.intensities[i], self.thetas[i], self.taus[i])
                idx = 1 + M + i*num_contacts
                gs = gs.at[idx:(idx+num_contacts)].set(
                    constraints - t_risk - ys[i] - slack_var)
        return gs

    def length_and_speed_constraints(self, Z):
        xs, us = self.convert_z_to_xs_us_mats(Z)
        # length_min <= length <= length_max
        length_min, length_max = 0.25, 1.
        gs = xs[1:, 3]
        gs_l = length_min * jnp.ones(S)
        gs_u = length_max * jnp.ones(S)

        speed_ell_max = 4.
        gs_speed = xs[1:, 7]
        gs_speed_l = -speed_ell_max * jnp.ones(S)
        gs_speed_u = speed_ell_max * jnp.ones(S)

        omega_max = 2.5
        gs_omega = xs[1:, 6]
        gs_omega_l = -omega_max * jnp.ones(S)
        gs_omega_u = omega_max * jnp.ones(S)

        gs = jnp.concatenate([gs, gs_speed, gs_omega])
        gs_l = jnp.concatenate([gs_l, gs_speed_l, gs_omega_l])
        gs_u = jnp.concatenate([gs_u, gs_speed_u, gs_omega_u])
        return gs, gs_l, gs_u

    # @partial(jit, static_argnums=(0,))
    def control_constraints(self, Z):
        xs, us = self.convert_z_to_xs_us_mats(Z)
        # Returns (gs(Z), gs_l, gs_u) corresponding to control constraints
        # such that gs_l <= gs(Z) <= gs_u.
        gs = jnp.zeros(n_u*S)
        gs_l = jnp.zeros(n_u*S)
        gs_u = jnp.zeros(n_u*S)
        # control bounds (motors)
        for i in range(2):
            for t in range(S):
                idx_con = t*n_u + i
                gs = gs.at[idx_con].set(us[t, i])
                gs_l = gs_l.at[idx_con].set(-u_max)
                gs_u = gs_u.at[idx_con].set(u_max)
        # contact forces pushing on the ground
        # (only if on contact, otherwise zero)
        for i in range(2, 4):
            for t in range(0, time_jump):
                    idx_con = t*n_u + i
                    # push on the ground
                    gs = gs.at[idx_con].set(us[t, i])
                    gs_l = gs_l.at[idx_con].set(0)
                    gs_u = gs_u.at[idx_con].set(max_contact_force)
            for t in range(time_jump, time_land):
                    idx_con = t*n_u + i
                    # in the air, impossible to push
                    gs = gs.at[idx_con].set(us[t, i])
                    gs_l = gs_l.at[idx_con].set(0)
                    gs_u = gs_u.at[idx_con].set(0)
            for t in range(time_land, S):
                    idx_con = t*n_u + i
                    # push on the ground
                    gs = gs.at[idx_con].set(us[t, i])
                    gs_l = gs_l.at[idx_con].set(0)
                    gs_u = gs_u.at[idx_con].set(max_contact_force)
        return gs, gs_l, gs_u

    def slack_constraints(self, Z):
        # Returns (gs(Z), gs_l, gs_u) corresponding 
        # to slack variable constraints
        # slack >= 0
        slack_var = Z[-2]
        gs = Z[-2] * jnp.ones(1)
        gs_l = jnp.zeros(1)
        gs_u = 1e6 * jnp.ones(1)
        return gs, gs_l, gs_u

    # Objective to minimize
    def f(self, Z):
        xs, us = self.convert_z_to_xs_us_mats(Z)
        R = 1.0
        obj = 0.
        # control cost
        for t in range(S):
            obj = obj + R*(us[t, 0]*us[t, 0])
            obj = obj + R*(us[t, 1]*us[t, 1])
        # max travel distance variable
        obj = obj - 10000 * xs[-1, 0]
        # slack variable
        obj = obj + 10000000 * Z[-2]
        return obj

# ***************************************

if B_compute_solution_baseline:
    alphas = [0.1]
for alpha in alphas:
    if B_compute_solution_baseline:
        model = Model(M, 'baseline')
        print("---------------------------------------")
        print("[hopper.py] >>> Solving (baseline) for alpha =", alpha)
        Z0 = model.initial_guess()
    elif B_compute_solution_saa:
        model = Model(M, 'saa', alpha)
        print("---------------------------------------")
        print("[hopper.py] >>> Solving (SAA) for alpha =", alpha)

        with open('results/hopper_base_results.npy', 'rb') as f:
            xs_guess = np.load(f)
            us_guess = np.load(f)
        Z0 = np.zeros((S+1)*n_x + S*n_u + M + 2)
        for t in range(S+1):
            idx_x = t*n_x
            Z0[idx_x:(idx_x+n_x)] = xs_guess[t, :]
        for t in range(S):
            idx_u = (S+1)*n_x + t*n_u
            Z0[idx_u:(idx_u+n_u)] = us_guess[t, :]
    else:
        continue


    print(">>> IPOPT: defining stochastic program")
    # Objective to minimize
    def f(Z):
        assert len(Z) == num_vars
        return model.f(Z)
    # Inequality Constraints
    # gL <= g(Z) <= gU
    def g(Z):
        assert len(Z) == num_vars
        gs_dyn = model.dynamics_constraints(Z)
        gs_x0 = model.initial_constraints(Z)
        gs_xf = model.final_constraints(Z)
        gs_slip = model.no_slip_constraints(Z)
        gs_contact = model.contact_constraints(Z)
        gs_over = model.leg_over_ground_constraints(Z)
        gs_slip_risk = model.slip_risk_constraints(Z)
        gs_control, _, _ = model.control_constraints(Z)
        gs_slack, _, _ = model.slack_constraints(Z)
        gs_len, _, _ = model.length_and_speed_constraints(Z)
        gs = jnp.concatenate([
            gs_dyn, 
            gs_x0, 
            gs_xf, 
            gs_slip,
            gs_contact,
            gs_over,
            gs_slip_risk,
            gs_control,
            gs_slack,
            gs_len])
        return gs
    def gL_gU(Z):
        # note: this code below, copy-pasted from g(Z),
        # is there to get the dimensions of each constraints
        # and define gL, gU such that
        #   gL <= gU(Z) <= gU
        # (gL, gU) are then passed to IPOPT
        gs_dyn = model.dynamics_constraints(Z)
        gs_x0 = model.initial_constraints(Z)
        gs_xf = model.final_constraints(Z)
        gs_slip = model.no_slip_constraints(Z)
        gs_contact = model.contact_constraints(Z)
        gs_over = model.leg_over_ground_constraints(Z)
        gs_slip_risk = model.slip_risk_constraints(Z)
        gs_control, gs_control_l, gs_control_u = model.control_constraints(Z)
        gs_slack, gs_slack_l, gs_slack_u = model.slack_constraints(Z)
        gs_len, gs_len_l, gs_len_u = model.length_and_speed_constraints(Z)
        gs = np.concatenate([
            gs_dyn, 
            gs_x0, 
            gs_xf, 
            gs_slip,
            gs_contact,
            gs_over,
            gs_slip_risk,
            gs_control,
            gs_slack,
            gs_len])
        g_L = np.zeros_like(gs)
        g_U = np.zeros_like(gs)
        num_equality_constraints = (
            len(gs_dyn)+
            len(gs_x0)+
            len(gs_xf)+
            len(gs_slip)+
            len(gs_contact))
        # inequality constraints (one-sided g(x) <= 0)
        g_L[num_equality_constraints:] = -1e15
        # inequality constraints (two-sided a <= g(x) <= b)
        idx_control = num_equality_constraints+len(gs_over)+len(gs_slip_risk)
        g_L[idx_control:(idx_control+len(gs_control))] = gs_control_l
        g_U[idx_control:(idx_control+len(gs_control))] = gs_control_u
        idx_slack = idx_control+len(gs_control)
        g_L[idx_slack:idx_slack+1] = gs_slack_l
        g_U[idx_slack:idx_slack+1] = gs_slack_u
        idx_len = idx_slack + 1
        g_L[idx_len:] = gs_len_l
        g_U[idx_len:] = gs_len_u
        return g_L, g_U

    print("Jitting functions (JAX)")
    g_L, g_U = gL_gU(Z0)
    f = jit(f)
    g = jit(g)
    grad_f_jax = jit(grad(f))
    grad_g_jax = jit(jacrev(g))
    _ = f(Z0), g(Z0), grad_f_jax(Z0), grad_g_jax(Z0)
    nvar = num_vars
    ncon = len(g(Z0))
    # Hessians
    hess_f = jit(hessian(f))
    def lagrange_dot_g(x, lagrange):
        return jnp.dot(lagrange, g(x))
    def hess_lagrange_dot_g(x, lagrange):
        hess = hessian(lagrange_dot_g)(x, lagrange)[:nvar,:nvar]
        return hess
    hess_lagrange_dot_g = jit(hess_lagrange_dot_g)
    _ = hess_f(Z0), hess_lagrange_dot_g(Z0, np.zeros_like(g(Z0)))
    print("Finished Jax-jitting.")

    # These functions are not jittable since they update an 
    # argument (out) in the function, as required by IPOPT.
    def eval_f(x):
        return f(x)
    def eval_grad_f(x, out):
        out[:] = grad_f_jax(x)
        return out
    def eval_g(x, out):
        out[:] = g(x)
        return out
    def eval_jac_g(x, out):
        out[:] = grad_g_jax(x).flatten()
        return out

    # Initial bounds for IPOPT
    x_L = -np.ones(nvar, dtype=np.float_) * 1000.0
    x_U = np.ones(nvar, dtype=np.float_) * 1000.0
    for t in range(S+1):
        # -3 <= x <= 3
        idx = t*n_x
        x_L[idx] = -3
        x_U[idx] = 3
        # 0.5 <= z <= 10
        idx = t*n_x + 1
        x_L[idx] = 0.5
        x_U[idx] = 10
        # pi/2 <= phi <= pi/2
        idx = t*n_x + 2
        x_L[idx] = -np.pi / 2
        x_U[idx] = np.pi / 2
        # ell >= 0
        idx = t*n_x + 3
        x_L[idx] = 0.1
        x_U[idx] = 3
        # velocity bounds
        x_L[(t*n_x+4):(t+1)*n_x] = -500
        x_U[(t*n_x+4):(t+1)*n_x] = 500

    def eval_h(x, lagrange, obj_factor, out):
        hess_cost = hess_f(x)
        hess_lagrange_prod = hess_lagrange_dot_g(x, lagrange)
        hess_cost = hess_cost[np.tril_indices(nvar)].flatten()
        hess_lagrange_prod = hess_lagrange_prod[np.tril_indices(nvar)].flatten()
        out[:] = obj_factor * hess_cost + hess_lagrange_prod
        return out

    # note: our implementation is very slow because we 
    # indicate here to ipopt that all matrices are dense,
    # which is not the case.
    indices_1, indices_2 = np.indices((ncon, nvar))
    eval_jac_g_sparsity_indices = (
        indices_1.flatten(), indices_2.flatten()
    )
    indices_1, indices_2 = np.tril_indices(nvar)
    eval_h_sparsity_indices = (
        indices_1.flatten(), indices_2.flatten()
    )

    options = {'max_iter': 3000,
               'tol': 1e-3,
               'kappa_d': 1e-5,
               }
    nlp = ipyopt.Problem(
        nvar,
        x_L,
        x_U,
        ncon,
        g_L,
        g_U,
        eval_jac_g_sparsity_indices,
        eval_h_sparsity_indices,
        eval_f,
        eval_grad_f,
        eval_g,
        eval_jac_g,
        ipopt_options=options,
        eval_h=eval_h,
    )
    zl = np.zeros(nvar)
    zu = np.zeros(nvar)
    constraint_multipliers = np.zeros(ncon)
    Z, obj, status = nlp.solve(
        Z0, 
        mult_g=constraint_multipliers, 
        mult_x_L=zl, 
        mult_x_U=zu)
    xs, us = model.convert_z_to_xs_us_mats(Z)

    if B_compute_solution_baseline:
        with open('results/hopper_base_results.npy', 'wb') as f:
            np.save(f, xs.to_py())
            np.save(f, us.to_py())
    if B_compute_solution_saa:
        with open('results/hopper_saa_alpha='+str(alpha)+'_results.npy', 
            'wb') as f:
            np.save(f, xs.to_py())
            np.save(f, us.to_py())
    print("---------------------------------------")





# def plot_results(xs, us, fname):
#     trajectory_center_of_mass = xs[:, :2]
#     trajectory_end_effector = vmap(model.end_effector_position)(xs)

#     fig = plt.figure(figsize=[6,3])
#     plt.scatter(
#         trajectory_center_of_mass[:, 0], 
#         trajectory_center_of_mass[:, 1], 
#         c='b')
#     plt.scatter(
#         trajectory_end_effector[:, 0], 
#         trajectory_end_effector[:, 1], 
#         c='g')
#     plt.scatter(xs[-1, 0], xs[-1, 1], 
#              c='r', label=r'$x_u(t)$')
#     plt.scatter(state_initial[0], state_initial[1], color='k')
#     plt.xlabel(r'$p_x$', fontsize=24)
#     plt.ylabel(r'$p_z$', fontsize=24, rotation=0)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     fig.savefig('figures/'+fname+'_traj.png')
#     plt.show()
#     plt.close()
#     # 
#     fig = plt.figure(figsize=[6,3])
#     plt.step(dt*np.arange(S), us[:, 0],
#         where='pre', color='r', label=r'$u_1(t)$')
#     plt.step(dt*np.arange(S), us[:, 1],
#         where='pre', color='b', label=r'$u_2(t)$')
#     plt.step(dt*np.arange(S), us[:, 2],
#         where='pre', color='g', label=r'$\lambda_x(t)$')
#     plt.step(dt*np.arange(S), us[:, 3],
#         where='pre', color='k', label=r'$\lambda_z(t)$')
#     plt.xlabel(r'$t$', fontsize=24)
#     plt.ylabel(r'$u(t)$', fontsize=24)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.xlim((0, T))
#     plt.legend(fontsize=18)
#     plt.grid()
#     fig.savefig('figures/'+fname+'_controls.png')
#     plt.show()
#     # 
#     fig = plt.figure(figsize=[6,3])
#     plt.scatter(dt*np.arange(S+1), xs[:, 2], 
#              c='b', label=r'$\phi$')
#     plt.scatter(dt*np.arange(S+1), xs[:, 3], 
#              c='g', label=r'$\ell$')
#     plt.legend()
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     fig.savefig('figures/'+fname+'_phi_ell.png')
#     plt.show()
#     plt.close()
#     # 
#     fig = plt.figure(figsize=[6,3])
#     plt.scatter(dt*np.arange(S+1), xs[:, 6], 
#              c='b', label=r'$\dot{\phi}$')
#     plt.scatter(dt*np.arange(S+1), xs[:, 7], 
#              c='g', label=r'$\dot{\ell}$')
#     plt.legend()
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     fig.savefig('figures/'+fname+'_vels.png')
#     plt.show()
#     plt.close()
#     # plot trajectory points one at a time
#     for t in range(S+1):
#         fig = plt.figure(figsize=[6,3])
#         xmin, xmax = -0.5, 3
#         ymin, ymax = -0.5, 3
#         xs_ground = [xmin, xmax] 
#         ys_ground = [ 0., 0.]
#         plt.plot(xs_ground, ys_ground, 'r--') 
#         plt.fill_between(
#             xs_ground, [-1, -1], ys_ground, 
#             color='r', alpha=0.2)
#         plt.scatter(xs[:, 0], xs[:, 1], 
#                  c='b', label=r'$x_u(t)$',
#                  alpha=0.3)
#         # at time t
#         center_position = xs[t, :2]
#         foot_position = model.end_effector_position(xs[t])
#         center = Circle(xs[t, :2], radius=0.05,
#             color='k', alpha=0.8)
#         plt.gca().add_patch(center)
#         foot = Circle(foot_position, radius=0.05,
#             color='g', alpha=0.8)
#         plt.gca().add_patch(foot)
#         plt.scatter(state_initial[0], state_initial[1], color='k')
#         plt.scatter(state_final[0], state_final[1], color='r')
#         plt.xlabel(r'$p_x$', fontsize=24)
#         plt.ylabel(r'$p_y$', fontsize=24, rotation=0)
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         plt.xlim((xmin, xmax))
#         plt.ylim((ymin, ymax))
#         fig.savefig('figures/'+fname+'_t='+str(t)+'.png')
#         plt.close()




if B_plot_all_results:
    print("---------------------------------------")
    print("[hopper.py] >>> Plotting")
    xs_all = np.zeros((len(alphas_to_plot)+1, S+1, n_x))
    us_all = np.zeros((len(alphas_to_plot)+1, S, n_u))
    for i, alpha in enumerate(alphas_to_plot):
        try:
            with open('results/hopper_saa_alpha='+str(alpha)+'_results.npy', 
                'rb') as f:
                xs_all[i, :, :] = np.load(f)
                us_all[i, :, :] = np.load(f)
        except:
            print("The file " + 'results/hopper_saa_alpha='+str(alpha)+'_results.npy' + ' does not exist.')
            print("On lines 25-26, change the values of B_compute_solution_baseline and B_compute_solution_saa,")
            print("and rerun the script: python hopper.py")
            with open('results/hopper_saa_alpha='+str(alpha)+'_results.npy', 
                'rb') as f:
                xs_all[i, :, :] = np.load(f)
                us_all[i, :, :] = np.load(f)
    with open('results/hopper_base_results.npy', 'rb') as f:
        xs_all[-1, :, :] = np.load(f)
        us_all[-1, :, :] = np.load(f)
    model = Model(M, 'saa', alpha)
    # center of mass trajectory
    trajs_com = xs_all[:, :, :2]
    # end effector trajectory
    trajs_ee = vmap(vmap(model.end_effector_position))(xs_all)
    # colors
    colors = pl.cm.bwr(np.linspace(0, 1, len(alphas_to_plot)))
    fig = plt.figure(figsize=[6,3])
    xmin, xmax = -0.1, 1.25
    ymin, ymax = -0.1, 1.75
    xs_ground = [xmin, xmax] 
    ys_ground = [ 0., 0.]
    plt.plot(xs_ground, ys_ground, 'r--') 
    plt.fill_between(xs_ground, [-1, -1], ys_ground, color='r', alpha=0.2)
    plt.plot(
        trajs_com[-1, :, 0], 
        trajs_com[-1, :, 1], 
        c='k', linestyle="--", linewidth=2,
        alpha=0.7)
    plt.plot(
        trajs_ee[-1, :, 0], 
        trajs_ee[-1, :, 1], 
        c='k', linestyle="-", linewidth=2,
        alpha=0.7)
    # plot links too
    ts_to_plot = [0, 6, 10, 12, 14, 16, 18, 20, 24, 30]
    print("ts_ =", ts_to_plot)
    plt.scatter(
        trajs_com[-1, ts_to_plot, 0], 
        trajs_com[-1, ts_to_plot, 1], 
        c='k', s=250)
    for t in ts_to_plot:
        plt.plot(
            np.array([trajs_com[-1, t, 0], trajs_ee[-1, t, 0]]), 
            np.array([trajs_com[-1, t, 1], trajs_ee[-1, t, 1]]), 
            c='k', linestyle="-", linewidth=3,
            alpha=1)
    plt.scatter(
        trajs_ee[-1, ts_to_plot, 0], 
        trajs_ee[-1, ts_to_plot, 1], 
        c='#9d7200', s=50)
    for i, alpha in enumerate(alphas_to_plot):
        plt.plot(
            trajs_com[i, :, 0], 
            trajs_com[i, :, 1], 
            c=colors[i], linestyle="--", linewidth=3)
        plt.plot(
            trajs_ee[i, :, 0], 
            trajs_ee[i, :, 1], 
            c=colors[i], linestyle="-", linewidth=3)
        if i == len(alphas_to_plot) - 1:
            plt.scatter(
                trajs_com[i, ts_to_plot, 0], 
                trajs_com[i, ts_to_plot, 1], 
                c=colors[i], s=250,
                alpha=0.5)
            for t in ts_to_plot:
                plt.plot(
                    np.array([trajs_com[i, t, 0], trajs_ee[i, t, 0]]), 
                    np.array([trajs_com[i, t, 1], trajs_ee[i, t, 1]]), 
                    c=colors[i], linestyle="-", linewidth=3,
                    alpha=0.5)
    Z = [[-100,-100],[-100,-100]]
    CS3 = plt.contourf(Z, 
        np.geomspace(np.min(alphas_to_plot), np.max(alphas_to_plot), num=201), 
        cmap=pl.cm.bwr)
    cbar = plt.colorbar(CS3, 
        ticks=alphas_to_plot)
    cbar.set_label(r'$\alpha$', 
        fontsize=20, rotation='horizontal', labelpad=16, y=0.53)
    cbar.ax.tick_params(labelsize=16) 
    plt.xlabel(r'$p_x$', fontsize=24)
    plt.ylabel(r'$p_z$', fontsize=24, rotation=0, labelpad=16)
    plt.yticks([0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0, 1.5])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    # fig.savefig('figures/all_trajs_alphas.png')
    plt.show()
    plt.close()
    print("---------------------------------------")




if B_validate_monte_carlo:
    print("---------------------------------------")
    print("[hopper.py] >>> Monte Carlo validation")
    def no_slip_constraint(
        position_x, contact_force, 
        intensity_vec, theta_vec, tau_vec):
        friction_coeff = friction_at_px(
            position_x, 
            intensity_vec, theta_vec, tau_vec)
        fx, fz = contact_force
        ineq = fx - friction_coeff * fz
        return ineq
    def no_slip_constraints_verification(
        positions_x, contact_forces, 
        intensity_vec, theta_vec, tau_vec):
        num_forces = contact_forces.shape[0]
        intensity_mat = jnp.repeat(
            intensity_vec[jnp.newaxis, :], num_forces, axis=0)
        theta_mat = jnp.repeat(
            theta_vec[jnp.newaxis, :], num_forces, axis=0)
        tau_mat = jnp.repeat(
            tau_vec[jnp.newaxis, :], num_forces, axis=0)
        cons = vmap(no_slip_constraint)(
            positions_x, contact_forces, 
            intensity_mat, theta_mat, tau_mat)
        max_constraint = jnp.max(cons)
        B_satisfied = max_constraint <= 1e-6
        return B_satisfied, max_constraint
    def avar(Z_samples, alpha):
        # estimates avar_alpha(Z)
        M = len(Z_samples)
        num_variables = M + 1 # (ys, t)
        P = np.zeros((num_variables, num_variables))
        q = np.zeros(num_variables)
        # inf_t t + (1/(M*alpha))sum_{i=1}^M yi
        q[:-1] = 1.0 / (M * alpha)
        q[-1] = 1.0
        # such that
        A = np.zeros((2*M, num_variables))
        l = np.zeros((2*M))
        u = np.zeros((2*M))
        # -yi <= 0 for all i
        l[:M] = -np.inf
        A[:M, :M] = -np.eye(M)
        # Z_i - t - yi <= 0 for all i
        l[M:(2*M)] = -np.inf
        u[M:(2*M)] = -Z_samples
        A[M:(2*M), :M] = -np.eye(M)
        A[M:(2*M), -1] = -1.0
        # risk estimation
        P = sp.csc_matrix(P)
        A = sp.csr_matrix(A)
        osqp_prob = osqp.OSQP()
        osqp_prob.setup(P, q, A, l, u,
                verbose=False,
                polish=True)
        result = osqp_prob.solve()
        t_risk = result.x[-1]        
        # return t_risk
        avar_value = t_risk + np.mean(np.maximum(Z_samples - t_risk, np.zeros(M)) / alpha)
        return avar_value

    xs_all = np.zeros((len(alphas)+1, S+1, n_x))
    us_all = np.zeros((len(alphas)+1, S, n_u))
    forces_all = np.zeros((len(alphas)+1, S, 2))
    for i, alpha in enumerate(alphas):
        with open('results/hopper_saa_alpha='+str(alpha)+'_results.npy', 
            'rb') as f:
            xs_all[i, :, :] = np.load(f)
            us_all[i, :, :] = np.load(f)
            forces_all[i, :, :] = us_all[i, :, 2:]
    with open('results/hopper_base_results.npy', 'rb') as f:
        xs_all[-1, :, :] = np.load(f)
        us_all[-1, :, :] = np.load(f)
        forces_all[-1, :, :] = us_all[-1, :, 2:]

    M = 10000
    intensities = np.random.uniform(0, 1, (M, num_mu_features))
    intensities = np.sqrt(2 / num_mu_features) * intensities
    intensities = 0.025 * intensities
    thetas = np.random.uniform(0, np.pi, (M, num_mu_features))
    taus = np.random.uniform(0, 2*np.pi, (M, num_mu_features))

    alphas_and_base = np.zeros(len(alphas)+1)
    alphas_and_base[:len(alphas)] = alphas
    for i, alpha in enumerate(alphas_and_base):
        print("---------------------------")
        print("Monte-Carlo: alpha =", alpha)
        xs = xs_all[i, :, :]
        print("jumped distance =", xs[-1, 0])
        forces = forces_all[i, :, :]
        px = vmap(model.end_effector_position)(xs)[:, 0]
        px = jnp.concatenate([
            px[:time_jump],
            px[time_land:-1]], axis=0)
        forces = jnp.concatenate([
            forces[:time_jump],
            forces[time_land:]], axis=0)
        px_vmapped = jnp.repeat(
            px[jnp.newaxis, :], M, axis=0) 
        forces_vmapped = jnp.repeat(
            forces[jnp.newaxis, :], M, axis=0) 
        B_satisfied_vec, max_con_vec = vmap(no_slip_constraints_verification)(
            px_vmapped, forces_vmapped,
            intensities, thetas, taus)
        if alpha > 0:
            avar_val = avar(max_con_vec, alpha)
        print("B_satisfied_vec =", jnp.mean(B_satisfied_vec))
        if alpha > 0:
            print("avar =", avar_val)
    print("---------------------------------------")