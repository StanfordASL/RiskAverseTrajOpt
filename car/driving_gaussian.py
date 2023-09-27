import scipy.sparse as sp
import numpy as np
import osqp
from time import time
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib import rc, rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from jax.lax import fori_loop
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from driving_utils import p_th_quantile_cdf_normal
from driving_utils import plot_gaussian_confidence_ellipse

import driving_params

# load driving parameters
OSQP_POLISH = driving_params.OSQP_POLISH
OSQP_TOL = 1e-8 # driving_params.OSQP_TOL
n_x = driving_params.n_x
n_u = driving_params.n_u
S = driving_params.S
# M = driving_params.M
T = driving_params.T
dt = driving_params.dt
R = driving_params.R
u_max = driving_params.u_max
omega_speed_nom = driving_params.omega_speed_nom
omega_speed_del = driving_params.omega_speed_del
omega_repulsive_nom = driving_params.omega_repulsive_nom
omega_repulsive_del = driving_params.omega_repulsive_del
ego_width = driving_params.ego_width
ego_height = driving_params.ego_height
ped_radius = driving_params.ped_radius
min_separation_distance = driving_params.min_separation_distance
speed_ped_des = driving_params.speed_ped_des
speed_ego_init = driving_params.speed_ego_init
position_ego_init = driving_params.position_ego_init
position_ped_init = driving_params.position_ped_init
velocity_ego_init = driving_params.velocity_ego_init
velocity_ped_init = driving_params.velocity_ped_init
position_ego_goal = driving_params.position_ego_goal
velocity_ego_goal = driving_params.velocity_ego_goal
state_init = driving_params.state_init
variance_ped_initial_state = driving_params.variance_ped_initial_state

B_compute_solution_gaussian = True
B_plot_results_gaussian = True
alphas = [0.01, 0.02, 0.05, 0.1]
num_scp_iters_max = 60
print("---------------------------------------")
print("[driving_gaussian_optimal_risk.py] >>> Running with parameters:")
print("B_compute_solution_gaussian =", B_compute_solution_gaussian)
print("B_plot_results_gaussian     =", B_plot_results_gaussian)
print("alphas =", alphas)
print("---------------------------------------")

class Model:
    def __init__(self, method='gaussian', alpha=0.1):
        print("Initializing Model with")
        print("> method     =", method)
        print("> alpha      =", alpha)
        self.method = method
        # control bounds
        self.u_max = u_max
        self.u_min = -self.u_max
        # uncertain parameters
        self.alpha = alpha # risk parameter
        self.beta = 3e-2 # diffusion term magnitude

        self.omega_speed_nominal = omega_speed_nom
        self.omega_repulsive_nominal = omega_repulsive_nom
        self.omegas_speed_variance = (
            2*omega_speed_del)**2 / 12.0
        self.omegas_repulsive_variance = (
            2*omega_repulsive_del)**2 / 12.0

        # initial mean and covariance state
        self.state_mean_init = state_init
        self.state_covariance_init = jnp.block([
            [jnp.zeros((4, 4)), jnp.zeros((4, 4))],
            [jnp.zeros((4, 4)), variance_ped_initial_state]
            ])

    def convert_us_vec_to_us_mat(self, us_vec):
        us_mat = jnp.reshape(us_vec, (n_u, S), 'F')
        us_mat = us_mat.T # (S, n_u)
        us_mat = jnp.array(us_mat)
        return us_mat

    def convert_us_mat_to_us_jaxvec(self, us_mat):
        us_vec = jnp.reshape(us_mat, (S*n_u), 'C')
        return us_vec

    def initial_guess_us_mat(self):
        us = jnp.zeros((S, n_u))
        u_initial_guess = (self.u_max + self.u_min) / 2.0
        u_initial_guess = u_initial_guess + 1e-2
        for t in range(S):
            us = us.at[t, :].set(u_initial_guess)
        return us

    def initial_guess_alphas_risk(self):
        alphas_uniform = (self.alpha / S) * jnp.ones(S)
        return alphas_uniform

    @partial(jit, static_argnums=(0,))
    def force_on_pedestrian(self, x, 
        omega_speed, omega_repulsive):
        position_ego = x[0:2]
        position_ped = x[4:6]
        speed_ego_along_y = x[7]
        # repulsion force
        positions_delta = position_ego - position_ped
        force = -omega_repulsive * positions_delta
        force = force / jnp.linalg.norm(positions_delta)
        # desired speed
        delta_speed = speed_ped_des - speed_ego_along_y
        force = force + omega_speed * delta_speed
        return force

    @partial(jit, static_argnums=(0,))
    def b(self, x, u, 
        omega_speed, omega_repulsive):
        vel_lin_ego, phi_ego = x[2:4] 
        vel_ped = x[6:]
        force_interaction = self.force_on_pedestrian(x, 
            omega_speed, omega_repulsive)
        bvec = jnp.array([
            # ego dynamics
            vel_lin_ego*jnp.cos(phi_ego),
            vel_lin_ego*jnp.sin(phi_ego),
            u[0],
            u[1],
            # pedestrian dynamics
            vel_ped[0],
            vel_ped[1],
            force_interaction[0],
            force_interaction[1]])
        return bvec

    @partial(jit, static_argnums=(0,))
    def b_dx(self, x, u):
        return jacfwd(self.b, argnums=(0))(x, u, 
            self.omega_speed_nominal, self.omega_repulsive_nominal)

    @partial(jit, static_argnums=(0,))
    def b_domega_speed(self, x, u):
        return jacfwd(self.b, argnums=(2))(x, u, 
            self.omega_speed_nominal, self.omega_repulsive_nominal)

    @partial(jit, static_argnums=(0,))
    def b_domega_repulsive(self, x, u):
        return jacfwd(self.b, argnums=(3))(x, u, 
            self.omega_speed_nominal, self.omega_repulsive_nominal)

    @partial(jit, static_argnums=(0,))
    def sigma(self, x, u):
        smat = jnp.zeros((n_x, n_x))
        smat = smat.at[6:,6:].set(self.beta * jnp.eye(2))
        return smat

    @partial(jit, static_argnums=(0,))
    def us_to_state_trajectory(self, us_mat):
        # returns the nominal (mean) trajectory
        # us_mat - (S, n_u)
        # mass   - scalar
        # dWs    - (S, n_x)
        xs = jnp.zeros((S+1, n_x))
        xs = xs.at[0, :].set(self.state_mean_init)
        for t in range(S):
            xt, ut = xs[t, :], us_mat[t, :]
            bt_dt = dt * self.b(xt, ut, 
                self.omega_speed_nominal, self.omega_repulsive_nominal)
            # Euler-Maruyama
            xn = xt + bt_dt
            xs = xs.at[t+1, :].set(xn)
        return xs

    @partial(jit, static_argnums=(0,))
    def us_to_covariance_trajectory(self, us_mat):
        # returns the covariance (Sigma) trajectory
        # us_mat - (S, n_u)
        xs = self.us_to_state_trajectory(us_mat)

        def compute_Sigmas_fori_loop(t, mus_Sigmas_us):
            mus, Sigmas, us = mus_Sigmas_us
            x_t, u_t, Sig_t = xs[t], us[t], Sigmas[t]

            # see comments in drone_gaussian.py

            # linearized dynamics
            A = dt * self.b_dx(x_t, u_t)
            A = jnp.eye(n_x) + A # Euler discretization
            # Brownian motion term
            Sigma_w = self.sigma(x_t, u_t)
            Sigma_w = dt * Sigma_w @ Sigma_w.T
            # term due to uncertain mass (iid approximation)
            b_ds = dt * self.b_domega_speed(x_t, u_t)
            b_dr = dt * self.b_domega_repulsive(x_t, u_t)
            Sigma_due_to_omega = (
                self.omegas_speed_variance * b_ds @ b_ds.T + 
                self.omegas_repulsive_variance * b_dr @ b_dr.T)

            # propagate covariance via Gaussian linearization
            Sig_next = A @ Sig_t @ A.T
            Sig_next += Sigma_w
            Sig_next += Sigma_due_to_omega

            Sigmas = Sigmas.at[t+1, :].set(Sig_next)
            return (mus, Sigmas, us)

        Sigmas = jnp.zeros((S+1, n_x, n_x))
        Sigmas = Sigmas.at[0].set(self.state_covariance_init)
        mus_Sigmas_us = (xs, Sigmas, us_mat)

        _, Sigmas, _ = fori_loop(0, S, 
            compute_Sigmas_fori_loop, mus_Sigmas_us)

        return Sigmas

    @partial(jit, static_argnums=(0,))
    def final_constraints(self, xs):
        state_ego_goal = jnp.concatenate((
            position_ego_goal,
            velocity_ego_goal), axis=-1)
        return (xs[-1, :4] - state_ego_goal)

    @partial(jit, static_argnums=(0,))
    def separation_distance_ego_to_pedestrian(self, 
        mu, Sigma, alpha_risk):
        Sigma_position_pedestrian = Sigma[4:6, 4:6]
        position_ego = mu[0:2]
        position_ped = mu[4:6]

        # distance
        positions_delta = position_ego - position_ped
        distance = jnp.linalg.norm(positions_delta)

        # variance term
        normal = positions_delta / distance
        quantile_value = p_th_quantile_cdf_normal(1 - alpha_risk)
        variance_padding = quantile_value * jnp.sqrt(
            normal.T @ Sigma_position_pedestrian @ normal)
        distance = distance - variance_padding

        # padding to keep a safe distance
        distance = distance - min_separation_distance

        return distance

    @partial(jit, static_argnums=(0,))
    def separation_distances_at_all_times(self, mus, Sigmas, alphas_risk):
        distances = vmap(self.separation_distance_ego_to_pedestrian)(
            mus[1:], Sigmas[1:], alphas_risk)
        return distances

    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### OSQP
    @partial(jit, static_argnums=(0,))
    def get_control_risk_constraints_coeffs_all(self):
        # Returns (A, l, u) corresponding to control constraints
        # such that l <= A uvec <= u.
        A = jnp.zeros((n_u*S + S + 1, n_u*S + S + 1))
        # S for alpha_i risk allocation probabilities
        l = jnp.zeros(n_u*S + S + 1)
        u = jnp.zeros(n_u*S + S + 1)
        # control bounds
        for t in range(S):
            for i in range(n_u):
                idx_u = t*n_u + i
                idx_con = idx_u
                A = A.at[idx_con, idx_u].set(1.0)
                l = l.at[idx_con].set(self.u_min)
                u = u.at[idx_con].set(self.u_max)
        # bounds on risk allocation
        # 0 <= alpha_i <= alpha
        for t in range(S):
            idx_alpha_i = S*n_u + t
            idx_con = idx_alpha_i
            A = A.at[idx_con, idx_alpha_i].set(1.0)
            l = l.at[idx_con].set(100 * OSQP_TOL)
            u = u.at[idx_con].set(self.alpha)
        # 0 <= sum_{t=1}^{S} alpha_i <= alpha
        for t in range(S):
            idx_alpha_i = S*n_u + t
            A = A.at[-1, idx_alpha_i].set(1.0)
        l = l.at[-1].set(100 * OSQP_TOL)
        u = u.at[-1].set(self.alpha)
        return A, l, u

    @partial(jit, static_argnums=(0,))
    def get_all_constraints_coeffs(self, us_mat, alphas_risk):
        # Returns (A, l, u) corresponding to all constraints
        # such that l <= A uvec <= u.
        def all_constraints_us_alphas(us_mat, alphas_risk):
            xs = self.us_to_state_trajectory(us_mat)
            Sigmas = self.us_to_covariance_trajectory(us_mat)
            val_final = self.final_constraints(xs)
            val_obs = -self.separation_distances_at_all_times(
                xs, Sigmas, alphas_risk)
            return (val_final, val_obs)
        def all_constraints_dus_dalphas(us_mat, alphas_risk):
            jac_dus = jacfwd(all_constraints_us_alphas, argnums=(0))(
                us_mat, alphas_risk)
            jac_dalphas = jacfwd(all_constraints_us_alphas, argnums=(1))(
                us_mat, alphas_risk)
            return jac_dus, jac_dalphas

        v_final, g_obs = all_constraints_us_alphas(us_mat, alphas_risk)
        v_final_g_obs_du, v_final_g_obs_dalphas = all_constraints_dus_dalphas(us_mat, alphas_risk)

        # reshape gradient
        v_final_du = v_final_g_obs_du[0]
        g_obs_du = v_final_g_obs_du[1]
        v_final_dalphas = v_final_g_obs_dalphas[0]
        g_obs_dalphas = v_final_g_obs_dalphas[1]
        v_final_du = jnp.reshape(v_final_du, (4, n_u*S), 'C')
        g_obs_du = jnp.reshape(g_obs_du, (S, n_u*S), 'C')
        v_final_du_dalphas = jnp.concatenate((
            v_final_du,
            v_final_dalphas),
            axis=-1)
        g_obs_du_dalphas = jnp.concatenate((
            g_obs_du,
            g_obs_dalphas),
            axis=-1)

        # final constraints
        val_final = -v_final + v_final_du @ self.convert_us_mat_to_us_jaxvec(
            us_mat)
        val_final_lower = val_final
        val_final_upper = val_final

        # obstacle_constraints
        # gi(u) <= 0
        # => ∇g(u_k) @ u <= -g(u_k) + ∇g(u_k) @ u_k (linearized)
        g_up = -g_obs + (
            g_obs_du @ self.convert_us_mat_to_us_jaxvec(us_mat) +
            g_obs_dalphas @ alphas_risk)

        return (v_final_du_dalphas, val_final_lower, val_final_upper, 
            g_obs_du_dalphas, g_up)


    @partial(jit, static_argnums=(0,))
    def get_all_constraints_coeffs_all(self, us_mat, alphas_risk):
        final_du_dalphas, final_low, final_up, gs_obs_du_dalphas, gs_obs_up = (
            self.get_all_constraints_coeffs)(us_mat, alphas_risk)

        # final constraints
        final_dparams = jnp.concatenate((
            final_du_dalphas, 
            jnp.zeros((final_du_dalphas.shape[0], 1))),
            axis=-1) # add slack variable

        # obstacle avoidance constraints
        obs_low = -jnp.inf* jnp.ones(S)
        obs_up = jnp.inf * jnp.ones(S)
        obs_dparams = jnp.zeros((S, n_u*S + S + 1))
        # pack variables
        obs_dparams = obs_dparams.at[:, :(n_u*S + S)].set(
            jnp.reshape(gs_obs_du_dalphas, (S, n_u*S + S), 'C'))
        obs_up = gs_obs_up.flatten()

        # combine final constraints and obstacle avoidance constraints
        constraints_dparams = jnp.vstack([final_dparams, obs_dparams])
        constraints_low = jnp.hstack([final_low, obs_low])
        constraints_up = jnp.hstack([final_up, obs_up])
        return constraints_dparams, constraints_low, constraints_up

    @partial(jit, static_argnums=(0,))
    def get_objective_coeffs_jax(self):
        # See description of @get_objective_coeffs
        P = jnp.zeros((n_u*S + S + 1, n_u*S + S + 1))
        q = jnp.zeros(n_u*S + S + 1)
        # control squared
        for t in range(S):
            idx = t*n_u
            # control norm squared
            P = P.at[idx:idx+n_u, idx:idx+n_u].set(2*dt*R)
        return P, q

    def get_objective_coeffs(self):
        # Returns (P, q) corresponding to objective
        #        min (1/2 z^T P z + q^T z)
        # where z = umat is the optimization variable.
        P, q = self.get_objective_coeffs_jax()
        P, q = sp.csc_matrix(P.to_py()), q.to_py()
        return P, q

    def get_constraints_coeffs(self, us_mat, alphas_risk, scp_iter):
        # Constraints: l <= A z <= u, with z = umat

        # control constraints
        A_con, l_con, u_con = self.get_control_risk_constraints_coeffs_all()
        # combined final + obstacle avoidance constraints
        As, ls, us = self.get_all_constraints_coeffs_all(
            us_mat, alphas_risk)

        # Jax => numpy
        A_con, l_con, u_con = A_con.to_py(), l_con.to_py(), u_con.to_py()
        As, ls, us = np.copy(As), np.copy(ls), np.copy(us)

        if scp_iter < 1:
            # remove separation distance avoidance
            As[n_x:] *= 0
            ls[n_x:] *= 0
            us[n_x:] *= 0

        As, A_con = sp.csr_matrix(As), sp.csr_matrix(A_con)
        A = sp.vstack([As, A_con], format='csc')
        l = np.hstack([ls, l_con])
        u = np.hstack([us, u_con])
        return A, l, u

    def define_problem(self, 
        us_mat_p, alphas_risk_p, scp_iter=0, verbose=False):
        # objective and constraints
        self.P, self.q = self.get_objective_coeffs()
        self.A, self.l, self.u = self.get_constraints_coeffs(
            us_mat_p, alphas_risk_p, scp_iter)
        # Setup OSQP problem
        if scp_iter==0 or scp_iter==1:
            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(
                self.P, self.q, self.A, self.l, self.u,
                eps_abs=OSQP_TOL, eps_rel=OSQP_TOL,
                linsys_solver="qdldl",
                # linsys_solver="mkl pardiso",
                warm_start=True, verbose=verbose,
                polish=OSQP_POLISH)
        else:
            self.osqp_prob.update(l=self.l, u=self.u)
            self.osqp_prob.update(Ax=self.A.data)
        return True

    def solve(self, verbose=False):
        self.res = self.osqp_prob.solve()
        if self.res.info.status != 'solved':
            print("[solve]: Problem infeasible.")
        us_sol = self.convert_us_vec_to_us_mat(
            self.res.x[:(n_u*S)])
        alphas_sol = self.res.x[(n_u*S):-1]
        return us_sol, alphas_sol

def L2_error_us(us_mat, us_mat_prev):
    # us_mat - (S, n_u)
    # us_mat_prev - (S, n_u)
    error = np.mean(np.linalg.norm(us_mat-us_mat_prev, axis=-1))
    error = error / np.mean(np.linalg.norm(us_mat, axis=-1))
    return error


if B_compute_solution_gaussian:
    print("---------------------------------------")
    print("[driving_gaussian_optimal_risk.py] >>> Computing solution")
    for alpha in alphas:
        model = Model(alpha=alpha)
        # Initial compilation (JAX)
        us_prev = model.initial_guess_us_mat()
        alphas_risk_prev = model.initial_guess_alphas_risk()
        model.define_problem(
            us_prev, alphas_risk_prev, verbose=False)
        us, alphas_risk = model.solve()
        model.define_problem(
            us, alphas_risk_prev, 1, verbose=False)
        us, alphas_risk = model.solve()

        us_prev = model.initial_guess_us_mat()
        alphas_risk_prev = model.initial_guess_alphas_risk()
        for scp_iter in range(num_scp_iters_max):
            model.define_problem(
                us_prev, 
                alphas_risk_prev,
                scp_iter,
                verbose=False)
            us, alphas_risk = model.solve()
            L2_error = L2_error_us(us, us_prev)
            us_prev = us
            alphas_risk_prev = alphas_risk
        print("alphas_risk =", alphas_risk)
        with open('results/driving_gaussian_alpha='+str(alpha)+'.npy', 
            'wb') as f:
            xs = model.us_to_state_trajectory(us)
            np.save(f, us.to_py())
            np.save(f, xs.to_py())
    print("---------------------------------------")


### PLOT
if B_plot_results_gaussian:
    print("---------------------------------------")
    print("[driving_gaussian_optimal_risk.py] >>> Plotting")
    alpha = alphas[0]
    with open('results/driving_gaussian_alpha='+str(alpha)+'.npy', 
        'rb') as f:
        model = Model(alpha=alphas[0])
        us = np.load(f)
        xs = np.load(f)
        Sigmas = model.us_to_covariance_trajectory(us)
    # plot
    fig = plt.figure(figsize=[6,3])
    ax = plt.gca()
    plt.grid()
    # plot ego car trajectory
    colors = pl.cm.winter(np.linspace(0, 1, S+1))
    for t in range(S+1):
        angle = xs[t, 3]
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])
        xy = xs[t, :2]
        xy = xy - rotation_matrix @ np.array(
            [0.5 * ego_width, 0.5 * ego_height])
        rectangle = Rectangle(xy, ego_width, ego_height, 
            angle=angle * 180. / np.pi,
            color=colors[t], alpha=0.8, fill=False,
            linewidth=2)
        plt.gca().add_patch(rectangle)
    plt.scatter(xs[:, 0], xs[:, 1], 
             c=colors, s=30, marker='+')
    # plot pedestrian trajectory
    for t in range(1, xs.shape[0]):
        mu, Sigma = xs[t, 4:6], Sigmas[t]
        Sigma = Sigma[4:6, 4:6]
        plot_gaussian_confidence_ellipse(ax, mu, Sigma, 1-alpha)
    # initial / final poses
    plt.text(position_ego_init[0]-3, position_ego_init[1]+2, 
        r'$p_\textrm{ego}(0)$', fontsize=18, weight="bold")
    plt.text(position_ped_init[0]+1.5, position_ped_init[1], 
        r'$p_\textrm{ped}(0)$', fontsize=18, weight="bold")
    plt.xlabel(r'$p^x$', fontsize=24)
    plt.ylabel(r'$p^y$', fontsize=24, rotation=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    plt.close()
    print("---------------------------------------")