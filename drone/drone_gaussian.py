from pathlib import Path
import scipy.sparse as sp
import numpy as np
import ipyopt
from time import time
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib import rc, rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import jacfwd, jacrev, grad, hessian, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from warnings import warn

from drone_utils import p_th_quantile_cdf_normal
from drone_utils import plot_gaussian_confidence_ellipse

import drone_params

# Load drone model parameters
# OSQP_POLISH = drone_params.OSQP_POLISH
# OSQP_TOL = drone_params.OSQP_TOL
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

# script parameters
B_compute_solution_gaussian = True
B_plot_results_gaussian = True
alphas = [0.05, 0.1, 0.2, 0.3]
alpha_to_plot = alphas[0]
print("---------------------------------------")
print("[drone_gaussian.py] >>> Running with parameters:")
print("B_compute_solution_gaussian =", B_compute_solution_gaussian)
print("B_plot_results_gaussian     =", B_plot_results_gaussian)
print("alphas                      =", alphas)
print("---------------------------------------")

num_vars = n_u*S + S*n_obs + n_obs

class Model:
    def __init__(self, S, method='gaussian', alpha=0.1):
        print("Initializing Model with")
        print("> method =", method)
        print("> alpha  =", alpha)
        print("> S      =", S)
        self.method = method
        self.S = S
        self.dt = T / S
        # control bounds
        self.u_max = u_max
        self.u_min = -self.u_max
        # uncertain parameters
        self.alpha = alpha # risk parameter
        self.beta = beta # diffusion term magnitude
        self.drag_coefficient = drag_coefficient 

        self.mass_nominal = mass_nom
        self.mass_variance = (2*mass_delta)**2 / 12.0
        self.obs_positions = obs_positions
        self.obs_radii = obs_radii

    def convert_z_to_variables(self, z):
        S = self.S
        us_vec = z[:(S*n_u)]
        alphas_risk = z[(S*n_u):]
        return us_vec, alphas_risk

    def convert_us_vec_to_us_mat(self, us_vec):
        S = self.S
        us_mat = jnp.reshape(us_vec, (n_u, S), 'F')
        us_mat = us_mat.T # (S, n_u)
        us_mat = jnp.array(us_mat)
        return us_mat

    def convert_us_mat_to_us_jaxvec(self, us_mat):
        S = self.S
        us_vec = jnp.reshape(us_mat, (S*n_u), 'C')
        return us_vec

    def initial_guess_us_mat(self):
        S = self.S
        print(">> Loading SAA solution as initial guess.")
        my_file = ('results/drone'+
            '_alpha='+str(self.alpha)+
            '_repeat=0.npy')
        if not(Path(my_file).is_file()):
            msg = my_file + " does not exist.\n"
            msg += "run drone_risk.py first."
            raise FileNotFoundError(msg)
        with open(my_file, 'rb') as f:
            us = np.load(f)
        return us

    def initial_guess_alphas_risk(self):
        S = self.S
        alphas_uniform = self.alpha / (S*n_obs + n_obs) 
        print("initial guess: p_th_quantile_cdf_normal =",
            p_th_quantile_cdf_normal(1 - alphas_uniform))
        alphas_uniform = alphas_uniform * jnp.ones(S*n_obs + n_obs)
        return alphas_uniform

    def initial_guess(self):
        S = self.S
        Zp = np.zeros(n_u*S + S*n_obs + n_obs)
        Zp[:(n_u*S)] = self.convert_us_mat_to_us_jaxvec(
            self.initial_guess_us_mat().flatten())
        Zp[(n_u*S):] = self.initial_guess_alphas_risk()

        return Zp

    @partial(jit, static_argnums=(0,))
    def b(self, x, u, mass):
        v = x[3:6]
        control_applied = u + feedback_gain @ x
        bvec = jnp.zeros(n_x)
        bvec = bvec.at[:3].set(v)
        bvec = bvec.at[3:6].set(control_applied / mass)
        bvec = bvec.at[3:6].set(
            bvec[3:6] - self.drag_coefficient * jnp.abs(v) * v / mass)
        return bvec

    @partial(jit, static_argnums=(0,))
    def b_dx(self, x, u):
        return jacfwd(self.b, argnums=(0))(x, u, self.mass_nominal)

    @partial(jit, static_argnums=(0,))
    def b_dmass(self, x, u):
        return jacfwd(self.b, argnums=(2))(x, u, self.mass_nominal)

    @partial(jit, static_argnums=(0,))
    def sigma(self, x, u):
        smat = jnp.zeros((n_x, n_x))
        smat = smat.at[3:6,3:6].set(
            (self.beta / self.mass_nominal) * jnp.eye(3))
        return smat

    @partial(jit, static_argnums=(0,))
    def us_to_state_trajectory(self, us_mat):
        # returns the nominal (mean) trajectory
        # us_mat - (S, n_u)
        S, dt = self.S, self.dt
        xs = jnp.zeros((S+1, n_x))
        xs = xs.at[0, :].set(x_init)
        for t in range(S):
            xt, ut = xs[t, :], us_mat[t, :]
            bt_dt = dt * self.b(xt, ut, self.mass_nominal)
            # Euler
            xn = xt + bt_dt
            xs = xs.at[t+1, :].set(xn)
        return xs

    @partial(jit, static_argnums=(0,))
    def us_to_covariance_trajectory(self, us_mat):
        # returns the covariance (Sigma) trajectory
        # us_mat - (S, n_u)
        S, dt = self.S, self.dt

        xs = self.us_to_state_trajectory(us_mat)

        def compute_Sigmas_fori_loop(t, mus_Sigmas_us):
            mus, Sigmas, us = mus_Sigmas_us
            x_t, u_t, Sig_t = xs[t], us[t], Sigmas[t]

            # see uncertainty propagation equations from
            # Lew, Bonalli, Pavone, "Chance-Constrained Sequential Convex Programming for 
            # for Robust Trajectory Optimization", European Control Conference (ECC), 2020
            # and the code at 
            # https://github.com/StanfordASL/ccscp/blob/a727dfc10d4acc43248c9a525a37279a70cecd80/exps/Models/astrobee.py#L327C61-L328C1
            # (Sig_next = AB_k@Sig_xu_k@(AB_k.T) + Sig_w + f_dw_k@mJ_var@(f_dw_k.T))
            # we make the following approximations (as in the ECC paper):
            # - the mass is randomized iid at each timestep (a typical approximation for this
            #   Gaussian linearization uncertainty propagation method that neglects time
            #   correlations of uncertainty)
            # - the covariance between the Brownian motion and mass parameters neglected 
            #   (we only take 1st order terms)

            # linearized dynamics
            A = dt * self.b_dx(x_t, u_t)
            A = jnp.eye(n_x) + A # Euler discretization
            # Brownian motion term
            Sigma_w = self.sigma(x_t, u_t)
            Sigma_w = dt * Sigma_w @ Sigma_w.T
            # term due to uncertain mass (iid approximation)
            b_dm = dt * self.b_dmass(x_t, u_t)
            Sigma_due_to_mass = self.mass_variance * b_dm @ b_dm.T

            # propagate covariance via Gaussian linearization
            Sig_next = A @ Sig_t @ A.T
            Sig_next += Sigma_w
            Sig_next += Sigma_due_to_mass

            Sigmas = Sigmas.at[t+1, :].set(Sig_next)

            return (mus, Sigmas, us)

        Sigmas = jnp.zeros((S+1, n_x, n_x))
        # Sigmas = Sigmas.at[0].set(0)
        mus_Sigmas_us = (xs, Sigmas, us_mat)

        _, Sigmas, _ = fori_loop(0, S, 
            compute_Sigmas_fori_loop, mus_Sigmas_us)

        return Sigmas

    @partial(jit, static_argnums=(0,))
    def final_constraints(self, xs):
        # constraint on the mean trajectory
        # us_vec, alphas_risk = self.convert_z_to_variables(Z)
        # us_mat = self.convert_us_vec_to_us_mat(us_vec)
        # xs = self.us_to_state_trajectory(us_mat)
        xT = xs[-1, :]
        return (xT - x_final)

    @partial(jit, static_argnums=(0,))
    def obstacle_avoidance_constraint_for_one_obstacle_at_x(self,
        mu, Sigma, 
        alpha_risk_state, alpha_risk_obs,
        obs_position, obs_mean_radius):
        # see Lew, Bonalli, Pavone, "Chance-Constrained Sequential Convex Programming for
        # for Robust Trajectory Optimization", European Control Conference (ECC), 2020

        p = mu[:2] # state position
        Sig = Sigma[:2, :2]
        obs_p = obs_position[:2]

        # take alpha quantile of obstacle radius
        rad_min = obs_mean_radius - obs_radii_deltas
        rad_max = obs_mean_radius + obs_radii_deltas

        # uniform risk allocation over 3 radii
        obs_radius = rad_max - (alpha_risk_obs / 3.0) * (rad_max - rad_min)

        # compute distance
        distance = jnp.linalg.norm(p - obs_p, 2)
        normal = (p - obs_p) / distance
        quantile_value = p_th_quantile_cdf_normal(
            1 - alpha_risk_state)
        variance_padding = quantile_value * jnp.sqrt(
            normal.T @ Sig @ normal)
        constraint = -(distance - variance_padding - obs_radius)

        return constraint

    @partial(jit, static_argnums=(0,))
    def obstacle_avoidance_constraints_for_one_obstacle(self,
        xs, Sigmas,
        alphas_risk_state, alphas_risk_obs,
        obs_position, obs_mean_radius):
        # return obstacle avoidance constraints g(x) <= 0
        # for all x in xs.
        # - xs must be of size (S, n_x)
        # - Sigmas must be of size (S, n_x, n_x)
        # - alphas_risk_state must be of size (S)
        # - alphas_risk_obs must be of size (S)
        # returns a vector of size S.
        S = self.S
        
        obs_positions = jnp.repeat(
            obs_position[jnp.newaxis, :], S, axis=0) 
        obs_mean_radii = jnp.repeat(
            obs_mean_radius, S)
        ineqs = vmap(self.obstacle_avoidance_constraint_for_one_obstacle_at_x)(
            xs, Sigmas, 
            alphas_risk_state, alphas_risk_obs,
            obs_positions, obs_mean_radii)
        return ineqs

    @partial(jit, static_argnums=(0,))
    def obstacle_avoidance_constraints(self, 
        xs, Sigmas,
        alphas_risk_state, alphas_risk_obs):
        # returns matrix g(x) such that g(x) <= 0
        # encodes all obstacle avoidance constraints
        # for one sample:
        #   xs is of shape (S+1, n_x)
        #   Sigmas is of shape (S+1, n_x, n_x)
        #   alphas_risk_state is of shape (S, n_obs)
        #   alphas_risk_obs is of shape (n_obs)
        # g(x) is of size (n_obs, S)
        S = self.S
        
        xs_vmapped = jnp.repeat(
            xs[jnp.newaxis, 1:, :], n_obs, axis=0)
        Sigmas_vmapped = jnp.repeat(
            Sigmas[jnp.newaxis, 1:, :, :], n_obs, axis=0)
        alphas_risk_obs_vmapped = jnp.repeat(
            alphas_risk_obs[:, jnp.newaxis], S, axis=1)
        cons = vmap(self.obstacle_avoidance_constraints_for_one_obstacle)(
            xs_vmapped, Sigmas_vmapped,
            alphas_risk_state.T, alphas_risk_obs_vmapped,
            self.obs_positions, self.obs_radii)
        return cons

    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------
    ### ----------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def get_control_and_risk_constraints(self, Z):
        # Returns (gs(Z), gs_l, gs_u) corresponding to 
        # control and risk constraints
        # such that gs_l <= gs(Z) <= gs_u.
        S = self.S
        
        us_vec, alphas_risk = self.convert_z_to_variables(Z)
        us_mat = self.convert_us_vec_to_us_mat(us_vec)

        gs = jnp.zeros(n_u*S + S*n_obs + n_obs + 1)
        gs_l = jnp.zeros(n_u*S + S*n_obs + n_obs + 1)
        gs_u = jnp.zeros(n_u*S + S*n_obs + n_obs + 1)
        # control bounds
        gs = gs.at[:(S*n_u)].set(us_vec)
        gs_l = gs_l.at[:(S*n_u)].set(self.u_min)
        gs_u = gs_u.at[:(S*n_u)].set(self.u_max)
        # bounds on risk allocation
        # 0 <= alpha_i <= alpha
        gs = gs.at[(S*n_u):-1].set(alphas_risk)
        gs_l = gs_l.at[(S*n_u):-1].set(1e-6)
        gs_u = gs_u.at[(S*n_u):-1].set(self.alpha)
        # 0 <= sum_{t=1}^{S} alpha_i <= alpha
        gs = gs.at[-1].set(jnp.sum(alphas_risk))
        gs_l = gs_l.at[-1].set(0)
        gs_u = gs_u.at[-1].set(self.alpha)
        return gs, gs_l, gs_u

    @partial(jit, static_argnums=(0,))
    def get_all_state_constraints(self, Z):
        S = self.S
        
        def all_constraints_us_alphas(us_mat, alphas_risk):
            alphas_risk_state = alphas_risk[:(S*n_obs)]
            alphas_risk_obs = alphas_risk[(S*n_obs):]
            alphas_risk_state = jnp.reshape( # (S, n_obs)
                alphas_risk_state, (n_obs, S), 'F').T

            xs = self.us_to_state_trajectory(us_mat)
            Sigmas = self.us_to_covariance_trajectory(us_mat)
            val_final = self.final_constraints(xs)
            val_obs = self.obstacle_avoidance_constraints(
                xs, Sigmas,
                alphas_risk_state, alphas_risk_obs)
            # add state bounds
            val_state_bounds_high = xs[:, :2] - jnp.array([0.5, 0.5])
            val_state_bounds_low = -xs[:, :2] + jnp.array([-2, -0.5])
            val_obs = jnp.concatenate([
                val_obs.flatten(), 
                val_state_bounds_high.flatten(), 
                val_state_bounds_low.flatten()
                ], axis=0)
            return (val_final, val_obs)

        us_vec, alphas_risk = self.convert_z_to_variables(Z)
        us_mat = self.convert_us_vec_to_us_mat(us_vec)

        g_final, g_obs = all_constraints_us_alphas(
            us_mat, alphas_risk)
        return g_final, g_obs

    # Objective to minimize
    def f(self, Z):
        S, dt = self.S, self.dt
        
        us_vec, _ = self.convert_z_to_variables(Z)
        us = self.convert_us_vec_to_us_mat(us_vec)

        obj = 0.
        # control cost
        for t in range(S):
            for i in range(n_u):
                obj = obj + 2 * dt * R[i, i] * (us[t, i]**2)
        return obj

# ***************************************

if B_compute_solution_gaussian:
    print("---------------------------------------")
    print("[drone_gaussian.py] >>> Computing solution")
    for alpha in alphas:
        model = Model(S=S, alpha=alpha)
        Z0 = model.initial_guess()

        print(">>> IPOPT: defining stochastic program")
        # Objective to minimize
        def f(Z):
            assert len(Z) == num_vars
            return model.f(Z)
        # Inequality Constraints
        # gL <= g(Z) <= gU
        def g(Z):
            assert len(Z) == num_vars
            gs_final, gs_obs = model.get_all_state_constraints(Z)
            gs_control, _, _ = model.get_control_and_risk_constraints(Z)
            gs = jnp.concatenate([
                gs_final,
                gs_obs,
                gs_control])
            return gs
        def gL_gU(Z):
            # note: this code below, copy-pasted from g(Z),
            # is there to get the dimensions of each constraints
            # and define gL, gU such that
            #   gL <= gU(Z) <= gU
            # (gL, gU) are then passed to IPOPT
            gs_final, gs_obs = model.get_all_state_constraints(Z)
            gs_control, gs_control_l, gs_control_u = model.get_control_and_risk_constraints(Z)
            gs = np.concatenate([
                gs_final,
                gs_obs,
                gs_control])
            g_L = np.zeros_like(gs)
            g_U = np.zeros_like(gs)
            num_equality_constraints = len(gs_final)
            # inequality constraints (one-sided g(x) <= 0)
            g_L[num_equality_constraints:] = -1e15
            # inequality constraints (two-sided a <= g(x) <= b)
            idx_control = num_equality_constraints+len(gs_obs)
            g_L[idx_control:(idx_control+len(gs_control))] = gs_control_l
            g_U[idx_control:(idx_control+len(gs_control))] = gs_control_u
            return g_L, g_U
        print("Jitting functions (JAX)")
        g_L, g_U = gL_gU(Z0)
        f = jit(f)
        g = jit(g)
        grad_f_jax = jit(grad(f))
        grad_g_jax = jit(jacfwd(g))
        _ = f(Z0), g(Z0), grad_f_jax(Z0), grad_g_jax(Z0)
        nvar = num_vars
        ncon = len(g(Z0))
        # Hessians
        hess_f = jit(jacfwd(jacfwd(f)))
        def lagrange_dot_g(x, lagrange):
            return jnp.dot(lagrange, g(x))
        def hess_lagrange_dot_g(x, lagrange):
            hess = jacfwd(jacfwd(lagrange_dot_g))(x, lagrange)[:nvar,:nvar]
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
                   'tol': 1e-8,
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
        us_vec, alphas_risk = model.convert_z_to_variables(jnp.array(Z))
        us = model.convert_us_vec_to_us_mat(us_vec)
        print("alphas_risk = ", alphas_risk)
        with open('results/drone_gaussian_alpha='+str(alpha)+'.npy',
            'wb') as f:
            xs = model.us_to_state_trajectory(us)
            np.save(f, us.to_py())
            np.save(f, xs.to_py())
    print("---------------------------------------")


if B_plot_results_gaussian:
    print("---------------------------------------")
    print("[drone_gaussian.py] >>> Plotting")
    alpha = alpha_to_plot
    with open('results/drone_gaussian_alpha='+str(alpha)+'.npy', 
        'rb') as f:
        model = Model(S=S, alpha=alpha)
        us = np.load(f)
        xs = np.load(f)
        Sigmas = model.us_to_covariance_trajectory(us)
    # plot
    fig = plt.figure(figsize=[6,3])
    ax = plt.gca()
    plt.plot(xs[:, 0], xs[:, 1], 
        c='b', alpha=0.3)
    for t in range(1, xs.shape[0]):
        mu, Sigma = xs[t, :2], Sigmas[t, :2, :2]
        plot_gaussian_confidence_ellipse(ax, mu, Sigma, 1-alpha)
    plt.scatter(x_init[0], x_init[1], color='k')
    plt.scatter(x_final[0], x_final[1], color='k')
    for i in range(n_obs):
        obs_position = obs_positions[i]
        obs_radius = obs_radii[i]
        obstacle = Circle(
            obs_position, 
            radius=obs_radius,
            color='r', alpha=0.3)
        plt.gca().add_patch(obstacle)
    offset = 0.15
    plt.text(0.05, 0.1, 
        r'$x_{g}$', fontsize=24, weight="bold")
    plt.text(-1.96, -0.1, 
        r'$x_{0}$', fontsize=24, weight="bold")
    plt.text(-0.75, 0.25, 
        r'$\mathcal{O}_j$', fontsize=28,
        color='#6d1300')
    plt.xlabel(r'$p_x$', fontsize=26)
    plt.ylabel(r'$p_y$', fontsize=26, rotation=0)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.close()
    print("---------------------------------------")