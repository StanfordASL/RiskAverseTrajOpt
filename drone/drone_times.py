import scipy.sparse as sp
import numpy as np
import osqp
from time import time
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib import rc, rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rc('text', usetex=True)
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import drone_params

# Load drone model parameters
OSQP_POLISH = drone_params.OSQP_POLISH
OSQP_TOL = drone_params.OSQP_TOL
n_x = drone_params.n_x
n_u = drone_params.n_u
S = drone_params.S
M = drone_params.M
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
B_compute_solution_saa = True
B_plot_computation_times = True
Ms = [20, 30, 50]
num_repeats_saa = 30
num_scp_iters_max = 15
alpha = 0.05
np.random.seed(0) # results may differ, the algorithm is randomized.
print("---------------------------------------")
print("[drone_times.py] with parameters:")
print("B_compute_solution_saa   =", B_compute_solution_saa)
print("B_plot_computation_times =", B_plot_computation_times)
print("Ms =", Ms)
print("num_repeats_saa =", num_repeats_saa)
print("alpha =", alpha)
print("---------------------------------------")

# save computation times
computation_times_define = np.zeros((
    num_repeats_saa, len(Ms), num_scp_iters_max))
computation_times_osqp = np.zeros((
    num_repeats_saa, len(Ms), num_scp_iters_max))
computation_times_cum = np.zeros((
    num_repeats_saa, len(Ms), num_scp_iters_max))
accuracy_error = np.zeros((
    num_repeats_saa, len(Ms), num_scp_iters_max))

for idx_M, M in enumerate(Ms):
    # Redefining the Model multiple times is not a clean way
    # of coding this, but it makes sure the class has the right
    # number of samples $M$ used to define constraints.
    class Model:
        def __init__(self, M, method='saa', alpha=0.1):
            print("Initializing Model with")
            print("> method =", method)
            print("> alpha  =", alpha)
            self.method = method
            # control bounds
            self.u_max = u_max
            self.u_min = -self.u_max
            # uncertain parameters
            self.alpha = alpha # risk parameter
            self.beta = 1e-2 # diffusion term magnitude
            self.drag_coefficient = 0.2
            if method == 'saa':
                # mass of the system
                self.masses = np.random.uniform(
                    mass_nom - mass_delta,
                    mass_nom + mass_delta, M)
                # semi axes of the obstacles
                self.obs_Qs = np.zeros((M, n_obs, 3, 3))
                for obs_i in range(n_obs):
                    for dim in range(3):
                        obs_delta_r = np.random.uniform(
                            -obs_radii_deltas,
                            obs_radii_deltas, M)
                        for i in range(M):
                            length = obs_radii[obs_i] + obs_delta_r[i]
                            self.obs_Qs[i, obs_i, dim, dim] = 1. / length**2
            if method == 'baseline':
                self.masses = np.random.uniform(
                    mass_nom - 0 * mass_delta,
                    mass_nom + 0 * mass_delta, M)
                self.obs_Qs = np.zeros((M, n_obs, 3, 3))
                for obs_i in range(n_obs):
                    for dim in range(3):
                        length = obs_radii[obs_i]
                        self.obs_Qs[:, obs_i, dim, dim] = 1. / length**2
            self.masses = jnp.array(self.masses)
            self.obs_Qs = jnp.array(self.obs_Qs)
            # Brownian motion increments
            self.DWs = np.zeros((M, S, n_x))
            for i in range(M):
                for t in range(S):
                    self.DWs[i, t, :] = np.random.randn(n_x)
            self.DWs = jnp.array(np.sqrt(dt) * self.DWs)
            if method == 'baseline':
                self.DWs = 0 * self.DWs 

        def convert_us_vec_to_us_mat(self, us_vec):
            us_mat = jnp.reshape(us_vec, (n_u, S), 'F')
            us_mat = us_mat.T # (S, n_u)
            us_mat = jnp.array(us_mat)
            return us_mat

        @partial(jit, static_argnums=(0,))
        def convert_us_mat_to_us_jaxvec(self, us_mat):
            us_vec = jnp.reshape(us_mat, (S*n_u), 'C')
            return us_vec

        def initial_guess_us_mat(self):
            us = jnp.zeros((S, n_u))
            u_initial_guess = (self.u_max + self.u_min) / 2.0
            # Add a small number to the initial guess.
            # This can help for many systems to avoid linearization
            # at a point where the dynamics are uncontrollable.
            # (e.g., linearizing cos(u) at u=0 (e.g., for
            #  Dubin's car dynamics) could cause issues)
            u_initial_guess = u_initial_guess + 1e-2
            for t in range(S):
                us = us.at[t, :].set(u_initial_guess)
            return us

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
        def sigma(self, x, u, mass):
            smat = jnp.zeros((n_x, n_x))
            smat = smat.at[3:6,3:6].set((self.beta/mass) * jnp.eye(3))
            return smat

        @partial(jit, static_argnums=(0,))
        def us_to_state_trajectory(self, us_mat, mass, dWs):
            # us_mat - (S, n_u)
            # mass   - scalar
            # dWs    - (S, n_x)
            # The dWs should already be indexed well, see @next_state
            xs = jnp.zeros((S+1, n_x))
            xs = xs.at[0, :].set(x_init)
            for t in range(S):
                xt, ut = xs[t, :], us_mat[t, :]
                dW_t = dWs[t]
                bt_dt = dt * self.b(xt, ut, mass)
                st_DWt = jnp.sqrt(dt) * self.sigma(xt, ut, mass) @ dW_t
                # Euler-Maruyama
                xn = xt + bt_dt + st_DWt
                xs = xs.at[t+1, :].set(xn)
            return xs

        def us_to_state_trajectories(self, us_mat):
            # us_mat - (S, n_u)
            # The dWs should already be indexed well, see @next_state
            Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
            Xs = vmap(self.us_to_state_trajectory)(
                Us, self.masses, self.DWs)
            return Xs

        @partial(jit, static_argnums=(0,))
        def final_constraints(self, xs):
            xT = xs[-1, :]
            return (xT - x_final)

        @partial(jit, static_argnums=(0,))
        def obstacle_avoidance_constraint_for_one_obstacle_at_x(self,
            x,
            obs_position, obs_Q_matrix):
            p = x[:2] # state position
            obs_p, obs_Q = obs_position[:2], obs_Q_matrix[:2, :2]
            # inequality constraint is 
            # (p-op).T @ oQ @ (p-op) >= 1
            # => 1 - (p-op).T @ oQ @ (p-op) <= 0
            constraint = 1. - (p - obs_p).T @ obs_Q @ (p - obs_p)
            return constraint

        @partial(jit, static_argnums=(0,))
        def obstacle_avoidance_constraints_for_one_obstacle(self,
            xs,
            obs_position, obs_Q_matrix):
            # return obstacle avoidance constraints g(x) <= 0
            # for all x in xs.
            # xs must be of size (S, n_x)
            # returns a vector of size S.
            obs_positions = jnp.repeat(
                obs_position[jnp.newaxis, :], S, axis=0) 
            obs_Q_matrices = jnp.repeat(
                obs_Q_matrix[jnp.newaxis, :, :], S, axis=0)
            ineqs = vmap(self.obstacle_avoidance_constraint_for_one_obstacle_at_x)(
                xs, obs_positions, obs_Q_matrices)
            return ineqs

        @partial(jit, static_argnums=(0,))
        def obstacle_avoidance_constraints(self, 
            xs, 
            obs_Q):
            # returns matrix g(x) such that g(x) <= 0
            # encodes all obstacle avoidance constraints
            # for one sample:
            #   xs is of shape (S+1, n_x)
            #   obs_Q is of shape (n_obs, 3, 3)
            # g(x) is of size (n_obs, S)
            xs_vmapped = jnp.repeat(
                xs[jnp.newaxis, 1:, :], n_obs, axis=0)
            cons = vmap(self.obstacle_avoidance_constraints_for_one_obstacle)(
                xs_vmapped,
                obs_positions, obs_Q)
            return cons

        ### ----------------------------------------------------
        ### ----------------------------------------------------
        ### ----------------------------------------------------
        ### ----------------------------------------------------
        ### OSQP

        @partial(jit, static_argnums=(0,))
        def get_control_constraints_coeffs_all(self):
            # Returns (A, l, u) corresponding to control constraints
            # such that l <= A uvec <= u.
            A = jnp.zeros((n_u*S, n_u*S + M + 2)) # (M + 2) are risk parameters
            l = jnp.zeros(n_u*S)
            u = jnp.zeros(n_u*S)
            # control bounds
            for t in range(S):
                for i in range(n_u):
                    idx_u = t*n_u + i
                    idx_con = idx_u
                    A = A.at[idx_con, idx_u].set(1.0)
                    l = l.at[idx_con].set(self.u_min)
                    u = u.at[idx_con].set(self.u_max)
            return A, l, u

        @partial(jit, static_argnums=(0,))
        def get_all_constraints_coeffs(self, 
            us_mat, 
            mass, dWs, obs_Q):
            # Returns (A, l, u) corresponding to all constraints
            # such that l <= A uvec <= u.
            def all_constraints_us(
                us_mat,
                mass, dWs, obs_Q):
                xs = self.us_to_state_trajectory(us_mat, mass, dWs)
                val_final = self.final_constraints(xs)
                val_obs = self.obstacle_avoidance_constraints(xs, obs_Q)
                return (val_final, val_obs)
            def all_constraints_dus(
                us_mat,
                mass, dWs, obs_Q):
                jac = jacfwd(all_constraints_us)(
                    us_mat,
                    mass, dWs, obs_Q)
                return jac

            v_final, g_obs = all_constraints_us(us_mat, mass, dWs, obs_Q)
            v_final_du, g_obs_du = all_constraints_dus(us_mat, mass, dWs, obs_Q)

            # reshape gradient
            v_final_du = jnp.reshape(v_final_du, (n_x, n_u*S), 'C')
            g_obs = jnp.reshape(g_obs, (n_obs, S), 'C')
            g_obs_du = jnp.reshape(g_obs_du, (n_obs, S, n_u*S), 'C')

            # final constraints
            val_final = -v_final + v_final_du @ self.convert_us_mat_to_us_jaxvec(us_mat)
            val_final_lower = val_final
            val_final_upper = val_final

            # obstacle_constraints
            # gi(u) <= 0
            # => ∇g(u_k) @ u - t - y <= -g(u_k) + ∇g(u_k) @ u_k (linearized)
            g_up = -g_obs + g_obs_du @ self.convert_us_mat_to_us_jaxvec(us_mat)

            return v_final_du, val_final_lower, val_final_upper, g_obs_du, g_up

        @partial(jit, static_argnums=(0,))
        def get_all_constraints_coeffs_all(self, us_mat):
            Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)

            final_du, final_low, final_up, g_obs_du, g_obs_up = vmap(
                self.get_all_constraints_coeffs)(Us, 
                self.masses, self.DWs, self.obs_Qs)

            # final constraints
            # sample average approximation with delta_M=0
            final_du = jnp.mean(final_du, axis=0)
            final_low = jnp.mean(final_low, axis=0)
            final_up = jnp.mean(final_up, axis=0)
            final_dparams = jnp.concatenate((
                final_du, 
                jnp.zeros((final_du.shape[0], M + 2))),
                axis=-1) # add risk parameters


            # obstacle avoidance constraints
            if self.method == 'baseline':
                print("baseline")
                obs_low = -jnp.inf* jnp.ones(M*n_obs*S)
                obs_up = jnp.inf * jnp.ones(M*n_obs*S)
                obs_dparams = jnp.zeros((M*n_obs*S, n_u*S + M + 2))
                for i in range(M):
                    idx = i*n_obs*S
                    idx_next = (i+1)*n_obs*S
                    obs_dparams = obs_dparams.at[idx:idx_next, :(n_u*S)].set(
                        jnp.reshape(g_obs_du[i], (n_obs*S, n_u*S), 'C'))
                    obs_up = obs_up.at[idx:idx_next].set(g_obs_up[i].flatten())

            elif self.method == 'saa':
                # pack so that 
                # val_low <= val_dparams @ params <= val_up
                # where params = (us, ys, t)
                obs_low = -jnp.inf* jnp.ones(1 + M + M*n_obs*S + 1)
                obs_up = jnp.inf * jnp.ones(1 + M + M*n_obs*S + 1)
                obs_dparams = jnp.zeros((1 + M + M*n_obs*S + 1, n_u*S + M + 2))

                # (M*alpha)t + sum_{i=1}^M yi <= 0
                obs_dparams = obs_dparams.at[0, -1].set(M*self.alpha)
                obs_dparams = obs_dparams.at[0, (n_u*S):-1].set(1.0)
                obs_up = obs_up.at[0].set(0.0)
                for i in range(M):
                    idx_yi = n_u*S + i

                    # -yi <= 0
                    idx = 1 + i
                    obs_dparams = obs_dparams.at[idx, idx_yi].set(-1.0)
                    obs_up = obs_up.at[idx].set(0.0)
                    # SLACK VARIABLE
                    # t - yi <= slack
                    obs_dparams = obs_dparams.at[idx, -2].set(-1.0)
                    # --------------------------------------------

                    # We multiply by MULTIPLIER due to low numerical precision
                    # the constraint is equivalent
                    MULTIPLIER = 0.01
                    # # gijk(u) - t - yi <= 0
                    # # => ∇g(u_k) @ u - t - yi <= -g(u_k) + ∇g(u_k) @ u_k (linearized)
                    # gijk(u) - yi <= 0
                    idx = 1 + M + i*n_obs*S
                    idx_next = 1 + M + (i+1)*n_obs*S
                    obs_dparams = obs_dparams.at[idx:idx_next, :(n_u*S)].set(
                        jnp.reshape(MULTIPLIER*g_obs_du[i], (n_obs*S, n_u*S), 'C'))
                    obs_dparams = obs_dparams.at[idx:idx_next, idx_yi].set(-MULTIPLIER)
                    obs_up = obs_up.at[idx:idx_next].set(MULTIPLIER*g_obs_up[i].flatten())
                    # gijk(u) - yi - t <= 0
                    obs_dparams = obs_dparams.at[idx:idx_next, -1].set(-MULTIPLIER)

                # 0 <= slack_variable
                obs_dparams = obs_dparams.at[-1, -2].set(-1.0)
                obs_up = obs_up.at[-1].set(0.0)

            # combine final constraints and obstacle avoidance constraints
            constraints_dparams = jnp.vstack([final_dparams, obs_dparams])
            constraints_low = jnp.hstack([final_low, obs_low])
            constraints_up = jnp.hstack([final_up, obs_up])
            return constraints_dparams, constraints_low, constraints_up
        # ------------------------------------------------------------

        @partial(jit, static_argnums=(0,))
        def get_objective_coeffs_jax(self):
            # See description of @get_objective_coeffs
            P = jnp.zeros((n_u*S + M + 2, n_u*S + M + 2)) # (M + 2) variables for risk constraint
            q = jnp.zeros(n_u*S + M + 2)
            # control squared
            for t in range(S):
                idx = t*n_u
                # control norm squared
                P = P.at[idx:idx+n_u, idx:idx+n_u].set(2*dt*R)
            # slack variable
            P = P.at[-2, -2].set(10000.0)
            q = q.at[-2].set(10000.0)
            return P, q

        def get_objective_coeffs(self):
            # Returns (P, q) corresponding to objective
            #        min (1/2 z^T P z + q^T z)
            # where z = umat is the optimization variable.
            P, q = self.get_objective_coeffs_jax()
            P, q = sp.csc_matrix(P.to_py()), q.to_py()
            return P, q

        def get_constraints_coeffs(self, us_mat, scp_iter):
            # Constraints: l <= A z <= u, with z = umat

            # control constraints
            A_con, l_con, u_con = self.get_control_constraints_coeffs_all()
            # combined final + obstacle avoidance constraints
            As, ls, us = self.get_all_constraints_coeffs_all(us_mat)

            # Jax => numpy
            A_con, l_con, u_con = A_con.to_py(), l_con.to_py(), u_con.to_py()
            As, ls, us = np.copy(As), np.copy(ls), np.copy(us)

            if scp_iter < 2:
                # remove obstacle avoidance
                As[n_x:] *= 1e-5
                ls[n_x:] = -10.0
                us[n_x:] = 10.0

            As, A_con = sp.csr_matrix(As), sp.csr_matrix(A_con)
            A = sp.vstack([As, A_con], format='csc')
            l = np.hstack([ls, l_con])
            u = np.hstack([us, u_con])
            return A, l, u

        def define_problem(self, us_mat_p, verbose=False):
            scp_iter = 2 # make sure that the osqp problem is defined 
                         # with collision avoidacne constraints
            # objective and constraints
            self.P, self.q = self.get_objective_coeffs()
            self.A, self.l, self.u = self.get_constraints_coeffs(
                us_mat_p, scp_iter)
            # Setup OSQP problem
            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(
                self.P, self.q, self.A, self.l, self.u,
                eps_abs=OSQP_TOL, eps_rel=OSQP_TOL,
                # linsys_solver="qdldl",
                linsys_solver="mkl pardiso",
                # to install MKL Pardiso on ubuntu:
                # https://github.com/eddelbuettel/mkl4deb
                warm_start=False, verbose=verbose,
                polish=OSQP_POLISH)
            return True

        def update_problem(self, us_mat_p, scp_iter=0, verbose=False):
            # -----------------------------------------------------------------
            # objective and constraints
            self.P, self.q = self.get_objective_coeffs()
            self.A, self.l, self.u = self.get_constraints_coeffs(
                us_mat_p, scp_iter)

            # Setup OSQP problem
            self.osqp_prob.update(l=self.l, u=self.u)
            self.osqp_prob.update(Ax=self.A.data)
            return True

        def solve(self, verbose=True):
            self.res = self.osqp_prob.solve()
            if self.res.info.status != 'solved':
                print("[solve]: Problem infeasible.")
            us_sol = self.convert_us_vec_to_us_mat(self.res.x[:(n_u*S)])
            ys, t_risk_sol = self.res.x[(n_u*S):-2], self.res.x[-1]
            # Note: we should have yi >= 0 for all i.
            # This may not be the case due to innaccuracies from OSQP.
            # Polishing (polish=True) should make it always true if it works
            # It may be easier (and faster) to pad the constraints for yi.
            if verbose:
                print("y_min =", np.min(ys))
                print("slack_var =", self.res.x[-2])
            return us_sol, t_risk_sol

    def L2_error_us(us_mat, us_mat_prev):
        # us_mat - (S, n_u)
        # us_mat_prev - (S, n_u)
        error = np.mean(np.linalg.norm(us_mat-us_mat_prev, axis=-1))
        error = error / np.mean(np.linalg.norm(us_mat, axis=-1))
        return error


    # -----------------------------------------
    if B_compute_solution_saa:
        print("---------------------------------------")
        print("[drone_times.py] >>> Computing SAA solution with M =", M)
        for idx_repeat in range(num_repeats_saa):
            model = Model(M, 'saa', alpha)

            # Initial compilation (JAX)
            us_prev = model.initial_guess_us_mat()
            model.define_problem(
                us_prev, 
                verbose=False)
            for scp_iter in range(5):
                model.update_problem(
                    us_prev, 
                    scp_iter, 
                    verbose=False)
                us, t_risk = model.solve(
                    verbose=False)
                us_prev = us

            # Solve the problem
            init_time = time()
            total_define_time = 0
            total_solve_time = 0
            us_prev = model.initial_guess_us_mat()
            for scp_iter in range(num_scp_iters_max):
                # define
                define_time = time()
                model.update_problem(
                    us_prev, 
                    scp_iter,
                    verbose=False)
                define_time = time()-define_time
                total_define_time += define_time
                # solve
                solve_time = time()
                us, t_risk = model.solve(
                    verbose=False)
                solve_time = time()-solve_time
                total_solve_time += solve_time
                # compute error
                L2_error = L2_error_us(us, us_prev)
                us_prev = us
                # save values
                computation_times_define[idx_repeat, idx_M, scp_iter] = define_time
                computation_times_osqp[idx_repeat, idx_M, scp_iter] = solve_time
                if scp_iter == 0:
                    computation_times_cum[idx_repeat, idx_M, scp_iter] = (
                    define_time + solve_time)
                else:
                    computation_times_cum[idx_repeat, idx_M, scp_iter] = (
                        computation_times_cum[idx_repeat, idx_M, scp_iter-1] + 
                        define_time + solve_time)
                accuracy_error[idx_repeat, idx_M, scp_iter] = L2_error

            with open('results/drone_computation_times.npy', 
                'wb') as f:
                np.save(f, Ms)
                np.save(f, computation_times_define)
                np.save(f, computation_times_osqp)
                np.save(f, computation_times_cum)
                np.save(f, accuracy_error)
    print("---------------------------------------")


if B_plot_computation_times:
    print("---------------------------------------")
    print("[drone_times.py] >>> Plotting computation times")
    with open('results/drone_computation_times.npy', 
        'rb') as f:
        Ms = np.load(f)
        computation_times_define = np.load(f)
        computation_times_osqp = np.load(f)
        computation_times_cum = np.load(f)
        accuracy_error = np.load(f)
        computation_times_define = computation_times_define[:, :, :num_scp_iters_max]
        computation_times_osqp = computation_times_osqp[:, :, :num_scp_iters_max]
        computation_times_cum = computation_times_cum[:, :, :num_scp_iters_max]
        accuracy_error = accuracy_error[:, :, :num_scp_iters_max]

    # scp errors
    idx = 1
    first_scp_iter = 2
    accuracy_error_median = np.median(accuracy_error, axis=0)
    accuracy_error_median = accuracy_error_median[idx, :]
    accuracy_error_median = accuracy_error_median[first_scp_iter:]
    scp_iters = np.arange(num_scp_iters_max)[first_scp_iter:] + 1
    print("plotting for M =", Ms[idx])
    print("plotting for scp_iters =", scp_iters)
    fig = plt.figure(figsize=[10, 3.7])
    plt.grid()
    plt.scatter(
        scp_iters,
        accuracy_error_median,
        color='k')
    plt.plot(
        scp_iters,
        accuracy_error_median,
        color='k')
    plt.yscale('log') 
    plt.xlabel(r'SCP Iteration $k$', fontsize=24)
    plt.ylabel(r'Relative error', fontsize=24)
    plt.ylabel(r'$\frac{\|u^{k}-u^{k-1}\|}{\|u^{k}\|}$', 
        fontsize=32, rotation=0, labelpad=60)
    plt.xticks([i for i in scp_iters], ([i for i in scp_iters]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close()

    # computation times as a function of scp iter
    width = 0.35 
    fig = plt.subplots(figsize = (10, 3.2))
    plt.grid(axis='y')
    computation_times_define = np.median(computation_times_define, axis=0)
    computation_times_osqp = np.median(computation_times_osqp, axis=0)
    computation_times_cum = np.median(computation_times_cum, axis=0)
    computation_times_define = computation_times_define[idx, :]
    computation_times_osqp = computation_times_osqp[idx, :]
    computation_times_cum = computation_times_cum[idx, :]
    computation_times_define = computation_times_define[first_scp_iter:]
    computation_times_osqp = computation_times_osqp[first_scp_iter:]
    computation_times_cum = computation_times_cum[first_scp_iter:]
    p1 = plt.bar(scp_iters, 
        1e3 * computation_times_define, 
        width,
        yerr =  0 * scp_iters,
        color='#0C7BDC')
    p2 = plt.bar(scp_iters, 
        1e3 * computation_times_osqp, 
        width, 
        bottom = 1e3 * computation_times_define,
        yerr =  0 * scp_iters,
        color='#FFC20A')
    plt.legend((p1[0], p2[0]), ('define', 'solve'),
        fontsize=16,
        loc='upper center')
    plt.xlabel(r'SCP iteration $k$', fontsize=24)
    plt.ylabel(r'Time / SCP iter. (ms)', fontsize=24)
    plt.xlim([np.min(scp_iters)-1, np.max(scp_iters)+1])
    plt.xticks([i for i in scp_iters], ([i for i in scp_iters]))
    plt.xticks([i for i in scp_iters], ([i for i in scp_iters]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax2 = plt.gca().twinx()
    ax2.plot(
        scp_iters, 
        1e3 * computation_times_cum, 
        'k--',
        linewidth=3)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel(r'Total time (ms)', fontsize=24)
    plt.tight_layout()
    plt.show()

    # computation times as a function of M
    with open('results/drone_computation_times.npy', 
        'rb') as f:
        Ms = np.load(f)
        computation_times_define = np.load(f)
        computation_times_osqp = np.load(f)
        computation_times_cum = np.load(f)
        accuracy_error = np.load(f)
        computation_times_define = computation_times_define[:, :, :num_scp_iters_max]
        computation_times_osqp = computation_times_osqp[:, :, :num_scp_iters_max]
        computation_times_cum = computation_times_cum[:, :, :num_scp_iters_max]
        accuracy_error = accuracy_error[:, :, :num_scp_iters_max]
    scp_iter = 10 - 1 # python indexing -> this gives the 10th scp iteration
    computation_times_cum = np.median(computation_times_cum, axis=0)
    computation_times_cum_all_Ms = computation_times_cum[:, scp_iter]
    fig = plt.subplots(figsize = (4, 3.2))
    plt.grid(axis='y')
    Ms_vec_nums = np.array([i for i in range(len(Ms))])
    plt.bar(Ms_vec_nums, 
        1e3 * computation_times_cum_all_Ms, 
        2*width,
        yerr =  0 * np.array(Ms),
        color='#0C7BDC')
    plt.xlabel(r'Sample size $M$', fontsize=24)
    plt.ylabel(r'Time (ms)', fontsize=24)
    plt.xticks(Ms_vec_nums, ([i for i in Ms]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()