from pathlib import Path
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
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import driving_params

# load driving parameters
OSQP_POLISH = driving_params.OSQP_POLISH
OSQP_TOL = driving_params.OSQP_TOL
n_x = driving_params.n_x
n_u = driving_params.n_u
S = driving_params.S
M = driving_params.M
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
std_matrix_ped_initial_state = jnp.sqrt(
    driving_params.variance_ped_initial_state)

B_compute_solution_saa = False
B_compute_solution_base = False
B_plot_results_saa = False
B_validate_monte_carlo = True
B_plot_computation_times = False
alphas = [0.01, 0.02, 0.05, 0.1]
num_repeats_saa = 30
num_scp_iters_max = 15
np.random.seed(0)
print("---------------------------------------")
print("[driving.py] >>> Running with parameters:")
print("B_compute_solution_saa   =", B_compute_solution_saa)
print("B_compute_solution_base  =", B_compute_solution_base)
print("B_plot_results_saa       =", B_plot_results_saa)
print("B_validate_monte_carlo   =", B_validate_monte_carlo)
print("B_plot_computation_times =", B_plot_computation_times)
print("alphas =", alphas)
print("num_repeats_saa =", num_repeats_saa)
print("---------------------------------------")

# save computation times
computation_times_define = np.zeros((
    num_repeats_saa, len(alphas), num_scp_iters_max))
computation_times_osqp = np.zeros((
    num_repeats_saa, len(alphas), num_scp_iters_max))
computation_times_cum = np.zeros((
    num_repeats_saa, len(alphas), num_scp_iters_max))
accuracy_error = np.zeros((
    num_repeats_saa, len(alphas), num_scp_iters_max))

class Model:
    def __init__(self, M, method='saa', alpha=0.05):
        print("Initializing Model with")
        print("> method =", method)
        print("> alpha  =", alpha)
        self.method = method
        # control bounds
        self.u_max = u_max
        self.u_min = -self.u_max
        # uncertain parameters
        self.alpha = alpha # risk parameter
        self.beta = 3e-2 # diffusion term magnitude
        self.omegas_speed = np.random.uniform(
            omega_speed_nom - omega_speed_del,
            omega_speed_nom + omega_speed_del, M)
        self.omegas_repulsive = np.random.uniform(
            omega_repulsive_nom - omega_repulsive_del,
            omega_repulsive_nom + omega_repulsive_del, M)
        self.omegas_speed = jnp.array(self.omegas_speed)
        self.omegas_repulsive = jnp.array(self.omegas_repulsive)
        # initial states
        states_init = jnp.repeat(state_init[jnp.newaxis, :], M, axis=0)
        if method=='saa':
            for i in range(M):
                samples = std_matrix_ped_initial_state @ np.random.randn(4)
                states_init = states_init.at[i, 4:].set(
                    states_init[i, 4:] + samples)
        self.states_init = states_init
        # Brownian motion (BM)
        self.DWs = np.zeros((M, S, n_x)) # Brownian motion increments
        for i in range(M):
            for t in range(S):
                self.DWs[i, t, :] = np.random.randn(n_x)
        self.DWs = jnp.array(np.sqrt(dt) * self.DWs)
        if method=='baseline':
            self.DWs = 0 * self.DWs 
            self.omegas_speed = 0 * self.omegas_speed 
            self.omegas_repulsive = 0 * self.omegas_repulsive 

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
    def sigma(self, x, u):
        smat = jnp.zeros((n_x, n_x))
        smat = smat.at[6:,6:].set(self.beta * jnp.eye(2))
        return smat

    @partial(jit, static_argnums=(0,))
    def us_to_state_trajectory(self, us_mat,
        state_init, omega_speed, omega_repulsive, dWs):
        # us_mat - (S, n_u)
        # mass   - scalar
        # dWs    - (S, n_x)
        xs = jnp.zeros((S+1, n_x))
        xs = xs.at[0, :].set(state_init)
        for t in range(S):
            xt, ut = xs[t, :], us_mat[t, :]
            dW_t = dWs[t]
            # predicts one-step for sample
            bt_dt = dt * self.b(xt, ut, 
                omega_speed, omega_repulsive)
            st_DWt = jnp.sqrt(dt) * self.sigma(xt, ut) @ dW_t
            # Euler-Maruyama
            xn = xt + bt_dt + st_DWt
            xs = xs.at[t+1, :].set(xn)
        return xs

    @partial(jit, static_argnums=(0,))
    def us_to_state_trajectories(self, us_mat):
        # us_mat  - (S, n_u)
        # masses  - (M,)
        # dWs_all - (M, S, n_x)
        Us = jnp.repeat(us_mat[jnp.newaxis, :, :], M, axis=0)
        Xs = vmap(self.us_to_state_trajectory)(
            Us, self.states_init, self.omegas_speed, self.omegas_repulsive, self.DWs)
        return Xs

    @partial(jit, static_argnums=(0,))
    def final_constraints(self, xs):
        state_ego_goal = jnp.concatenate((
            position_ego_goal,
            velocity_ego_goal), axis=-1)
        return (xs[-1, :4] - state_ego_goal)

    @partial(jit, static_argnums=(0,))
    def separation_distance_ego_to_pedestrian(self, x):
            position_ego = x[0:2]
            position_ped = x[4:6]
            positions_delta = position_ego - position_ped
            distance = jnp.linalg.norm(positions_delta)
            distance = distance - min_separation_distance
            return distance

    @partial(jit, static_argnums=(0,))
    def separation_distances_at_all_times(self, xs):
        distances = vmap(self.separation_distance_ego_to_pedestrian)(
            xs[1:])
        return distances

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
    def get_all_constraints_coeffs(self, us_mat, 
        state_init, omega_speed, omega_repulsive, dWs):
        # Returns (A, l, u) corresponding to all constraints
        # such that l <= A uvec <= u.
        def all_constraints_us(us_mat, 
            state_init, omega_speed, omega_repulsive, dWs):
            xs = self.us_to_state_trajectory(us_mat,
                state_init, omega_speed, omega_repulsive, dWs)
            val_final = self.final_constraints(xs)
            val_obs = -self.separation_distances_at_all_times(xs)
            return (val_final, val_obs)
        def all_constraints_dus(us_mat,
            state_init, omega_speed, omega_repulsive, dWs):
            jac = jacfwd(all_constraints_us)(
                us_mat, state_init, omega_speed, omega_repulsive, dWs)
            return jac

        v_final, g_obs = all_constraints_us(us_mat, 
            state_init, omega_speed, omega_repulsive, dWs)
        v_final_du, g_obs_du = all_constraints_dus(us_mat, 
            state_init, omega_speed, omega_repulsive, dWs)

        # reshape gradient
        v_final_du = jnp.reshape(v_final_du, (4, n_u*S), 'C')
        g_obs = jnp.reshape(g_obs, (S), 'C')
        g_obs_du = jnp.reshape(g_obs_du, (S, n_u*S), 'C')

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

        final_du, final_lower, final_upper, gs_obs_du, gs_obs_up = vmap(
            self.get_all_constraints_coeffs)(Us, 
            self.states_init, self.omegas_speed, self.omegas_repulsive, self.DWs)

        # final constraints
        # sample average approximation with delta_M=0
        final_du = jnp.mean(final_du, axis=0)
        final_low = jnp.mean(final_lower, axis=0)
        final_up = jnp.mean(final_upper, axis=0)
        final_dparams = jnp.concatenate((
            final_du, 
            jnp.zeros((final_du.shape[0], M + 2))),
            axis=-1) # add risk parameter

        # obstacle avoidance constraints
        if self.method == 'baseline':
            obs_low = -jnp.inf* jnp.ones(M*S)
            obs_up = jnp.inf * jnp.ones(M*S)
            obs_dparams = jnp.zeros((M*S, n_u*S + M + 2))
            for i in range(M):
                idx = i*S
                idx_next = (i+1)*S
                obs_dparams = obs_dparams.at[idx:idx_next, :(n_u*S)].set(
                    jnp.reshape(gs_obs_du[i], (S, n_u*S), 'C'))
                obs_up = obs_up.at[idx:idx_next].set(gs_obs_up[i].flatten())

        elif self.method == 'saa':
            # pack so that 
            # val_low <= val_dparams @ params <= val_up
            # where params = (us, ys, t)
            obs_low = -jnp.inf* jnp.ones(1 + M + M*S + 1)
            obs_up = jnp.inf * jnp.ones(1 + M + M*S + 1)
            obs_dparams = jnp.zeros((1 + M + M*S + 1, n_u*S + M + 2))

            # (N*alpha)t + sum_{i=1}^M yi <= 0
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

                # gijk(u) - yi <= 0
                idx = 1 + M + i*S
                idx_next = 1 + M + (i+1)*S
                obs_dparams = obs_dparams.at[idx:idx_next, :(n_u*S)].set(
                    jnp.reshape(gs_obs_du[i], (S, n_u*S), 'C'))
                obs_dparams = obs_dparams.at[idx:idx_next, idx_yi].set(-1.0)
                obs_up = obs_up.at[idx:idx_next].set(gs_obs_up[i].flatten())
                # gijk(u) - yi - t <= 0
                obs_dparams = obs_dparams.at[idx:idx_next, -1].set(-1.0)

            # 0 <= slack_variable
            obs_dparams = obs_dparams.at[-1, -2].set(-1.0)
            obs_up = obs_up.at[-1].set(0.0)

        # combine final constraints and obstacle avoidance constraints
        constraints_dparams = jnp.vstack([final_dparams, obs_dparams])
        constraints_low = jnp.hstack([final_low, obs_low])
        constraints_up = jnp.hstack([final_up, obs_up])
        return constraints_dparams, constraints_low, constraints_up

    @partial(jit, static_argnums=(0,))
    def get_objective_coeffs_jax(self):
        # See description of @get_objective_coeffs
        # (M + 2) variables for risk constraint
        P = jnp.zeros((n_u*S + M + 2, n_u*S + M + 2))
        q = jnp.zeros(n_u*S + M + 2)
        # control squared
        for t in range(S):
            idx = t*n_u
            # control norm squared
            P = P.at[idx:idx+n_u, idx:idx+n_u].set(2*dt*R)
        # slack variable
        P = P.at[-2, -2].set(1000.0)
        q = q.at[-2].set(1000.0)
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

    def define_problem(self, us_mat_p, scp_iter=0, verbose=False):
        # objective and constraints
        self.P, self.q = self.get_objective_coeffs()
        self.A, self.l, self.u = self.get_constraints_coeffs(
            us_mat_p, scp_iter)
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


if B_compute_solution_saa:
    print("---------------------------------------")
    print("[driving.py] >>> Computing SAA solution")
    for idx_alpha, alpha in enumerate(alphas):
        for idx_repeat in range(num_repeats_saa):
            model = Model(M, 'saa', alpha)

            # Initial compilation (JAX)
            us_prev = model.initial_guess_us_mat()
            model.define_problem(us_prev, verbose=False)
            us, t_risk = model.solve()
            model.define_problem(us, 1, verbose=False)
            us, t_risk = model.solve()
            us_prev = us

            start_time = time()
            total_define_time = 0
            total_osqp_solve_time = 0
            us_prev = model.initial_guess_us_mat()
            for scp_iter in range(num_scp_iters_max):
                define_time = time()
                model.define_problem(
                    us_prev, 
                    scp_iter,
                    verbose=False)
                define_time = time()-define_time
                total_define_time += define_time

                solve_time = time()
                us, t_risk = model.solve()
                solve_time = time()-solve_time
                total_osqp_solve_time += solve_time

                L2_error = L2_error_us(us, us_prev)
                us_prev = us

                # save values
                computation_times_define[idx_repeat, idx_alpha, scp_iter] = define_time
                computation_times_osqp[idx_repeat, idx_alpha, scp_iter] = solve_time
                if scp_iter == 0:
                    computation_times_cum[idx_repeat, idx_alpha, scp_iter] = (
                    define_time + solve_time)
                else:
                    computation_times_cum[idx_repeat, idx_alpha, scp_iter] = (
                        computation_times_cum[idx_repeat, idx_alpha, scp_iter-1] + 
                        define_time + solve_time)
                accuracy_error[idx_repeat, idx_alpha, scp_iter] = L2_error
                

            with open('results/driving_alpha='+str(alpha)+
                '_repeat='+str(idx_repeat)+'.npy', 
                'wb') as f:
                xs = model.us_to_state_trajectories(us)
                np.save(f, us.to_py())
                np.save(f, xs.to_py())

    with open('results/driving_computation_times.npy', 
        'wb') as f:
        np.save(f, alphas)
        np.save(f, computation_times_define)
        np.save(f, computation_times_osqp)
        np.save(f, computation_times_cum)
        np.save(f, accuracy_error)
    print("---------------------------------------")



if B_compute_solution_base:
    print("---------------------------------------")
    print("[driving.py] >>> Computing baseline solution")
    model = Model(M, 'baseline')

    # Initial compilation
    us_prev = model.initial_guess_us_mat()
    model.define_problem(us_prev, verbose=False)
    us, t_risk = model.solve()
    model.define_problem(us, 1, verbose=False)
    us, t_risk = model.solve()
    us_prev = us

    us_prev = model.initial_guess_us_mat()

    for scp_iter in range(15):
        model.define_problem(us_prev, scp_iter,
            verbose=False)
        us, t_risk = model.solve()
        us_prev = us
        
    with open('results/driving_baseline.npy', 
        'wb') as f:
        xs = model.us_to_state_trajectories(us)
        np.save(f, us.to_py())
        np.save(f, xs.to_py())
    print("---------------------------------------")



### PLOT
if B_plot_results_saa:
    print("---------------------------------------")
    print("[driving.py] >>> Plotting")
    alpha = 0.01
    idx_repeat = 3
    with open('results/driving_alpha='+str(alpha)+
        '_repeat='+str(idx_repeat)+'.npy', 
        'rb') as f:
        us = np.load(f)
        xs = np.load(f)

    fig = plt.figure(figsize=[6,3])
    plt.grid()
    x_traj_mean = np.mean(xs, axis=0)
    # plot ego car trajectory
    colors = pl.cm.winter(np.linspace(0, 1, S+1))
    for t in range(S+1):
        angle = x_traj_mean[t, 3]
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])
        xy = x_traj_mean[t, :2]
        xy = xy - rotation_matrix @ np.array(
            [0.5 * ego_width, 0.5 * ego_height])
        rectangle = Rectangle(xy, ego_width, ego_height, 
            angle=angle * 180. / np.pi,
            color=colors[t], alpha=0.8, fill=False,
            linewidth=2)
        plt.gca().add_patch(rectangle)
    plt.scatter(x_traj_mean[:, 0], x_traj_mean[:, 1], 
             c=colors, s=30, marker='+')
    # plot pedestrian trajectory
    colors = pl.cm.cool(1-np.linspace(0, 1, S+1))
    for i in range(M):
        plt.scatter(xs[i, :, 4], xs[i, :, 5], 
                 c=colors,
                 s=10, alpha=0.3)
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




if B_validate_monte_carlo:
    print("---------------------------------------")
    print("[driving.py] >>> Monte Carlo")
    M = 10000
    model = Model(M)
    def monte_carlo_cost(us_mat):
        value = 0.
        for t in range(S):
            value = value + R[0, 0] * us_mat[t, 0] * us_mat[t, 0]
            value = value + R[1, 1] * us_mat[t, 1] * us_mat[t, 1]
        value = dt * value
        return value
    def monte_carlo_separation_constraints_verification(us_mat, 
        state_init, omega_speed, omega_repulsive, dWs):
        xs = model.us_to_state_trajectory(us_mat,
            state_init, omega_speed, omega_repulsive, dWs)
        val_obs = -model.separation_distances_at_all_times(xs)
        val_constraint = np.max(val_obs) - OSQP_TOL
        # val_constraint should be <= 0
        B_satisfied = val_constraint <= 1e-6
        return B_satisfied, val_constraint
    def monte_carlo_avar(Z_samples, alpha):
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

    for alpha in alphas:
        print("---------------------------")
        print("Monte-Carlo: alpha =", alpha)
        us_mat_all = jnp.zeros((num_repeats_saa, S, n_u))
        B_satisfied_mat = jnp.zeros((num_repeats_saa, M))
        avar_val_vec = jnp.zeros(num_repeats_saa)
        for idx_repeat in range(num_repeats_saa):
            with open('results/driving_alpha='+str(alpha)+
                '_repeat='+str(idx_repeat)+'.npy', 
                'rb') as f:
                us = np.load(f)
                xs = model.us_to_state_trajectories(us)

            us_vmapped = jnp.repeat(
                us[jnp.newaxis, :, :], M, axis=0) 
            B_satisfied_vec, val_constraint_vec = vmap(monte_carlo_separation_constraints_verification)(
                us_vmapped,
                model.states_init, model.omegas_speed, model.omegas_repulsive, model.DWs)
            avar_val = monte_carlo_avar(val_constraint_vec, alpha)
            # pack results
            us_mat_all = us_mat_all.at[idx_repeat, :, :].set(us)
            B_satisfied_mat = B_satisfied_mat.at[idx_repeat, :].set(B_satisfied_vec)
            avar_val_vec = avar_val_vec.at[idx_repeat].set(avar_val)
            print("B_satisfied_vec =", jnp.mean(B_satisfied_vec))
            # print("avar =", avar_val)
        print("percentage safe (mean) =", jnp.mean(B_satisfied_mat))
        print("avar (mean) =", jnp.mean(avar_val_vec))
        print("cost (mean) =", jnp.mean(vmap(monte_carlo_cost)(us_mat_all)))
        print("percentage safe (median) =", jnp.median(jnp.mean(B_satisfied_mat, axis=1)))
        print("avar (median) =", jnp.median(avar_val_vec))
        print("cost (median) =", jnp.median(vmap(monte_carlo_cost)(us_mat_all)))

    print("---------------------------")
    print("Monte-Carlo: deterministic baseline")
    with open('results/driving_baseline.npy', 
        'rb') as f:
        us = np.load(f)
    us_vmapped = jnp.repeat(
        us[jnp.newaxis, :, :], M, axis=0) 
    B_satisfied_vec, val_constraint_vec = vmap(monte_carlo_separation_constraints_verification)(
        us_vmapped,
        model.states_init, model.omegas_speed, model.omegas_repulsive, model.DWs)
    print("percentage safe =", jnp.mean(B_satisfied_vec))
    print("cost =", monte_carlo_cost(us))
    print("---------------------------------------")

    print("---------------------------")
    print("Monte-Carlo: Gaussian baseline")
    for alpha in alphas:
        print("---------------------------")
        print("Monte-Carlo: alpha =", alpha)
        my_file = "results/driving_gaussian_alpha="+str(alpha)+".npy"
        if not(Path(my_file).is_file()):
            msg = my_file + " does not exist.\n"
            msg += "run driving_gaussian.py first."
            raise FileNotFoundError(msg)
        with open(my_file, 'rb') as f:
            us = np.load(f)
            _ = np.load(f)
        us_vmapped = jnp.repeat(
            us[jnp.newaxis, :, :], M, axis=0) 
        B_satisfied_vec, val_constraint_vec = vmap(monte_carlo_separation_constraints_verification)(
            us_vmapped,
            model.states_init, model.omegas_speed, model.omegas_repulsive, model.DWs)
        print("percentage safe =", jnp.mean(B_satisfied_vec))
        print("cost =", monte_carlo_cost(us))
    print("---------------------------------------")


if B_plot_computation_times:
    print("---------------------------------------")
    print("[driving.py] >>> Computation times")
    with open('results/driving_computation_times.npy', 
        'rb') as f:
        alphas = np.load(f)
        computation_times_define = np.load(f)
        computation_times_osqp = np.load(f)
        computation_times_cum = np.load(f)
        accuracy_error = np.load(f)

        computation_times_define = computation_times_define[:, :, :num_scp_iters_max]
        computation_times_osqp = computation_times_osqp[:, :, :num_scp_iters_max]
        computation_times_cum = computation_times_cum[:, :, :num_scp_iters_max]
        accuracy_error = accuracy_error[:, :, :num_scp_iters_max]
    idx = 1
    first_scp_iter = 2
    accuracy_error_median = np.median(accuracy_error, axis=0)
    accuracy_error_median = accuracy_error_median[idx, :]
    accuracy_error_median = accuracy_error_median[first_scp_iter:]
    scp_iters = np.arange(num_scp_iters_max)[first_scp_iter:] + 1
    print("plotting for alpha =", alphas[idx])
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
    print("scp_iters =", scp_iters)
    print("computation_times_define =", computation_times_define)
    print("computation_times_osqp =", computation_times_osqp)
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
    print("computation_times_cum =", computation_times_cum)
    plt.tight_layout()
    plt.show()


    with open('results/driving_computation_times.npy', 
        'rb') as f:
        alphas = np.load(f)
        computation_times_define = np.load(f)
        computation_times_osqp = np.load(f)
        computation_times_cum = np.load(f)
        accuracy_error = np.load(f)

        computation_times_define = computation_times_define[:, :, :num_scp_iters_max]
        computation_times_osqp = computation_times_osqp[:, :, :num_scp_iters_max]
        computation_times_cum = computation_times_cum[:, :, :num_scp_iters_max]
        accuracy_error = accuracy_error[:, :, :num_scp_iters_max]
    scp_iter = 9
    computation_times_cum = np.median(computation_times_cum, axis=0)
    computation_times_cum_all_alphas = computation_times_cum[:, scp_iter]
    print("scp_iter =", scp_iter + 1)
    print("computation_times_cum_all_Ms =", computation_times_cum_all_alphas)
    fig = plt.subplots(figsize = (4, 3.2))
    plt.grid(axis='y')
    alphas_vec_nums = np.array([i for i in range(len(alphas))])
    plt.bar(alphas_vec_nums, 
        1e3 * computation_times_cum_all_alphas, 
        2*width,
        yerr =  0 * np.array(alphas),
        color='#0C7BDC')
    plt.xlabel(r'Risk parameter $\alpha$', fontsize=24)
    plt.ylabel(r'Time (ms)', fontsize=24)
    plt.xticks(alphas_vec_nums, ([i for i in alphas]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    print("---------------------------------------")
