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
from warnings import warn

B_compute_solution_saa = True
B_compute_solution_base = True
B_plot_results_saa = True
B_plot_results_base = True
B_validate_monte_carlo = True
alphas = [0.05, 0.1, 0.2, 0.3]
num_repeats_saa = 30
num_scp_iters_max = 20
np.random.seed(0) # results may differ, the algorithm is randomized.
print("---------------------------------------")
print("[drone_risk.py] >>> Running with parameters:")
print("B_compute_solution_saa   =", B_compute_solution_saa)
print("B_compute_solution_base  =", B_compute_solution_base)
print("B_plot_results_saa       =", B_plot_results_saa)
print("B_plot_results_base      =", B_plot_results_base)
print("B_validate_monte_carlo   =", B_validate_monte_carlo)
print("alphas =", alphas)
print("num_repeats_saa =", num_repeats_saa)
print("---------------------------------------")

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

class Model:
    def __init__(
        self, 
        DWs, masses, obs_Qs,
        method='saa', alpha=0.1):
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

        self.DWs = DWs
        self.masses = masses
        self.obs_Qs = obs_Qs

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
            us = us.at[t, :(n_u-1)].set(u_initial_guess)
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
            obs_low = -jnp.inf* jnp.ones(M*n_obs*S)
            obs_up = jnp.inf * jnp.ones(M*n_obs*S)
            obs_dparams = jnp.zeros((M*n_obs*S, n_u*S + M + 2))

            # We multiply by MULTIPLIER due to potentially low numerical precision
            # note that the constraint is equivalent.
            MULTIPLIER = 0.01

            for i in range(M):
                idx = i*n_obs*S
                idx_next = (i+1)*n_obs*S
                obs_dparams = obs_dparams.at[idx:idx_next, :(n_u*S)].set(
                    jnp.reshape(MULTIPLIER*g_obs_du[i], (n_obs*S, n_u*S), 'C'))
                obs_up = obs_up.at[idx:idx_next].set(
                    MULTIPLIER*g_obs_up[i].flatten())
                # add a small padding to make the baseline a little bit safer   
                # otherwise, collision rates are about 90% in this scenario, due
                # to the configuration of the obstacles (which is probably
                # not representative of usual collision rates with a deterministic
                # baseline)
                obs_up = obs_up.at[idx:idx_next].set(
                    obs_up[idx:idx_next] - 1e-3) 

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

                # We multiply by MULTIPLIER due to potentially low numerical precision
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
            As[n_x:] *= 1e-7
            ls[n_x:] = -0.1
            us[n_x:] = 0.1

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
            linsys_solver="qdldl",
            # linsys_solver="mkl pardiso",
            warm_start=True, verbose=verbose,
            polish=OSQP_POLISH)
        return True

    def update_problem(self, us_mat_p, scp_iter=0, verbose=False):
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
def sample_uncertain_parameters(method='saa', M=M):
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

DWs_all = np.zeros((num_repeats_saa, M, S, n_x))
masses_all = np.zeros((num_repeats_saa, M))
obs_Qs_all = np.zeros((num_repeats_saa, M, n_obs, 3, 3))
for idx_repeat in range(num_repeats_saa):
    DWs, masses, obs_Qs = sample_uncertain_parameters('saa')
    DWs_all[idx_repeat] = DWs
    masses_all[idx_repeat] = masses
    obs_Qs_all[idx_repeat] = obs_Qs
DWs_all = jnp.array(DWs_all)
masses_all = jnp.array(masses_all)
obs_Qs_all = jnp.array(obs_Qs_all)
# -----------------------------------------


# -----------------------------------------
if B_compute_solution_saa:
    print("---------------------------------------")
    print("[drone_risk.py] >>> Computing SAA solution")
    for idx_alpha, alpha in enumerate(alphas):
        for idx_repeat in range(num_repeats_saa):
            DWs = DWs_all[idx_repeat]
            masses = masses_all[idx_repeat]
            obs_Qs = obs_Qs_all[idx_repeat]
            model = Model(DWs, masses, obs_Qs, 'saa', alpha)

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
            us_prev = model.initial_guess_us_mat()
            for scp_iter in range(num_scp_iters_max):
                # define
                model.update_problem(
                    us_prev, 
                    scp_iter,
                    verbose=False)
                # solve
                us, t_risk = model.solve(
                    verbose=False)
                # compute error
                L2_error = L2_error_us(us, us_prev)
                us_prev = us
            
            with open('results/drone_alpha='+str(alpha)+
                '_repeat='+str(idx_repeat)+'.npy', 
                'wb') as f:
                xs = model.us_to_state_trajectories(us)
                np.save(f, us.to_py())
                np.save(f, xs.to_py())
    print("---------------------------------------")

if B_compute_solution_base:
    print("---------------------------------------")
    print("[drone_risk.py] >>> Computing baseline solution")
    # run baseline without uncertainty
    DWs, masses, obs_Qs = sample_uncertain_parameters('baseline')
    model = Model(DWs, masses, obs_Qs, 'baseline')
    # initial compilation
    us_prev = model.initial_guess_us_mat()
    model.define_problem(us_prev, verbose=False)
    for scp_iter in range(5):
        model.update_problem(
            us_prev, 
            scp_iter, 
            verbose=False)
        us, t_risk = model.solve(
            verbose=False)
        us_prev = us
    # solve
    us_prev = model.initial_guess_us_mat()
    for scp_iter in range(num_scp_iters_max):
        # define
        model.update_problem(
            us_prev, 
            scp_iter,
            verbose=False)
        # solve
        us, t_risk = model.solve(
            verbose=False)
        us_prev = us
    xs = model.us_to_state_trajectories(us)
    with open('results/drone_baseline.npy', 
        'wb') as f:
        np.save(f, us.to_py())
        np.save(f, xs.to_py())
print("---------------------------------------")




if B_plot_results_base or B_plot_results_saa:
    print("---------------------------------------")
    print("[drone_risk.py] >>> Plotting")
    for counter in range(2): # saa and baseline
        if counter == 0 and not(B_plot_results_base):
            continue
        if counter == 1 and not(B_plot_results_saa):
            continue
        if counter == 0 and B_plot_results_base:
            with open('results/drone_baseline.npy', 
                'rb') as f:
                us = np.load(f)
                xs = np.load(f)
            DWs, masses, obs_Qs = sample_uncertain_parameters('baseline')
            model = Model(DWs, masses, obs_Qs, 'baseline')
        if counter == 1 and B_plot_results_saa:
            alpha = 0.05
            idx_repeat = 1
            with open('results/drone_alpha='+str(alpha)+
                '_repeat='+str(idx_repeat)+'.npy', 
                'rb') as f:
                us = np.load(f)
                xs = np.load(f)
            DWs, masses, obs_Qs = sample_uncertain_parameters('saa')
            model = Model(DWs, masses, obs_Qs, 'saa')
        # plot
        fig = plt.figure(figsize=[6,3])
        for i in range(M):
            plt.plot(xs[i, :, 0], xs[i, :, 1], 
                c='b', alpha=0.3)
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
            r'$\mathcal{O}_j(\omega^i)$', fontsize=28,
            color='#6d1300')
        plt.text(-1.0, -0.32, 
            r'$x_u(t, \omega^j)$', fontsize=20,
            color='b')
        plt.xlabel(r'$p_x$', fontsize=26)
        plt.ylabel(r'$p_y$', fontsize=26, rotation=0)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
        plt.close()
    print("---------------------------------------")




if B_validate_monte_carlo:
    print("---------------------------------------")
    print("[drone_risk.py] >>> Monte Carlo")
    M = 10000
    DWs, masses, obs_Qs = sample_uncertain_parameters('saa', M=M)
    model = Model(DWs, masses, obs_Qs, 'saa')
    def cost(us_mat):
        value = 0.
        for t in range(S):
            for i in range(n_u):
                value = value + R[i, i] * us_mat[t, i] * us_mat[t, i]
        value = dt * value
        return value
    def no_collisions_constraint_verification(
        us_mat, mass, dWs, obs_Q):
        xs = model.us_to_state_trajectory(us_mat, mass, dWs)
        ineqs = model.obstacle_avoidance_constraints(xs, obs_Q)
        max_constraint = jnp.max(ineqs) - OSQP_TOL
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
        A = sp.csc_matrix(A)
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
            with open('results/drone_alpha='+str(alpha)+
                '_repeat='+str(idx_repeat)+'.npy', 
                'rb') as f:
                us = np.load(f)
                _ = np.load(f)
            us_vmapped = jnp.repeat(
                us[jnp.newaxis, :, :], M, axis=0) 
            B_satisfied_vec, constraints_vec = vmap(no_collisions_constraint_verification)(
                us_vmapped,
                model.masses, model.DWs, model.obs_Qs)
            avar_val = avar(constraints_vec, alpha)
            # pack results
            us_mat_all = us_mat_all.at[idx_repeat, :, :].set(us)
            B_satisfied_mat = B_satisfied_mat.at[idx_repeat, :].set(B_satisfied_vec)
            avar_val_vec = avar_val_vec.at[idx_repeat].set(avar_val)
            print("B_satisfied_vec =", jnp.mean(B_satisfied_vec))
        print("percentage safe (mean) =", jnp.mean(B_satisfied_mat))
        print("avar (mean) =", jnp.mean(avar_val_vec))
        print("cost (mean) =", jnp.mean(vmap(cost)(us_mat_all)))
        print("percentage safe (median) =", jnp.median(jnp.mean(B_satisfied_mat, axis=-1)))
        print("avar (median) =", jnp.median(avar_val_vec))
        print("cost (median) =", jnp.median(vmap(cost)(us_mat_all)))

    print("---------------------------")
    print("Monte-Carlo: baseline")
    with open('results/drone_baseline.npy', 
        'rb') as f:
        us = np.load(f)
        _ = np.load(f)
    us_vmapped = jnp.repeat(
        us[jnp.newaxis, :, :], M, axis=0) 
    B_satisfied_vec, constraints_vec = vmap(no_collisions_constraint_verification)(
        us_vmapped,
        model.masses, model.DWs, model.obs_Qs)
    avar_val = avar(constraints_vec, alpha)
    print("percentage safe =", jnp.mean(B_satisfied_vec))
    print("cost =", cost(us))
    print("---------------------------------------")