# 
from hyperparameters import *
from taichi_utils import *
from init_conditions import *
from io_utils import *
import torch
import sys
from lp import *
from lp_init import * 

# dx = 1./res_y

ti.init(arch=ti.cuda, device_memory_GB=16.0, debug = False)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# solver
boundary_types = ti.Matrix([[2, 2], [2, 2], [2, 2]], ti.i32) # boundaries: 1 means Dirichlet, 2 means Neumann
#

# undeformed coordinates (cell center and faces)
X = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
X_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
X_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
X_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
center_coords_func(X, dx)
x_coords_func(X_x, dx)
y_coords_func(X_y, dx)
z_coords_func(X_z, dx)
#

# back flow map
T_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z)) # d_psi / d_x
T_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z)) # d_psi / d_y
T_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1)) # d_psi / d_z
psi = ti.Vector.field(3, float, shape=(res_x, res_y, res_z)) # x coordinate
psi_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z)) # x coordinate
psi_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z)) # y coordinate
psi_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1)) # z coordinate
# fwrd flow map
F_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z)) # d_phi / d_x
F_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z)) # d_phi / d_y
F_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1)) # d_phi / d_z
phi = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
phi_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
phi_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
phi_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
#

# velocity storage
u = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
sizing = ti.field(float, shape=(res_x, res_y, res_z)) # sizing value corresponding to u
tmp_sizing = ti.field(float, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, float, shape=(res_x, res_y, res_z)) # curl of u
u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
# 

# some helper storage
init_u_x = ti.field(float, shape=(res_x+1, res_y, res_z)) # stores the "m0"
init_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
init_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
err_u_x = ti.field(float, shape=(res_x+1, res_y, res_z)) # stores the roundtrip "m0"
err_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
err_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
tmp_u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
tmp_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
tmp_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))


#

# CFL related
max_speed = ti.field(float, shape=())
max_speed2 = ti.field(float, shape=())

dts = torch.zeros(reinit_every)
#

# smoke
init_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
tmp_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
err_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
#

# neural buffer
x_coord_flat = X_x.to_torch().view(-1, 3)
y_coord_flat = X_y.to_torch().view(-1, 3)
z_coord_flat = X_z.to_torch().view(-1, 3)
#
tmp_h = ti.field(float, shape=(res_x , res_y, res_z))
err_h = ti.field(float, shape=(res_x , res_y, res_z))
init_h = ti.field(float, shape=(res_x , res_y, res_z))
h = ti.field(float, shape=(res_x , res_y, res_z))
u_square = ti.field(float, shape=(res_x, res_y, res_z))
acc_int = ti.field(float, shape=(res_x, res_y, res_z))
acc_int_new = ti.field(float, shape=(res_x, res_y, res_z))


laden_particles_pos = ti.Vector.field(
    3, float, shape=laden_particles_max_num)
laden_particles_vel = ti.Vector.field(
    3, float, shape=laden_particles_max_num)
laden_particles_vel_new=ti.Vector.field(
    3, float, shape=laden_particles_max_num)
laden_particles_pos_new=ti.Vector.field(
    3, float, shape=laden_particles_max_num)
delete_flag = ti.field(
    int, shape=laden_particles_max_num)

laden_particles_vel_tem = ti.Vector.field(
    3, float, shape=laden_particles_max_num)
laden_particles_vel_diff = ti.Vector.field(
    3, float, shape=laden_particles_max_num)
laden_particles_acc = ti.Vector.field(
    3, float, shape=laden_particles_max_num)
drag_x = ti.field(float, shape=(res_x + 1, res_y, res_z))
drag_y = ti.field(float, shape=(res_x, res_y + 1, res_z))
drag_z = ti.field(float, shape=(res_x, res_y, res_z + 1))
vis_x = ti.field(float, shape=(res_x + 1, res_y, res_z))
vis_y = ti.field(float, shape=(res_x, res_y + 1, res_z))
vis_z = ti.field(float, shape=(res_x, res_y, res_z + 1))

laden_particle_num = ti.field(int, shape=())
old_laden_particle_num = ti.field(int, shape=())
laden_particle_num[None] = laden_particles_init_num
laden_particle_ref_radius = ti.field(float, shape=laden_particles_max_num)

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.e-3 # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i+1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j+1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k+1])
        speed = ti.sqrt(u ** 2 + v ** 2 + w ** 2)
        ti.atomic_max(max_speed[None], speed)
@ti.kernel
def calc_max_particle_speed(u: ti.template()):
    max_speed2[None] = 1.0e-3
    for I in ti.grouped(u):
        ti.atomic_max(max_speed2[None], u[I].norm())
# set to undeformed
@ti.kernel
def reset_to_identity(psi:ti.template(),psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), 
                    T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(psi):
        psi[I] = X[I]
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

# backtrack to step 0
def backtrack_psi_grid(curr_step,rootdir):
    reset_to_identity(psi, psi_x, psi_y, psi_z, T_x, T_y, T_z)
    # first step is done on grid
    RK4_grid_only_psi(psi, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid(psi_x, T_x, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid(psi_y, T_y, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid(psi_z, T_z, u_x, u_y, u_z, dts[curr_step].item())
    # previous steps are done on neural
    for step in reversed(range(curr_step)):
        march_no_neural(psi_x, T_x, psi_y, T_y, psi_z, T_z,psi, step, rootdir)
        #march_neural(psi_x, T_x, step, rootdir)
        #march_neural(psi_y, T_y, step, rootdir)
        #march_neural(psi_z, T_z, step, rootdir)
        #march_neural_only_psi(psi,step, rootdir)

# march forward for 1 step
def march_phi_grid(curr_step):
    RK4_grid_only_psi(phi, u_x, u_y, u_z, -1 * dts[curr_step].item())
    RK4_grid(phi_x, F_x, u_x, u_y, u_z, -1 * dts[curr_step].item())
    RK4_grid(phi_y, F_y, u_x, u_y, u_z, -1 * dts[curr_step].item())
    RK4_grid(phi_z, F_z, u_x, u_y, u_z, -1 * dts[curr_step].item())

@ti.func
def interp_u_MAC_grad(u_x, u_y, u_z, p, dx):
    u_x_p, grad_u_x_p = interp_grad_2(u_x, p, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.5)
    u_y_p, grad_u_y_p = interp_grad_2(u_y, p, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.5)
    u_z_p, grad_u_z_p = interp_grad_2(u_z, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p, grad_u_z_p])

@ti.func
def interp_u_MAC_vector(u_x, u_y, u_z, p, dx):
    u_x_p = interp_2_vector(u_x, p, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.5)
    u_y_p= interp_2_vector(u_y, p, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.5)
    u_z_p = interp_2_vector(u_z, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.0)
    return u_x_p, u_y_p, u_z_p

@ti.kernel
def RK4_grid(psi_x: ti.template(), T_x: ti.template(), 
            u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * neg_dt * dT_x_dt2 
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2 # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] + 1.0 * neg_dt * u3
        T_x3 = T_x[I] + 1.0 * neg_dt * dT_x_dt3 # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + neg_dt * 1./6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4) # advance full

@ti.kernel
def RK4_grid_only_psi(psi: ti.template(), 
            u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi[I], dx)
        # prepare second
        psi1 = psi[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi1, dx)
        # prepare third
        psi2 = psi[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi2, dx)
        # prepare fourth
        psi3 = psi[I] + 1.0 * neg_dt * u3
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi3, dx)
        # final advance
        psi[I] = psi[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)

# assumes data to be one dimensional
@ti.kernel
def unravel_to_ti(data: ti.types.ndarray(), field: ti.template(), start: int):
    num_x, num_y, num_z = field.shape
    num_yz = num_y * num_z
    for _idx in data:
        idx = start + _idx
        idx_x = idx // num_yz
        idx_yz = idx % num_yz
        idx_y = idx_yz // num_z
        idx_z = idx_yz % num_z
        field[idx_x, idx_y, idx_z] = data[_idx]

def query_in_chunks(x_coord_flat, y_coord_flat, z_coord_flat, tmp_u_x, tmp_u_y, tmp_u_z, step):
    with torch.no_grad():
        encoder_max_num = nb.model.fp.max_num_queries
        # X
        num_queries = x_coord_flat.shape[0]
        num_divisions = int(np.ceil(num_queries/encoder_max_num))
        for chunk in range(num_divisions):
            chunk_begin = chunk * encoder_max_num
            chunk_end = min((chunk+1) * encoder_max_num, num_queries)
            u_x_torch = nb.pred_u_x(x_coord_flat[chunk_begin:chunk_end], nb.mid_ts[step]).squeeze()
            unravel_to_ti(u_x_torch, tmp_u_x, chunk_begin)
        # Y
        num_queries = y_coord_flat.shape[0]
        num_divisions = int(np.ceil(num_queries/encoder_max_num))
        for chunk in range(num_divisions):
            chunk_begin = chunk * encoder_max_num
            chunk_end = min((chunk+1) * encoder_max_num, num_queries)
            u_y_torch = nb.pred_u_y(y_coord_flat[chunk_begin:chunk_end], nb.mid_ts[step]).squeeze()
            unravel_to_ti(u_y_torch, tmp_u_y, chunk_begin)
        # Z
        num_queries = z_coord_flat.shape[0]
        num_divisions = int(np.ceil(num_queries/encoder_max_num))
        for chunk in range(num_divisions):
            chunk_begin = chunk * encoder_max_num
            chunk_end = min((chunk+1) * encoder_max_num, num_queries)
            u_z_torch = nb.pred_u_z(z_coord_flat[chunk_begin:chunk_end], nb.mid_ts[step]).squeeze()
            unravel_to_ti(u_z_torch, tmp_u_z, chunk_begin)

def march_neural(psi_x, T_x, step):
    # query neural buffer
    query_in_chunks(x_coord_flat, y_coord_flat, z_coord_flat, tmp_u_x, tmp_u_y, tmp_u_z, step)
    # time integration
    RK4_grid(psi_x, T_x, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())

def march_no_neural(psi_x, T_x, psi_y, T_y, psi_z, T_z, psi, step,rootdir):
    # query neural buffer
    tmp_u_x.from_numpy(np.load(os.path.join(rootdir, f"u_x_{step}.npy")))
    tmp_u_y.from_numpy(np.load(os.path.join(rootdir, f"u_y_{step}.npy")))
    tmp_u_z.from_numpy(np.load(os.path.join(rootdir, f"u_z_{step}.npy")))
    #query_in_chunks(x_coord_flat, y_coord_flat, z_coord_flat, tmp_u_x, tmp_u_y, tmp_u_z, step)
    # time integration
    RK4_grid(psi_x, T_x, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())
    RK4_grid(psi_y, T_y, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())
    RK4_grid(psi_z, T_z, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())
    RK4_grid_only_psi(psi, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())

def march_neural_only_psi(psi, step):
    # query neural buffer
    query_in_chunks(x_coord_flat, y_coord_flat, z_coord_flat, tmp_u_x, tmp_u_y, tmp_u_z, step)
    # time integration
    RK4_grid_only_psi(psi, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())



# covector advection
# u_x0, u_y0, u_z0 are the initial time quantities
# u_x1, u_y1, u_z0 are the current time quantities (to modify)
@ti.kernel
def advect_u(u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
            u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(),
            T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)
    # z velocity
    for I in ti.grouped(u_z1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = T_z[I].dot(u_at_psi)

@ti.kernel
def advect_h(h0: ti.template(), h1: ti.template(),
            T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
            psi: ti.template(), dx: float):
    # x velocity
    for i,j,k in h0:
        pos = ti.Vector([i+0.5,j+0.5,k+0.5])*dx
        T_x0, T_y0,T_z0 = interp_u_MAC_vector(T_x, T_y,T_z, pos, dx)
        T = ti.Matrix.cols([T_x0, T_y0, T_z0])
        h0_at_psi,tem=interp_grad_2(h0, psi[i,j,k], dx)
        h1[i,j,k]=ti.math.determinant(T)* h0_at_psi


@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(), 
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] + \
                        psi_y[i, j, k] + psi_y[i, j+1, k] + \
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        smoke1[i,j,k] = interp_1(smoke0, psi_c, dx)

def diffuse_sizing():
    for _ in range(1024):
        diffuse_grid(sizing, tmp_sizing)

@ti.kernel
def clamp_smoke(smoke0: ti.template(), smoke1: ti.template(), 
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] + \
                        psi_y[i, j, k] + psi_y[i, j+1, k] + \
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        mini, maxi = sample_min_max_1(smoke0, psi_c, dx)
        smoke1[i,j,k] = ti.math.clamp(smoke1[i,j,k], mini, maxi)

# limit u with u1, write to u2
@ti.kernel
def clamp_u(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i,j,k in u:
        u1_l = sample(u1, i-1, j, k)
        u1_r = sample(u1, i+1, j, k)
        u1_b = sample(u1, i, j-1, k)
        u1_t = sample(u1, i, j+1, k)
        u1_a = sample(u1, i, j, k-1)
        u1_c = sample(u1, i, j, k+1)
        maxi = ti.math.max(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        mini = ti.math.min(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        u2[i,j,k] = ti.math.clamp(u[i,j,k], mini, maxi)


@ti.kernel
def calculate_h(
    psi:ti.template(),
    T_x:ti.template(),T_y:ti.template(),T_z:ti.template(),
    h:ti.template(),init_h:ti.template(),dx:float):
    for i,j,k in h:
        pos = ti.Vector([i+0.5,j+0.5,k+0.5])*dx
        T_x0, T_y0,T_z0 = interp_u_MAC_vector(T_x, T_y,T_z, pos, dx)
        T = ti.Matrix.cols([T_x0, T_y0, T_z0])
        h0,tem=interp_grad_2(init_h, psi[i,j,k], dx)
        h[i,j,k]=ti.math.determinant(T)* h0

@ti.kernel
def g2p_lamb(phi:ti.template(),acc_int_new:ti.template() , acc_int:ti.template(), dx:float):
    for i,j,k in acc_int_new:
        v, tem = interp_grad_2(acc_int_new,phi[i,j,k],dx)
        acc_int[i,j,k] = v

@ti.kernel
def enforce_boundary(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), h:ti.template()):
    ii,jj,kk=u_x.shape
    for i,j,k in u_x:
        if(i==0 or i==ii-1):
            u_x[i,j,k]=0

    ii,jj,kk=u_y.shape
    for i,j,k in u_y:
        if(j==0 or j==jj-1):
            u_y[i,j,k]=0

    ii,jj,kk=u_z.shape
    for i,j,k in u_z:
        if(k==0 or k==kk-1):
            u_z[i,j,k]=0 

@ti.kernel
def enforce_velocity_new(
    u_x: ti.template(), 
    u_y: ti.template(),
    u_z: ti.template(),
    acc_int:ti.template(),
    acc_int_new:ti.template(),
    psi:ti.template(),
    u_square:ti.template(),
    h:ti.template(),
    dx:float,
    dt:float
):
    for I in ti.grouped(h):
        v,tem=interp_grad_2(acc_int,psi[I],dx)
        acc_int_new[I]=v -dt*gravity_fake*h[I]**gamma + dt*u_square[I] 
    
    for i,j,k in u_x:
        grad = (sample(acc_int_new,i,j,k)-sample(acc_int_new,i-1,j,k))/dx
        u_x[i,j,k]+=grad
    

    for i,j,k in u_y:
        grad=(sample(acc_int_new,i,j,k)-sample(acc_int_new,i,j-1,k))/dx
        u_y[i,j,k]+=grad

    for i,j,k in u_z:
        grad=(sample(acc_int_new,i,j,k)-sample(acc_int_new,i,j,k-1))/dx
        u_z[i,j,k]+=grad

@ti.kernel
def get_grid_usqure(
    u_x: ti.template(),
    u_y: ti.template(), 
    u_z: ti.template(), 
    u_square: ti.template(),
    dx:float
):
    for i,j,k in u_square:
        pos=ti.Vector([i+0.5,j+0.5,k+0.5])*dx
        u1, tem= interp_u_MAC_grad(u_x, u_y,u_z, pos, dx)
        u_square[i,j,k] = 0.5*u1.norm()*u1.norm()

@ti.kernel
def add_gravity_force(
    u_x:ti.template(), 
    u_y:ti.template(), 
    u_z:ti.template(), 
    h:ti.template() , 
    dx:float, 
    dt:float
):
    for i,j,k in u_x:
        grad_h_x=(sample(h,i,j,k)-sample(h,i-1,j,k))/dx
        u_x[i,j,k]+=-gravity_fake* grad_h_x**gamma*dt

    for i,j,k in u_y:
        grad_h_y=(sample(h,i,j,k)-sample(h,i,j-1,k))/dx
        u_y[i,j,k]+=-gravity_fake* grad_h_y**gamma*dt

    for i,j,k in u_z:
        grad_h_z=(sample(h,i,j,k)-sample(h,i,j,k-1))/dx
        u_z[i,j,k]+=-gravity_fake* grad_h_z**gamma*dt

def write_velocity(u_x,u_y,u_z,rootdir,ind):
    u_x_np= u_x.to_numpy()
    u_y_np= u_y.to_numpy()
    u_z_np= u_z.to_numpy()
    
    np.save(os.path.join(rootdir, f"u_x_{ind}.npy"),u_x_np)
    np.save(os.path.join(rootdir, f"u_y_{ind}.npy"),u_y_np)
    np.save(os.path.join(rootdir, f"u_z_{ind}.npy"),u_z_np)

def delete_velocity(rootdir,all_ind):
    for ind in range(all_ind):
        if(os.path.exists(os.path.join(rootdir, f"u_x_{ind}.npy"))) :
            os.remove(os.path.join(rootdir, f"u_x_{ind}.npy"))
        if(os.path.exists(os.path.join(rootdir, f"u_y_{ind}.npy"))) :
            os.remove(os.path.join(rootdir, f"u_y_{ind}.npy"))
        if(os.path.exists(os.path.join(rootdir, f"u_z_{ind}.npy"))) :
            os.remove(os.path.join(rootdir, f"u_z_{ind}.npy"))

def init_vorts():
    init_vorts_oblique(X, u, smoke, tmp_smoke)


def write_w_and_laden_particles(pos, outdir, i, laden_particle_num):
    write_to_vtks_lp(
        pos.to_numpy()[:laden_particle_num, :],
        scalar_data={},
        vector_data={},
        file_path=os.path.join(outdir, "field_{:03d}".format(i)),
    )

@ti.kernel
def map_u(u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
          u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(),
          T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
          psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    for I in ti.grouped(u_x1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    for I in ti.grouped(u_y1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)
    for I in ti.grouped(u_z1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = T_z[I].dot(u_at_psi)

# main function
def main(from_frame = 0, testing = False):
    from_frame = max(0, from_frame)
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vtkdir = "vtks"
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)
    save_u_dir = "save_u"
    save_u_dir = os.path.join(logsdir, save_u_dir)
    os.makedirs(save_u_dir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    particle_vtkdir = "particle"
    particle_vtkdir = os.path.join(logsdir, particle_vtkdir)
    os.makedirs(particle_vtkdir, exist_ok=True)

    if testing:
        testdir = 'test_buffer'
        testdir = os.path.join(logsdir, testdir)
        os.makedirs(testdir, exist_ok=True)
        remove_everything_in(testdir)
        GTdir = os.path.join(testdir, "GT")
        os.makedirs(GTdir, exist_ok=True)
        preddir = os.path.join(testdir, "pred")
        os.makedirs(preddir, exist_ok=True)

    # initial condition
    if from_frame <= 0:
        ink_init_case0(laden_particles_pos, laden_particles_vel, laden_particle_num, u,
                        X, drag_x, drag_y, drag_z, laden_particle_ref_radius, compute_laden_radius)
        split_central_vector(u, u_x, u_y, u_z)
    else:
        u_x.from_numpy(np.load(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        u_z.from_numpy(np.load(os.path.join(ckptdir, "vel_z_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))
    #

    # visualization
    get_central_vector(u_x, u_y, u_z, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_norm = np.linalg.norm(w_numpy, axis = -1)
    smoke_numpy = smoke.to_numpy()
    smoke_norm = smoke_numpy[...,-1]
    write_vtks(w_norm, smoke_norm, vtkdir, from_frame)
    # 

    # save init checkpoint
    np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
    np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
    np.save(os.path.join(ckptdir, "vel_z_numpy_" + str(from_frame)), u_z.to_numpy())
    np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke_numpy) 
    #

    sub_t = 0. # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0 # number of reinits already performed
    i = -1
    h.fill(1.0)

    while True:
        i += 1
        j = i % reinit_every
        i_next = i + 1
        j_next = i_next % reinit_every
        print("[Simulate] Running step: ", i, " / substep: ", j)

        # determine dt
        calc_max_speed(u_x, u_y, u_z)  # saved to max_speed[None]

        calc_max_particle_speed(laden_particles_vel)
        curr_dt = min(CFL * dx / max_speed[None], CFL * dx / max_speed2[None])
        if sub_t+curr_dt >= visualize_dt: # if over
            curr_dt = visualize_dt-sub_t
            sub_t = 0. # empty sub_t
            frame_idx += 1
            output_frame = True
        else:
            sub_t += curr_dt
            output_frame = False
        dts[j] = curr_dt
        #

        # reinitialize flow map if j == 0:
        if j == 0:
            print("[Simulate] Reinitializing the flow map for the: ", num_reinits, " time!")
            reset_to_identity(phi, phi_x, phi_y, phi_z, F_x, F_y, F_z) # reset phi, F
            # set current values as init
            copy_to(u_x, init_u_x)
            copy_to(u_y, init_u_y)
            copy_to(u_z, init_u_z)
            copy_to(smoke, init_smoke)
            # reinit neural buffer
            delete_velocity(save_u_dir,reinit_every)
            """
            get_central_vector(u_x, u_y, u_z, u)
            sizing_function(u, sizing, dx) # compute sizing
            diffuse_sizing()
            nb.reinit(sizing)
            nb.set_magnitude_scale(max_speed[None])
            print("[Neural Buffer] New magnitude scale: ", nb.magnitude_scale)
            """
            # increment
            num_reinits += 1
            copy_to(h,init_h)
            acc_int.fill(0.0)
        #

        update_particles_acc(u_x, u_y, u_z, laden_particles_vel, laden_particles_pos,
                            laden_particles_acc, dx, curr_dt, laden_particle_num[None], St, gravity)
        update_particles_velocity(laden_particles_acc, laden_particles_vel, laden_particles_pos,
                                curr_dt, dx, laden_particle_num[None], domain_range_x, domain_range_y, domain_range_z)
        calculate_velocity_diff(u_x, u_y, u_z, laden_particles_vel, laden_particles_vel_diff,
                                laden_particles_pos, dx, laden_particle_num[None], St, density_ratio, epiral_radius_ratio)
        laden_particles_back_with_computing_particle(drag_x, drag_y, drag_z, u_x, u_y, u_z, laden_particles_vel,
                                                    laden_particles_vel_diff, laden_particles_pos, laden_particle_ref_radius, dx, laden_particle_num[None], laden_radius, domain_range_x, domain_range_y, domain_range_z, res_x, res_y, res_z)
        update_particles_position(laden_particles_vel, laden_particles_pos, curr_dt,
                                laden_particle_num[None], domain_range_x, domain_range_y, domain_range_z)
        calculate_viscous_force(u_x, u_y, u_z, vis_x, vis_y, vis_z, dx, vis_coef)
        add_fields(vis_x, drag_x, vis_x, 1.0)
        add_fields(vis_y, drag_y, vis_y, 1.0)
        add_fields(vis_z, drag_z, vis_z, 1.0)


        # start midpoint
        reset_to_identity(psi, psi_x, psi_y, psi_z, T_x, T_y, T_z) # reset psi, T
        RK4_grid(psi_x, T_x, u_x, u_y, u_z, 0.5 * curr_dt)
        RK4_grid(psi_y, T_y, u_x, u_y, u_z, 0.5 * curr_dt)
        RK4_grid(psi_z, T_z, u_x, u_y, u_z, 0.5 * curr_dt)
        copy_to(u_x, tmp_u_x)
        copy_to(u_y, tmp_u_y)
        copy_to(u_z, tmp_u_z)
        advect_u(tmp_u_x, tmp_u_y, tmp_u_z, u_x, u_y, u_z, \
                T_x, T_y, T_z, psi_x, psi_y, psi_z, dx)
        add_gravity_force(u_x, u_y, u_z, h, dx, 0.5* curr_dt)

        
        #solver.Poisson(u_x, u_y, u_z)
        # done midpoint
        get_grid_usqure(u_x, u_y,u_z, u_square,dx)

        # store midpoint u to nb
        write_velocity(u_x,u_y,u_z,save_u_dir, j)
        """
        if not j_next == 0: # only store if not reinit-pending
            # grow buffer with new sizing
            get_central_vector(u_x, u_y, u_z, u)
            sizing_function(u, sizing, dx)
            diffuse_sizing()
            nb.grow(sizing)
            #

            final_loss = nb.store_u(u_x.to_torch(), u_y.to_torch(), u_z.to_torch(),\
                    init_u_x.to_torch(), init_u_y.to_torch(), init_u_z.to_torch(), dts[:j+1])

            print("[Simulation] Stored in neural buffer with final loss:", final_loss)
        """

        # backtrack
        backtrack_psi_grid(j,save_u_dir)
        march_phi_grid(j)

        # ink force
        map_u(vis_x, vis_y, vis_z, tmp_u_x, tmp_u_y, tmp_u_z,
                F_x, F_y, F_z, phi_x, phi_y, phi_z, dx)
        add_fields(init_u_x, tmp_u_x, init_u_x, curr_dt)
        add_fields(init_u_y, tmp_u_y, init_u_y, curr_dt)
        add_fields(init_u_z, tmp_u_z, init_u_z, curr_dt)

        advect_u(init_u_x, init_u_y, init_u_z, u_x, u_y, u_z,\
                T_x, T_y, T_z, psi_x, psi_y, psi_z, dx)
        advect_h(init_h, h, T_x, T_y, T_z, psi, dx)
        
        advect_smoke(init_smoke, smoke, psi_x, psi_y, psi_z, dx)
        # # Begin BFECC
        advect_u(u_x, u_y, u_z, err_u_x, err_u_y, err_u_z,\
                F_x, F_y, F_z, phi_x, phi_y, phi_z, dx)
        advect_h(h, err_h, F_x, F_y, F_z, phi, dx)
        
        advect_smoke(smoke, err_smoke, phi_x, phi_y, phi_z, dx)
        add_fields(err_u_x, init_u_x, err_u_x, -1.) # subtract init_u_x from back_u_x
        add_fields(err_u_y, init_u_y, err_u_y, -1.)
        add_fields(err_u_z, init_u_z, err_u_z, -1.)
        add_fields(err_h, init_h, err_h, -1.)

        add_fields(err_smoke, init_smoke, err_smoke, -1.)
        scale_field(err_u_x, 0.5, err_u_x) # halve error
        scale_field(err_u_y, 0.5, err_u_y)
        scale_field(err_u_z, 0.5, err_u_z)
        scale_field(err_h, 0.5, err_h)
        scale_field(err_smoke, 0.5, err_smoke)
        advect_u(err_u_x, err_u_y, err_u_z,\
                tmp_u_x, tmp_u_y, tmp_u_z,\
                T_x, T_y, T_z, psi_x, psi_y, psi_z, dx) # advect error (tmp_u_x is the advected error)
        advect_h(err_h, tmp_h, T_x, T_y, T_z, psi, dx)
        advect_smoke(err_smoke, tmp_smoke, psi_x, psi_y, psi_z, dx) # advect error
        add_fields(u_x, tmp_u_x, err_u_x, -1.) # subtract err
        add_fields(u_y, tmp_u_y, err_u_y, -1.)
        add_fields(u_z, tmp_u_z, err_u_z, -1.)
        add_fields(h, tmp_h, err_h, -1.)
        add_fields(smoke, tmp_smoke, smoke, -1.)
        # clamp smoke
        clamp_smoke(init_smoke, smoke, psi_x, psi_y, psi_z, dx)
        # clamp u
        if BFECC_clamp:
            clamp_u(err_u_x, u_x, tmp_u_x)
            clamp_u(err_u_y, u_y, tmp_u_y)
            clamp_u(err_u_z, u_z, tmp_u_z)
            copy_to(tmp_u_x, u_x)
            copy_to(tmp_u_y, u_y)
            copy_to(tmp_u_z, u_z)
        else:
            copy_to(err_u_x, u_x)
            copy_to(err_u_y, u_y)
            copy_to(err_u_z, u_z)  
        copy_to(err_h, h)
        #
        #calculate_h(psi,T_x,T_y,T_z,h,init_h,dx)

        vis_x.fill(0.0)
        vis_y.fill(0.0)
        vis_z.fill(0.0)

        enforce_velocity_new(u_x, u_y, u_z, acc_int,acc_int_new, psi ,u_square,h,dx,curr_dt)
        enforce_boundary(u_x, u_y,u_z, h)
        g2p_lamb(phi,acc_int_new , acc_int, dx)

        #solver.Poisson(u_x, u_y, u_z)

        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush = True)

        if output_frame:
            # visualization
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, dx)
            w_numpy = w.to_numpy()
            w_norm = np.linalg.norm(w_numpy, axis = -1)
            smoke_numpy = smoke.to_numpy()
            smoke_norm = smoke_numpy[...,-1]
            write_vtks(w_norm, smoke_norm, vtkdir, frame_idx)

            write_w_and_laden_particles(
                    laden_particles_pos, particle_vtkdir, frame_idx, laden_particle_num[None])

            # if frame_idx % ckpt_every == 0:
                # np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                # np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                # np.save(os.path.join(ckptdir, "vel_z_numpy_" + str(frame_idx)), u_z.to_numpy())
                # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke_numpy)   

            print("[Simulate] Finished frame: ", frame_idx, " in ", i-last_output_substep, "substeps \n\n")
            last_output_substep = i

            # if reached desired number of frames
            if frame_idx >= total_frames:
                break
    
    
        
if __name__ == '__main__':
    print("[Main] Begin")
    main(from_frame = from_frame)
    print("[Main] Complete")
