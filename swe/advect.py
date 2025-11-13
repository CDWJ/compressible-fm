#
from hyperparameters import *
from taichi_utils import *
from io_utils import *
import sys
import shutil
import time
#

"""
1. Reset the flow map to identity

"""


##################################################################
################## 1. Reset the flow map #########################
##################################################################

@ti.kernel
def g2p_scalar(field:ti.template(), particles_scalar:ti.template(), particles_pos:ti.template(), dx:float):
    for I in ti.grouped(particles_pos):
        scalar, n1, n2 = interp_grad_2(field, particles_pos[I], dx, BL_x=0.5, BL_y=0.5, is_y=False)
        particles_scalar[I] = scalar


@ti.kernel
def g2p_vel(particles_vel: ti.template(), particles_pos: ti.template(),
                       u_x: ti.template(), u_y: ti.template(), dx: float):
    for i in particles_vel:
        particles_vel[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[i], dx)

@ti.kernel
def reset_to_identity_new(T_x: ti.template(), T_y: ti.template()):
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)

@ti.kernel
def reset_to_identity(particles_pos:ti.template(),psi: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i in psi:
        psi[i] = particles_pos[i]
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template()):
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)
        
@ti.kernel
def reset_F_to_identity(F_x: ti.template(), F_y: ti.template()):
    for i in F_x:
        F_x[i] = ti.Vector.unit(2, 0)
    for i in F_y:
        F_y[i] = ti.Vector.unit(2, 1)

##################################################################
################## 2. Advect the flow map ########################
##################################################################

def stretch_FT_and_advect_particles(particles_pos, T_x, T_y, F_x, F_y, u_x, u_y, dt, dx, particles_active):
    TVD_RK3_FT_forward(particles_pos, T_x, T_y, F_x, F_y, u_x, u_y, dt, dx, 1, particles_active)


@ti.kernel
def TVD_RK3_FT_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), F_x: ti.template(), F_y: ti.template(), 
u_x0: ti.template(), u_y0: ti.template(), dt: float,dx:float, advect_psi: int, particles_active:ti.template()):
    
    for i in psi:
        if particles_active[i] == 1:
            # first
            # u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            # divu1, tem1, tem2 = interp_grad_2(divU, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi.transpose() @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi.transpose() @ T_y[i]  # time derivative of T
            
            dF_x_dt1 = grad_u_at_psi @ F_x[i]  # time derivative of F
            dF_y_dt1 = grad_u_at_psi @ F_y[i]  # time derivative of F
            
            # prepare second
            psi_x1 = psi[i] + 1 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 1 * dt * dT_x_dt1
            T_y1 = T_y[i] - 1 * dt * dT_y_dt1
            
            F_x1 = F_x[i] + 1 * dt * dF_x_dt1
            F_y1 = F_y[i] + 1 * dt * dF_y_dt1

            # second
            u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            # divu2, tem1, tem2  = interp_grad_2(divU, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1  # time derivative of T
            
            dF_x_dt2 = grad_u_at_psi @ F_x1 # time derivative of F
            dF_y_dt2 = grad_u_at_psi @ F_y1  # time derivative of F

            # prepare third
            psi_x2 = (psi_x1 + dt * u2) * 0.25 + 0.75 * psi[i]  # advance 0.5 again
            T_x2 = (T_x1 - dt * dT_x_dt2) * 0.25 + 0.75 * T_x[i]
            T_y2 = (T_y1 - dt * dT_y_dt2) * 0.25 + 0.75 * T_y[i]

            F_x2 = (F_x1 + dt * dF_x_dt2) * 0.25 + 0.75 * F_x[i]
            F_y2 = (F_y1 + dt * dF_y_dt2) * 0.25 + 0.75 * F_y[i]
            

            # third
            u3, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
            # divu3, tem1, tem2  = interp_grad_2(divU, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi.transpose() @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi.transpose() @ T_y2  # time derivative of T
            
            dF_x_dt3 = grad_u_at_psi @ F_x2  # time derivative of T
            dF_y_dt3 = grad_u_at_psi @ F_y2  # time derivative of T


            # final advance
            if advect_psi:
                psi[i] = 1. / 3 * psi[i] + 2. / 3 * (psi_x2 + dt * u3)
            T_x[i] = 1. / 3 * T_x[i] + 2. / 3 * (T_x2 - dt * dT_x_dt3)
            T_y[i] = 1. / 3 * T_y[i] + 2. / 3 * (T_y2 - dt * dT_y_dt3)
            
            F_x[i] = 1. / 3 * F_x[i] + 2. / 3 * (F_x2 + dt * dF_x_dt3)
            F_y[i] = 1. / 3 * F_y[i] + 2. / 3 * (F_y2 + dt * dF_y_dt3)




@ti.kernel
def RK4_FT_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), F_x: ti.template(), F_y: ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), dt: float,dx:float, advect_psi: int, particles_active:ti.template()):
    for i in psi:
        if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi.transpose() @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi.transpose() @ T_y[i]  # time derivative of T
            
            dF_x_dt1 = grad_u_at_psi @ F_x[i]  # time derivative of F
            dF_y_dt1 = grad_u_at_psi @ F_y[i]  # time derivative of F
            
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            
            F_x1 = F_x[i] + 0.5 * dt * dF_x_dt1
            F_y1 = F_y[i] + 0.5 * dt * dF_y_dt1
            # second
            u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1  # time derivative of T
            
            dF_x_dt2 = grad_u_at_psi @ F_x1 # time derivative of F
            dF_y_dt2 = grad_u_at_psi @ F_y1  # time derivative of F
            # prepare third
            psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
            T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
            
            F_x2 = F_x[i] + 0.5 * dt * dF_x_dt2
            F_y2 = F_y[i] + 0.5 * dt * dF_y_dt2
            # third
            u3, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi.transpose() @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi.transpose() @ T_y2  # time derivative of T
            
            dF_x_dt3 = grad_u_at_psi @ F_x2  # time derivative of T
            dF_y_dt3 = grad_u_at_psi @ F_y2  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * dt * u3
            T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3  # advance 1.0
            
            F_x3 = F_x[i] + 1.0 * dt * dF_x_dt3  # advance 1.0
            F_y3 = F_y[i] + 1.0 * dt * dF_y_dt3  # advance 1.0
            # fourth
            u4, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi.transpose() @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi.transpose() @ T_y3  # time derivative of T
            
            dF_x_dt4 = grad_u_at_psi @ F_x3  # time derivative of T
            dF_y_dt4 = grad_u_at_psi @ F_y3  # time derivative of T
            # final advance
            if advect_psi:
                psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full
            
            F_x[i] = F_x[i] + dt * 1. / 6 * (dF_x_dt1 + 2 * dF_x_dt2 + 2 * dF_x_dt3 + dF_x_dt4)  # advance full
            F_y[i] = F_y[i] + dt * 1. / 6 * (dF_y_dt1 + 2 * dF_y_dt2 + 2 * dF_y_dt3 + dF_y_dt4)  # advance full

@ti.kernel
def RK2_FT_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), F_x: ti.template(), F_y: ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), dt: float,dx:float, advect_psi: int, particles_active:ti.template()):
    
    for i in psi:
        if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi.transpose() @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi.transpose() @ T_y[i]  # time derivative of T
            
            dF_x_dt1 = grad_u_at_psi @ F_x[i]  # time derivative of F
            dF_y_dt1 = grad_u_at_psi @ F_y[i]  # time derivative of F
            
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            
            F_x1 = F_x[i] + 0.5 * dt * dF_x_dt1
            F_y1 = F_y[i] + 0.5 * dt * dF_y_dt1
            # second
            u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1  # time derivative of T
            
            dF_x_dt2 = grad_u_at_psi @ F_x1 # time derivative of F
            dF_y_dt2 = grad_u_at_psi @ F_y1  # time derivative of F
            
            # final advance
            if advect_psi:
                psi[i] = psi[i] + dt * u2

            T_x[i] = T_x[i] - dt  * dT_x_dt2   # advance full
            T_y[i] = T_y[i] - dt  * dT_y_dt2  # advance full
            
            F_x[i] = F_x[i] + dt  * dF_x_dt2   # advance full
            F_y[i] = F_y[i] + dt  * dF_y_dt2 # advance full

@ti.kernel
def advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), dx : float, dt : float,
    X_horizontal:ti.template(),X_vertical:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_horizontal[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p, dx)
        u_x1[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_vertical[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p, dx)
        u_y1[I] = v5[1]

@ti.kernel
def advect_u_grid_euler(
    u_x_save: ti.template(), u_y_save: ti.template(),
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), dx : float, dt : float,
    X_horizontal:ti.template(),X_vertical:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_horizontal[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x_save, u_y_save, p, dx)
        u_x1[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_vertical[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x_save, u_y_save, p, dx)
        u_y1[I] = v5[1]

@ti.kernel
def advect_h_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    h1: ti.template(), h0: ti.template(), dx : float, dt : float
):
    for I in ti.grouped(h0):
        p1 = (I+0.5)*dx
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        h_at_psi,tem1,tem2 = interp_grad_2(h0,p,dx)
        h1[I] = h_at_psi

@ti.kernel
def advect_u_grid_RK2(
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), dx : float, dt : float,
    X_horizontal:ti.template(),X_vertical:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_horizontal[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p = p1 - v2 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p, dx)
        u_x1[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_vertical[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p = p1 - v2 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p, dx)
        u_y1[I] = v5[1]

@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(),
                 psi_x: ti.template(), psi_y: ti.template(), dx: float):
    # horizontal velocity
    for i, j in ti.ndrange(res_x, res_y):
        psi_c = 0.25 * (psi_x[i, j] + psi_x[i + 1, j] + psi_y[i, j] + psi_y[i, j + 1])
        smoke1[i, j] = interp_1(smoke0, psi_c, dx)

##################################################################
################## 3. update of impulse    #######################
##################################################################

@ti.kernel
def update_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(), grad_half_usquare: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), particles_active:ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_imp[i] = T @ (particles_init_imp[i] + grad_lamb[i]) + grad_half_usquare[i]

@ti.kernel
def update_particles_imp_no_u2(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), particles_active:ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_imp[i] = T @ (particles_init_imp[i] + grad_lamb[i])

@ti.kernel
def update_particles_imp_pure(particles_imp: ti.template(), particles_init_imp: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), particles_active:ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_imp[i] = T @ (particles_init_imp[i])

@ti.kernel
def update_particles_h_pure(particles_h: ti.template(), particles_init_h: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), particles_active:ti.template()):
    for i in particles_h:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_h[i] = ti.math.determinant(T) * (particles_init_h[i])

@ti.kernel
def update_particles_together(
    particles_imp: ti.template(), particles_init_imp: ti.template(),
    particles_h: ti.template(), particles_init_h: ti.template(), particles_e: ti.template(), particles_init_e: ti.template(), 
    T_x: ti.template(), T_y: ti.template(), particles_active:ti.template()
):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_imp[i] = T @ (particles_init_imp[i])
            particles_h[i] = ti.math.determinant(T) * (particles_init_h[i])
            particles_e[i] = ti.math.determinant(T)**(1.4 - 1) * (particles_init_e[i])
            
@ti.kernel
def update_particles_imp_together(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(), grad_half_usquare: ti.template(),
                        particles_h: ti.template(), particles_init_h: ti.template(), particles_e: ti.template(), particles_init_e: ti.template(), 
                         T_x: ti.template(), T_y: ti.template(), particles_active:ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_imp[i] = T @ (particles_init_imp[i] + grad_lamb[i]) + grad_half_usquare[i]
            particles_h[i] = ti.math.determinant(T) * (particles_init_h[i])
            particles_e[i] = (ti.math.determinant(T)**(1.4 - 1)) * (particles_init_e[i])
            # particles_e[i] = ti.math.determinant(T)

@ti.kernel
def update_particles_water(
    particles_imp: ti.template(), 
    particles_init_imp: ti.template(), 
    grad_lamb: ti.template(), 
    grad_half_usquare: ti.template(),
    particles_h: ti.template(), 
    particles_init_h: ti.template(), 
    T_x: ti.template(), 
    T_y: ti.template(), 
    particles_active:ti.template()
):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            particles_imp[i] = T @ (particles_init_imp[i] + grad_lamb[i]) + grad_half_usquare[i]
            particles_h[i] = ti.math.determinant(T) * (particles_init_h[i])

@ti.kernel
def calculate_particle_u_square(
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(), 
    u_sqaure: ti.template(), 
    curr_dt: ti.template(),
    dx:float,
    particles_active:ti.template()
):
    for i in particles_grad_half_u:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            nouse, grad_u, _ = interp_grad_2(u_sqaure, p, dx, BL_x=0.5, BL_y=0.5)
            particles_grad_half_u[i] = grad_u * curr_dt

@ti.kernel
def get_grad_usqure(u: ti.template(), u_square: ti.template()):
    for I in ti.grouped(u):
        u_square[I] = 0.5 * (u[I][0]**2 + u[I][1]**2)

@ti.kernel
def get_grid_usqure(
    u_x: ti.template(),
    u_y: ti.template(), 
    u_square: ti.template(),
    dx:float
):
    for i,j in u_square:
        pos=ti.Vector([i+0.5,j+0.5])*dx
        u1, tem1, tem2, tem3 = interp_u_MAC_grad(u_x, u_y, pos, dx)
        u_square[i,j] = 0.5*u1.norm()*u1.norm()

"""
@ti.kernel
def get_particle_usqure(
    particles_pos:ti.template(),
    u_x: ti.template(),
    u_y: ti.template(), 
    u_square: ti.template(),
    dx:float
):
    for i in particles_pos:
        pos=particles_pos[i]
        u1, tem1, tem2, tem3 = interp_u_MAC_grad(u_x, u_y, pos, dx)
        u_square[i,j] = 0.5*u1.norm()*u1.norm()
"""

@ti.kernel
def enforce_boundary(u_x:ti.template(), u_y:ti.template(), h:ti.template()):
    ii,jj=u_x.shape
    for i,j in u_x:
        if(case == 5 or case == 6 or case == 11):
            if(i == 0 or i == ii-1):
                u_x[i,j] = 0
        """
        elif(case == 6):
            if i == 0:
                if(u_x[i, j]<0):
                    u_x[i, j]=-u_x[i, j]
                else:
                    u_x[i, j]=0.0
            if i == ii-1:
                if(u_x[i, j]>0):
                    u_x[i, j]=-u_x[i, j]
                else:
                    u_x[i, j]=0.0"""


    ii,jj=u_y.shape
    for i, j in u_y:
        if(case == 6 or case == 7 or case == 11 or case == 12):
            if(j == 0 or j == jj-1):
                u_y[i,j] = 0
        if(case == 3 or case == 5):
            if j == 0:
                if(u_y[i, j]<0):
                    u_y[i, j]=-u_y[i, j]
                else:
                    u_y[i, j]=0.0
            if j == jj-1:
                if(u_y[i, j]>0):
                    u_y[i, j]=-u_y[i, j]
                else:
                    u_y[i, j]=0.0

@ti.kernel
def enforce_boundary_diamond(u_x:ti.template(), u_y:ti.template(), h:ti.template()):
    ii,jj=u_x.shape
    res_x, res_y = h.shape
    for i,j in u_x:
        if i == 0 and j <= res_y // 2 + 1.0 / tunnel_width * res_y and j >= res_y // 2 - 1.0 / tunnel_width * res_y:
            u_x[i,j]=3
        # elif i == 0:
        #     u_x[i,j]=0

    ii,jj=u_y.shape
    # for i,j in u_y:
    #     if(j==0 or j==jj-1):
    #         u_y[i,j]=0
    for i, j in u_y:
        if j == 0:
            u_y[i, j] = u_y[i, j+1]
        if j == jj-1:
            u_y[i, j] = u_y[i, j-1]
    
            

@ti.kernel
def enforce_velocity_pre(
    u_x: ti.template(), 
    u_y: ti.template(),
    acc_int:ti.template(),
    dx:float
):  
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        u_x[i,j]+=(sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
    

    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        u_y[i,j]+=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx

@ti.kernel
def enforce_velocity(
    u_x: ti.template(), 
    u_y: ti.template(),
    acc_int:ti.template(),
    u_square:ti.template(),
    h:ti.template(),
    dx:float,
    dt:float
):
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        acc_int[I]=-dt*gravity*h[I]**gamma+ dt*u_square[I]
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        u_x[i,j]+=(sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
    

    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        u_y[i,j]+=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx

@ti.kernel
def enforce_velocity_new(
    u_x: ti.template(), 
    u_y: ti.template(),
    acc_int:ti.template(),
    u_square:ti.template(),
    h:ti.template(),
    dx:float,
    dt:float,
    boundary_mask:ti.template()
):
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        if(boundary_mask[I]==0):
            # acc_int[I]+= -dt*gravity*h[I]**gamma + dt*u_square[I] 
            acc_int[I]+= -dt*gravity*h[I]**gamma + dt*u_square[I] 
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            u_x[i,j]+=grad
    

    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
            u_y[i,j]+=grad

@ti.kernel
def enforce_velocity_shock(
    u_x: ti.template(), 
    u_y: ti.template(),
    acc_int:ti.template(),
    u_square:ti.template(),
    X_vertical:ti.template(),
    X_horizontal:ti.template(),
    h:ti.template(),
    e:ti.template(),
    dx:float,
    dt:float,
    boundary_mask:ti.template()
):
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        acc_int[I] = -dt * (1.4 - 1) * e[I]  * h[I]

    # for I in ti.grouped(h):
    #     divu = interp_MAC_divergence_u(u_x, u_y, ti.Vector([I[0] + 0.5, I[1] + 0.5]) * dx, dx)
    #     gamma_term = (1.4 - 1) / 4
    #     if divu < 0:
    #         # divu = ti.abs(divu)
    #         # acc_int[I] += dt * h[I] * (gamma_term * divu + ti.math.sqrt(gamma_term ** 2 * divu**2 + ti.abs(1.4 * acc_int[I] / h[I]))) * divu * 0.25
    #         acc_int[I] += dt * 1 * divu * h[I] * dx
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        h_x = (sample(h,i,j)+sample(h,i-1,j))/2
        # tem1, grad, tem2 = interp_grad_2(acc_int, X_horizontal[i, j], dx, BL_x=0.5, BL_y=0.5)
        # h_x, tem3, tem2 = interp_grad_2(h, X_horizontal[i, j], dx, BL_x=0.5, BL_y=0.5)
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            # u_x[i,j]+=grad[0] / h_x
            u_x[i,j]+=grad / h_x

    ii, jj = u_y.shape
    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        h_y=(sample(h,i,j)+sample(h,i,j-1))/2
        if j >= 1 and j <= jj - 2:
            # tem1, grad, tem2 = interp_grad_2(acc_int, X_vertical[i, j], dx, BL_x=0.5, BL_y=0.5)
            # h_y, tem3, tem2 = interp_grad_2(h, X_vertical[i, j], dx, BL_x=0.5, BL_y=0.5)
            if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
                # u_y[i,j]+=grad[1] / h_y
                u_y[i,j]+=grad / h_y

@ti.kernel
def enforce_velocity_water_shock(
    u_x: ti.template(), 
    u_y: ti.template(),
    acc_int:ti.template(),
    u_square:ti.template(),
    X_vertical:ti.template(),
    X_horizontal:ti.template(),
    h:ti.template(),
    e:ti.template(),
    dx:float,
    dt:float,
    boundary_mask:ti.template()
):
    print("real_gravity",real_gravity)
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        acc_int[I] = -dt * h[I]* real_gravity

    # for I in ti.grouped(h):
    #     divu = interp_MAC_divergence_u(u_x, u_y, ti.Vector([I[0] + 0.5, I[1] + 0.5]) * dx, dx)
    #     gamma_term = (1.4 - 1) / 4
    #     if divu < 0:
    #         # divu = ti.abs(divu)
    #         # acc_int[I] += dt * h[I] * (gamma_term * divu + ti.math.sqrt(gamma_term ** 2 * divu**2 + ti.abs(1.4 * acc_int[I] / h[I]))) * divu * 0.25
    #         acc_int[I] += dt * 1 * divu * h[I] * dx
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        h_x = (sample(h,i,j)+sample(h,i-1,j))/2
        # tem1, grad, tem2 = interp_grad_2(acc_int, X_horizontal[i, j], dx, BL_x=0.5, BL_y=0.5)
        # h_x, tem3, tem2 = interp_grad_2(h, X_horizontal[i, j], dx, BL_x=0.5, BL_y=0.5)
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            # u_x[i,j]+=grad[0] / h_x
            u_x[i,j]+=grad 

    ii, jj = u_y.shape
    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        h_y=(sample(h,i,j)+sample(h,i,j-1))/2
        if j >= 1 and j <= jj - 2:
            # tem1, grad, tem2 = interp_grad_2(acc_int, X_vertical[i, j], dx, BL_x=0.5, BL_y=0.5)
            # h_y, tem3, tem2 = interp_grad_2(h, X_vertical[i, j], dx, BL_x=0.5, BL_y=0.5)
            if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
                # u_y[i,j]+=grad[1] / h_y
                u_y[i,j]+=grad 

@ti.kernel
def enforce_velocity_water_shock_ibm(
    u_x: ti.template(), 
    u_y: ti.template(),
    acc_int:ti.template(),
    ibm_force_x:ti.template(),
    ibm_force_y:ti.template(),
    u_square:ti.template(),
    X_vertical:ti.template(),
    X_horizontal:ti.template(),
    h:ti.template(),
    e:ti.template(),
    dx:float,
    dt:float,
    boundary_mask:ti.template(),
    ibm_boundary_mask:ti.template()
):
    print("real_gravity",real_gravity)
    ibm_force_x.fill(0.0)
    ibm_force_y.fill(0.0)
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        acc_int[I] = -dt * h[I]* real_gravity

    # for I in ti.grouped(h):
    #     divu = interp_MAC_divergence_u(u_x, u_y, ti.Vector([I[0] + 0.5, I[1] + 0.5]) * dx, dx)
    #     gamma_term = (1.4 - 1) / 4
    #     if divu < 0:
    #         # divu = ti.abs(divu)
    #         # acc_int[I] += dt * h[I] * (gamma_term * divu + ti.math.sqrt(gamma_term ** 2 * divu**2 + ti.abs(1.4 * acc_int[I] / h[I]))) * divu * 0.25
    #         acc_int[I] += dt * 1 * divu * h[I] * dx
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        h_x = (sample(h,i,j)+sample(h,i-1,j))/2
        # tem1, grad, tem2 = interp_grad_2(acc_int, X_horizontal[i, j], dx, BL_x=0.5, BL_y=0.5)
        # h_x, tem3, tem2 = interp_grad_2(h, X_horizontal[i, j], dx, BL_x=0.5, BL_y=0.5)
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            # u_x[i,j]+=grad[0] / h_x
            if(sample(ibm_boundary_mask,i-1,j)>=1 and  sample(ibm_boundary_mask,i-1,j)>=1):
                u_x[i,j]+=grad+ibm_coef*(0.8-u_x[i,j])
                ibm_force_x[i,j]=ibm_coef*(0.8-u_x[i,j])
            else:
                u_x[i,j]+=grad
            #u_x[i,j]+=grad+ ibm_coef*(0.8-u_x[i,j])

    ii, jj = u_y.shape
    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        h_y=(sample(h,i,j)+sample(h,i,j-1))/2
        if j >= 1 and j <= jj - 2:
            # tem1, grad, tem2 = interp_grad_2(acc_int, X_vertical[i, j], dx, BL_x=0.5, BL_y=0.5)
            # h_y, tem3, tem2 = interp_grad_2(h, X_vertical[i, j], dx, BL_x=0.5, BL_y=0.5)
            if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
                # u_y[i,j]+=grad[1] / h_y
                u_y[i,j]+=grad 


@ti.kernel
def update_h(
    u_x:ti.template(),
    u_y:ti.template(),
    h:ti.template(),
    dx:float,
    dt:float
):
    for i,j in h:
        pos=ti.Vector([i+0.5,j+0.5])*dx
        div = interp_MAC_divergence_u(u_x, u_y, pos, dx)
        h[i,j]-=h[i,j]*div*dt

@ti.kernel
def g2p_update_h(particles_h:ti.template(),  particles_pos:ti.template(), h:ti.template(),u_x:ti.template(), u_y:ti.template(), dx:float,dt:float, particles_active:ti.template()):
    for i in particles_h:
        if(particles_active[i]>=1):
            div = interp_MAC_divergence_u(u_x, u_y, particles_pos[i], dx)
            particles_h[i]-=particles_h[i]* div *dt
##################################################################
################## 4. calculate of F and T #######################
##################################################################

@ti.kernel
def update_T(T_x: ti.template(), T_y: ti.template(), T_x_init: ti.template(), T_y_init: ti.template(),
             T_x_grad_m: ti.template(), T_y_grad_m: ti.template()):
    for i in T_x:
        T_grad_m = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i]])
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i]])
        T = T_grad_m @ T_init
        T_x[i] = T[:, 0]
        T_y[i] = T[:, 1]

@ti.kernel
def check_FT(T_x: ti.template(), T_y: ti.template(),F_x: ti.template(), F_y: ti.template())->bool:
    epsilon=1e-4
    pass_check=True
    for i in T_x:
        T = ti.Matrix.cols([T_x[i], T_y[i]])
        F = ti.Matrix.rows([F_x[i], F_y[i]])
        I=ti.Matrix([[1.0,0.0],[0.0,1.0]])
        if((F@T-I).norm()>epsilon):
            pass_check=False
            #print((F@T-I).norm(),end=",")
    return pass_check

@ti.kernel
def check_FT_with_max(T_x: ti.template(), T_y: ti.template(),F_x: ti.template(), F_y: ti.template(),max_speed:ti.template())->bool:
    epsilon=1e-4
    pass_check=True
    max_speed[None]=0.0
    for i in T_x:
        T = ti.Matrix.cols([T_x[i], T_y[i]])
        F = ti.Matrix.rows([F_x[i], F_y[i]])
        I=ti.Matrix([[1.0,0.0],[0.0,1.0]])
        if((F@T-I).norm()>epsilon):
            pass_check=False
            ti.atomic_max(max_speed[None], (F@T-I).norm())
    return pass_check


###################################################################
########################### 5. p2g ################################
###################################################################

@ti.kernel
def compute_particle_grad_u(
    particles_pos: ti.template(), 
    u_x: ti.template(), 
    u_y: ti.template(),
    C_x: ti.template(), 
    C_y: ti.template(),
    dx:float,
    particles_active:ti.template()
):
    C_x.fill(0.0)
    C_y.fill(0.0)
    for i in particles_pos:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            u_x_p, grad_u_x_p, _ = interp_grad_2(u_x, p, dx, BL_x=0.0, BL_y=0.5, is_y=False)
            u_y_p, grad_u_y_p, _ = interp_grad_2(u_y, p, dx, BL_x=0.5, BL_y=0.0, is_y=True)
            C_x[i] = grad_u_x_p*0.5
            C_y[i] = grad_u_y_p*0.5

@ti.kernel
def compute_particle_grad_scalar(
    particles_pos: ti.template(), 
    h: ti.template(), 
    grad_h: ti.template(),
    dx:float,
    particles_active:ti.template()
):
    grad_h.fill(0.0)
    for i in particles_pos:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem, grad_h_tem, _ = interp_grad_2(h, p, dx, BL_x=0.5, BL_y=0.5, is_y=False)
            grad_h[i] = grad_h_tem*0.5

@ti.kernel
def stretach_grad_scalar(
    T_x:ti.template(), 
    T_y:ti.template(),
    grad_s:ti.template(), 
    particles_active:ti.template()
):
    for i in T_x:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            grad_s[i]=T@grad_s[i]


@ti.kernel
def P2G(particles_imp: ti.template(), particles_pos: ti.template(), u_x: ti.template(), u_y: ti.template(), C_x: ti.template(), C_y: ti.template(),
        psi: ti.template(), psi_x_grid: ti.template(), psi_y_grid: ti.template(),
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), dx:float, particles_active:ti.template()):
    
    u_x.fill(0.0)
    u_y.fill(0.0)
    psi_x_grid.fill(0.0)
    psi_y_grid.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    # print(particles_imp[i][0], weight, delta)
                    if use_APIC:
                        u_x[face_id] += (particles_imp[i][0] + delta) * weight
                    else:
                        u_x[face_id] += (particles_imp[i][0]) * weight

                    psi_x_grid[face_id] += psi[i] * weight

            # vertical impulse
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    p2g_weight_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    if use_APIC:
                        u_y[face_id] += (particles_imp[i][1] + delta) * weight
                    else:
                        u_y[face_id] += (particles_imp[i][1]) * weight

                    psi_y_grid[face_id] += psi[i] * weight
        """
        else:
            # horizontal impulse
            pos = particles_pos[i] / dx
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    # print(particles_imp[i][0], weight, delta)
                    if use_APIC:
                        u_x[face_id] += (particles_imp[i][0]) * weight
                    else:
                        u_x[face_id] += (particles_imp[i][0]) * weight

                    psi_x_grid[face_id] += psi[i] * weight

            # vertical impulse
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    p2g_weight_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    if use_APIC:
                        u_y[face_id] += (particles_imp[i][1]) * weight
                    else:
                        u_y[face_id] += (particles_imp[i][1]) * weight

                    psi_y_grid[face_id] += psi[i] * weight
        """
    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale
            psi_x_grid[I] *= scale

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale
            psi_y_grid[I] *= scale

@ti.kernel
def P2G_new(particles_imp: ti.template(), particles_pos: ti.template(), u_x: ti.template(), u_y: ti.template(), C_x: ti.template(), C_y: ti.template(),
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), dx:float, particles_active:ti.template()):
    
    u_x.fill(0.0)
    u_y.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        if  particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    u_x[face_id] += (particles_imp[i][0]+ delta) * weight

            # vertical impulse
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    p2g_weight_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    u_y[face_id] += (particles_imp[i][1] + delta) * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale

@ti.kernel
def P2G_new_1(particles_imp: ti.template(), particles_pos: ti.template(), u_x: ti.template(), u_y: ti.template(), C_x: ti.template(), C_y: ti.template(),
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), dx:float, particles_active:ti.template()):
    
    u_x.fill(0.0)
    u_y.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if abs(pos[0] - face_id[0])<1 and abs((pos[1] - face_id[1] - 0.5))<=1 and 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_1(pos[0] - face_id[0]) * N_1(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    u_x[face_id] += (particles_imp[i][0]+ delta) * weight

            # vertical impulse
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if abs(pos[0] - face_id[0] - 0.5)<1 and abs(pos[1] - face_id[1])<1 and 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_1(pos[0] - face_id[0] - 0.5) * N_1(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    p2g_weight_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    u_y[face_id] += (particles_imp[i][1] + delta) * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale

@ti.kernel
def P2G_height(particles_h: ti.template(), particles_pos: ti.template(), h: ti.template(), grad_h: ti.template(),
        p2g_weight: ti.template(),dx:float, particles_active:ti.template()):
    
    h.fill(0.0)
    p2g_weight.fill(0.0)

    for i in particles_h:
        if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]-0.5) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0]+0.5 - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight[face_id] += weight
                    delta = grad_h[i].dot(dpos)
                    if use_APIC:
                        h[face_id] += (particles_h[i] + delta) * weight
                    else:
                        h[face_id] += particles_h[i] * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            scale = 1. / p2g_weight[I]
            h[I] *= scale


@ti.kernel
def P2G_height_new(particles_h: ti.template(), particles_pos: ti.template(), h: ti.template(), 
        p2g_weight: ti.template(),dx:float, particles_active:ti.template()):
    
    h.fill(0.0)
    p2g_weight.fill(0.0)

    for i in particles_h:
        if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]-0.5) * N_2(pos[1] - face_id[1] - 0.5)
                    
                    p2g_weight[face_id] += weight
                    h[face_id] += particles_h[i] * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            scale = 1. / p2g_weight[I]
            h[I] *= scale

@ti.kernel
def P2G_height_and_acc(
    particles_h: ti.template(),particles_e:ti.template(), particles_lamb: ti.template(), particles_pos: ti.template(), 
    h: ti.template(), e:ti.template(), acc_int: ti.template(),
    p2g_weight: ti.template(),dx:float, particles_active:ti.template(), T_x:ti.template(), T_y:ti.template()):
    
    h.fill(0.0)
    p2g_weight.fill(0.0)
    acc_int.fill(0.0)

    for i in particles_h:
        T = ti.Matrix.cols([T_x[i], T_y[i]])
        if particles_active[i] == 1:
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]-0.5) * N_2(pos[1] - face_id[1] - 0.5)
                    p2g_weight[face_id] += weight
                    h[face_id] += particles_h[i] * weight
                    e[face_id] += particles_e[i] * weight
                    acc_int[face_id] += (T @ particles_lamb[i]) * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            scale = 1. / p2g_weight[I]
            h[I] *= scale
            acc_int[I] *= scale
            e[I] *= scale


@ti.kernel
def P2G_height_and_e(
    particles_h: ti.template(),particles_e:ti.template(), particles_pos: ti.template(), 
    h: ti.template(), e:ti.template(),
    p2g_weight: ti.template(),dx:float, particles_active:ti.template(), T_x:ti.template(), T_y:ti.template()):
    
    h.fill(0.0)
    e.fill(0.0)
    p2g_weight.fill(0.0)

    for i in particles_h:
        T = ti.Matrix.cols([T_x[i], T_y[i]])
        if particles_active[i] == 1:
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]-0.5) * N_2(pos[1] - face_id[1] - 0.5)
                    p2g_weight[face_id] += weight
                    h[face_id] += particles_h[i] * weight
                    e[face_id] += particles_e[i] * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            scale = 1. / p2g_weight[I]
            h[I] *= scale
            e[I] *= scale

@ti.kernel
def P2G_height_water(
    particles_h: ti.template(),
    grad_h:ti.template(),
    particles_pos: ti.template(), 
    h: ti.template(),
    p2g_weight: ti.template(),
    dx:float, 
    particles_active:ti.template()
):
    
    h.fill(0.0)
    p2g_weight.fill(0.0)

    for i in particles_h:
        if particles_active[i] == 1:
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]-0.5) * N_2(pos[1] - face_id[1] - 0.5)
                    p2g_weight[face_id] += weight
                    dpos = ti.Vector([face_id[0]+0.5 - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    delta = grad_h[i].dot(dpos)
                    h[face_id] += (particles_h[i]+delta) * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            scale = 1. / p2g_weight[I]
            h[I] *= scale


@ti.kernel
def P2G_height_and_acc_1(
    particles_h: ti.template(), particles_lamb: ti.template(), particles_pos: ti.template(), 
    h: ti.template(), acc_int: ti.template(),
    p2g_weight: ti.template(),dx:float, particles_active:ti.template()):
    
    h.fill(0.0)
    p2g_weight.fill(0.0)
    acc_int.fill(0.0)

    for i in particles_h:
        if particles_active[i] == 1:
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y:
                    weight = N_1(pos[0] - face_id[0]-0.5) * N_1(pos[1] - face_id[1] - 0.5)
                    p2g_weight[face_id] += weight
                    h[face_id] += particles_h[i] * weight
                    acc_int[face_id]+=particles_lamb[i] * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            scale = 1. / p2g_weight[I]
            h[I] *= scale
            acc_int[I] *= scale
##########################################################################
##########################################################################
##########################################################################

@ti.kernel
def rK4_after_advect_acc(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template(),
    acc_int_tem:ti.template(),
    u2_square:ti.template(),

    u_x_acc:ti.template(), 
    u_y_acc:ti.template(), 
    h_acc:ti.template(),
    acc_int_tem_acc:ti.template(),

    dx:ti.template(),
):
    for I in ti.grouped(h):
        acc_int_tem_acc[I]=-gravity*h[I]+ u2_square[I]
    
    for i,j in u_x:
        u_x_acc[i,j]=(sample(acc_int_tem_acc,i,j)-sample(acc_int_tem_acc,i-1,j))/dx
    

    for i,j in u_y:
        u_y_acc[i,j]=(sample(acc_int_tem_acc,i,j)-sample(acc_int_tem_acc,i,j-1))/dx

    for i,j in h:
        pos=ti.Vector([i+0.5,j+0.5])*dx
        div = interp_MAC_divergence_u(u_x, u_y, pos, dx)
        #h_acc[i,j]=-h[i,j]*div

@ti.kernel
def rK4_after_advect_final(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template(),
    acc_int_tem:ti.template(),

    u_x_save:ti.template(), 
    u_y_save:ti.template(), 
    h_save:ti.template(),
    acc_int_tem_save:ti.template(),

    u_x_buffer:ti.template(), 
    u_y_buffer:ti.template(), 
    h_buffer:ti.template(),
    acc_int_tem_buffer:ti.template(),

    dt:ti.template()
):
    for I in ti.grouped(u_x):
        u_x[I]=u_x_save[I]+dt/6*(
            u_x_buffer[I][0]+2*u_x_buffer[I][1]+2*u_x_buffer[I][2]+u_x_buffer[I][3]
        )

    for I in ti.grouped(u_y):
        u_y[I]=u_y_save[I]+dt/6*(
            u_y_buffer[I][0]+2*u_y_buffer[I][1]+2*u_y_buffer[I][2]+u_y_buffer[I][3]
        )

    for I in ti.grouped(h):
        h[I]=h_save[I]+dt/6*(
            h_buffer[I][0]+2*h_buffer[I][1]+2*h_buffer[I][2]+h_buffer[I][3]
        )
    
    for I in ti.grouped(acc_int_tem):
        acc_int_tem[I]=acc_int_tem_save[I]+dt/6*(
            acc_int_tem_buffer[I][0]+2*acc_int_tem_buffer[I][1]+2*acc_int_tem_buffer[I][2]+acc_int_tem_buffer[I][3]
        )

def RK4_after_advect(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template(),
    acc_int_tem:ti.template(),
    u2_square:ti.template(),

    u_x_buffer:ti.template(), 
    u_y_buffer:ti.template(), 
    h_buffer:ti.template(),
    acc_int_tem_buffer:ti.template(),

    u_x_save:ti.template(), 
    u_y_save:ti.template(), 
    h_save:ti.template(),
    acc_int_tem_save:ti.template(),

    u_x_acc:ti.template(), 
    u_y_acc:ti.template(), 
    h_acc:ti.template(),
    acc_int_tem_acc:ti.template(),

    dx:ti.template(),
    curr_dt:ti.template()
):
    copy_to(u_x,u_x_save)
    copy_to(u_y,u_y_save)
    copy_to(h,h_save)
    copy_to(acc_int_tem,acc_int_tem_save)

    # first round
    rK4_after_advect_acc(
        u_x,  u_y,  h, acc_int_tem, u2_square,
        u_x_acc,  u_y_acc,  h_acc, acc_int_tem_acc,
        dx
    )
    scalar2vector(u_x_acc,u_x_buffer,0)
    scalar2vector(u_y_acc,u_y_buffer,0)
    scalar2vector(h_acc,h_buffer,0)
    scalar2vector(acc_int_tem_acc,acc_int_tem_buffer,0)
    
    add_fields(u_x_save,u_x_acc,u_x,0.5*curr_dt)
    add_fields(u_y_save,u_y_acc,u_y,0.5*curr_dt)
    add_fields(h_save,h_acc,h,0.5*curr_dt)
    add_fields(acc_int_tem_save,acc_int_tem_acc,acc_int_tem,0.5*curr_dt)
    
    # second round
    rK4_after_advect_acc(
        u_x,  u_y,  h, acc_int_tem, u2_square,
        u_x_acc,  u_y_acc,  h_acc, acc_int_tem_acc,
        dx
    )
    scalar2vector(u_x_acc,u_x_buffer,1)
    scalar2vector(u_y_acc,u_y_buffer,1)
    scalar2vector(h_acc,h_buffer,1)
    scalar2vector(acc_int_tem_acc,acc_int_tem_buffer,1)
    
    add_fields(u_x_save,u_x_acc,u_x,0.5*curr_dt)
    add_fields(u_y_save,u_y_acc,u_y,0.5*curr_dt)
    add_fields(h_save,h_acc,h,0.5*curr_dt)
    add_fields(acc_int_tem_save,acc_int_tem_acc,acc_int_tem,0.5*curr_dt)

    # thirst round
    rK4_after_advect_acc(
        u_x,  u_y,  h, acc_int_tem,  u2_square,
        u_x_acc,  u_y_acc,  h_acc, acc_int_tem_acc,
        dx
    )
    scalar2vector(u_x_acc,u_x_buffer,2)
    scalar2vector(u_y_acc,u_y_buffer,2)
    scalar2vector(h_acc,h_buffer,2)
    scalar2vector(acc_int_tem_acc,acc_int_tem_buffer,2)

    add_fields(u_x_save,u_x_acc,u_x,curr_dt)
    add_fields(u_y_save,u_y_acc,u_y,curr_dt)
    add_fields(h_save,h_acc,h,curr_dt)
    add_fields(acc_int_tem_save,acc_int_tem_acc,acc_int_tem,curr_dt)
    
    # fourth round
    rK4_after_advect_acc(
        u_x,  u_y,  h, acc_int_tem,     u2_square,
        u_x_acc,  u_y_acc,  h_acc, acc_int_tem_acc,
        dx
    )
    scalar2vector(u_x_acc,u_x_buffer,3)
    scalar2vector(u_y_acc,u_y_buffer,3)
    scalar2vector(h_acc,h_buffer,3)
    scalar2vector(acc_int_tem_acc,acc_int_tem_buffer,3)

    # RK4 final
    rK4_after_advect_final(
        u_x, u_y, h,acc_int_tem,
        u_x_save, u_y_save, h_save,acc_int_tem_save,
        u_x_buffer, u_y_buffer, h_buffer,acc_int_tem_buffer,curr_dt
    )

@ti.kernel
def compute_artificial_viscosity(pressure:ti.template(), u_x:ti.template(), u_y:ti.template(), X:ti.template(), rho:ti.template(), dx:float):
    for I in ti.grouped(pressure):
        divu = interp_MAC_divergence_u(u_x, u_y, X[I], dx)
        gamma_term = (1.4 - 1) / 4
        if divu < 0:
            divu = ti.abs(divu)
            pressure[I] += rho[I] * (gamma_term * divu + ti.math.sqrt(gamma_term ** 2 * divu**2 + ti.abs(1.4 * pressure[I] / rho[I]))) * divu* 0.25