#
from hyperparameters import *
from taichi_utils import *
from io_utils import *
import sys
import shutil
import time
#

# @ti.kernel
# def g2p_lamb(
#     particles_lamb: ti.template(), 
#     particles_pos: ti.template(), 
#     acc_int:ti.template(),
#     dx: float,
#     particles_active:ti.template()
# ):
#     for i in particles_lamb:
#         if particles_active[i] >= 1:
#             p = particles_pos[i]
#             inerp,tem1, tem2 = interp_grad_2(acc_int, p, dx, BL_x=0.5, BL_y=0.5)
#             particles_lamb[i] = inerp 


@ti.kernel
def g2p_lamb(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    pressure:ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem1, grad_p_u, tem2 = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( -grad_p_u)

@ti.kernel
def accumulate_lamb(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(), 
    pressure: ti.template(), 
    F_x:ti.template(),
    F_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem1, grad_p_u, tem2 = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_x, vis_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( -grad_p_u + particles_grad_half_u[i]+ vis* curr_dt) 

@ti.kernel
def accumulate_lamb_visc(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(), 
    pressure: ti.template(), 
    particles_rho:ti.template(),
    F_x:ti.template(),
    F_y:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem1, grad_p_u, tem2 = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            # vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_x, vis_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( grad_p_u/ particles_rho[i] + particles_grad_half_u[i]) 


@ti.kernel
def accumulate_force_water(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(),     
    F_x:ti.template(),
    F_y:ti.template(),
    f_x:ti.template(),
    f_y:ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            f, tem3, tem4, tem5 = interp_u_MAC_grad(f_x, f_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ (particles_grad_half_u[i]+ f) 

@ti.kernel
def accumulate_lamb_visc_ibm(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(), 
    pressure: ti.template(), 
    particles_rho:ti.template(),
    F_x:ti.template(),
    F_y:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    ibm_boundary_mask:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem1, grad_p_u, tem2 = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            grid_idx = ti.round((particles_pos[i]) / dx - 0.5, int)
            u = ti.Vector([0.0,0.0])
            if(ibm_boundary_mask[grid_idx]>=1):
                u, tem3, tem4, tem5 = interp_u_MAC_grad(u_x, u_y, p, dx)
                u = ti.Vector([-boat_v,0.0])-u
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( grad_p_u/ particles_rho[i] + particles_grad_half_u[i]+ibm_coef*u) 

@ti.kernel
def accumulate_lamb_visc_ibm2(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(), 
    pressure: ti.template(), 
    particles_rho:ti.template(),
    F_x:ti.template(),
    F_y:ti.template(),
    ibm_force_x:ti.template(),
    ibm_force_y:ti.template(),
    ibm_boundary_mask:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem1, grad_p_u, tem2 = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            grid_idx = ti.round((particles_pos[i]) / dx - 0.5, int)
            u = ti.Vector([0.0,0.0])
            if(ibm_boundary_mask[grid_idx]>=1):
                u, tem3, tem4, tem5 = interp_u_MAC_grad(ibm_force_x, ibm_force_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( grad_p_u/ particles_rho[i] + particles_grad_half_u[i]+u) 

@ti.kernel
def accumulate_lamb_no_u2(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    pressure: ti.template(), 
    F_x:ti.template(),
    F_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            tem1, grad_p_u, tem2 = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_x, vis_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( -grad_p_u + vis* curr_dt) 

@ti.kernel
def accumulate_lamb_no_u2_no_p(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    F_x:ti.template(),
    F_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_x, vis_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ vis* curr_dt

@ti.kernel
def accumulate_lamb_no_u2_no_p_non_slip(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    F_x:ti.template(),
    F_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    non_slip_m_x:ti.template(),
    non_slip_m_y:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_x, vis_y, p, dx)
            impulse, tem3, tem4, tem5 = interp_u_MAC_grad(non_slip_m_x, non_slip_m_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ (vis* curr_dt+impulse)

@ti.kernel
def accumulate_lamb_whole(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    F_x:ti.template(),
    F_y:ti.template(),
    vis_acc_x:ti.template(),
    vis_acc_y:ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_acc_x, vis_acc_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] = F @ vis 

@ti.kernel       
def accumulate_lamb2(
    particles_grad_lamb: ti.template(), 
    particles_pos: ti.template(), 
    particles_grad_half_u: ti.template(), 
    grad_p_x: ti.template(),
    grad_p_y: ti.template(), 
    F_x:ti.template(),
    F_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    curr_dt: ti.template(),
    dx: float,
    particles_active:ti.template()
):
    for i in particles_grad_lamb:
        if particles_active[i] >= 1:
            p = particles_pos[i]
            grad_p_u, tem3, tem4, tem5 = interp_u_MAC_grad(grad_p_x, grad_p_y, p, dx)
            vis, tem3, tem4, tem5 = interp_u_MAC_grad(vis_x, vis_y, p, dx)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ ( -grad_p_u + particles_grad_half_u[i]+ vis* curr_dt) 

@ti.kernel
def calculate_viscous_force(
    u_x:ti.template(),
    u_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    viscosity:float,
    dx:float
):
    for i,j in u_x:
        vis_x[i,j]= viscosity*(sample(u_x,i+1,j)+sample(u_x,i-1,j)+sample(u_x,i,j+1)+sample(u_x,i,j-1)-4*sample(u_x,i,j))/dx/dx

    for i,j in u_y:
        vis_y[i,j]= viscosity*(sample(u_y,i+1,j)+sample(u_y,i-1,j)+sample(u_y,i,j+1)+sample(u_y,i,j-1)-4*sample(u_y,i,j))/dx/dx

@ti.kernel
def calculate_viscous_force2(
    u_x:ti.template(),
    u_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    viscosity_new:float,
    dx:float
): #https://www.math.purdue.edu/~zhan1966/research/paper/DMP_Laplacian.pdf
    for i,j in u_x:
        vis_x[i,j]= viscosity_new*(
            -0.25*sample(u_x,i+2,j)-0.25*sample(u_x,i-2,j)-0.25*sample(u_x,i,j+2)-0.25*sample(u_x,i,j-2)+
            2*sample(u_x,i+1,j)+2*sample(u_x,i-1,j)+2*sample(u_x,i,j+1)+2*sample(u_x,i,j-1)-7*sample(u_x,i,j))/dx/dx

    for i,j in u_y:
        vis_y[i,j]= viscosity_new*(
            -0.25*sample(u_y,i+2,j)-0.25*sample(u_y,i-2,j)-0.25*sample(u_y,i,j+2)-0.25*sample(u_y,i,j-2)+
            2*sample(u_y,i+1,j)+2*sample(u_y,i-1,j)+2*sample(u_y,i,j+1)+2*sample(u_y,i,j-1)-7*sample(u_y,i,j))/dx/dx

@ti.kernel
def calculate_viscous_force3(
    u_x:ti.template(),
    u_y:ti.template(),
    vis_x:ti.template(),
    vis_y:ti.template(),
    viscosity:float,
    dx:float
): #https://www.math.purdue.edu/~zhan1966/research/paper/DMP_Laplacian.pdf
    for i,j in u_x:
        vis_x[i,j]= viscosity*(
            -0.25*sample(u_x,i+2,j)-0.25*sample(u_x,i-2,j)+2*sample(u_x,i+1,j)+2*sample(u_x,i-1,j)+
            1*sample(u_x,i,j+1)+1*sample(u_x,i,j-1)-5.5*sample(u_x,i,j))/dx/dx

    for i,j in u_y:
        vis_y[i,j]= viscosity*(
            -0.25*sample(u_y,i,j+2)-0.25*sample(u_y,i,j-2)+2*sample(u_y,i,j+1)+2*sample(u_y,i,j-1)+
            1*sample(u_y,i+1,j)+1*sample(u_y,i-1,j)-5.5*sample(u_y,i,j))/dx/dx

@ti.kernel
def apply_bc_w_cavity_second(stream:ti.template(), w_n: ti.template(), dx:float):
    u_dim, v_dim = w_n.shape
    inv_dx = 1.0/dx
    h2=inv_dx*inv_dx
    for i, j in w_n:
        if j == v_dim - 1:
            w_n[i, j] = -3 * inv_dx *  cavity_vel - h2 * (4 * stream[i, j-1] - 0.5 * stream[i, j-2])
        if i == 0:
            w_n[i, j] = -h2 * (4 * stream[i + 1, j] - 0.5 * stream[i + 2, j])
        if j == 0:
            w_n[i, j] = -h2 * (4 * stream[i, j + 1] - 0.5 * stream[i, j + 2])
        if i == u_dim - 1:
            w_n[i, j] = -h2 * (4 * stream[i - 1, j] - 0.5 * stream[i - 2, j])

@ti.kernel
def apply_bc_w_cavity_first(stream:ti.template(), w_n: ti.template(), dx:float):
    u_dim, v_dim = w_n.shape
    inv_dx = 1.0/dx
    h2=inv_dx*inv_dx
    for i, j in w_n:
        if j == v_dim - 1:
            w_n[i, j] = -2.0 * stream[i, j-1] * h2 - 2.0 * cavity_vel * inv_dx
            # w_n[i, j] = -3 * inv_dx *  cavity_vel - h2 * (4 * stream[i, j-1] - 0.5 * stream[i, j-2])
            continue
        if i == 0:
            w_n[i, j] = -2.0 * h2 * stream[i + 1, j]
            # w_n[i, j] = h2 * (4 * stream[i + 1, j] - 0.5 * stream[i + 2, j])
            continue
        if j == 0:
            # w_n[i, j] = -h2 * (4 * stream[i, j + 1] - 0.5 * stream[i, j + 2])
            w_n[i, j] = -2.0 * h2 * stream[i, j + 1]
            continue
        if i == u_dim - 1:
            w_n[i, j] = -2.0 * h2 * stream[i - 1, j]
            # w_n[i, j] = h2 * (4 * stream[i - 1, j] - 0.5 * stream[i - 2, j])

@ti.kernel
def set_first_guess(u_x:ti.template(),u_y:ti.template()):
    i_shape,j_shape=u_x.shape
    for i,j in u_x:
        if(j==j_shape-1):
            u_x[i,j]=1.0
        if(j==0):
            u_x[i,j]=0.0

    i_shape,j_shape=u_y.shape
    for i,j in u_y:
        if(i==i_shape-1):
            u_y[i,j]=1.0
        if(i==0):
            u_y[i,j]=0.0



@ti.kernel
def calculate_vortex(
    u_x:ti.template(),
    u_y:ti.template(),
    w:ti.template(),
    dx:float
):
    for i,j in w:
        w[i,j]=(sample(u_y,i,j)-sample(u_y,i-1,j))/dx-(sample(u_x,i,j)-sample(u_x,i,j-1))/dx

@ti.kernel
def no_mask(boundary_mask: ti.template(), boundary_vel: ti.template()):
    boundary_mask.fill(0.0)
    boundary_vel.fill(0.0)

@ti.kernel
def point_from_center_boundary(center_boundary_mask:ti.template(),point_boundary_mask:ti.template()):
    point_boundary_mask.fill(0.0)
    for i,j in center_boundary_mask:
        if(center_boundary_mask[i,j]>=1):
            point_boundary_mask[i,j]=1.0
            point_boundary_mask[i+1,j]=1.0
            point_boundary_mask[i,j+1]=1.0
            point_boundary_mask[i+1,j+1]=1.0
    shape_x,shape_y=point_boundary_mask.shape
    for i,j in point_boundary_mask:
        if(i == 0 or j == 0 or i==shape_x-1 or j==shape_y-1):
            point_boundary_mask[i,j]=1.0

@ti.kernel
def edge_from_center_boundary(center_boundary_mask:ti.template(),edge_x_boundary_mask:ti.template(),edge_y_boundary_mask:ti.template()):
    edge_x_boundary_mask.fill(0.0)
    edge_y_boundary_mask.fill(0.0)
    for i,j in center_boundary_mask:
        if(center_boundary_mask[i,j]>=1):
            edge_x_boundary_mask[i,j]=1.0
            edge_x_boundary_mask[i+1,j]=1.0
            edge_y_boundary_mask[i,j]=1.0
            edge_y_boundary_mask[i,j+1]=1.0
    shape_x,shape_y=edge_x_boundary_mask.shape
    for i,j in edge_x_boundary_mask:
        if(i==0 or i== shape_x-1):
            edge_x_boundary_mask[i,j]=1.0
    shape_x,shape_y=edge_y_boundary_mask.shape
    for i,j in edge_y_boundary_mask:
        if(j==0 or j== shape_y-1):
            edge_y_boundary_mask[i,j]=1.0

@ti.kernel
def extend_boundary_field(a:ti.template(),b:ti.template()):
    shape_x,shape_y=a.shape
    for i,j in b:
        if(i<shape_x and j<shape_y):
            b[i,j]=a[i,j]
        else:
            b[i,j]=1

@ti.kernel
def check_all_zero(w:ti.template())->bool:
    all_zero=True
    for I in ti.grouped(w):
        if(w[I]!=0):
            all_zero=False
            print(I,w[I])
    return all_zero


@ti.kernel
def apply_bc_v(boundary_mask:ti.template(),u_horizontal: ti.template(), u_vertical: ti.template()):
        for i, j in boundary_mask:
            if boundary_mask[i, j] > 0:
                u_vertical[i, j] = 0
                u_vertical[i, j + 1] = 0
                u_horizontal[i, j] = 0
                u_horizontal[i + 1, j] = 0
        
        u_dim, v_dim = u_horizontal.shape
        
        for i, j in u_horizontal:
            if i == 0:
                u_horizontal[i,j] = 0
            if i == u_dim - 1:
                u_horizontal[i,j] = 0
            if j == 0:
                u_horizontal[i,j] = 1
            if j == v_dim - 1:
                u_horizontal[i,j] = 0

        for i, j in u_horizontal:
            if j == 1:
                u_horizontal[i,j] = 2 - u_horizontal[i,j - 1]
            if j == v_dim - 2:
                u_horizontal[i,j] = 0 - u_horizontal[i,j + 1]

        u_dim, v_dim = u_vertical.shape
        for i, j in u_vertical:
            if j == 0:
                u_vertical[i,j] = 0.0
            if j == v_dim - 1:
                u_vertical[i,j] = 0
            if i == 0:
                u_vertical[i,j] = 0.0
            if i == u_dim - 1:
                u_vertical[i,j] = 0.0

        for i, j in u_vertical:
            if i == 1:
                u_vertical[i,j] = 0 - u_vertical[i-1,j]
            if i == u_dim - 2:
                u_vertical[i,j] = 0 - u_vertical[i + 1,j]


@ti.func
def trace(F):
    return F[0,0]+F[1,1]

@ti.func
def proj_FT(F):
    max_iter = 10  # Maximum number of iterations for convergence
    tol = 1e-6  # Tolerance for convergence

    lamda = 0.0
    for _ in range(max_iter):
        F_invT = F.transpose().inverse()
        F_plus_lambda_F_invT = F + lamda * F_invT
        det_F_plus_lambda_F_invT = ti.math.determinant(F_plus_lambda_F_invT)
        if abs(det_F_plus_lambda_F_invT - 1.0) < tol:
            break
        trace_F_invT = trace((F.transpose() @ F).inverse())
        lamda -= (det_F_plus_lambda_F_invT - 1.0) / (trace_F_invT * det_F_plus_lambda_F_invT)
        F += lamda * F_invT
    return F

@ti.kernel
def project_FT_field(T_x:ti.template(),T_y:ti.template(),F_x:ti.template(),F_y:ti.template()):
    for i in T_x:
        T = ti.Matrix.cols([T_x[i], T_y[i]])
        F = ti.Matrix.rows([F_x[i], F_y[i]])
        T=proj_FT(T)#T/ti.math.determinant(T)#proj_FT(T)
        F=proj_FT(F)#F/ti.math.determinant(F)#proj_FT(F)
        T_x[i]=T[:, 0]
        T_y[i]=T[:, 1]
        F_x[i]=F[0,:]
        F_y[i]=F[1,:]

@ti.kernel
def calculate_non_slip_impulse_and_enforce_boundary(
    u_x:ti.template(),
    u_y:ti.template(),
    boundary_mask:ti.template(),
    non_slip_m_x:ti.template(),
    non_slip_m_y:ti.template()
):
    non_slip_m_x.fill(0.0)
    non_slip_m_y.fill(0.0)

    for i, j in boundary_mask:
        if(valid_center(i-1,j,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_y[i-1,j] = hybrid_add(non_slip_m_y[i-1,j],-u_y[i-1,j]*lamb)
            u_y[i-1,j]=0.0
        if(valid_center(i,j-1,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_x[i,j-1] = hybrid_add(non_slip_m_x[i,j-1] , -u_x[i,j-1]*lamb)
            u_x[i,j-1]=0.0


    for i, j in boundary_mask:
        if(valid_center(i-1,j,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_y[i-1,j+1] = hybrid_add(non_slip_m_y[i-1,j+1],-u_y[i-1,j+1]*lamb)
            u_y[i-1,j+1]=0.0
        if(valid_center(i,j-1,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_x[i+1,j-1] = hybrid_add(non_slip_m_x[i+1,j-1] ,-u_x[i+1,j-1]*lamb)
            u_x[i+1,j-1]=0.0

    for i, j in boundary_mask:
        if(valid_center(i+1,j,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_y[i+1,j] = hybrid_add(non_slip_m_y[i+1,j] , -u_y[i+1,j]*lamb)
            u_y[i+1,j]=0.0
        if(valid_center(i,j+1,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_x[i,j+1] = hybrid_add(non_slip_m_x[i,j+1] , -u_x[i,j+1]*lamb)
            u_x[i,j+1]=0.0

    for i, j in boundary_mask:
        if(valid_center(i+1,j,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_y[i+1,j+1]=hybrid_add(non_slip_m_y[i+1,j+1],-u_y[i+1,j+1]*lamb)
            u_y[i+1,j+1]=0.0
        if(valid_center(i,j+1,boundary_mask) and boundary_mask[i,j]>0):
            non_slip_m_x[i+1,j+1]=hybrid_add(non_slip_m_x[i+1,j+1] ,-u_x[i+1,j+1]*lamb)
            u_x[i+1,j+1]=0.0
        
@ti.kernel
def stat_boundary_avg(boundary_mask:ti.template(),bm_pos_x:ti.template(), bm_num_x:ti.template(),bm_num1:ti.template(),bm_num2:ti.template()):   
    bm_num_x.fill(0.0)
    bm_num1.fill(0.0)
    bm_num2.fill(0.0)
    bm_pos_x.fill(0.0)
    for i,j in boundary_mask:
        if(boundary_mask[i,j]>0):
            bm_num_x[i]+=1
            bm_pos_x[i]+=j
            if(j+1<=256):
                bm_num1[i]+=1
            else:
                bm_num2[i]+=1
    
    for i in bm_num_x:
        if(bm_num_x[i]>0):
            bm_pos_x[i]=bm_pos_x[i]/bm_num_x[i]
            print(bm_pos_x[i],bm_num1[i],bm_num2[i])
    

@ti.kernel
def add_gravity_force(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template() , 
    dx:float, 
    dt:float,
    boundary_mask:ti.template()
):
    for i,j in u_x:
        pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
        grad_h_x=(sample(h,i,j)-sample(h,i-1,j))/dx
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            u_x[i,j]+=-gravity* grad_h_x**gamma*dt
            
    ii, jj = u_y.shape
    for i,j in u_y:
        if j >= 1 and j <= jj - 2:
            pos= ti.Vector([i+0.5,j])*dx
            #tem1,grad_h_y,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
            grad_h_y=(sample(h,i,j)-sample(h,i,j-1))/dx
            if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
                u_y[i,j]+=-gravity* grad_h_y**gamma*dt


@ti.kernel
def add_gravity_force_shock(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template(),
    e:ti.template(), 
    acc_int:ti.template(),
    dx:float, 
    dt:float,
    boundary_mask:ti.template()
):
    # for i,j in u_x:
    #     pos= ti.Vector([i,j+0.5])*dx
    #     #tem1,grad_h_x,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
    #     grad_h_x=(sample(h,i,j)-sample(h,i-1,j))/dx
    #     if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
    #         u_x[i,j]+=-gravity* grad_h_x**gamma*dt

    # for i,j in u_y:
    #     pos= ti.Vector([i+0.5,j])*dx
    #     #tem1,grad_h_y,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
    #     grad_h_y=(sample(h,i,j)-sample(h,i,j-1))/dx
    #     if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
    #         u_y[i,j]+=-gravity* grad_h_y**gamma*dt


    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        if(boundary_mask[I]==0):
            acc_int[I] = -dt * (1.4 - 1) * e[I] * h[I]
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        h_x = (sample(h,i,j)+sample(h,i-1,j))/2
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            u_x[i,j]+=grad / h_x
    

    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        h_y=(sample(h,i,j)+sample(h,i,j-1))/2
        if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
            u_y[i,j]+=grad / h_y


@ti.kernel
def add_drag_force(
    u_x:ti.template(), 
    u_y:ti.template(),
    wind_u_x:ti.template(), 
    wind_u_y:ti.template(),
    f_x:ti.template(), 
    f_y:ti.template(), 
    h:ti.template(),
    dt:float
):
    for i,j in u_x:
        if(case == 3 or case == 7 or case == 8 or case == 9 or case == 12):
            h0 = (sample(h,i-1,j)+sample(h,i,j))/2
            f_x[i,j] += drag_coef/h0*(wind_u_x[None]-u_x[i,j])*dt
            u_x[i,j] += drag_coef/h0*(wind_u_x[None]-u_x[i,j])*dt

    for i,j in u_y:
        if(case == 3 or case == 7 or case == 8 or case == 9 or case == 12):
            h0 = (sample(h,i,j-1)+sample(h,i,j))/2
            f_y[i,j] += drag_coef/h0*(wind_u_y[None]-u_y[i,j])*dt
            u_y[i,j] += drag_coef/h0*(wind_u_y[None]-u_y[i,j])*dt

@ti.kernel
def add_gravity_force_water(
    u_x:ti.template(), 
    u_y:ti.template(), 
    f_x:ti.template(), 
    f_y:ti.template(), 
    h:ti.template(),
    eta:ti.template(),
    acc_int:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_velocity:ti.template(),
    ibm_boundary_h:ti.template(), 
    ibm_boundary_coef:ti.template(),
    ibm_force_y:ti.template(),
    dx:float, 
    dt:float
):
    print("real_gravity",real_gravity)
    for I in ti.grouped(h):
        acc_int[I] = -dt * real_gravity * (h[I]+eta[I])
    
    for i,j in u_x:
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        u_x[i,j]+=grad
        f_x[i,j] = grad
        if(sample(ibm_boundary_mask,i,j) >= 1 or sample(ibm_boundary_mask,i-1,j) >= 1):
            v = (sample(ibm_boundary_velocity,i,j)+ sample(ibm_boundary_velocity,i-1,j))[0]/2
            coef = (sample(ibm_boundary_coef,i,j)+ sample(ibm_boundary_coef,i-1,j))/2
            f_x[i,j] += (v-u_x[i,j])*coef
            #u_x[i,j] = v
    
    ibm_force_y.fill(0.0)
    for i,j in u_y:
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        u_y[i,j]+=grad 
        f_y[i,j] = grad 
        if(sample(ibm_boundary_mask,i,j) >= 1 or sample(ibm_boundary_mask,i,j-1) >= 1):
            v = (sample(ibm_boundary_velocity,i,j)+ sample(ibm_boundary_velocity,i,j-1))[1]/2
            coef = (sample(ibm_boundary_coef,i,j)+ sample(ibm_boundary_coef,i,j-1))/2
            f_y[i,j] += (v-u_y[i,j])*coef
            ibm_force_y[i,j] = (v-u_y[i,j])*coef
            #u_y[i,j] = v

@ti.kernel
def add_gravity_force_water_others(
    u_x:ti.template(), 
    u_y:ti.template(), 
    f_x:ti.template(), 
    f_y:ti.template(), 
    h:ti.template(),
    eta:ti.template(),
    acc_int:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_velocity:ti.template(),
    ibm_boundary_h:ti.template(), 
    ibm_boundary_coef:ti.template(),
    ibm_force_y:ti.template(),
    dx:float, 
    dt:float
):
    print("real_gravity",real_gravity)
    for I in ti.grouped(h):
        acc_int[I] = -dt * real_gravity * (h[I]+eta[I])
    
    for i,j in u_x:
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        u_x[i,j]+=grad
        f_x[i,j] = grad
        if(sample(ibm_boundary_mask,i,j) >= 1 or sample(ibm_boundary_mask,i-1,j) >= 1):
            v = (sample(ibm_boundary_velocity,i,j)+ sample(ibm_boundary_velocity,i-1,j))[0]/2
            coef = (sample(ibm_boundary_coef,i,j)+ sample(ibm_boundary_coef,i-1,j))/2
            f_x[i,j] += (v-u_x[i,j])*coef
            #u_x[i,j] = v
    
    ibm_force_y.fill(0.0)
    for i,j in u_y:
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        u_y[i,j]+=grad 
        f_y[i,j] = grad 
        if(sample(ibm_boundary_mask,i,j) >= 1 or sample(ibm_boundary_mask,i,j-1) >= 1):
            v = (sample(ibm_boundary_velocity,i,j)+ sample(ibm_boundary_velocity,i,j-1))[1]/2
            coef = (sample(ibm_boundary_coef,i,j)+ sample(ibm_boundary_coef,i,j-1))/2
            f_y[i,j] += (v-u_y[i,j])*coef
            ibm_force_y[i,j] = (v-u_y[i,j])*coef
            #u_y[i,j] = v
    
    for I in ti.grouped(h):
        p = (I+0.5)*dx
        div = interp_MAC_divergence_u(u_x, u_y, p, dx)
        h[I] += -dt * h[I] * div

@ti.kernel
def add_gravity_force_water_shock(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template(),
    e:ti.template(), 
    acc_int:ti.template(),
    dx:float, 
    dt:float,
    boundary_mask:ti.template()
):
    # for i,j in u_x:
    #     pos= ti.Vector([i,j+0.5])*dx
    #     #tem1,grad_h_x,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
    #     grad_h_x=(sample(h,i,j)-sample(h,i-1,j))/dx
    #     if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
    #         u_x[i,j]+=-gravity* grad_h_x**gamma*dt

    # for i,j in u_y:
    #     pos= ti.Vector([i+0.5,j])*dx
    #     #tem1,grad_h_y,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
    #     grad_h_y=(sample(h,i,j)-sample(h,i,j-1))/dx
    #     if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
    #         u_y[i,j]+=-gravity* grad_h_y**gamma*dt

    print("real_gravity",real_gravity)
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        if(boundary_mask[I]==0):
            acc_int[I] = -dt * real_gravity * h[I]
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        h_x = (sample(h,i,j)+sample(h,i-1,j))/2
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            u_x[i,j]+=grad 
    

    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        h_y=(sample(h,i,j)+sample(h,i,j-1))/2
        if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
            u_y[i,j]+=grad 

@ti.kernel
def add_gravity_force_water_shock_ibm(
    u_x:ti.template(), 
    u_y:ti.template(), 
    h:ti.template(),
    e:ti.template(), 
    acc_int:ti.template(),
    dx:float, 
    dt:float,
    boundary_mask:ti.template(),
    ibm_boundary_mask:ti.template()
):
    # for i,j in u_x:
    #     pos= ti.Vector([i,j+0.5])*dx
    #     #tem1,grad_h_x,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
    #     grad_h_x=(sample(h,i,j)-sample(h,i-1,j))/dx
    #     if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
    #         u_x[i,j]+=-gravity* grad_h_x**gamma*dt

    # for i,j in u_y:
    #     pos= ti.Vector([i+0.5,j])*dx
    #     #tem1,grad_h_y,tem2=interp_grad_2(h, pos, dx, BL_x=0.5, BL_y=0.5)
    #     grad_h_y=(sample(h,i,j)-sample(h,i,j-1))/dx
    #     if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
    #         u_y[i,j]+=-gravity* grad_h_y**gamma*dt

    print("real_gravity",real_gravity)
    for I in ti.grouped(h):
        #acc_int[I]-=dt*gravity*h[I]- dt*u_square[I]
        if(boundary_mask[I]==0):
            acc_int[I] = -dt * real_gravity * h[I]
    
    for i,j in u_x:
        #pos= ti.Vector([i,j+0.5])*dx
        #tem1,grad_h_x,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad = (sample(acc_int,i,j)-sample(acc_int,i-1,j))/dx
        h_x = (sample(h,i,j)+sample(h,i-1,j))/2
        if(sample(boundary_mask,i-1,j)<=0 and sample(boundary_mask,i,j)<=0):
            if(sample(ibm_boundary_mask,i-1,j)>=1 and  sample(ibm_boundary_mask,i,j)>=1):
                u_x[i,j]+=grad+ibm_coef*(0.8-u_x[i,j])
            else:
                u_x[i,j]+=grad
    

    for i,j in u_y:
        #pos= ti.Vector([i+0.5,j])*dx
        #tem1,grad_h_y,tem2=interp_grad_2(acc_int, pos, dx, BL_x=0.5, BL_y=0.5)
        grad=(sample(acc_int,i,j)-sample(acc_int,i,j-1))/dx
        h_y=(sample(h,i,j)+sample(h,i,j-1))/2
        if(sample(boundary_mask,i,j-1)<=0 and sample(boundary_mask,i,j)<=0):
            u_y[i,j]+=grad 

@ti.kernel
def construct_3D_surfae(
    vis_h:ti.template(),
    vis_w:ti.template(),
    h:ti.template(),
    w:ti.template(),
    dx:float
):  
    for i,j,k in vis_h:
        if(k*dx*2 < h[i*2,j*2]):
            vis_h[i,j,k]=1.0
            vis_w[i,j,k]=abs(w[i*2,j*2])
        else:
            vis_h[i,j,k]=0.0
            vis_w[i,j,k]=0.0
    
        