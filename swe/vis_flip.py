from hyperparameters import *
from taichi_utils import *
import numpy as np
#####################################################################
#################     Initialization     ############################
#####################################################################

def flip_particles_init(
    flip_particles_pos:ti.template(),
    flip_particles_valid:ti.template(),
    ibm_boundary_mask: ti.template(),
    res_x:int,
    res_y:int,
    particles_per_cell_axis:int,
    dist_between_neighbor:float
):
    flip_particles_init_position(
        flip_particles_pos,
        res_x,
        res_y,
        particles_per_cell_axis,
        dist_between_neighbor
    )
    set_flip_particles_valid(
        flip_particles_pos,
        flip_particles_valid,
        ibm_boundary_mask,
    )

@ti.kernel
def flip_particles_init_position(
    flip_particles_pos:ti.template(),
    res_x:int,
    res_y:int,
    particles_per_cell_axis:int,
    dist_between_neighbor:float
):
    particles_x_num = particles_per_cell_axis * res_x
    particles_y_num = particles_per_cell_axis * res_y
    print(flip_particles_pos.shape,particles_x_num*particles_y_num,particles_x_num,particles_y_num,dist_between_neighbor,dx,particles_per_cell_axis,"particles_x_num")
    for i in flip_particles_pos:
        id_x = i % particles_x_num
        id_yz = i // particles_x_num
        id_y = id_yz % particles_y_num
        id_z = id_yz // particles_y_num
        flip_particles_pos[i] = (ti.Vector([id_x, id_y, id_z]) + 0.5) * dist_between_neighbor + ti.Vector([flip_particle_begin_x*dx,0.0,init_h-flip_particles_layer*dx])

@ti.kernel
def set_flip_particles_valid(
    flip_particles_pos:ti.template(),
    flip_particles_valid:ti.template(),
    ibm_boundary_mask: ti.template(),
):
    for I in flip_particles_pos:
        pos = flip_particles_pos[I]
        pos2D = ti.Vector([pos[0],pos[1]])
        pos_grid = ti.cast(ti.floor(pos2D/dx), ti.int32)
        if(ibm_boundary_mask[pos_grid]==1):
            flip_particles_valid[I] = 0
        else:
            flip_particles_valid[I] = 1
        
#####################################################################
#################      Visualiztion      ############################
#####################################################################

@ti.kernel
def flip_particles_vis(
    vis_particle_num:ti.template(),
    vis_particle_mapping:ti.template(),
    vis_particle_pos:ti.template(),
    flip_particles_pos:ti.template(),
    flip_particles_valid:ti.template(),
):
    vis_particle_num[None] = 0
    for I in flip_particles_pos:
        if(flip_particles_valid[I]==1):
            ind = ti.atomic_add(vis_particle_num[None],1)
            vis_particle_mapping[ind] = I

    #print(begin_res_x*vis_h_dx,"begin_res_x*vis_h_dx,")
    for I in vis_particle_pos:
        if(I<vis_particle_num[None]):
            vis_particle_pos[I] = ti.Vector(
                [
                    flip_particles_pos[vis_particle_mapping[I]][0]/vis_h_dx-begin_res_x/vis_h_dx_ratio,
                    flip_particles_pos[vis_particle_mapping[I]][1]/vis_h_dx,
                    flip_particles_pos[vis_particle_mapping[I]][2]/dx_z
                ]
            )

#####################################################################
#################         Moving         ############################
#####################################################################

@ti.kernel
def flip_partiles_moving_RK3(
    flip_particles_pos:ti.template(),
    flip_particles_v:ti.template(),
    flip_particles_valid:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    h:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_u_x:ti.template(),
    ibm_boundary_u_y:ti.template(),
    ibm_levelset:ti.template(),
    dx:float,
    dt:float
):
    for I in flip_particles_pos:
        if(flip_particles_valid[I]==1):
            if(flip_particles_pos[I][0]>0):
                if(in_fluid_domain(flip_particles_pos[I],h,dx)):            
                    psi_x0 = flip_particles_pos[I]
                    u1 = vel_3D(u_x,u_y,h,psi_x0,dx)
                    psi_x1 = psi_x0 + 1 * dt * u1  # advance 0.5 steps
                    u2 = vel_3D(u_x,u_y,h,psi_x1,dx)
                    psi_x2 = (psi_x1 + dt * u2) * 0.25 + 0.75 * psi_x0  # advance 0.5 again
                    u3 = vel_3D(u_x,u_y,h,psi_x2,dx)
                    flip_particles_pos[I] = 1. / 3 * psi_x0 + 2. / 3 * (psi_x2 + dt * u3)
                    flip_particles_v[I] = vel_3D(u_x,u_y,h,flip_particles_pos[I],dx)
                else:
                    for j in range(10):
                        flip_particles_pos[I]+= flip_particles_v[I]*dt*0.1
                        flip_particles_v[I]+= ti.Vector([0.0,0.0,-real_gravity*dt*0.1])
                #if(out_of_fluid_domain(flip_particles_pos[I])):
                #    flip_particles_valid[I] = 0
            
                if(in_solid_domain(flip_particles_pos[I],ibm_boundary_mask,dx)):
                    flip_particles_pos[I],flip_particles_v[I] = move_out_solid_domain(
                        flip_particles_pos[I],
                        flip_particles_v[I],
                        ibm_boundary_mask, 
                        ibm_boundary_u_x, 
                        ibm_boundary_u_y,
                        ibm_levelset,
                        dx
                    )

            flip_particles_pos[I]+=ti.Vector([boat_v,0.0, 0.0])*dt
            
@ti.func
def out_of_fluid_domain(pos):
    res = False
    if(pos[0]/dx>res_x):
        res = True
    return res
    
@ti.func
def in_fluid_domain(pos,h,dx):
    pos2D = ti.Vector([pos[0],pos[1]])
    h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
    res = False
    if(pos[2]<h_u_at_psi):
        res = True
    return res

@ti.func
def in_solid_domain(pos,ibm_boundary_mask,dx):
    pos2D = ti.Vector([pos[0],pos[1]])
    pos_grid = ti.cast(ti.floor(pos2D/dx), ti.int32)
    res = False
    if(ibm_boundary_mask[pos_grid]==1):
        res = True
    return res

@ti.func
def move_out_solid_domain(pos,v,ibm_boundary_mask, ibm_boundary_u_x, ibm_boundary_u_y,ibm_levelset_mask,dx):
    mdx = dx/10
    pos2D,boundary_v = ti.Vector([pos[0],pos[1]]), ti.Vector([0.0, 0.0])
    for i in range(100):
        boundary_v,tem1 = interp_u_MAC_grad_1(ibm_boundary_u_x, ibm_boundary_u_y, pos2D, dx)
        pos_grid = ti.cast(ti.floor(pos2D/dx), ti.int32)
        if(ibm_boundary_mask[pos_grid]==1):
            break        
        tem2, ls_grad, tem3 = interp_grad_2(ibm_levelset_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
        pos2D+=ls_grad*mdx
    return ti.Vector([pos2D[0],pos2D[1],pos[2]]), ti.Vector([boundary_v[0],boundary_v[1],v[2]])

@ti.func
def vel_3D(u_x,u_y,h,pos,dx):
    pos2D = ti.Vector([pos[0],pos[1]])
    u_at_psi, tem0, tem1, tem2  = interp_u_MAC_grad(u_x, u_y, pos2D, dx)
    div_u_at_psi = interp_MAC_divergence_u(u_x, u_y, pos2D, dx)
    h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
    real_h = pos[2]
    if(real_h>h_u_at_psi):
        real_h=h_u_at_psi
    return ti.Vector([u_at_psi[0],u_at_psi[1],-div_u_at_psi*real_h])