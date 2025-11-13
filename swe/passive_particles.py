from hyperparameters import *
from taichi_utils import *
import numpy as np
#####################################################################
#################     Reseed     ############################
#####################################################################
@ti.kernel
def set_vis_h(
    vis_h:ti.template(),
    vis_u:ti.template(),
    vis_u_3D:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    h:ti.template(),
    eta:ti.template(),
    flip_paticles_type:ti.template(),
    flip_paticles_pos:ti.template(),
    flip_particles_num:ti.template(),
    dx:float
):
    for i,j,k in vis_h:
        z = k*vol_dx_z+vol_z_base
        pos2D = ti.Vector([i+0.5,j+0.5])*vol_dx_xy
        h_at_pos, tem1, tem2 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
        eta_at_pos, tem1, tem2 = interp_grad_2(eta, pos2D, dx, BL_x=0.5, BL_y=0.5)
        u_at_pos, tem3, tem4, tem5 = interp_u_MAC_grad(u_x, u_y, pos2D, dx)
        vis_u[i,j,k] = u_at_pos 
        pos3D = ti.Vector([pos2D[0],pos2D[1],z])    
        vis_u_3D[i,j,k] = vel_3D(u_x,u_y,h,pos3D,dx)
        if(z<h_at_pos+eta_at_pos):
            vis_h[i,j,k]=1
        else:
            vis_h[i,j,k]=0

    ii,jj,kk = vis_h.shape
    for I in flip_paticles_pos:
        if(I<flip_particles_num[None]):
            pos = flip_paticles_pos[I]
            pos2D= ti.Vector([pos[0],pos[1]])
            ind2D = ti.floor(pos2D/vol_dx_xy,int)
            ind_z = ti.floor((pos[2]-vol_z_base)/vol_dx_z,int)
            ind = ti.Vector([ind2D[0],ind2D[1],ind_z])
            if(ind[0]>=0 and ind[1]>=0 and ind[0]<ii and ind[1]<jj and ind[2]>=0 and ind[2]<kk):
                if(flip_paticles_type[I] == 2):
                    vis_h[ind]  = 2
                elif(flip_paticles_type[I] == 3):
                    vis_h[ind]  = 3


@ti.kernel
def curl_3D(vf: ti.template(), cf: ti.template(), dx_xy: float, dx_z: float):
    inv_dist_xy = 1./(2*dx_xy)
    inv_dist_z = 1./(2*dx_z)
    for i, j, k in cf:
        vr = sample_3D(vf, i+1, j, k)
        vl = sample_3D(vf, i-1, j, k)
        vt = sample_3D(vf, i, j+1, k)
        vb = sample_3D(vf, i, j-1, k)
        vc = sample_3D(vf, i, j, k+1)
        va = sample_3D(vf, i, j, k-1)

        d_vx_dz = inv_dist_z * (vc.x - va.x)
        d_vx_dy = inv_dist_xy * (vt.x - vb.x)
        
        d_vy_dx = inv_dist_xy * (vr.y - vl.y)
        d_vy_dz = inv_dist_z * (vc.y - va.y)

        d_vz_dx = inv_dist_xy * (vr.z - vl.z)
        d_vz_dy = inv_dist_xy * (vt.z - vb.z)

        cf[i,j,k][0] = d_vz_dy - d_vy_dz
        cf[i,j,k][1] = d_vx_dz - d_vz_dx
        cf[i,j,k][2] = d_vy_dx - d_vx_dy

@ti.kernel
def init_flip_particles(
    ibm_boundary_mask:ti.template(),
    flip_particles_pos:ti.template(),
    flip_particles_vel:ti.template(),
    flip_particles_life:ti.template(),
    flip_particles_type:ti.template(),    
    flip_particles_num:ti.template(),
    flip_particles_h:ti.template(),
    h:ti.template(),
    dx:float
):
    flip_particles_vel.fill(0.0)
    flip_particles_life.fill(0.0)
    flip_particles_num[None] = pp_res_x*pp_res_y*pp_res_z*npc
    print(pp_res_z*pp_dx_z,"pp_res_z*pp_dx_z")
    for I in flip_particles_pos:
        grid_ind = I//npc
        #grid_ind_in = i%npc
        grid_ind_z = grid_ind//(pp_res_x*pp_res_y)
        grid_ind_xy = grid_ind%(pp_res_x*pp_res_y)
        grid_ind_y = grid_ind_xy//pp_res_x
        grid_ind_x = grid_ind_xy%pp_res_x
        flip_particles_pos[I] = ti.Vector(
            [
               (grid_ind_x+ti.random())*pp_dx_xy,
               (grid_ind_y+ti.random())*pp_dx_xy,
                (grid_ind_z+ti.random())*pp_dx_z+ init_h- pp_res_z*pp_dx_z,
            ]
        )
        flip_particles_type[I] = 1
        pos2D = ti.Vector([flip_particles_pos[I][0],flip_particles_pos[I][1]])
        h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
        flip_particles_h[I] = h_u_at_psi
        bm_at_psi, tem4, tem5 = interp_grad_2(ibm_boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
        if(bm_at_psi>0.0):
            flip_particles_type[I] = 0


@ti.kernel
def advect_passive_particles(
    flip_particles_pos:ti.template(),
    flip_particles_vel:ti.template(),
    flip_particles_life:ti.template(),
    flip_particles_type:ti.template(),
    flip_particles_num:ti.template(),
    ibm_boundary_mask:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    flip_particles_h:ti.template(),
    h:ti.template(),
    dt:float
):
    for I in flip_particles_pos:
        if(I<flip_particles_num[None] and flip_particles_type[I]!=0):
            if(flip_particles_type[I]==3):
                #if(abs(flip_particles_vel[I][2])>0.1):
                #    sign = flip_particles_vel[I][2]/abs(flip_particles_vel[I][2])
                #    flip_particles_vel[I][2] = 0.1*sign
                if(flip_particles_vel[I][2]>vertical_velocity_limit):
                    flip_particles_vel[I][2]=vertical_velocity_limit
                v0 = flip_particles_vel[I]
                drag_coef = 0.8
                drag = -v0.norm()*drag_coef*v0
                flip_particles_vel[I]+=(ti.Vector([0.0,0.0,-flip_gravity_coef*real_gravity])+drag)*dt
                flip_particles_pos[I]+=0.5*(v0+flip_particles_vel[I])*dt
            # elif (flip_particles_type[I]==2):
            #     v0 = flip_particles_vel[I]
            #     v = vel_3D(u_x,u_y,h,flip_particles_pos[I],dx)
            #     v0[2] = v[2]
            #     pos2D = ti.Vector([flip_particles_pos[I][0],flip_particles_pos[I][1]])
            #     h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
            #     flip_particles_h[I] = h_u_at_psi
            #     bm_at_psi, tem4, tem5 = interp_grad_2(ibm_boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
            #     flip_particles_life[I]-=dt
            #     drag_coef = 0.8
            #     flip_particles_vel[I] = 0.5*(v0+v)
            #     if(bm_at_psi>0.0):
            #         flip_particles_vel[I][2] = 0.0
            #     flip_particles_pos[I]+=flip_particles_vel[I]*dt
            else:
                #flip_particles_vel[I] = vel_3D2(u_x,u_y,h,flip_particles_h[I],flip_particles_pos[I],dx,dt)
                flip_particles_vel[I] = vel_3D(u_x,u_y,h,flip_particles_pos[I],dx)
                pos2D = ti.Vector([flip_particles_pos[I][0],flip_particles_pos[I][1]])
                h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
                flip_particles_h[I] = h_u_at_psi
                bm_at_psi, tem4, tem5 = interp_grad_2(ibm_boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
                if(bm_at_psi>0.0):
                    flip_particles_vel[I][2] = 0.0
                if(flip_particles_type[I]==2):
                    flip_particles_life[I]-=dt
                    flip_particles_pos[I]+=flip_particles_vel[I]*dt
                else:
                    flip_particles_pos[I]+=flip_particles_vel[I]*dt

@ti.kernel
def reseed_passive_particles(
    flip_particles_pos:ti.template(),
    flip_particles_vel:ti.template(),
    flip_particles_life:ti.template(),
    flip_particles_type:ti.template(), # 1 fluid, 2 foam, 3 spray, 0 invalid
    flip_particles_h:ti.template(),

    flip_particles_pos_new:ti.template(),
    flip_particles_vel_new:ti.template(),
    flip_particles_life_new:ti.template(),
    flip_particles_type_new:ti.template(), # 1 fluid, 2 foam, 3 spray, 0 invalid
    flip_particles_h_new:ti.template(),

    flip_particles_num:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    boundary_mask: ti.template(),
    grid_particles_num:ti.template(),
    grid_particles_base:ti.template(),
    flip_particle_delete_flag:ti.template(),
    
    h:ti.template()
):
    for i,j in grid_particles_base:
        pos2D = ti.Vector([i+0.5,j+0.5])*pp_dx_xy
        h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
        grid_particles_base[i,j] = h_u_at_psi-pp_dx_z*pp_res_z
    # first convert type of particles and delete particles
    flip_particle_delete_flag.fill(0)
    grid_particles_num.fill(0)
    for I in flip_particles_pos:
        if(I < flip_particles_num[None] and flip_particles_type[I]!=0):
            pos2D = ti.Vector([flip_particles_pos[I][0],flip_particles_pos[I][1]])
            h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
            bm_at_psi, tem4, tem5 = interp_grad_2(boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
            ind = ti.floor(pos2D/pp_dx_xy, int)
            if(flip_particles_type[I]==3):
                if(flip_particles_pos[I][2]<h_u_at_psi):
                    prob = 0.5
                    if(ti.random()>prob):
                        flip_particles_type[I] = 1
                    else:
                        flip_particles_type[I] = 2
                        flip_particles_life[I] = (1+ti.random())* life_time_init
                        flip_particles_pos[I][2] = h_u_at_psi+foam_init_height - (ti.random()) * dx
                        flip_particles_pos[I][0] += (ti.random() - 0.5)* 4 * dx
                        flip_particles_pos[I][1] += (ti.random() - 0.5) * 4 * dx
                else:
                    if(
                        pos2D[0]>(pp_res_x-boarder_ratio)*pp_dx_xy or pos2D[0]< boarder_ratio*pp_dx_xy or 
                       pos2D[1]>(pp_res_y-boarder_ratio)*pp_dx_xy or pos2D[1]< boarder_ratio*pp_dx_xy or
                        bm_at_psi>0.5
                    ):
                        flip_particle_delete_flag[I] = 1
            else:
                if(ind[0]< 0  or ind[0]>= pp_res_x or  ind[1]< 0  or ind[1]>= pp_res_y or bm_at_psi>0.1):
                        flip_particle_delete_flag[I] = 1
                else:
                    if(flip_particles_pos[I][2]>h_u_at_psi+spray_bandwidth):
                        flip_particles_type[I] = 3
                        old_pos = ti.Vector([flip_particles_pos[I][0], flip_particles_pos[I][1], flip_particles_pos[I][2]])
                        flip_particles_pos[I][0] += (ti.random() - 0.5)* 3 * dx
                        flip_particles_pos[I][1] += (ti.random() - 0.5) * 3 * dx
                        # flip_particles_pos[I][2] += ti.random() * dx * 0.1
                        bm_at_psi, tem4, tem5 = interp_grad_2(boundary_mask, ti.Vector([flip_particles_pos[I][0], flip_particles_pos[I][1]]), dx, BL_x=0.5, BL_y=0.5)
                        if bm_at_psi > 0.5:
                            flip_particles_pos[I] = old_pos

                    else:
                        #if(flip_particles_pos[I][2]<h_u_at_psi):
                        #    flip_particles_type[I] = 1
                        if(flip_particles_life[I]<0):
                            flip_particles_type[I] = 1
                            flip_particles_life[I] = 0
                        if(flip_particles_pos[I][2]<grid_particles_base[ind] or bm_at_psi>0.5):
                            flip_particle_delete_flag[I] = 1
                        else:
                            ind_z =  ti.floor((flip_particles_pos[I][2]-grid_particles_base[ind])/pp_dx_z,int)
                            if(ind_z<pp_res_z):
                                grid_ind = ti.atomic_add(grid_particles_num[ind[0],ind[1],ind_z], 1)
                                if(grid_ind>=npc+npc_tl):
                                    flip_particle_delete_flag[I] = 1
    old_flip_particles_num = flip_particles_num[None]
    flip_particles_num[None] = 0
    flip_particles_pos_new.fill(0.0)
    flip_particles_vel_new.fill(0.0)
    flip_particles_life_new.fill(0.0)
    flip_particles_type_new.fill(0.0)
    flip_particles_h_new.fill(0.0)
    for I in flip_particles_pos:
        if(I < old_flip_particles_num and flip_particles_type[I]!=0 and flip_particle_delete_flag[I]==0):
            ind = ti.atomic_add(flip_particles_num[None], 1)
            flip_particles_pos_new[ind] = flip_particles_pos[I]
            flip_particles_vel_new[ind] = flip_particles_vel[I]
            flip_particles_life_new[ind] = flip_particles_life[I]
            flip_particles_type_new[ind] = flip_particles_type[I]
            flip_particles_h_new[ind] = flip_particles_h[I]


    for i,j,k in grid_particles_num:
        pos2D = ti.Vector([i+0.5,j+0.5])*pp_dx_xy
        bm_at_psi, tem4, tem5 = interp_grad_2(boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
        if(grid_particles_num[i,j,k]<npc-npc_tl and bm_at_psi<0.1):
            n = npc-npc_tl - grid_particles_num[i,j,k]
            ind = ti.atomic_add(flip_particles_num[None], n)
            for I in range(ind,ind+n):
                flip_particles_pos_new[I] = ti.Vector([(i+ti.random())*pp_dx_xy,(j+ti.random())*pp_dx_xy,(k+ti.random())*pp_dx_z+grid_particles_base[i,j]])
                flip_particles_vel_new[I] = vel_3D(u_x,u_y,h,flip_particles_pos_new[I],dx)
                flip_particles_life_new[I] = 0
                flip_particles_type_new[I] = 1
                pos2D_tem = ti.Vector([flip_particles_pos_new[I][0],flip_particles_pos_new[I][1]])
                h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D_tem, dx, BL_x=0.5, BL_y=0.5)
                flip_particles_h_new[I] = h_u_at_psi

    # for i,j,k in grid_particles_num:
    #     pos2D = ti.Vector([i+0.5,j+0.5])*pp_dx_xy
    #     bm_at_psi, tem4, tem5 = interp_grad_2(boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
    #     if(grid_particles_num[i,j,k]<npc-npc_tl and bm_at_psi<0.1):
    #         n = npc-npc_tl - grid_particles_num[i,j,k]
    #         ind = ti.atomic_add(flip_particles_num[None], n)
    #         for I in range(ind,ind+n):
    #             pos = ti.Vector([0.0,0.0,0.0])
    #             while(True):
    #                 pos = ti.Vector([(i+ti.random())*pp_dx_xy,(j+ti.random())*pp_dx_xy,(k+ti.random())*pp_dx_z+grid_particles_base[i,j]])
    #                 pos2D_new = ti.Vector([pos[0],pos[1]])
    #                 h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D_new, dx, BL_x=0.5, BL_y=0.5)
    #                 if( pos[2]< h_u_at_psi):
    #                     break
    #             flip_particles_pos_new[I] = pos
    #             flip_particles_vel_new[I] = vel_3D(u_x,u_y,h,flip_particles_pos_new[I],dx)
    #             flip_particles_life_new[I] = 0
    #             flip_particles_type_new[I] = 1
    #             pos2D_tem = ti.Vector([flip_particles_pos_new[I][0],flip_particles_pos_new[I][1]])
    #             h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D_tem, dx, BL_x=0.5, BL_y=0.5)
    #             flip_particles_h_new[I] = h_u_at_psi

    flip_particles_pos.fill(0.0)
    flip_particles_vel.fill(0.0)
    flip_particles_life.fill(0.0)
    flip_particles_type.fill(0.0)  
    flip_particles_h.fill(0.0) 
    for I in flip_particles_pos_new:
        if(I < flip_particles_num[None]):
            flip_particles_pos[I] = flip_particles_pos_new[I]
            flip_particles_vel[I] = flip_particles_vel_new[I]
            flip_particles_life[I] = flip_particles_life_new[I]
            flip_particles_type[I] = flip_particles_type_new[I]  
            flip_particles_h[I] = flip_particles_h_new[I]        
    print("num",flip_particles_num[None])
            
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

@ti.func
def vel_3D2(u_x,u_y,h,p_h,pos,dx,dt):
    pos2D = ti.Vector([pos[0],pos[1]])
    u_at_psi, tem0, tem1, tem2  = interp_u_MAC_grad(u_x, u_y, pos2D, dx)
    h_u_at_psi, tem4, tem5 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
    real_h = pos[2]
    if(real_h>h_u_at_psi):
        real_h=h_u_at_psi
    return ti.Vector([u_at_psi[0],u_at_psi[1],(h_u_at_psi- p_h)/dt*real_h/h_u_at_psi])

                    



                        

                    
