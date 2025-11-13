from hyperparameters import *
from taichi_utils import *
import numpy as np

###################################################################################
######################First Option: For Paraview Visualization ####################
###################################################################################
@ti.kernel
def set_vis_h(
    vis_h:ti.template(),
    surf_h:ti.template(),
    ibm_boundary_mask:ti.template(),
    h:ti.template(),
    dx:float
):
    for i,j,k in vis_h:
        z = k*dx_z+0.5
        pos2D = ti.Vector([i+0.5,j+0.5])*vis_h_dx + ti.Vector([dx*begin_res_x,0])
        h_at_pos, tem1, tem2 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
        bm_at_pos, tem1, tem2 = interp_grad_2(ibm_boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
        if(bm_at_pos>0.5 and False):
            vis_h[i,j,k]=0
            surf_h[i,j,k]=0
        else:
            if(abs(z-h_at_pos)<2*dx_z):
                surf_h[i,j,k]=1
            else:
                surf_h[i,j,k]=0
            if(z<h_at_pos):
                vis_h[i,j,k]=1
            else:
                vis_h[i,j,k]=0

###################################################################################
######################Second Option: For Houdini Visualization ####################
###################################################################################

def mesh_for_height_field(mesh_h_v,mesh_h_f,tem_mesh_h_v,tem_mesh_h_f,h,ibm_boundary_mask):

    set_mesh_h(mesh_h_v,mesh_h_f,h,ibm_boundary_mask,dx)
    v_array,f_array = mesh_h_v.to_numpy(),mesh_h_f.to_numpy()
    
    set_bottom_mesh_h(v_array.shape[0],mesh_h_v,mesh_h_f,dx)
    v_array=np.append(v_array,mesh_h_v.to_numpy(),axis=0)
    f_array=np.append(f_array,mesh_h_f.to_numpy(),axis=0)

    set_mesh_h_side(v_array.shape[0],0, 0,tem_mesh_h_v,tem_mesh_h_f,h,dx)
    v_array=np.append(v_array,tem_mesh_h_v.to_numpy(),axis=0)
    f_array=np.append(f_array,tem_mesh_h_f.to_numpy(),axis=0)

    set_mesh_h_side(v_array.shape[0],0, 1,tem_mesh_h_v,tem_mesh_h_f,h,dx)
    v_array=np.append(v_array,tem_mesh_h_v.to_numpy(),axis=0)
    f_array=np.append(f_array,tem_mesh_h_f.to_numpy(),axis=0)

    set_mesh_h_side(v_array.shape[0],1, 0,tem_mesh_h_v,tem_mesh_h_f,h,dx)
    v_array=np.append(v_array,tem_mesh_h_v.to_numpy(),axis=0)
    f_array=np.append(f_array,tem_mesh_h_f.to_numpy(),axis=0)

    set_mesh_h_side(v_array.shape[0],1, 1,tem_mesh_h_v,tem_mesh_h_f,h,dx)
    v_array=np.append(v_array,tem_mesh_h_v.to_numpy(),axis=0)
    f_array=np.append(f_array,tem_mesh_h_f.to_numpy(),axis=0)

    return v_array,f_array

@ti.kernel
def set_mesh_h(
    mesh_h_v:ti.template(),
    mesh_h_f:ti.template(),
    h:ti.template(),
    ibm_boundary_mask:ti.template(),
    dx:float
):
    for I in mesh_h_v:
        i, j = int(I%mesh_size_x), int(I/mesh_size_x)
        pos2D = ti.Vector([i+0.5,j+0.5])*mesh_dx + ti.Vector([dx*begin_res_x,dx*begin_res_y])
        h_at_pos, tem1, tem2 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
        bm_at_pos, tem1, tem2 = interp_grad_2(ibm_boundary_mask, pos2D, dx, BL_x=0.5, BL_y=0.5)
        mesh_h_v[I] = ti.Vector([pos2D[0],pos2D[1],boat_height]) if(bm_at_pos>0.5) else ti.Vector([pos2D[0],pos2D[1],h_at_pos])

    for I in mesh_h_f:
        v_ind = int(I/2)
        i, j = int(v_ind%(mesh_size_x-1)), int(v_ind/(mesh_size_x-1))
        v1, v2 = i+ j*mesh_size_x, i+1+ j*mesh_size_x
        v3, v4 = i+ (j+1)*mesh_size_x, i+1+ (j+1)*mesh_size_x
        mesh_h_f[I]= ti.Vector([v1,v2,v4]) if(I%2==0) else ti.Vector([v1,v4,v3])

@ti.kernel
def set_bottom_mesh_h(
    begin_ind:int,
    mesh_h_v:ti.template(),
    mesh_h_f:ti.template(),
    dx:float
):
    for I in mesh_h_v:
        i, j = int(I%mesh_size_x), int(I/mesh_size_x)
        pos2D = ti.Vector([i+0.5,j+0.5])*mesh_dx + ti.Vector([dx*begin_res_x,dx*begin_res_y])
        mesh_h_v[I] = ti.Vector([pos2D[0],pos2D[1],0.0])

    for I in mesh_h_f:
        v_ind = int(I/2)
        i, j = int(v_ind%(mesh_size_x-1)), int(v_ind/(mesh_size_x-1))
        v1, v2 = i+ j*mesh_size_x, i+1+ j*mesh_size_x
        v3, v4 = i+ (j+1)*mesh_size_x, i+1+ (j+1)*mesh_size_x
        mesh_h_f[I]=(
            change_order(ti.Vector([v1+begin_ind,v2+begin_ind,v4+begin_ind]))
            if(I%2==0) else change_order(ti.Vector([v1+begin_ind,v4+begin_ind,v3+begin_ind]))
        )

@ti.kernel
def set_mesh_h_side(
    begin_v_ind:int,
    axis:int, 
    axis_type:int,
    tem_mesh_h_v:ti.template(),
    tem_mesh_h_f:ti.template(),
    h:ti.template(),
    dx:float
):
    tem_mesh_h_v.fill(-1.0)
    tem_mesh_h_f.fill(-1)

    actual_mesh_size_side,another_ind = 0,0    
    if(axis == 0):
        actual_mesh_size_side = mesh_size_x    
        another_ind= 0 if(axis_type == 0) else int(mesh_size_y-1)
    else:
        actual_mesh_size_side = mesh_size_y
        another_ind= 0 if(axis_type == 0) else int(mesh_size_x-1)

    actual_mesh_v_size, actual_mesh_f_size = actual_mesh_size_side*height_len, (actual_mesh_size_side-1)*(height_len-1)*2
    for I in tem_mesh_h_v:
        if(I<actual_mesh_v_size):
            i,j = int(I%actual_mesh_size_side),int(I/actual_mesh_size_side)
            if(axis == 0):
                pos = ti.Vector([(i+0.5)*mesh_dx,(another_ind+0.5)*mesh_dx,(j+0.5)*height_dx]) + ti.Vector([dx*begin_res_x,dx*begin_res_y,0])
                pos2D =ti.Vector([pos[0],pos[1]])
                h_at_pos, tem1, tem2 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
                tem_mesh_h_v[I] = pos if(pos[2]<h_at_pos) else ti.Vector([-1,-1,-1])

            else:
                pos = ti.Vector([(another_ind+0.5)*mesh_dx,(i+0.5)*mesh_dx,(j+0.5)*height_dx]) + ti.Vector([dx*begin_res_x,dx*begin_res_y,0])
                pos2D =ti.Vector([pos[0],pos[1]])
                h_at_pos, tem1, tem2 = interp_grad_2(h, pos2D, dx, BL_x=0.5, BL_y=0.5)
                tem_mesh_h_v[I] = pos if(pos[2]<h_at_pos) else ti.Vector([-1,-1,-1])

    for I in tem_mesh_h_f:
        if(I<actual_mesh_f_size):
            v_ind = int(I/2)
            i, j = int(v_ind%(actual_mesh_size_side-1)), int(v_ind/(actual_mesh_size_side-1))
            v1, v2 = i+ j*actual_mesh_size_side, i+1+ j*actual_mesh_size_side
            v3, v4 = i+ (j+1)*actual_mesh_size_side, i+1+ (j+1)*actual_mesh_size_side
            if(I%2==0 and valid_vertex(tem_mesh_h_v[v1]) and valid_vertex(tem_mesh_h_v[v2]) and valid_vertex(tem_mesh_h_v[v4])):
                tem_mesh_h_f[I]= ti.Vector([v1+begin_v_ind,v2+begin_v_ind,v4+begin_v_ind])
            elif(I%2==1 and valid_vertex(tem_mesh_h_v[v1]) and valid_vertex(tem_mesh_h_v[v4]) and valid_vertex(tem_mesh_h_v[v3])):
                tem_mesh_h_f[I]= ti.Vector([v1+begin_v_ind,v4+begin_v_ind,v3+begin_v_ind])
            if((axis == 0 and axis_type==1) or (axis == 1 and axis_type==0)):
                tem_mesh_h_f[I]=change_order(tem_mesh_h_f[I])

@ti.func
def change_order(v):
    return ti.Vector([v[0],v[2],v[1]])

@ti.func
def valid_vertex(v):
    res= False
    if(v[0]>=0 and v[1]>=0 and v[2]>=0):
        res = True
    return res

