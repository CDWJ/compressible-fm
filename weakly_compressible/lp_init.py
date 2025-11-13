import taichi as ti
import numpy as np
import random
from taichi_utils import *

#########################################################################################
#########################################################################################
# For Dynamic Management of particles
@ti.kernel
def ink_bottle_src_set(
    boundary_mask:ti.template(),
    ink_source:ti.template(),
    dx:float,
    domain_range_y:float
):
    for i,j,k in boundary_mask:
        pos = ti.Vector([i+0.5,j+0.5,k+0.5])*dx
        pos2D = ti.Vector([i+0.5,k+0.5])*dx
        c = ti.Vector([domain_range_y/2*0.55,domain_range_y/2*0.55])
        r = domain_range_y/2*0.07
        if(pos[1]<0.55*domain_range_y and pos[1]>0.45*domain_range_y and (pos2D-c).norm()<r):
            if(boundary_mask[i,j,k]<=0 and boundary_mask[i,j-1,k]>=1):
                ink_source[i,j,k]=1
            else:
                ink_source[i,j,k]=0

@ti.kernel
def ink_propeller_src_set(
    boundary_mask:ti.template(),
    ink_source:ti.template(),
    dx:float,
    domain_range_y:float
):
    for i,j,k in boundary_mask:
        pos = ti.Vector([i+0.5,j+0.5,k+0.5])*dx
        pos2D = ti.Vector([j+0.5,k+0.5])*dx
        c_2D = ti.Vector([0.5,0.5])
        bm = interp_2(boundary_mask,pos,dx)
        if(pos[0]<1.49 and 0<bm<0.1 and 0.3>(pos2D-c_2D).norm()>0.1):
        #if(pos[0]>1.55 and 0<bm<0.1 and (pos2D-c_2D).norm()>0.1):
            ink_source[i,j,k] = 1
        else:
            ink_source[i,j,k] = 0
            

@ti.kernel
def manage_grid_src_propeller(
    smoke:ti.template(),
    smoke_init:ti.template(),
    ink_source:ti.template()
):
    for I in ti.grouped(ink_source):
        if(ink_source[I]==1):
            smoke[I] = ti.Vector([0,0,0,1.0])
            smoke_init[I] = ti.Vector([0,0,0,1.0])

@ti.kernel
def manage_src_laden_particle(
    u_x:ti.template(), 
    u_y:ti.template(), 
    u_z:ti.template(),
    ink_source:ti.template(),
    ink_grid_num:ti.template(),
    laden_particles_vel:ti.template(), 
    laden_particles_pos:ti.template(),
    laden_particles_vel_new:ti.template(), 
    laden_particles_pos_new:ti.template(),
    delete_flag:ti.template(),
    laden_particle_num:ti.template(),
    old_laden_particle_num:ti.template(),
    min_num:int,
    dx:float,
    laden_particles_max_num:int,
    domain_range_y:float
):
    ink_grid_num.fill(0)
    delete_flag.fill(0)
    for I in laden_particles_pos:
        if(I<laden_particle_num[None]):
            idx = ti.floor(laden_particles_pos[I]/dx,int)
            ink_grid_num[idx]+=1
    for i,j,k in ink_source:
        if(ink_source[i,j,k]==1 and ink_grid_num[i,j,k]<min_num):
            n = min_num-ink_grid_num[i,j,k]
            ind = ti.atomic_add(laden_particle_num[None],n)
            for I in range(ind,ind+n):
                pos = ti.Vector([i+ti.random(),j+ti.random(),k+ti.random()])*dx
                if(I<laden_particles_max_num):
                    laden_particles_pos[I] = pos
                    laden_particles_vel[I],_ = interp_u_MAC_grad(u_x, u_y, u_z, pos, dx)
    if(laden_particle_num[None]>laden_particles_max_num):
        laden_particle_num[None]=laden_particles_max_num

    for I in laden_particles_pos:
        if(I<laden_particle_num[None]):
            if(laden_particles_pos[I][1]<domain_range_y*0.1):
                delete_flag[I] = 1

    old_laden_particle_num[None] = laden_particle_num[None]
    laden_particle_num[None] = 0
    for I in laden_particles_pos:
        if(I<old_laden_particle_num[None] and delete_flag[I]==0):
            ind = ti.atomic_add(laden_particle_num[None],1)
            laden_particles_pos_new[ind] = laden_particles_pos[I]
            laden_particles_vel_new[ind] = laden_particles_vel[I]

    for I in laden_particles_pos:
        if(I<laden_particle_num[None]):
            laden_particles_pos[I] = laden_particles_pos_new[I]
            laden_particles_vel[I] = laden_particles_vel_new[I]

#########################################################################################
#########################################################################################




def ink_drop_particle_case0(laden_particle_pos,num):
    particle_num=0
    laden_particle_pos_np=np.zeros(shape=(num,3), dtype=float)
    random.seed(0)
    while(True):
        r=3.0
        center=64.0
        domain_range_y=256.0
        x,y,z=random.random()*r*2+center-r,random.random()*r*2+0.05*domain_range_y,random.random()*r*2+center-r
        if((x-center)**2+(y-0.05*domain_range_y-r)**2+(z-center)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num):
            break
    laden_particle_pos.from_numpy(laden_particle_pos_np)  

@ti.kernel
def ink_drop_vel_case0(laden_particle_vel:ti.template(),laden_particle_pos:ti.template(),laden_particle_num:int):
    for I in ti.grouped(laden_particle_vel):
        if(I[0]<laden_particle_num):
            laden_particle_vel[I]=ti.Vector([0.0,1.0,0.0])

@ti.kernel
def ink_drop_vel_grid_case0(u:ti.template(),X:ti.template()):
    r=3.0
    center=64.0
    domain_range_y=256.0
    for I in ti.grouped(X):
        x,y,z=X[I][0],X[I][1],X[I][2]
        if((x-center)**2+(y-0.05*domain_range_y-r)**2+(z-center)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])

@ti.kernel
def init_ref_radius(
    laden_particle_ref_radius:ti.template(),
    laden_particle_num:int,
    compute_laden_radius:float
):
    for I in ti.grouped(laden_particle_ref_radius):
        if(I[0]<laden_particle_num):
            laden_particle_ref_radius[I]=compute_laden_radius

def ink_init_case0(
    laden_particles_pos:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particle_num:ti.template(),
    u:ti.template(),
    X:ti.template(),
    drag_x:ti.template(),
    drag_y:ti.template(),
    drag_z:ti.template(),
    laden_particle_ref_radius:ti.template(),
    compute_laden_radius:float
):    
    ink_drop_particle_case0(laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_case0(laden_particles_vel,laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_grid_case0(u,X)
    drag_x.fill(0.0)
    drag_y.fill(0.0)
    drag_z.fill(0.0)
    init_ref_radius(laden_particle_ref_radius,laden_particle_num[None],compute_laden_radius)


def ink_drop_particle_case16(laden_particle_pos,num, DropR, L, DomainRange):
    particle_num=0
    laden_particle_pos_np=np.zeros(shape=(num,3), dtype=float)
    
    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r
        if((x-center)**2+(y-0.05*DomainRange-r)**2+(z-center)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r-L*2/3.0
        if((x-center)**2+(y-0.05*DomainRange-r)**2+(z-center+L*2/3.0)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*2/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r+L*2/3.0
        if((x-center)**2+(y-0.05*DomainRange-r)**2+(z-center-L*2/3.0)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*3/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r-L*2/3.0,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r
        if((x-center+L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*4/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r-L*2/3.0,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r-L*2/3.0
        if((x-center+L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center+L*2/3.0)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*5/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r-L*2/3.0,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r+L*2/3.0
        if((x-center+L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center-L*2/3.0)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*6/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r+L*2/3.0,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r
        if((x-center-L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*7/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r+L*2/3.0,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r-L*2/3.0
        if((x-center-L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center+L*2/3.0)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num*8/9.0):
            break

    while(True):
        r=DropR
        center=L
        x,y,z=random.random()*r*2+center-r+L*2/3.0,random.random()*r*2+0.05*DomainRange,random.random()*r*2+center-r+L*2/3.0
        if((x-center-L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center-L*2/3.0)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num):
            break
    laden_particle_pos.from_numpy(laden_particle_pos_np)  

@ti.kernel
def ink_drop_vel_case16(laden_particle_vel:ti.template(),laden_particle_pos:ti.template(),laden_particle_num:int, DropR:float, L:float, DomainRange:float):
    for I in ti.grouped(laden_particle_vel):
        if(I[0]<laden_particle_num):
            laden_particle_vel[I]=ti.Vector([0.0,1.0,0.0])

@ti.kernel
def ink_drop_vel_grid_case16(u:ti.template(),X:ti.template(), DropR:float, L:float, DomainRange:float):
    r=DropR
    
    for I in ti.grouped(X):
        x,y,z=X[I][0],X[I][1],X[I][2]
        
        center=L
        if((x-center)**2+(y-0.05*DomainRange-r)**2+(z-center)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center)**2+(y-0.05*DomainRange-r)**2+(z-center+L*2/3.0)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center)**2+(y-0.05*DomainRange-r)**2+(z-center-L*2/3.0)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center-L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center-L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center+L*2/3.0)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center-L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center-L*2/3.0)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])

        center=L
        if((x-center+L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center+L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center+L*2/3.0)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])
        
        center=L
        if((x-center+L*2/3.0)**2+(y-0.05*DomainRange-r)**2+(z-center-L*2/3.0)**2<r**2):
            u[I]=ti.Vector([0.0,1.0,0.0])

def ink_init_case_porus(
    laden_particles_pos:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particle_num:ti.template(),
    u:ti.template(),
    X:ti.template(),
    drag_x:ti.template(),
    drag_y:ti.template(),
    drag_z:ti.template(),
    laden_particle_ref_radius:ti.template(),
    compute_laden_radius:float
):    
    DropR = 3.0
    L = 64
    DomainRange = 256.0
    ink_drop_particle_case16(laden_particles_pos,laden_particle_num[None],DropR, L, DomainRange)
    ink_drop_vel_case16(laden_particles_vel,laden_particles_pos,laden_particle_num[None],DropR, L, DomainRange)
    ink_drop_vel_grid_case16(u,X,DropR, L, DomainRange)
    drag_x.fill(0.0)
    drag_y.fill(0.0)
    drag_z.fill(0.0)
    init_ref_radius(laden_particle_ref_radius,laden_particle_num[None],compute_laden_radius)

def ink_drop_particle_case1(laden_particle_pos,num):
    particle_num=0
    laden_particle_pos_np=np.zeros(shape=(num,3), dtype=float)
    random.seed(0)
    while(True):
        r=8.0
        domain_range_x=128.0
        domain_range_y=256.0
        domain_range_z=128.0
        center_x = domain_range_x * 0.3
        center_z = domain_range_z * 0.3
        x,y,z=random.random()*r*2+center_x-r,random.random()*r*2+0.05*domain_range_y,random.random()*r*2+center_z-r
        if((x-center_x)**2+(y-0.05*domain_range_y-r)**2+(z-center_z)**2<r**2):
            laden_particle_pos_np[particle_num,:]=np.array([x,y,z])
            particle_num+=1
        if(particle_num>=num):
            break
    laden_particle_pos.from_numpy(laden_particle_pos_np)  

@ti.kernel
def ink_drop_vel_case1(laden_particle_vel:ti.template(),laden_particle_pos:ti.template(),laden_particle_num:int):
    for I in ti.grouped(laden_particle_vel):
        if(I[0]<laden_particle_num):
            laden_particle_vel[I]=ti.Vector([0.0,0.0,0.0])

@ti.kernel
def ink_drop_vel_grid_case1(u:ti.template(),X:ti.template()):
    r=8.0
    domain_range_x=128.0
    domain_range_y=256.0
    domain_range_z=128.0
    center_x = domain_range_x * 0.3
    center_z = domain_range_z * 0.3
    for I in ti.grouped(X):
        x,y,z=X[I][0],X[I][1],X[I][2]
        if((x-center_x)**2+(y-0.05*domain_range_y-r)**2+(z-center_z)**2<r**2):
            u[I]=ti.Vector([0.3,0.7,0.4])

def ink_init_case1(
    laden_particles_pos:ti.template(),
    laden_particles_vel:ti.template(),
    laden_particle_num:ti.template(),
    u:ti.template(),
    X:ti.template(),
    drag_x:ti.template(),
    drag_y:ti.template(),
    drag_z:ti.template(),
    laden_particle_ref_radius:ti.template(),
    compute_laden_radius:float
):    
    ink_drop_particle_case1(laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_case1(laden_particles_vel,laden_particles_pos,laden_particle_num[None])
    ink_drop_vel_grid_case1(u,X)
    drag_x.fill(0.0)
    drag_y.fill(0.0)
    drag_z.fill(0.0)
    init_ref_radius(laden_particle_ref_radius,laden_particle_num[None],compute_laden_radius)