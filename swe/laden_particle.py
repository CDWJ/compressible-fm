from hyperparameters import *
from taichi_utils import *
from passive_particles import *
import numpy as np

@ti.kernel
def advect_laden_particles(
        laden_particles_pos:ti.template(),
        laden_particles_vel:ti.template(),
        u_x:ti.template(),
        u_y:ti.template(),
        h:ti.template(),
        dx:float,
        dt:float,
        t:float
):
    for I in laden_particles_pos:
        if(t>1.5):
            u_w = vel_3D(u_x,u_y,h,laden_particles_pos[I],dx)
            a = (
                ti.Vector([0.0,0.0,-laden_gravity]) + laden_drag*(u_w-laden_particles_vel[I])
            )/(1+laden_drag*dt)
            laden_particles_vel[I]+=a*dt
            laden_particles_pos[I]+= laden_particles_vel[I]*dt
            if(laden_particles_pos[I][2]<0):
                laden_particles_pos[I][2]=2

@ti.kernel
def init_laden_particles(
    laden_particles_pos:ti.template(),
    laden_particles_vel:ti.template(),
):
    laden_particles_vel.fill(0.0)
    for I in laden_particles_pos:
        laden_particles_pos[I]=ti.Vector(
            [
                ti.random()*res_x*dx,
                ti.random()*res_y*dx,
                ti.random()*init_h*0.1
            ]
        )