#
from hyperparameters import *
from taichi_utils import *
from io_utils import *
import sys
import shutil
import time
#

@ti.kernel
def up_wind_update_u(
    new_u_x:ti.template(),
    new_u_y:ti.template(),
    h:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    du_x:ti.template(),
    du_y:ti.template(),
    q_x:ti.template(),
    q_y:ti.template(),
    dx:float,
    dt:float
):
    # first calculate q_x, q_y
    for i,j in q_x:
        q_x[i,j] =  up_wind_h(i,j, h,u_x, False)*u_x[i,j]
    for i,j in q_y:
        q_y[i,j] =  up_wind_h(i,j, h,u_y, True) *u_y[i,j]

    for i,j in u_x:
        h0 = up_wind_h(i,j, h,u_x, False)
        q_x_positive = (sample(q_x,i,j)+sample(q_x,i-1,j))/2
        q_x_negative = (sample(q_x,i,j)+sample(q_x,i+1,j))/2
        delta1,delta2 =0.0,0.0
        if(q_x_positive+q_x_negative>=0):
            delta1 = q_x_positive/h0*(sample(u_x,i,j)-sample(u_x,i-1,j))/dx
        else:
            delta1 = q_x_negative/h0*(sample(u_x,i+1,j)-sample(u_x,i,j))/dx
        q_y_flow = (
            sample(q_y,i-1,j)+sample(q_y,i,j)+
            sample(q_y,i-1,j+1)+sample(q_y,i,j+1)
        )/4
        if(q_y_flow>0):
            if(q_x_positive+q_x_negative>=0):
                delta2 = sample(q_y,i-1,j)/h0 * (sample(u_x,i,j)-sample(u_x,i,j-1))/dx
            else:
                delta2 = sample(q_y,i,j)/h0 * (sample(u_x,i,j)-sample(u_x,i,j-1))/dx
        else:
            if(q_x_positive+q_x_negative>=0):
                delta2 = sample(q_y,i-1,j+1)/h0 * (sample(u_x,i,j+1)-sample(u_x,i,j))/dx
            else:
                delta2 = sample(q_y,i,j+1)/h0 * (sample(u_x,i,j+1)-sample(u_x,i,j))/dx
        delta = delta1+delta2+real_gravity*(sample(h,i,j)-sample(h,i-1,j))/dx
        new_u_x[i,j] = u_x[i,j]-(3.0/2*delta-1.0/2*du_x[i,j])*dt
        du_x[i,j] =delta 

    for i,j in u_y:
        h0 = up_wind_h(i,j, h,u_y, True)
        q_y_positive = (sample(q_y,i,j)+sample(q_y,i,j-1))/2
        q_y_negative = (sample(q_y,i,j)+sample(q_y,i,j+1))/2
        delta1,delta2 =0.0,0.0
        if(q_y_positive+q_y_negative>=0):
            delta1 = q_y_positive/h0*(sample(u_y,i,j)-sample(u_y,i,j-1))/dx
        else:
            delta1 = q_y_negative/h0*(sample(u_y,i,j+1)-sample(u_y,i,j))/dx
        q_x_flow = (
            sample(q_x,i,j-1)+sample(q_x,i,j)+
            sample(q_x,i+1,j-1)+sample(q_x,i+1,j)
        )/4
        if(q_x_flow>0):
            if(q_y_positive+q_y_negative>=0):
                delta2 = sample(q_x,i,j-1)/h0 * (sample(u_y,i,j)-sample(u_y,i-1,j))/dx
            else:
                delta2 = sample(q_x,i,j)/h0 * (sample(u_y,i,j)-sample(u_y,i-1,j))/dx
        else:
            if(q_y_positive+q_y_negative>=0):
                delta2 = sample(q_x,i+1,j-1)/h0 * (sample(u_y,i+1,j)-sample(u_y,i,j))/dx
            else:
                delta2 = sample(q_x,i+1,j)/h0 * (sample(u_y,i+1,j)-sample(u_y,i,j))/dx
        delta = delta1+delta2+real_gravity*(sample(h,i,j)-sample(h,i,j-1))/dx
        new_u_y[i,j] = u_y[i,j]-(3.0/2*delta-1.0/2*du_y[i,j])*dt
        du_y[i,j] =delta 

@ti.kernel
def up_wind_update_h(
    new_h:ti.template(),
    h:ti.template(),
    dh:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    dx:float,
    dt:float
):
    for i, j in h:
        delta_h = (
            up_wind_h(i+1,j, h,u_x, False)*u_x[i+1,j]-
            up_wind_h(i,j, h,u_x, False)*u_x[i,j]
        )/dx+(
            up_wind_h(i,j+1, h,u_y, True)*u_y[i,j+1]-
            up_wind_h(i,j, h,u_y, True)*u_y[i,j]
        )/dx
        new_h[i,j] = h[i,j]-(3.0/2*delta_h-1.0/2*dh[i,j])*dt
        dh[i,j] = delta_h

@ti.func
def up_wind_h(i,j,h,u, is_y):
    res = 0.0
    if(not is_y):
        h_0 = sample(h,i-1,j)
        h_1 = sample(h,i,j)
        if(u[i,j]>0):
            res = h_0
        elif(u[i,j]<0):
            res = h_1
        else:
            res = ti.max(h_0,h_1)
    else:
        h_0 = sample(h,i,j-1)
        h_1 = sample(h,i,j)
        if(u[i,j]>0):
            res = h_0
        elif(u[i,j]<0):
            res = h_1
        else:
            res = ti.max(h_0,h_1)
    return res
