from taichi_utils import *
from hyperparameters import *

############################################################################################################
##############################Begin: Solid Boundary Moving ################################################
############################################################################################################
############################################################################################################

@ti.kernel
def init_velocity_by_ibm_boundary_mask(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_velocity:ti.template(),
    u_x:ti.template(),u_y:ti.template()
):

    for i,j in u_x:
        if(sample(ibm_boundary_mask,i,j) >= 1 or sample(ibm_boundary_mask,i-1,j) >= 1):
            v = (sample(ibm_boundary_velocity,i,j)+ sample(ibm_boundary_velocity,i-1,j))[0]/2
            u_x[i,j] = v
    

    for i,j in u_y:
        if(sample(ibm_boundary_mask,i,j) >= 1 or sample(ibm_boundary_mask,i,j-1) >= 1):
            v = (sample(ibm_boundary_velocity,i,j)+ sample(ibm_boundary_velocity,i,j-1))[1]/2
            u_y[i,j] = v

@ti.kernel
def set_case_3(
    h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        eta[i,j] = 0.0
        if(i<=2):
            boundary_mask[i,j]=2
        #if(i>=res_x-3):
        #    boundary_mask[i,j]=2
        """
        if(i<200 and j<200):
            boundary_mask[i,j]=1
        elif(i<200 and j>res_y-200):
            boundary_mask[i,j]=1
        elif(i>res_x-200 and j<200):
            boundary_mask[i,j]=1
        elif(i>res_x-200 and j>res_y-200):
            boundary_mask[i,j]=1"""

@ti.kernel
def set_case_9(
    h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 0.75
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        eta[i,j] = 0.0
        if(i<=50):
            boundary_mask[i,j]=2

@ti.kernel
def set_case_8(
    h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        eta[i,j] = 0.0
        if(i<=50):
            boundary_mask[i,j]=2

@ti.kernel
def set_case_7(
    h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        eta[i,j] = 0.0
        if(i<=50):
            boundary_mask[i,j]=2

@ti.kernel
def set_case_4(
    h:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        if(ibm_boundary_mask[i,j]>=1):
            h[i, j] = ibm_boundary_h[i,j]
        eta[i,j] = 0.0

@ti.kernel
def set_case_10(
    h:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        if(ibm_boundary_mask[i,j]>=1):
            h[i, j] = ibm_boundary_h[i,j]
        eta[i,j] = 0.0

@ti.kernel
def set_case_11(
    h:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        if(ibm_boundary_mask[i,j]>=1):
            h[i, j] = ibm_boundary_h[i,j]
        eta[i,j] = 0.0

@ti.kernel
def set_case_12(
    h:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_h:ti.template(),
    eta:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    wind_u_x:ti.template(),
    wind_u_y:ti.template(),
    boundary_mask:ti.template()
):
    wind_u_x[None] = 1.0
    wind_u_y[None] = 0.0
    u_x.fill(0.0)
    u_y.fill(0.0)
    for i, j in h:
        h[i, j] = init_h#+eta[i,j]
        eta[i,j] = 0.0
        #if(ibm_boundary_mask[i,j]>=1):
        #    h[i, j] = ibm_boundary_h[i,j]
        if(i<=2):
            boundary_mask[i,j]=2

def moving_ibm_obstacle(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    y_sum:ti.template(),
    new_c:ti.template(),
    t:float
):
    if(case == 3):
        moving_ibm_obstacle_case_3(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,t)
    elif(case == 4):
        moving_ibm_obstacle_case_4_new(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,t)
    elif(case == 7):
        moving_ibm_obstacle_case_7(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,y_sum,new_c,t)
    elif(case == 8):
        moving_ibm_obstacle_case_8(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,y_sum,new_c,t)
    elif(case == 9):
        moving_ibm_obstacle_case_9(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,y_sum,new_c,t)
    elif(case == 10):
        moving_ibm_obstacle_case_10_new(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,t)
    elif(case == 11):
        moving_ibm_obstacle_case_11_new(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,t)
    elif(case == 12):
        moving_ibm_obstacle_case_12(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h,ibm_boundary_coef,ibm_boundary_velocity,t)


@ti.kernel
def moving_ibm_obstacle_case_11_new(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    # domain: 6 * 1.5
    A = 0.5
    v = 1.5/2
    Tx = 2
    w = 2*3.14159/Tx          
    for i, j in ibm_boundary_mask:   
        p = ti.Vector([i+0.5,j+0.5])*dx      
        dt = 0.001
        c0,boat_c0,pos0,rot_mat0, rot_mat_inverse0 = S_shape_position_new(t, A, w, v)
        c1,boat_c1,pos1,rot_mat1, rot_mat_inverse1 = S_shape_position_new(t+dt, A, w, v)
        pos = rot_mat0@(p-c0)-boat_c0
        flag, cood1, cood2= in_boat(pos)
        if(i == 0 and j == 0):
            print(rot_mat0,c0,boat_c0)
        if(flag):
            max_depth = 0.1*init_h
            #if(cood1>0.5):
            #    cood1 = 1-cood1
            ibm_boundary_h[i,j] = init_h - (1-cood2)*cood1* max_depth
            ibm_boundary_coef[i,j] = (init_h - ibm_boundary_h[i,j])/init_h #0.1
            ibm_boundary_mask[i,j]=1
            new_pos = rot_mat_inverse1@(pos+boat_c1)+c1
            ibm_boundary_velocity[i,j] = (new_pos-p)/dt 
            ibm_boundary_mask_extend[i,j] = 1

@ti.func
def S_shape_position_new(t, A, w, v):
    x = v*t
    y = A*ti.sin(w*x)
    pos = ti.Vector([x,y])
    theta = ti.atan2(A*w*ti.cos(w*x),1)
    boat_c = ti.Vector([0.0,0.0])
    c = pos
    boat_c = pos
    rot_mat = theta2Matrix(3.1415-theta)
    rot_mat_inverse = theta2InverseMatrix(3.1415-theta)
    c+=ti.Vector([0.5,1.5/2])
    boat_c+=ti.Vector([0.5,1.5/2])
    pos+=ti.Vector([0.5,1.5/2])
    return c,boat_c-c,pos,rot_mat,rot_mat_inverse

@ti.kernel
def moving_ibm_obstacle_case_11(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    # domain: 6 * 1.5
    
    r = 0.5
    w = 1.5/r        
    for i, j in ibm_boundary_mask:   
        p = ti.Vector([i+0.5,j+0.5])*dx      
        dt = 0.001
        c0,boat_c0,pos0,rot_mat0, rot_mat_inverse0 = S_shape_position(t, r, w)
        c1,boat_c1,pos1,rot_mat1, rot_mat_inverse1 = S_shape_position(t+dt, r, w)
        pos = rot_mat0@(p-c0)-boat_c0
        flag, cood1, cood2= in_boat(pos)
        if(flag):
            max_depth = 0.1*init_h
            #if(cood1>0.5):
            #    cood1 = 1-cood1
            ibm_boundary_h[i,j] = init_h - (1-cood2)*cood1* max_depth
            ibm_boundary_coef[i,j] = (init_h - ibm_boundary_h[i,j])/init_h #0.1
            ibm_boundary_mask[i,j]=1
            new_pos = rot_mat_inverse1@(pos+boat_c1)+c1
            ibm_boundary_velocity[i,j] = (new_pos-p)/dt 
            ibm_boundary_mask_extend[i,j] = 1

@ti.func
def S_shape_position(t, r, w):
    c1 = ti.Vector([1.0,1.5/2])
    c2 = ti.Vector([2.0,1.5/2])
    c3 = ti.Vector([3.0,1.5/2])
    c4 = ti.Vector([4.0,1.5/2])
    c5 = ti.Vector([5.0,1.5/2])
    pi = 3.1415926
    T = pi/w
    cn = int(t/T)
    theta1 =  w*(t-cn)
    pos = ti.Vector([0.0,0.0])
    c = ti.Vector([0.0,0.0]) 
    boat_c = ti.Vector([0.0,0.0]) 
    rot_mat = ti.Matrix([[1.0,0.0],[0.0,1.0]])
    rot_mat_inverse = ti.Matrix([[1.0,0.0],[0.0,1.0]])
    if(cn%2 == 0):
        pos = ti.Vector([cn*2*r+c1[0]-r,0]) + r*ti.Vector([1-ti.cos(theta1),ti.sin(theta1)])
        rot_mat = theta2Matrix(-(pi/2-theta1))
        rot_mat_inverse = theta2InverseMatrix(-(pi/2-theta1))
    else:
        pos = ti.Vector([cn*2*r+c1[0]-r,0]) + r*ti.Vector([1-ti.cos(theta1),-ti.sin(theta1)])
        rot_mat = theta2Matrix((pi/2-theta1))
        rot_mat_inverse = theta2InverseMatrix((pi/2-theta1))
    if(cn == 0):
        c = c1
        boat_c = c+ti.Vector([0,r])
    elif(cn == 1):
        c = c2
        boat_c = c-ti.Vector([0,r])
    elif(cn == 2):
        c = c3
        boat_c = c+ti.Vector([0,r])
    elif(cn == 3):
        c = c4
        boat_c = c-ti.Vector([0,r])
    else:
        c = c5
        boat_c = c+ti.Vector([0,r])
    return c,boat_c,pos,rot_mat, rot_mat_inverse


@ti.kernel
def moving_ibm_obstacle_case_7(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    y_sum:ti.template(),
    new_c:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    for i, j in ibm_boundary_mask:
        ibm_boundary_h[i,j] = init_h - 0.1
        ibm_boundary_coef[i,j] = 1.0
        pos =ti.Vector([i+0.5,j+0.5])*dx
        c = ti.Vector([0.5,1.0])
        r = 0.070
        if((pos - c).norm()<r):
            ibm_boundary_mask[i,j]=1
            ibm_boundary_velocity[i,j] = ti.Vector([0.0,0.0])
            ibm_boundary_mask_extend[i,j] = 1

@ti.kernel
def moving_ibm_obstacle_case_8(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    y_sum:ti.template(),
    new_c:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    if(t>4.5):
        new_c[None]+=-y_sum[None]*5
    print("c",new_c[None],y_sum[None])
    for i, j in ibm_boundary_mask:
        ibm_boundary_h[i,j] = init_h - 0.1
        ibm_boundary_coef[i,j] = 1.0
        pos =ti.Vector([i+0.5,j+0.5])*dx
        c = ti.Vector([0.5,new_c[None]])
        r = 0.070
        if((pos - c).norm()<r):
            ibm_boundary_mask[i,j]=1
            ibm_boundary_velocity[i,j] = ti.Vector([0.0,0.0])
            ibm_boundary_mask_extend[i,j] = 1

@ti.kernel
def moving_ibm_obstacle_case_9(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    y_sum:ti.template(),
    new_c:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    if(t>4.5):
        new_c[None]+=-y_sum[None]*5
    print("c",new_c[None],y_sum[None])
    for i, j in ibm_boundary_mask:
        ibm_boundary_h[i,j] = init_h - 0.1
        ibm_boundary_coef[i,j] = 1.0
        pos =ti.Vector([i+0.5,j+0.5])*dx
        c = ti.Vector([0.5,new_c[None]])
        r = 0.070
        if((pos - c).norm()<r):
            ibm_boundary_mask[i,j]=1
            ibm_boundary_velocity[i,j] = ti.Vector([0.0,0.0])
            ibm_boundary_mask_extend[i,j] = 1

@ti.kernel
def moving_ibm_obstacle_case_12(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    for i, j in ibm_boundary_mask:
        ibm_boundary_h[i,j] = init_h - 0.1
        ibm_boundary_coef[i,j] = 1.0
        pos =ti.Vector([i+0.5,j+0.5])*dx
        # domain 3*3
        all_c = ti.Vector([1.0,1.5])
        r = 0.070
        c1 = all_c + ti.Vector([0.0,r*2])
        c2 = all_c - ti.Vector([0.0,r*2])

        c3 = all_c - ti.Vector([r*4,0.0])

        c4 = all_c + ti.Vector([r*4,-r*4])
        c5 = all_c + ti.Vector([r*4,0.0])
        c6 = all_c + ti.Vector([r*4,r*4])
        
        for k in range(6):
            c = c6
            if(k==0):
                c = c1
            elif(k==1):
                c = c2
            elif(k==2):
                c = c3
            elif(k==3):
                c = c4
            elif(k==4):
                c = c5
            if((pos - c).norm()<r):
                ibm_boundary_mask[i,j]=1
                ibm_boundary_velocity[i,j] = ti.Vector([0.0,0.0])
                ibm_boundary_mask_extend[i,j] = 1

@ti.kernel
def moving_ibm_obstacle_case_3(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    for i, j in ibm_boundary_mask:
        ibm_boundary_h[i,j] = 0.9*init_h
        ibm_boundary_coef[i,j] = 1.0

        
        for kk in range(3):
            c = ti.Vector([1.5- 0.1*t,0.5])
            T = 1.25
            dt = 0.001
            A = 3.1415/12
            new_c = ti.Vector([1.5- 0.1*(t+dt),0.5])
            if(kk==1):
                c = ti.Vector([1.5- 0.1*t,1.0])
                new_c = ti.Vector([1.5- 0.1*(t+dt),1.0])
            elif(kk==2):
                c = ti.Vector([1.5- 0.1*t,2.0])
                new_c = ti.Vector([1.5- 0.1*(t+dt),2.0])
            elif(kk ==0):
                A/=2
                T/=2
                c = ti.Vector([1.5- 0.1*t*2,0.5+1])
                new_c =ti.Vector([1.5- 0.1*(t+dt)*2,0.5+1])
            mat = theta2InverseMatrix(A*ti.sin(2*3.1415/T*t))
            mat2 = theta2Matrix(A*ti.sin(2*3.1415/T*(t+dt)))
            p = ti.Vector([i+0.5,j+0.5])*dx
            pos = mat@(p-c)
            if(0.0<pos[0]<0.3 and abs(pos[1])<0.025):
                ibm_boundary_mask[i,j]=1
                ibm_boundary_velocity[i,j] = ((mat2@pos+new_c)-p)/dt
            if(0.0-pp_dx_xy<pos[0]<0.3+pp_dx_xy and abs(pos[1])<0.025+pp_dx_xy):
                ibm_boundary_mask_extend[i,j] = 1
        
        """
        T = 0.5
        A = 0.01
        board_x = A*(1-ti.cos(2*3.1415/T*t))
        v = A*2*3.1415/T*(ti.sin(2*3.1415/T*t))
        if((i+0.5)*dx<board_x):
            ibm_boundary_mask[i,j]=2
            if(v<0):
                v = 0.0
            ibm_boundary_velocity[i,j] = ti.Vector([v,0.0])
        """     

@ti.kernel
def moving_ibm_obstacle_case_4_new(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    for i, j in ibm_boundary_mask:
        dt = 0.001
        r = 1.0
        w = 1.5/r        
        mat = theta2InverseMatrix(t*w+3.1415/2)
        mat2 = theta2Matrix((t+dt)*w+3.1415/2)
        mat90 = theta2Matrix(3.1415/2)
        imat90 = theta2InverseMatrix(3.1415/2)
        c = ti.Vector([0.0,r])
        c0 = ti.Vector([1.5,1.5])
        p = ti.Vector([i+0.5,j+0.5])*dx        
        pos = mat90@(mat@(p-c0))-c
        flag, cood1, cood2= in_boat(pos)
        if(flag):
            max_depth = 0.1*init_h
            #if(cood1>0.5):
            #    cood1 = 1-cood1
            ibm_boundary_h[i,j] = init_h - (1-cood2)*cood1* max_depth
            ibm_boundary_coef[i,j] = (init_h - ibm_boundary_h[i,j])/init_h #0.1
            ibm_boundary_mask[i,j]=1
            ibm_boundary_velocity[i,j] = (mat2@(imat90@(c+pos))+c0-p)/dt 
            ibm_boundary_mask_extend[i,j] = 1

@ti.kernel
def moving_ibm_obstacle_case_10_new(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    for i, j in ibm_boundary_mask:
        dt = 0.001
        r = 1
         # w = 1.5/r # case 1
        w = 1.65/r                
        for k in range(3):
            mat = theta2InverseMatrix(t*w+3.1415/2)
            mat2 = theta2Matrix((t+dt)*w+3.1415/2)
            mat90 = theta2Matrix(3.1415/2)
            imat90 = theta2InverseMatrix(3.1415/2)
            c = ti.Vector([0.0,r])
            if(k == 1):
                #c = ti.Vector([-ti.sqrt(3)/2*r,-0.5*r])
                mat = theta2InverseMatrix(t*w+3.1415*(1+1.0/6))
                mat2 = theta2Matrix((t+dt)*w+3.1415*(1+1.0/6))
            elif(k == 2):
                #c = ti.Vector([ti.sqrt(3)/2*r,-0.5*r])
                mat = theta2InverseMatrix(t*w+3.1415*(2-1.0/6))
                mat2 = theta2Matrix((t+dt)*w+3.1415*(2-1.0/6))
            # c0 = ti.Vector([1.5,1.5])

            c0 = ti.Vector([2.5,2.5])

            p = ti.Vector([i+0.5,j+0.5])*dx        
            pos = mat90@(mat@(p-c0))-c
            flag, cood1, cood2= in_boat(pos)
            if(flag):
                # max_depth = 0.07*init_h # case 2
                max_depth = 0.065*init_h # case 2
                # max_depth = 0.1*init_h # case1
                #if(cood1>0.5):
                #    cood1 = 1-cood1
                ibm_boundary_h[i,j] = init_h - (1-cood2)*cood1* max_depth
                ibm_boundary_coef[i,j] = (init_h - ibm_boundary_h[i,j])/init_h #0.1
                ibm_boundary_mask[i,j]=1
                ibm_boundary_velocity[i,j] = (mat2@(imat90@(c+pos))+c0-p)/dt 
                ibm_boundary_mask_extend[i,j] = 1

@ti.kernel
def moving_ibm_obstacle_case_4(
    ibm_boundary_mask:ti.template(),
    ibm_boundary_mask_extend:ti.template(),
    ibm_boundary_h:ti.template(),
    ibm_boundary_coef:ti.template(),
    ibm_boundary_velocity:ti.template(),
    t:float
):
    ibm_boundary_mask_extend.fill(0)
    ibm_boundary_mask.fill(0)
    ibm_boundary_velocity.fill(0)
    for i, j in ibm_boundary_mask:
        dt = 0.001
        c = ti.Vector([pos_x_case_4(t),1.0])
        new_c = ti.Vector([pos_x_case_4(t+dt),1.0])
        p = ti.Vector([i+0.5,j+0.5])*dx
        pos = (p-c)
        flag, cood1, cood2= in_boat(pos)
        if(flag):
            max_depth = 0.2*init_h
            if(cood1>0.5):
                cood1 = 1-cood1
            ibm_boundary_h[i,j] = init_h - (1-cood2)*cood1* max_depth
            ibm_boundary_coef[i,j] = (init_h - ibm_boundary_h[i,j])/init_h #0.1
            ibm_boundary_mask[i,j]=1
            ibm_boundary_velocity[i,j] = ((pos+new_c)-p)/dt
            ibm_boundary_mask_extend[i,j] = 1
        
@ti.func
def pos_x_case_4(t):
    bT = 0.5
    v = 1.5#3.0#1.5
    return 2.7 - v*t
    #x = 0.0
    #if(t<bT):
    #    a = v/bT
    #    x =  2.7- 0.5*a*t**2
    #else:
    #    a = v/bT
    #    x =  2.7- 0.5*a*bT**2 - v*(t-bT)
    #return x

@ti.func
def in_boat(pos):
    x = pos[0]
    y = pos[1]

    inshape = False
    # Bullet shape parameters
    bullet_center_x = 0.0
    bullet_center_y = 0.0
    bullet_length = 0.15*3
    bullet_width = 0.03*3

    # Bullet rear rectangle parameters
    rect_length = bullet_length * 0.5  # Half the total length is the rectangular section
    rect_half_width = bullet_width / 2

    # Bullet front triangle parameters
    tri_length = bullet_length * 0.5  # Half the total length is the triangular section

    # Define boundaries of the rectangle (tail of the bullet)
    rect_left = bullet_center_x
    rect_right = bullet_center_x + rect_length
    rect_top = bullet_center_y + rect_half_width
    rect_bottom = bullet_center_y - rect_half_width

    # Check if the point is in the rectangular section
    #if rect_left <= x <= rect_right and rect_bottom <= y <= rect_top:
    #    inshape= True

    # Check if the point is in the triangular section (sharp front of the bullet)
    # The triangle's tip is at rect_left - tri_length, and its base is at rect_left
    cood1,cood2 = 0.0,0.0
    if (rect_left - tri_length) <= x <= rect_left:
        # Calculate the width of the triangle at the given x value
        triangle_width_at_x = rect_half_width * (1 - (rect_left - x) / tri_length)
        if (bullet_center_y - triangle_width_at_x) <= y <= (bullet_center_y + triangle_width_at_x):
            inshape= True
            cood1 = (x-(rect_left - tri_length))/tri_length
            cood2 = abs((y-bullet_center_y)/triangle_width_at_x)

    return inshape,cood1,cood2         
        

@ti.func
def theta2Matrix(theta):
    return ti.Matrix(
        [
            [ti.cos(theta),-ti.sin(theta)],
            [ti.sin(theta),ti.cos(theta)]
        ]
    )

@ti.func
def theta2InverseMatrix(theta):
    return ti.Matrix(
        [
            [ti.cos(theta),ti.sin(theta)],
            [-ti.sin(theta),ti.cos(theta)]
        ]
    )

@ti.kernel
def init_bottom(eta:ti.template(),dx:float):
    ii,jj = eta.shape
    for i,j in eta:
        pos = ti.Vector([i+0.5,j+0.5])*dx
        #if(i>100 and i<ii-100 and j>100 and j<jj-100):
        eta[i,j] = 0.1 * noise_octave(3*pos, 10)
        if(eta[i,j]<0.0):
            eta[i,j] = 0.0

@ti.func
def hash2(v):
    rand = ti.Vector([0.0,0.0])
    rand  = 52.5 * ti.math.fract(ti.Vector([v[1],v[0]]) * 0.31 + ti.Vector([0.31, 0.113]))
    rand = -1.0 + 3.1 * ti.math.fract(rand[0] * rand[1] * ti.Vector([rand[1],rand[0]]))
    return rand


@ti.func
def perlin_noise(p): 
    noise = 0.0
    i = ti.floor(p,int)
    f =ti.math.fract(p) 
    m = f*f*(3.0-2.0*f)
    noise = ti.math.mix( ti.math.mix( ti.math.dot( hash2(i + ti.Vector([0.0, 0.0])), f - ti.Vector([0.0,0.0])),
					                  ti.math.dot( hash2(i + ti.Vector([1.0, 0.0])), f - ti.Vector([1.0,0.0])), m[0]),
				        ti.math.mix(  ti.math.dot( hash2(i + ti.Vector([0.0, 1.0])), f - ti.Vector([0.0,1.0])),
					                  ti.math.dot( hash2(i + ti.Vector([1.0, 1.0])), f - ti.Vector([1.0,1.0])), m[0]), m[1])
    return noise

@ti.func
def noise_octave(p, num):
	sum_v = 0.0
	for i in range(3):
		sum_v += ti.pow(2,-1.0*i) * perlin_noise(ti.pow(2,i) * p)
	return sum_v
############################################################################################################
##############################End: Solid Boundary Moving ################################################
############################################################################################################
############################################################################################################

############################################################################################################
##############################Begin: Solid Boundary Setting ################################################
############################################################################################################
############################################################################################################
@ti.kernel
def set_bullet_boundary_Case0(boundary:ti.template(),levelset:ti.template(), X:ti.template()):
    for I in ti.grouped(boundary):
        x = X[I][0]
        y = X[I][1]

        inshape = False
        # Bullet shape parameters
        bullet_center_x = 2.0
        bullet_center_y = 0.5
        bullet_length = 0.15*2#*4
        bullet_width = 0.03*2#*4

        # Bullet rear rectangle parameters
        rect_length = bullet_length * 0.5  # Half the total length is the rectangular section
        rect_half_width = bullet_width / 2

        # Bullet front triangle parameters
        tri_length = bullet_length * 0.5  # Half the total length is the triangular section

        # Define boundaries of the rectangle (tail of the bullet)
        rect_left = bullet_center_x
        rect_right = bullet_center_x + rect_length
        rect_top = bullet_center_y + rect_half_width
        rect_bottom = bullet_center_y - rect_half_width

        # Check if the point is in the rectangular section
        if rect_left <= x <= rect_right and rect_bottom <= y <= rect_top:
            inshape= True

        # Check if the point is in the triangular section (sharp front of the bullet)
        # The triangle's tip is at rect_left - tri_length, and its base is at rect_left
        if (rect_left - tri_length) <= x <= rect_left:
            # Calculate the width of the triangle at the given x value
            triangle_width_at_x = rect_half_width * (1 - (rect_left - x) / tri_length)
            if (bullet_center_y - triangle_width_at_x) <= y <= (bullet_center_y + triangle_width_at_x):
                inshape= True

        # then calculate the levelset 
        pos = ti.Vector([x,y])
        levelset[I] = ti.min(
            point_segment_distance(pos, ti.Vector([bullet_center_x-tri_length,bullet_center_y]),ti.Vector([bullet_center_x-tri_length,bullet_center_y+rect_half_width])),
            point_segment_distance(pos, ti.Vector([bullet_center_x-tri_length,bullet_center_y]),ti.Vector([bullet_center_x-tri_length,bullet_center_y-rect_half_width])),
            point_segment_distance(pos, ti.Vector([bullet_center_x,bullet_center_y+rect_half_width]),ti.Vector([bullet_center_x+rect_length,bullet_center_y+rect_half_width])),
            point_segment_distance(pos, ti.Vector([bullet_center_x,bullet_center_y-rect_half_width]),ti.Vector([bullet_center_x+rect_length,bullet_center_y-rect_half_width])),
            point_segment_distance(pos, ti.Vector([bullet_center_x+rect_length,bullet_center_y-rect_half_width]),ti.Vector([bullet_center_x+rect_length,bullet_center_y+rect_half_width])),
        )

        if inshape:
            levelset[I]*=-1
        if inshape:
            boundary[I] = 1

@ti.kernel
def set_bullet_boundary_Case1(boundary:ti.template(), X:ti.template()):
    for I in ti.grouped(boundary):
        x = X[I][0]
        y = X[I][1]

        inshape = False
        # Bullet shape parameters
        bullet_center_x = 2.0
        bullet_center_y = 0.5
        bullet_length = 0.15*4
        bullet_width = 0.03*4

        # Bullet rear rectangle parameters
        rect_length = bullet_length * 0.5  # Half the total length is the rectangular section
        rect_half_width = bullet_width / 2

        # Bullet front triangle parameters
        tri_length = bullet_length * 0.5  # Half the total length is the triangular section

        # Define boundaries of the rectangle (tail of the bullet)
        rect_left = bullet_center_x
        rect_right = bullet_center_x + rect_length
        rect_top = bullet_center_y + rect_half_width
        rect_bottom = bullet_center_y - rect_half_width

        # Check if the point is in the rectangular section
        if rect_left <= x <= rect_right and rect_bottom <= y <= rect_top:
            inshape= True

        # Check if the point is in the triangular section (sharp front of the bullet)
        # The triangle's tip is at rect_left - tri_length, and its base is at rect_left
        if (rect_left - tri_length) <= x <= rect_left:
            # Calculate the width of the triangle at the given x value
            triangle_width_at_x = rect_half_width * (1 - (rect_left - x) / tri_length)
            if (bullet_center_y - triangle_width_at_x) <= y <= (bullet_center_y + triangle_width_at_x):
                inshape= True

        if inshape:
            boundary[I] = 1

@ti.kernel
def set_triangle_boundary_Case2(boundary:ti.template(), X:ti.template()):
    for I in ti.grouped(boundary):
        x = X[I][0]
        y = X[I][1]

        inshape = False
        # Bullet shape parameters
        bullet_center_x = 2.0
        bullet_center_y = 0.5
        bullet_length = 0.15*2/2/2#1#*4
        bullet_width = 0.03*4/2#2.5#*4

        # Bullet rear rectangle parameters
        rect_length = bullet_length * 0.5  # Half the total length is the rectangular section
        rect_half_width = bullet_width / 2

        # Bullet front triangle parameters
        tri_length = bullet_length * 0.5  # Half the total length is the triangular section

        # Define boundaries of the rectangle (tail of the bullet)
        rect_left = bullet_center_x
        rect_right = bullet_center_x + rect_length
        rect_top = bullet_center_y + rect_half_width
        rect_bottom = bullet_center_y - rect_half_width

        # Check if the point is in the rectangular section
        #if rect_left <= x <= rect_right and rect_bottom <= y <= rect_top:
        #    inshape= True

        # Check if the point is in the triangular section (sharp front of the bullet)
        # The triangle's tip is at rect_left - tri_length, and its base is at rect_left
        if (rect_left - tri_length) <= x <= rect_left:
            # Calculate the width of the triangle at the given x value
            triangle_width_at_x = rect_half_width * (1 - (rect_left - x) / tri_length)
            if (bullet_center_y - triangle_width_at_x) <= y <= (bullet_center_y + triangle_width_at_x):
                inshape= True

        if inshape:
            boundary[I] = 1

@ti.func
def point_segment_distance(p,seg1,seg2):
    dist = 0.0
    if(ti.math.cross(seg1-seg2, p-seg2)>0 and ti.math.cross(seg2-seg1, p-seg1)>0):
        x1,x2,x3 = p[0],seg1[0],seg2[0]
        y1,y2,y3 = p[1],seg1[1],seg2[1]
        A = (y1-y2)*x3+(x2-x1)*y3+x1*y2-y1*x2
        B = (y1-y2)**2 +(x1-x2)**2
        dist = ti.abs(A)/ti.sqrt(B)
    else:
        dist = ti.min(
            (p-seg1).norm(),
            (p-seg2).norm(),
        )
    return dist
############################################################################################################
##############################  End: Solid Boundary Setting ################################################
############################################################################################################
############################################################################################################

def init_all_conditions(h:ti.template(),vf: ti.template(), dx:float):
    if(init_experiment_name == "one_vortex"):
        h.fill(init_h)
        one_vortex_vel_func(vf,dx)
    elif(init_experiment_name == "two_vortex"):
        h.fill(init_h)
        two_vortex_vel_func(vf,dx)
    elif(init_experiment_name == "four_vortex"):
        h.fill(init_h)
        four_vortex_vel_func(vf,dx)
    elif(init_experiment_name == "wave_generator"):
        h.fill(init_h)
        wave_generator_vel_func(vf,dx)
    return init_h


@ti.kernel
def one_vortex_vel_func(vf: ti.template(), dx:float):
    c1 = ti.Vector([0.5, 0.5])
    w1 = 0.5
    for i, j in vf:
        pos = ti.Vector([i+0.5,j+0.5])*dx 
        p = pos - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition

@ti.kernel
def two_vortex_vel_func(vf: ti.template(), dx:float):
    c1 = ti.Vector([0.4, 0.5])
    c2 = ti.Vector([0.6, 0.5])
    w1 = 0.5
    for i, j in vf:
        pos = ti.Vector([i+0.5,j+0.5])*dx 
        p = pos - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition

        p2 = pos - c2
        r2 = ti.sqrt(p2.x * p2.x + p2.y * p2.y)
        addition2 = angular_vel_func(r2, 0.02, -0.01) * w1 * ti.Vector([-p2.y, p2.x])
        vf[i, j] += addition2

@ti.kernel
def three_vortex_vel_func(vf: ti.template(), dx:float):
    c= ti.Vector([0.5,0.5])
    r = 0.2*1.5
    c1 = c+ti.Vector([0.0, r])
    c2 = c+ti.Vector([-r*ti.sqrt(3)/2, -r*0.5])
    c3 = c+ti.Vector([r*ti.sqrt(3)/2, -r*0.5])
    w1 = 0.5*2*1.5
    for i, j in vf:
        pos = ti.Vector([i+0.5,j+0.5])*dx 
        p = pos - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02*2*1.5, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition

        p2 = pos - c2
        r2 = ti.sqrt(p2.x * p2.x + p2.y * p2.y)
        addition2 = angular_vel_func(r2, 0.02*2*1.5, -0.01) * w1 * ti.Vector([-p2.y, p2.x])
        vf[i, j] += addition2

        p3 = pos - c3
        r3 = ti.sqrt(p3.x * p3.x + p3.y * p3.y)
        addition3 = angular_vel_func(r3, 0.02*2*1.5, -0.01) * w1 * ti.Vector([-p3.y, p3.x])
        vf[i, j] += addition3

@ti.kernel
def four_vortex_vel_func(vf: ti.template(), dx:float):
    c1 = ti.Vector([0.25, 0.62])
    c2 = ti.Vector([0.25, 0.38])
    c3 = ti.Vector([0.25, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    w1 = -0.5#*5
    w2 = 0.5#*5
    w3 = -0.5#*5
    w4 = 0.5#*5
    for i, j in vf:
        pos = ti.Vector([i+0.5,j+0.5])*dx 
        p = pos - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition

        p2 = pos - c2
        r2 = ti.sqrt(p2.x * p2.x + p2.y * p2.y)
        addition2 = angular_vel_func(r2, 0.02, -0.01) * w2 * ti.Vector([-p2.y, p2.x])
        vf[i, j] += addition2

        p3 = pos - c3
        r3 = ti.sqrt(p3.x * p3.x + p3.y * p3.y)
        addition3 = angular_vel_func(r3, 0.02, -0.01) * w3 * ti.Vector([-p3.y, p3.x])
        vf[i, j] += addition3

        p4 = pos - c4
        r4 = ti.sqrt(p4.x * p4.x + p4.y * p4.y)
        addition4 = angular_vel_func(r4, 0.02, -0.01) * w4 * ti.Vector([-p4.y, p4.x])
        vf[i, j] += addition4

@ti.kernel
def wave_generator_vel_func(vf:ti.template(),dx:float):
    vf.fill(0.0)

# single vortex fields
@ti.func
def angular_vel_func(r, rad, strength):
    r = r + 1e-6
    linear_vel = strength * 1./r * (1.-ti.exp(-(r**2)/(rad**2)))
    return 1./r * linear_vel

@ti.func
def dvdr(r, rad, strength):
    r = r + 1e-6
    result = strength * (-(2*(1-ti.exp(-(r**2)/(rad**2))))/(r**3) + (2*ti.exp(-(r**2)/(rad**2)))/(r*(rad**2)))
    return result    

# vortex velocity field
@ti.kernel
def vortex_vel_func(vf: ti.template(), pf: ti.template()):
    c = ti.Vector([0.5, 0.5])
    for i, j in vf:
        p = pf[i, j] - c
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j].y = p.x
        vf[i, j].x = -p.y
        vf[i, j] *= angular_vel_func(r, 0.02, -0.01)

# vortex velocity field
@ti.kernel
def leapfrog_vel_func(vf: ti.template(), pf: ti.template()):
    c1 = ti.Vector([0.25, 0.62])
    c2 = ti.Vector([0.25, 0.38])
    c3 = ti.Vector([0.25, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    cs = [c1, c2, c3, c4]
    w1 = -0.5
    w2 = 0.5
    w3 = -0.5
    w4 = 0.5
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w2 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w3 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w4 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
# # # #

@ti.kernel
def set_boundary(field: ti.template()):
    if(case == 0):
        pass
    elif(case == 1):
        
        offset_x = 1.0
        offset_y = 0.5
        dx = 1.0 / field.shape[1]
        r = 0.03
        for I in ti.grouped(field):
            field[I] = 0.0
            loc = (I + 0.5) * dx
            if (loc[0] - offset_x)**2 + (loc[1] - offset_y)**2 <= r**2:
                field[I] = 1.0
            if(I[0]<=100 or I[0]>=field.shape[0]-101):
                field[I] = 2.0
    
    # field[0, 0] = 1
    # field[0, field.shape[1] - 1] = 1
    # field[field.shape[0] - 1, 0] = 1
    # field[field.shape[0] - 1, field.shape[1] - 1] = 1
    pass
    #for I in ti.grouped(field):
    #    if I[0] == 0:
    #        field[I] = 1
    #    if I[0] == field.shape[0] - 1:
    #        field[I] = 1
    #    if I[1] == 0:
    #        field[I] = 1
    #    if I[1] == field.shape[1] - 1:
    #        field[I] = 1



# some shapes (checkerboards...)
@ti.kernel
def checkerboard_func(qf: ti.template(), pf: ti.template()):
    thickness = 0.1
    for i, j in qf:
        p = int(pf[i,j] / thickness)
        if (p.x + p.y) % 2 > 0:
            qf[i, j] = ti.Vector([1.0, 1.0, 1.0])
        else:
            qf[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def stripe_func(qf: ti.template(), pf: ti.template(), x_start: float, x_end: float):
    for i, j in qf:
        if x_start <= pf[i,j].x <= x_end and 0.15 <= pf[i,j].y <= 0.85:
            qf[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            qf[i, j] = ti.Vector([1.0, 1.0, 1.0])

@ti.kernel
def init_impulse(imp_x: ti.template(), imp_y: ti.template(), u_x: ti.template(), u_y: ti.template()):
    for I in ti.grouped(imp_x):
        imp_x[I] = u_x[I]
    for I in ti.grouped(imp_y):
        imp_y[I] = u_y[I]

@ti.kernel
def active_init_particles(particles_active: ti.template(), initial_particle_num: int):
    particles_active.fill(0)
    for i in particles_active:
        if i < initial_particle_num:
            particles_active[i] = 1

@ti.kernel
def init_particles_pos(particles_pos: ti.template(), particles_active: ti.template(), X: ti.template(),
                       res_x: int, particles_per_cell: int, dx: float):
    for i in particles_pos:
        if particles_active[i] == 1:
            cell = int(i / particles_per_cell)
            id_x = cell % res_x
            id_y = cell // res_x
            particles_pos[i] = X[id_x, id_y] + ti.Vector([(ti.random() - 0.5) for _ in ti.static(range(2))]) * dx

@ti.kernel
def init_particles_pos_uniform(particles_pos: ti.template(), X: ti.template(),
                       res_x: int, particles_per_cell: int, dx: float, particles_per_cell_axis: int,
                       dist_between_neighbor: float):
    # particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
    # dist_between_neighbor = dx / particles_per_cell_axis
    # for i in particles_pos:
    #     if particles_active[i] == 1:
    #         cell = int(i / particles_per_cell)
    #         id_x = cell % res_x
    #         id_y = cell // res_x
    #
    #         particle_id_in_cell = i % particles_per_cell
    #         particle_id_x_in_cell = particle_id_in_cell % particles_per_cell_axis
    #         particle_id_y_in_cell = particle_id_in_cell // particles_per_cell_axis
    #
    #         particles_pos[i] = X[id_x, id_y] - ti.Vector([0.5, 0.5]) * dx + \
    #                            ti.Vector([particle_id_x_in_cell + 0.5, particle_id_y_in_cell + 0.5]) * dist_between_neighbor

    # particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
    # dist_between_neighbor = dx / particles_per_cell_axis
    particles_x_num = particles_per_cell_axis * (res_x + 20)
    # particles_x_num = particles_per_cell_axis * (res_x)

    for i in particles_pos:
        # if particles_active[i] == 1:
            id_x = i % particles_x_num
            id_y = i // particles_x_num
            particles_pos[i] = (ti.Vector([id_x, id_y]) + 0.5) * dist_between_neighbor
            particles_pos[i] -= 10 * dx

@ti.kernel
def init_particles_pos_in_a_rectangle(particles_pos: ti.template(), particles_active: ti.template(), X: ti.template(),
                       initial_fluid_res_x: int, particles_per_cell: int, dx: float, fluid_origin_grid: ti.types.ndarray()):
    for i in particles_pos:
        if particles_active[i] == 1:
            cell = int(i / particles_per_cell)
            id_x = cell % initial_fluid_res_x + fluid_origin_grid[0]
            id_y = cell // initial_fluid_res_x + fluid_origin_grid[1]
            particles_pos[i] = X[id_x, id_y] + ti.Vector([(ti.random() - 0.5) for _ in ti.static(range(2))]) * dx

@ti.kernel
def backup_particles_pos(particles_pos: ti.template(), particles_pos_backup: ti.template()):
    for i in particles_pos:
        particles_pos_backup[i] = particles_pos[i]

@ti.kernel
def init_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(), particles_pos: ti.template(),
                       u_x: ti.template(), u_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[i], dx)
        particles_init_imp[i] = particles_imp[i]

@ti.kernel
def init_particles_imp_grad_m(particles_imp: ti.template(), particles_pos: ti.template(),
                                u_x: ti.template(), u_y: ti.template(),
                                C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[i], dx)
        C_x[i] = new_C_x
        C_y[i] = new_C_y

@ti.kernel
def init_particles_imp_one_step(particles_imp: ti.template(), particles_pos: ti.template(),
                                u_x: ti.template(), u_y: ti.template(), imp_x: ti.template(), imp_y: ti.template(),
                                C_x: ti.template(), C_y: ti.template(), dx: float):
    for i in particles_imp:
        particles_imp[i], _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, imp_x, imp_y, particles_pos[i], dx)
        C_x[i] = new_C_x
        C_y[i] = new_C_y