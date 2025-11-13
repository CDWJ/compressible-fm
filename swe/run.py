#
from hyperparameters import *
from taichi_utils import *
#from mgpcg_solid import *
from init_conditions import *
# from boundary_conditions import *
from io_utils import *
import sys
import shutil
import time
from advect import *
from force import *
#from simple_mesh import *
#from vis_flip import *
from passive_particles import *
from mgpcg import MGPCG_2
from laden_particle import *
#


half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx


ti.init(arch=ti.cuda, device_memory_GB=16.0, debug=False, advanced_optimization=False, fast_math=False)

# uniform distribute particles
particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
dist_between_neighbor = dx / particles_per_cell_axis

print("res_x, res_y",res_x, res_y)
# solver
boundary_mask = ti.field(ti.f32, shape=(res_x, res_y))
ibm_boundary_mask = ti.field(ti.f32, shape=(res_x, res_y))
ibm_boundary_h = ti.field(ti.f32, shape=(res_x, res_y))
ibm_boundary_coef = ti.field(ti.f32, shape=(res_x, res_y))
ibm_boundary_mask_extend = ti.field(ti.f32, shape=(res_x, res_y))
ibm_boundary_velocity = ti.Vector.field(2,ti.f32, shape=(res_x, res_y))
ibm_boundary_u_x = ti.field(ti.f32, shape=(res_x+1, res_y))
ibm_boundary_u_y = ti.field(ti.f32, shape=(res_x, res_y+1))
ibm_levelset = ti.field(ti.f32, shape=(res_x, res_y))

boundary_vel = ti.Vector.field(2, float, shape=(res_x, res_y))
boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32) # boundaries: 1 means Dirichlet, 2 means Neumann
# set_boundary(boundary_mask)


# undeformed coordinates (cell center and faces)
X = ti.Vector.field(2, float, shape=(res_x, res_y))
X_horizontal = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
X_vertical = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
center_coords_func(X, dx)
horizontal_coords_func(X_horizontal, dx)
vertical_coords_func(X_vertical, dx)

tem_p = ti.field(float, shape=(res_x , res_y))
real_p = ti.field(float, shape=(res_x , res_y))
# P2G weight storage
p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1))
p2g_weight = ti.field(float, shape=(res_x , res_y))

# velocity storage
u = ti.Vector.field(2, float, shape=(res_x, res_y))
w = ti.field(float, shape=(res_x, res_y))
# sizing = ti.field(float, shape=(res_x, res_y))  # sizing value corresponding to u
# tmp_sizing = ti.field(float, shape=(res_x, res_y))
h = ti.field(float, shape=(res_x , res_y))
h_dis = ti.field(float, shape=(res_x , res_y))

u_x = ti.field(float, shape=(res_x + 1, res_y))
u_y = ti.field(float, shape=(res_x, res_y + 1))
f_x = ti.field(float, shape=(res_x + 1, res_y))
f_y = ti.field(float, shape=(res_x, res_y + 1))

save_u_x = ti.field(float, shape=(res_x + 1, res_y))
save_u_y = ti.field(float, shape=(res_x, res_y + 1))

acc_int = ti.field(float, shape=(res_x, res_y))
acc_int_tem = ti.field(float, shape=(res_x, res_y))

u_square = ti.field(float, shape=(res_x, res_y))

initial_particle_num = (res_x + 20) * (res_y + 20) * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio
current_particle_num = ti.field(int, shape=1)
particles_active = ti.field(float, shape=particle_num)
particles_active.fill(1)
particles_pos = ti.Vector.field(2, float, shape=particle_num)
particles_imp = ti.Vector.field(2, float, shape=particle_num)
particles_acc_vis = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp = ti.Vector.field(2, float, shape=particle_num)
particles_h = ti.field(float, shape=particle_num)
particles_init_h = ti.field(float, shape=particle_num)
# added for zhiqi's method
particles_grad_lamb = ti.Vector.field(2, float, shape=particle_num)

particles_half_usquare = ti.field(float, shape=particle_num)
particles_grad_half_usquare = ti.Vector.field(2, float, shape=particle_num)
particles_weight = ti.Vector.field(2, float, shape=particle_num)

# back flow map
T_x = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y

F_x = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
F_y = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y


gradm_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # d_psi / d_x
gradm_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # d_psi / d_
gradm_x = ti.Vector.field(2, float, shape=particle_num)
gradm_y = ti.Vector.field(2, float, shape=particle_num)

# paticles in each cell
cell_max_particle_num = int(cell_max_particle_num_ratio * particles_per_cell)
cell_particle_num = ti.field(int, shape=(res_x, res_y))
cell_particles_id = ti.field(int, shape=(res_x, res_y, cell_max_particle_num))

# APIC
C_x = ti.Vector.field(2, float, shape=particle_num)
C_y = ti.Vector.field(2, float, shape=particle_num)

# CFL related
max_speed = ti.field(float, shape=())
max_err = ti.field(float, shape=())
# smoke
init_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
smoke = ti.Vector.field(3, float, shape=(res_x, res_y))


# RK4
u_x_buffer = ti.Vector.field(4,float, shape=(res_x + 1, res_y))
u_y_buffer = ti.Vector.field(4,float, shape=(res_x, res_y + 1))
h_buffer = ti.Vector.field(4,float, shape=(res_x , res_y))
acc_int_tem_buffer = ti.Vector.field(4,float, shape=(res_x, res_y))

u_x_save = ti.field(float, shape=(res_x + 1, res_y))
u_y_save = ti.field(float, shape=(res_x, res_y + 1))
h_save = ti.field(float, shape=(res_x , res_y))
acc_int_tem_save = ti.field(float, shape=(res_x, res_y))

u_x_acc = ti.field(float, shape=(res_x + 1, res_y))
u_y_acc = ti.field(float, shape=(res_x, res_y + 1))
acc_int_tem_acc = ti.field(float, shape=(res_x, res_y))


psi_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # x coordinate
psi_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # y coordinate
psi = ti.Vector.field(2, float, shape=particle_num)  # x coordinate
grad_h = ti.Vector.field(2, float, shape=particle_num)
grad_acc_int= ti.Vector.field(2, float, shape=particle_num)

max_speed_x = ti.field(float, shape=())
max_speed_y = ti.field(float, shape=())
        
flip_particle_max_num = pp_res_x*pp_res_y*pp_res_z*npc*2
flip_particles_pos = ti.Vector.field(3, float, shape=(flip_particle_max_num,))
flip_particles_vel = ti.Vector.field(3, float, shape=(flip_particle_max_num,))
flip_particles_life = ti.field(float, shape=(flip_particle_max_num,))
flip_particles_type  = ti.field(int, shape=(flip_particle_max_num,))
flip_particles_h = ti.field(float, shape=(flip_particle_max_num,))
flip_particles_pos_new = ti.Vector.field(3, float, shape=(flip_particle_max_num,))
flip_particles_vel_new = ti.Vector.field(3, float, shape=(flip_particle_max_num,))
flip_particles_life_new = ti.field(float, shape=(flip_particle_max_num,))
flip_particles_type_new  = ti.field(int, shape=(flip_particle_max_num,))
flip_particles_h_new = ti.field(float, shape=(flip_particle_max_num,))
flip_particles_num = ti.field(int,shape=())

grid_particles_num  = ti.field(int,shape=(pp_res_x,pp_res_y,pp_res_z))
grid_particles_base =  ti.field(float, shape=(pp_res_x,pp_res_y))
flip_particle_delete_flag = ti.field(int, shape=(flip_particle_max_num,))

flip_particles_vis_type  = ti.field(int, shape=(flip_particle_max_num,))
vis_h = ti.field(int, shape = (vol_res_x,vol_res_y,vol_res_z))
vis_u = ti.Vector.field(2,float, shape = (vol_res_x,vol_res_y,vol_res_z))
vis_u_3D = ti.Vector.field(3,float, shape = (vol_res_x,vol_res_y,vol_res_z))
vis_w_3D = ti.Vector.field(3,float, shape = (vol_res_x,vol_res_y,vol_res_z))
eta = ti.field(float, shape= (res_x,res_y))
wind_u_x = ti.field(float, shape=())
wind_u_y = ti.field(float, shape=())



#################################################################
########################Laden Particles #########################
#################################################################
if(laden_particles_on):
    laden_particles_num = 100000
    laden_particles_pos = ti.Vector.field(3,float,shape=(laden_particles_num))
    laden_particles_vel = ti.Vector.field(3,float,shape=(laden_particles_num))




#################################################################

ibm_force_y = ti.field( float, shape=(res_x , res_y+ 1)) 
y_sum = ti.field(float, shape=())
new_c  = ti.field(float, shape=())
new_c[None] =1.0
@ti.kernel
def add_pressure_force():
    y_sum[None]=0
    for I in ti.grouped(ibm_force_y):
        if(abs(ibm_force_y[I])>0):
            y_sum[None]+=ibm_force_y[I]*dx*dx

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i, j in ti.ndrange(res_x, res_y):
        u = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        speed = ti.sqrt(u ** 2 + v ** 2)
        ti.atomic_max(max_speed[None], speed)

@ti.kernel
def calc_max_imp_particles(particles_imp: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i in particles_imp:
        # if particles_active[i] == 1:
            imp = particles_imp[i].norm()
            ti.atomic_max(max_speed[None], imp)
        
@ti.kernel
def mask_particles():
    particles_active.fill(1)
    for i in particles_active:
        grid_idx = ti.round((particles_pos[i]) / dx - 0.5, int)
        if (grid_idx[0] >= 0 and grid_idx[0] < res_x - 0 and grid_idx[1] >= 0 and grid_idx[1] < res_y - 0) and boundary_mask[grid_idx] == 1:
            particles_active[i] = 0

@ti.kernel
def set_grid_boundary(
    boundary_mask:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_h:ti.template(), 
    ibm_boundary_coef:ti.template(),
    h:ti.template(),
):
    for i,j in boundary_mask:
        if( ibm_boundary_mask[i,j] == 1):
            h[i,j] = ibm_boundary_h[i,j]
        elif(ibm_boundary_mask[i,j] == 2):
            h[i,j] = init_h
        if(boundary_mask[i,j] == 1):
            u_x[i,j]=0.0
            u_x[i+1,j]=0.0
            u_y[i,j]=0.0
            u_y[i,j+1]=0.0
            h[i,j] = init_h
        elif(boundary_mask[i,j] == 2):
            pass

@ti.kernel
def set_boundary_particle(
    particles_pos:ti.template(),
    particles_h:ti.template(),
    particles_init_h:ti.template(),
    ibm_boundary_mask:ti.template(),
    ibm_boundary_h:ti.template(),
    T_x:ti.template(), 
    T_y:ti.template(),
    F_x:ti.template(), 
    F_y:ti.template(),
    dx:float,
    particles_active:ti.template()
):
    ii,jj = ibm_boundary_mask.shape
    for I in particles_pos:
        if(particles_active[I] == 1):
            idx = ti.floor(particles_pos[I]/dx,int)
            if(idx[0]>=0 and idx[1]>=0 and idx[0]<ii and idx[1]<jj and ibm_boundary_mask[idx] >=1):
                new_h = ibm_boundary_h[idx]
                F = ti.Matrix.rows([F_x[I], F_y[I]])
                T = ti.Matrix.cols([T_x[I], T_y[I]])
                particles_init_h[I]=particles_init_h[I]+ti.math.determinant(F)*(new_h - particles_h[I])
                particles_h[I] = new_h

@ti.kernel
def set_grid_boundary_h(boundary_mask:ti.template(),h:ti.template()):
    for i,j in boundary_mask:
        if(boundary_mask[i,j] == 1):
            h[i,j]=1.0
        elif(boundary_mask[i,j] == 2):
            if(case == 1):
                h[i,j]=1.0                 
        
@ti.kernel
def mask_by_boundary(field: ti.template()):
    for I in ti.grouped(field):
        if boundary_mask[I] > 0:
            field[I] *= 0

        
@ti.kernel
def set_h_dis(h: ti.template(),h_dis: ti.template(),ibm_boundary_mask: ti.template()):
    for I in ti.grouped(h):
        if(ibm_boundary_mask[I]>=1):
            h_dis[I]=10
        else:
            h_dis[I]=h[I]

@ti.kernel
def calculate_new_pos(particle_pos:ti.template(),curr_dt:float):
    for I in ti.grouped(particle_pos):
        particle_pos[I]+= ti.Vector([boat_v,0.0])*curr_dt


@ti.kernel
def source_set(
    h:ti.template(),
    u_x:ti.template(),
    u_y:ti.template(),
    boundary_mask:ti.template(),
    dt:float,
    t:float
):
    for i,j in boundary_mask:
        if(boundary_mask[i,j]==2):
            if(case == 3):
                T = 1.0
                #sv = 1*(2*ti.sin(2*3.14/T))
                hT = init_h+0.05*ti.sin(2*3.14/T*t)
                #u_x[i,j] = sv*(1+0.1*(ti.random()-0.5))
                #u_x[i+1,j] = sv*(1+0.1*(ti.random()-0.5))
                #u_y[i,j] = 0.02*(ti.random()-0.5)
                #u_y[i,j+1] = 0.02*(ti.random()-0.5)
                if(i<10):
                    h[i,j] = hT
                    #h[i,j]+=h[i,j]*dt*sv
                else:
                    pass
                    #h[i,j]-=h[i,j]*dt*sv

            elif(case == 4):
                h[i,j] = init_h

            elif(case == 7):
                h[i,j] = init_h
                u_x[i,j] =u_x[51,j]
            
            elif(case == 8):
                h[i,j] = init_h
                u_x[i,j] =u_x[51,j]

            elif(case == 9):
                h[i,j] = init_h
                u_x[i,j] =u_x[51,j]

            elif(case == 10):
                h[i,j] = init_h
            
            elif(case == 12):
                T = 1.0
                hT = init_h+0.05*ti.sin(2*3.14/T*t)
                if(i<10):
                    h[i,j] = hT
                else:
                    pass




    

# main function
def main(from_frame=0, testing=False):
    from_frame = max(0, from_frame)
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vortdir = 'vorticity'
    vortdir = os.path.join(logsdir, vortdir)
    os.makedirs(vortdir, exist_ok=True)
    
    hdir = 'h'
    hdir = os.path.join(logsdir, hdir)
    os.makedirs(hdir, exist_ok=True)

    vtkdir = 'vtk'
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)

    objdir = 'obj'
    objdir = os.path.join(logsdir, objdir)
    os.makedirs(objdir, exist_ok=True)

    h_render_dir = "h_exr"
    h_render_dir = os.path.join(logsdir, h_render_dir)
    os.makedirs(h_render_dir, exist_ok=True)

    smokedir = 'smoke'
    smokedir = os.path.join(logsdir, smokedir)
    os.makedirs(smokedir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    levelsdir = 'levels'
    levelsdir = os.path.join(logsdir, levelsdir)
    os.makedirs(levelsdir, exist_ok=True)
    modeldir = 'model'  # saves the model
    modeldir = os.path.join(logsdir, modeldir)
    os.makedirs(modeldir, exist_ok=True)
    velocity_buffer_dir = 'velocity_buffer'
    velocity_buffer_dir = os.path.join(logsdir, velocity_buffer_dir)
    os.makedirs(velocity_buffer_dir, exist_ok=True)
    particles_dir = 'particles'
    particles_dir = os.path.join(logsdir, particles_dir)
    os.makedirs(particles_dir, exist_ok=True)

    shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')

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
    
    _t = 0.0
    if from_frame <= 0: 
        if(case == 3):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_3(h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
           
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 4):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_4(h,ibm_boundary_mask,ibm_boundary_h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
            
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 5):
            ibm_boundary_mask.fill(0.0)
            eta.fill(0.0)
            wind_u_x.fill(0.0)
            wind_u_y.fill(0.0)
            boundary_mask.fill(0.0)
            h.fill(init_h)
            
            boundary_vel = ti.Vector.field(2, float, shape=(res_x, res_y))
            boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)
            solver = MGPCG_2(boundary_types, boundary_mask, boundary_vel, N = [res_x, res_y])

            u_x.fill(0.0)
            u_y.fill(0.0)
            four_vortex_vel_func(u, dx)
            h.fill(init_h)
            split_central_vector(u,u_x,u_y)
            solver.Poisson(u_x, u_y, tem_p, dx)

        elif(case == 6):
            ibm_boundary_mask.fill(0.0)
            eta.fill(0.0)
            wind_u_x.fill(0.0)
            wind_u_y.fill(0.0)
            boundary_mask.fill(0.0)
            h.fill(init_h)
            
            boundary_vel = ti.Vector.field(2, float, shape=(res_x, res_y))
            boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)
            solver = MGPCG_2(boundary_types, boundary_mask, boundary_vel, N = [res_x, res_y])

            u_x.fill(0.0)
            u_y.fill(0.0)
            if(suboption == 0):
                three_vortex_vel_func(u,dx)
            else:
                two_vortex_vel_func(u,dx)
            h.fill(init_h)
            split_central_vector(u,u_x,u_y)
            solver.Poisson(u_x, u_y, tem_p, dx)

            if(laden_particles_on):
                init_laden_particles(laden_particles_pos,laden_particles_vel)

        elif(case == 7):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_7(h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
           
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 8):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_8(h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
           
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 9):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_9(h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
           
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 10):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_10(h,ibm_boundary_mask,ibm_boundary_h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
            
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 11):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_11(h,ibm_boundary_mask,ibm_boundary_h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
            
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

        elif(case == 12):
            #init_bottom(eta,dx)
            moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
            set_case_12(h,ibm_boundary_mask,ibm_boundary_h,eta,u_x,u_y,wind_u_x,wind_u_y,boundary_mask)  
            
            write_levelset_field(ibm_boundary_mask.to_numpy(), "./", particles_pos, vmin=0.0, vmax=1.0,  dpi=512//8)
            init_velocity_by_ibm_boundary_mask(
                ibm_boundary_mask,
                ibm_boundary_velocity,
                u_x,u_y
            )
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_flip_particles(ibm_boundary_mask_extend, flip_particles_pos, flip_particles_vel, flip_particles_life, flip_particles_type, flip_particles_num,flip_particles_h,h,dx)

    else:
        u_x.from_numpy(np.load(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))

    # for visualization
    get_central_vector(u_x, u_y, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_max = max(np.abs(w_numpy.max()), np.abs(w_numpy.min()))
    w_min = -1 * w_max
    write_field(w_numpy, vortdir, from_frame, particles_pos.to_numpy() / dx, vmin=w_min,
                vmax=w_max,
                plot_particles=plot_particles, dpi=dpi_vor)
    
    write_image(smoke.to_numpy(), smokedir, from_frame)

    set_vis_h(vis_h,vis_u,vis_u_3D,u_x,u_y, h,eta,flip_particles_type,flip_particles_pos,flip_particles_num, dx)
    #write_vol_h_vel(vis_h.to_numpy(),vis_u.to_numpy(),vtkdir,  from_frame)
    write_vol_h(vis_h.to_numpy(),vtkdir,  from_frame)
    h_numpy = h.to_numpy()
    write_h_field(h_numpy, hdir, from_frame, particles_pos.to_numpy() / dx, vmin=0.65*init_h, vmax=1.35*init_h,
                        plot_particles=plot_particles, dpi=dpi_vor)

    if save_ckpt:
        np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
        np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
        np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke.to_numpy())

    curr_t= 0 
    sub_t = 0.  # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0  # number of reinitializations already performed
    i = -1

    frame_times = np.zeros(total_steps)
    prev_dt = 0
    while True:
        i += 1
        j = i % reinit_every
        k = i % reinit_every_grad_m
        i_next = i + 1
        j_next = i_next % reinit_every

        # determine dt
        if(use_cfl):
            calc_max_speed(u_x, u_y)  # saved to max_speed[None]
            curr_dt = ti.min(CFL * dx / max_speed[None],CFL*dx/(real_gravity*init_h)**0.5)
            print("cfl dt", curr_dt )
        else:
            curr_dt = fixed_dt
            calc_max_speed(u_x, u_y)  # saved to max_speed[None]
            tem_dt = ti.min(CFL * dx / max_speed[None],CFL*dx/(real_gravity*init_h)**0.5)
            if(tem_dt<curr_dt):
                curr_dt = tem_dt
            print("cfl dt", tem_dt)

        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t + curr_dt >= visualize_dt:  # if over
                curr_dt = visualize_dt - sub_t
                sub_t = 0.  # empty sub_t
                frame_idx += 1

                output_frame = True

            else:
                sub_t += curr_dt
                print(f'Visualize time {sub_t}/{visualize_dt}')
                output_frame = False

        if j == 0:
            if use_reseed_particles:
                pass
            else:
                if reinit_particle_pos:
                    # pass
                    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                                               particles_per_cell_axis, dist_between_neighbor)
            mask_particles()
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y,  dx)
            g2p_scalar(h, particles_init_h, particles_pos, dx)

            reset_T_to_identity(T_x, T_y)
            reset_F_to_identity(F_x, F_y)
            particles_grad_lamb.fill(0.0)
            num_reinits += 1
        
        copy_to(u_x,save_u_x)
        copy_to(u_y,save_u_y)
        
        add_pressure_force()

        if use_midpoint_vel:
            advect_u_grid(save_u_x, save_u_y, u_x, u_y, dx, 0.5 * curr_dt, X_horizontal,X_vertical)
            add_gravity_force_water(u_x, u_y, f_x, f_y, h, eta, acc_int_tem, ibm_boundary_mask, ibm_boundary_velocity,ibm_boundary_h, ibm_boundary_coef,ibm_force_y, dx, 0.5* curr_dt)
            add_drag_force(u_x, u_y,wind_u_x,wind_u_y,f_x, f_y, h,0.5 * curr_dt)
            enforce_boundary(u_x, u_y, h)
            set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
            source_set(h,u_x,u_y,boundary_mask,0.5 * curr_dt,_t)

        get_grid_usqure(u_x, u_y, u_square,dx)
        compute_particle_grad_u(particles_pos, save_u_x, save_u_y, C_x, C_y, dx, particles_active)        
        compute_particle_grad_scalar(particles_pos,h,grad_h,dx,particles_active)
        stretch_FT_and_advect_particles(particles_pos, T_x, T_y, F_x, F_y, u_x, u_y, curr_dt, dx, particles_active)
        mask_particles()
        moving_ibm_obstacle(ibm_boundary_mask,ibm_boundary_mask_extend,ibm_boundary_h, ibm_boundary_coef, ibm_boundary_velocity,y_sum,new_c,_t)
        calculate_particle_u_square(particles_pos, particles_grad_half_usquare, u_square, curr_dt, dx, particles_active)

        update_particles_water(particles_imp, particles_init_imp, particles_grad_lamb, particles_grad_half_usquare, \
            particles_h, particles_init_h, T_x, T_y,particles_active)        

        P2G_new(particles_imp, particles_pos, u_x, u_y, C_x, C_y, p2g_weight_x, p2g_weight_y, dx, particles_active)
        P2G_height_water(particles_h, grad_h,  particles_pos, h,  p2g_weight,dx, particles_active)
        set_boundary_particle(
            particles_pos,
            particles_h,
            particles_init_h,
            ibm_boundary_mask,
            ibm_boundary_h,
            T_x, 
            T_y,
            F_x, 
            F_y,
            dx,
            particles_active
        )
        add_gravity_force_water(u_x, u_y,f_x, f_y, h, eta, acc_int_tem, ibm_boundary_mask, ibm_boundary_velocity, ibm_boundary_h, ibm_boundary_coef,ibm_force_y,dx, curr_dt)    
        add_drag_force(u_x, u_y,wind_u_x,wind_u_y,f_x, f_y, h,curr_dt)
        enforce_boundary(u_x, u_y, h)
        set_grid_boundary(boundary_mask,u_x,u_y,ibm_boundary_mask,ibm_boundary_h, ibm_boundary_coef,h)
        accumulate_force_water(particles_grad_lamb, particles_pos, particles_grad_half_usquare,
                        F_x, F_y,f_x,f_y, dx, particles_active)

        source_set(h,u_x,u_y,boundary_mask,curr_dt,_t) 
        advect_passive_particles(
            flip_particles_pos, flip_particles_vel, flip_particles_life,
            flip_particles_type, flip_particles_num, ibm_boundary_mask_extend,u_x, u_y,
            flip_particles_h,h, curr_dt)   
        
        reseed_passive_particles(
            flip_particles_pos,
            flip_particles_vel,
            flip_particles_life,
            flip_particles_type,
            flip_particles_h,
            flip_particles_pos_new,
            flip_particles_vel_new,
            flip_particles_life_new,
            flip_particles_type_new,     
            flip_particles_h_new,
            flip_particles_num,
            u_x,
            u_y,
            ibm_boundary_mask_extend,
            grid_particles_num,
            grid_particles_base,
            flip_particle_delete_flag,
            
            h
        )       
        prev_dt = curr_dt
        _t+=curr_dt
        if(print_log):
            print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush=True)
        curr_t+=curr_dt
        if(laden_particles_on):
            advect_laden_particles( laden_particles_pos, laden_particles_vel, u_x, u_y, h, dx, curr_dt,curr_t)
        if output_frame:
            # for visualization
            get_central_vector(u_x, u_y, u)
            # write_image(levels_display[..., np.newaxis], levelsdir, frame_idx)
            curl(u, w, dx)
            w_numpy =w.to_numpy()
            write_field(w_numpy, vortdir, frame_idx, particles_pos.to_numpy() / dx, vmin=w_min, vmax=w_max,
                        plot_particles=plot_particles, dpi=dpi_vor)
            
            #set_h_dis(h,h_dis,ibm_boundary_mask)
            h_numpy = h.to_numpy()

            write_h_field(h_numpy, hdir, frame_idx, particles_pos.to_numpy() / dx, vmin=0.65*init_h, vmax=1.35*init_h,
                        plot_particles=plot_particles, dpi=dpi_vor)

            write_image(smoke.to_numpy(), smokedir, frame_idx)
            write_render_h(h_numpy, h_render_dir, frame_idx)
            #set_vis_type(flip_particles_type,flip_particles_vis_type)
            if(laden_particles_on):
                write_laden_particles(
                    laden_particles_pos.to_numpy(),                    
                    particles_dir,  frame_idx, laden_particles_num
                )
            """write_flip_particles(
                flip_particles_pos.to_numpy(),
                flip_particles_vel.to_numpy(),
                flip_particles_type.to_numpy(),
                flip_particles_vis_type.to_numpy(),
                flip_particles_life.to_numpy(),
                particles_dir,  frame_idx, flip_particles_num[None]
            )"""
            set_vis_h(vis_h,vis_u,vis_u_3D,u_x,u_y, h,eta,flip_particles_type,flip_particles_pos,flip_particles_num, dx)
            #curl_3D(vis_u_3D, vis_w_3D, vol_dx_xy, vol_dx_z)
            #w_3D_numpy = vis_w_3D.to_numpy()
            #w_3D_norm = np.linalg.norm(w_3D_numpy, axis = -1)
            #write_vol_h_w(vis_h.to_numpy(),w_3D_norm,vtkdir,  frame_idx)
            write_vol_h(vis_h.to_numpy(),vtkdir,  frame_idx)

            write_flip_particles_render(
                flip_particles_pos.to_numpy(),
                flip_particles_type.to_numpy(),
                flip_particles_life.to_numpy(),
                particles_dir,  frame_idx, flip_particles_num[None]
            )

            if frame_idx % ckpt_every == 0 and save_ckpt:
                np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                np.save(os.path.join(ckptdir, "w_numpy_" + str(frame_idx)), w.to_numpy())

            
            print("\n[Simulate] Finished frame: ", frame_idx, " in ", i - last_output_substep, "substeps \n\n")
            last_output_substep = i

            # if reached desired number of frames
            if frame_idx >= total_frames:
                break

        if use_total_steps and i >= total_steps - 1:
            frame_time_dir = 'frame_time'
            frame_time_dir = os.path.join(logsdir, frame_time_dir)
            os.makedirs(f'{frame_time_dir}', exist_ok=True)
            np.save(f'{frame_time_dir}/frame_times.npy', frame_times)
            break


if __name__ == '__main__':
    print("[Main] Begin")
    if len(sys.argv) <= 1:
        main(from_frame=from_frame)
    else:
        main(from_frame=from_frame, testing=testing)
    print("[Main] Complete")