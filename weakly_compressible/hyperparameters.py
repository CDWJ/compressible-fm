
use_midpoint_vel = True
forward_update_T = True
use_APIC = True
use_APIC_smoke = True
save_particle_pos_numpy = False
save_frame_each_step = False
reinit_particle_pos = True

gravity_fake = 0.8#1.1#10.0
gamma = 1

# encoder hyperparameters
min_res = (16, 16, 16)
num_levels = 4
feat_dim = 4
activate_threshold = 0.028 * 13
# neural buffer hyperparameters
N_iters = 3000
N_batch = 80000
success_threshold = 2e-6
num_chunks = 2 # query buffer in chunks (as small as machine memory permits)
# simulation hyperparameters
res_x = 128
res_y = 256
res_z = 128
visualize_dt = 0.0075
reinit_every = 9
reinit_every_grad_m = 2
ckpt_every = 1
CFL = 0.5
from_frame = 0
total_frames = 300
BFECC_clamp = True
use_total_steps = False
total_steps = 1

particles_per_cell = 8
total_particles_num_ratio = 1
cell_max_particle_num_ratio = 1.6

mach_number = 1

res_x = 128
res_y = 256
res_z = 128

domain_range_y = 256.0
domain_range_x = domain_range_y/res_y*res_x
domain_range_z = domain_range_y/res_y*res_z
dx = domain_range_y/res_y
Re = 10  # 4
vis_coef = 1/Re
# Re=5.5#8
# Re=5.0#6
St = 6.6e-5
gravity = 1
density_ratio = 2.0/3
epiral_radius_ratio = 35
laden_particles_max_num = 1000000
compute_ratio = 1
laden_particles_init_num = int(laden_particles_max_num/compute_ratio**3)
laden_radius = 0.01
compute_laden_radius = compute_ratio*laden_radius

visualize_dt = 10
reinit_every = 25
ckpt_every = 1
CFL = 0.25
from_frame = 0
total_frames = 800
BFECC_clamp = True
exp_name = "ink-garmed4fu4"
ed_eps = 0.02 * dx
lp = True
lp_passive= False
solid = False
moving_solid = False
