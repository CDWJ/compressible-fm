#case = 3 # fish tail like Clebsch free surface
#case = 4 # single boat
#case = 5 # leapfrog
#case = 6 # two vortex rotation
#case = 7 # karman vortex
#case = 8 # swing karman vortex, fast
#case = 9 # swing karman vortex, slow
#case = 10 # three boat
#case = 11 # single boat with S
#case = 12 # multi lighthouse


# # some hyperparameters
case = 10
euler_method = False
advanced_euler_method = False
suboption = 0

if case == 3:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 2.4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1200
    res_y = 600*3
    dx = 3. / res_y
    init_h = 1.0
    visualize_dt =0.02# 0.1#0.05
    reinit_every = 40
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 0.25
    use_cfl = False
    fixed_dt= 0.0001
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "rotating_triangle2" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 1
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 1000
    vertical_velocity_limit = 1000
    foam_init_height = 1.0/128 /4
    life_time_init = 1.0
    drag_coef = 0.1

    laden_particles_on = False

elif case == 4:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 2.4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1800
    res_y = 1800
    dx = 3. / res_y
    init_h = 0.4
    visualize_dt =0.05# 0.1#0.05
    reinit_every = 80
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 0.25
    use_cfl = False
    fixed_dt= 0.00005
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "single_boat_higher" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128/2

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 10
    vertical_velocity_limit = 0.1
    foam_init_height =0#1.0/128*1/4
    life_time_init = 1.0
    drag_coef = 0.1

    laden_particles_on = False

elif case == 5:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 0.8#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1024
    res_y = 512
    dx = 1. / res_y
    init_h = 0.5
    visualize_dt =0.2# 0.1#0.05
    reinit_every = 160
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 1.0
    if(euler_method):
        CFL = 0.5
    use_cfl = True
    fixed_dt= 0.00005
    if(advanced_euler_method):
        use_cfl = False
        CFL = 0.5
        fixed_dt= 0.00005
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "leapfrog" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 1000
    vertical_velocity_limit = 1000
    foam_init_height = 1.0/128 /4
    life_time_init = 1.0
    drag_coef = 0.1

    laden_particles_on = False

elif case == 6:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 0.8#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 512
    res_y = 512
    dx = 1. / res_y
    init_h = 0.5 #0.5
    visualize_dt =0.2#0.2# 0.1#0.05
    reinit_every = 160
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 1.0
    if(euler_method):
        CFL = 0.5    
    use_cfl = True
    fixed_dt= 0.0001
    if(advanced_euler_method):
        use_cfl = False
        fixed_dt= 0.0001
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    
    if(suboption == 0):
        exp_name = "three_vortex" # prev: 0.2
    else:
        exp_name = "two_vortex" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 1000
    vertical_velocity_limit = 1000
    foam_init_height = 1.0/128 /4
    life_time_init = 1.0
    drag_coef = 0.1

    laden_particles_on = False
    laden_gravity = 0.0
    laden_drag = 1.0

elif case == 7:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 0.8*4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1024*3
    res_y = 512*2
    dx = 2. / res_y
    init_h = 0.5#1.0
    visualize_dt =0.05# 0.1#0.05
    reinit_every = 20
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 1.0
    use_cfl = False
    fixed_dt= 0.001
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "Karman2" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 4#1000
    vertical_velocity_limit = 1.0#1000
    foam_init_height = 1.0/128/4.0
    life_time_init = 5.0/2/1.5
    drag_coef = 0.1

    laden_particles_on = False

elif case == 8:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 0.8*4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1024*3
    res_y = 512*2
    dx = 2. / res_y
    init_h = 0.5#1.0
    visualize_dt =0.05# 0.1#0.05
    reinit_every = 20
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 1.0
    use_cfl = False
    fixed_dt= 0.001
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "Karman_vibrant_fast" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 1000
    vertical_velocity_limit = 1000
    foam_init_height = 1.0/128/4
    life_time_init = 5.0
    drag_coef = 0.1

    laden_particles_on = False

elif case == 9:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 0.8*4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1024*3
    res_y = 512*2
    dx = 2. / res_y
    init_h = 0.5#1.0
    visualize_dt =0.05# 0.1#0.05
    reinit_every = 20
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 1.0
    use_cfl = False
    fixed_dt= 0.001
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "Karman_vibrant_slow" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 1000
    vertical_velocity_limit = 1000
    foam_init_height = 1.0/128 /4
    life_time_init = 5.0
    drag_coef = 0.1

    laden_particles_on = False

elif case == 10:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 2.4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1800
    res_y = 1800
    dx = 5 / res_y
    init_h = 0.4
    visualize_dt =0.015# 0.1#0.05
    reinit_every = 70
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 0.25
    use_cfl = False
    fixed_dt= 0.00005
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 1000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "three_boat_higher_3" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 10
    vertical_velocity_limit = 0.05
    foam_init_height =0#1.0/128*1/4
    life_time_init = 4
    drag_coef = 0.1

    laden_particles_on = False

elif case == 11:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 2.4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 3600
    res_y = 900
    dx = 1.5 / res_y
    init_h = 0.4
    visualize_dt =0.05# 0.1#0.05
    reinit_every = 80
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 0.25
    use_cfl = False
    fixed_dt= 0.00005
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "single_boat_S" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 4
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128/2

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 10
    vertical_velocity_limit = 0.1
    foam_init_height =0#1.0/128*1/4
    life_time_init = 1.0
    drag_coef = 0.1

    laden_particles_on = False

elif case == 12:
    dim = 2
    testing = False
    use_neural = False
    save_ckpt = True
    save_frame_each_step = False
    use_BFECC = False
    use_midpoint_vel = True
    use_APIC = True
    forward_update_T = True
    plot_particles = False
    use_reseed_particles = False
    reinit_particle_pos = True
    use_diff_init_cfl = False
    dpi_vor = 512 if plot_particles else 512 // 8
    print_log =   False

    ibm_coef = 1.0
    real_gravity = 2.4#4.8#1.8#2.4#1.2#0.8#(1.4 - 1)
    # encoder hyperparameters
    min_res = (128, 256)
    num_levels = 4
    feat_dim = 2
    activate_threshold = 0.03
    # neural buffer hyperparameters
    N_iters = 2000
    N_batch = 40000 #25000
    success_threshold = 3.e-8
    # simulation hyperparameters
    gamma = 1
    res_x = 1800
    res_y = 1800
    dx = 3. / res_y
    init_h = 0.4
    visualize_dt =0.02# 0.1#0.05
    reinit_every = 40
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 0.25
    use_cfl = False
    fixed_dt= 0.00005
    viscosity=0.05/9500
    from_frame = 0
    total_frames = 10000
    use_total_steps = False
    
    total_steps = 500
    exp_name = "lighthouse" # prev: 0.2
    init_experiment_name = "" #

    vis_num=1
    lamb = 1.0

    particles_per_cell = 1
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

    npc = 64
    npc_tl = 2
    h_grid_size = 0.1
    pp_grid_ratio = 3
    pp_res_x = res_x // pp_grid_ratio
    pp_res_y = res_y // pp_grid_ratio
    pp_res_z = 3
    pp_dx_xy = dx*pp_grid_ratio    
    pp_dx_z = dx*3
    boarder_ratio = 0.03
    spray_bandwidth = 1.0/128

    vol_grid_ratio = 5
    vol_res_x = res_x // pp_grid_ratio
    vol_res_y = res_y // pp_grid_ratio
    vol_dx_xy = dx*pp_grid_ratio   
    vol_dx_z = vol_dx_xy 
    vol_z_base = 0.2*init_h
    vol_z_max = 1.3*init_h
    vol_res_z = int((vol_z_max-vol_z_base)/vol_dx_z)

    flip_gravity_coef = 1000
    vertical_velocity_limit = 1000
    foam_init_height = 1.0/128 /4
    life_time_init = 1.0
    drag_coef = 0.1

    laden_particles_on = False