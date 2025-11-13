case = 1
# simulation hyperparameters
if(case == 0):
    res_x = 128
    res_y = 128
    res_z = 256
    visualize_dt = 0.1
    reinit_every = 20
    ckpt_every = 10
    CFL = 0.25
    from_frame = 0
    total_frames = 500
    BFECC_clamp = True
    exp_name = "four_vortices_wc_3"

    # learning hyperparameters
    min_res = (16, 16, 32) # encoder base resolutions
    num_levels = 4 # number of refining levels
    feat_dim = 4 # feature vector size (per anchor vector)
    activate_threshold = 0.045 # smaller means more cells are activated
    N_iters = 200
    N_batch = 240000
    success_threshold = 2e-6 # smaller means later termination
    num_chunks = 2 # query buffer in chunks (as small as machine memory permits)

    gravity = 0.2#1.1#10.0
    gamma = 1
elif(case == 1):
    use_midpoint_vel = True
    forward_update_T = True
    use_APIC = True
    use_APIC_smoke = True
    save_particle_pos_numpy = False
    save_particle_imp_numpy = False
    save_frame_each_step = True
    reinit_particle_pos = True

    gravity = 0.1#1.1#10.0
    gamma = 500
    mach_number = 2
    acceleration=0
    ibm_coeff=2

    # encoder hyperparameters
    min_res = (32, 16, 16)
    num_levels = 4
    feat_dim = 4
    activate_threshold = 0.032
    # neural buffer hyperparameters
    N_iters = 1500
    N_batch = 240000
    success_threshold = 2.e-7  # 5.e-8
    num_chunks = 2 # query buffer in chunks (as small as machine memory permits)
    # simulation hyperparameters
    res_x = 400
    res_y = 200
    res_z = 200
    visualize_dt = 0.05
    reinit_every = 8  # 12 #20
    reinit_every_grad_m = 1
    ckpt_every = 1
    CFL = 0.4
    from_frame = 0  # 0
    total_frames = 500
    BFECC_clamp = False
    use_total_steps = False
    total_steps = 1
    move_vel = 0.0
    use_cfl = True
    fixed_dt = 0.00001
    exp_name = "3D_lander_2"

    particles_per_cell = 8
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

elif case == 2:
    use_midpoint_vel = True
    forward_update_T = True
    use_APIC = True
    use_APIC_smoke = True
    save_particle_pos_numpy = True
    save_frame_each_step = False
    reinit_particle_pos = False

    gravity = 0.4#1.1#10.0
    gamma = 1

    # encoder hyperparameters
    min_res = (16, 16, 16)
    num_levels = 4
    feat_dim = 4
    activate_threshold = 0.036
    # neural buffer hyperparameters


    N_iters = 3000
    N_batch = 80000
    success_threshold = 1.5e-7
    num_chunks = 2 # query buffer in chunks (as small as machine memory permits)
    # simulation hyperparameters
    res_x = 128
    res_y = 128
    res_z = 128
    visualize_dt = 0.05
    reinit_every = 12
    reinit_every_grad_m = 4
    ckpt_every = 1
    CFL = 0.1
    from_frame = 0
    total_frames = 800
    exp_name = "3D_oblique_wc_nfm"
    BFECC_clamp = True
    use_total_steps = False
    total_steps = 1

    particles_per_cell = 8
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6

elif case == 3:
    use_midpoint_vel = True
    forward_update_T = True
    use_APIC = True
    use_APIC_smoke = True
    save_particle_pos_numpy = False
    save_frame_each_step = False
    reinit_particle_pos = True

    gravity = 5#1.1#10.0
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
    res_y = 128
    res_z = 128
    visualize_dt = 0.0075
    reinit_every = 9
    reinit_every_grad_m = 2
    ckpt_every = 1
    CFL = 0.5
    from_frame = 0
    total_frames = 800
    exp_name = "3D_trefoil_wc_nfm"
    BFECC_clamp = True
    use_total_steps = False
    total_steps = 1

    particles_per_cell = 8
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6