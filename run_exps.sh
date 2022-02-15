
##########################
######### D = 20 #########
##########################

# baselines
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp default
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --exp default
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_latent_path --exp default
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --exp default

# unseen_amp (needs investigation what about 1D?)
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp unseen_amp
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --exp unseen_amp
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_latent_path --exp unseen_amp
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --exp unseen_amp

# unseen_shift -- does not fit ??
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp unseen_shift
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --attn_det_path --exp unseen_shift
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --attn_latent_path --exp unseen_shift
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --exp unseen_shift

# # unseen_freq
# CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp unseen_freq
# CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --attn_det_path --exp unseen_freq
# CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --attn_latent_path --exp unseen_freq
# CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --exp unseen_freq

# context_1_3
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp context_1_3
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --attn_det_path --exp context_1_3
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --attn_latent_path --exp context_1_3
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512   --dim 20 --lr 3e-4 --exp context_1_3

# context_1_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp context_1_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --attn_det_path --exp context_1_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --attn_latent_path --exp context_1_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512  --dim 20 --lr 3e-4 --exp context_1_10

# context_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp context_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --exp context_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_latent_path --exp context_10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --exp context_10

# context_1_50
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp context_1_50
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --exp context_1_50
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_latent_path --exp context_1_50
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --exp context_1_50

# context_50
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --attn_latent_path --exp context_50
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_det_path --exp context_50
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --attn_latent_path --exp context_50
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 20 --lr 3e-4 --exp context_50

#########################
######### D = 1 #########
#########################
# baselines
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --attn_latent_path --exp default
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --exp default
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_latent_path --exp default
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --exp default

# unseen_amp
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --attn_latent_path --exp unseen_amp
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --exp unseen_amp
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_latent_path --exp unseen_amp
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --exp unseen_amp

# unseen_shift
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --attn_latent_path --exp unseen_shift
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --exp unseen_shift
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_latent_path --exp unseen_shift
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --exp unseen_shift

# # unseen_freq
# CUDA_VISIBLE_DEVICES=3 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --attn_latent_path --exp unseen_freq
# CUDA_VISIBLE_DEVICES=3 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_det_path --exp unseen_freq
# CUDA_VISIBLE_DEVICES=3 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --attn_latent_path --exp unseen_freq
# CUDA_VISIBLE_DEVICES=3 python npc/scripts/train_np.py -wb --max_steps 10000 --bsize 32  --z_dim 16 --r_dim 16 --h_dim 512 --dim 1 --lr 3e-4 --exp unseen_freq



# unseen amp, D=10
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 10 --lr 3e-4 --attn_det_path --attn_latent_path --exp amp_seen
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 10 --lr 3e-4 --attn_det_path --attn_latent_path --exp amp_unseen

# unseen amp, D=5
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 5 --lr 3e-4 --attn_det_path --attn_latent_path --exp amp_seen
CUDA_VISIBLE_DEVICES=0 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 5 --lr 3e-4 --attn_det_path --attn_latent_path --exp amp_unseen

# unseen amp, D=2
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 2 --lr 3e-4 --attn_det_path --attn_latent_path --exp amp_seen
CUDA_VISIBLE_DEVICES=1 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 2 --lr 3e-4 --attn_det_path --attn_latent_path --exp amp_unseen

##################

# unseen shift, D=10
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 10 --lr 3e-4 --attn_det_path --attn_latent_path --exp shift_seen
CUDA_VISIBLE_DEVICES=2 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 10 --lr 3e-4 --attn_det_path --attn_latent_path --exp shift_unseen

# unseen shift, D=5
CUDA_VISIBLE_DEVICES=3 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 5 --lr 3e-4 --attn_det_path --attn_latent_path --exp shift_seen
CUDA_VISIBLE_DEVICES=3 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 5 --lr 3e-4 --attn_det_path --attn_latent_path --exp shift_unseen

# unseen shift, D=2
CUDA_VISIBLE_DEVICES=4 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 2 --lr 3e-4 --attn_det_path --attn_latent_path --exp shift_seen
CUDA_VISIBLE_DEVICES=4 python npc/scripts/train_np.py -wb --max_steps 20000 --bsize 512  --z_dim 16 --r_dim 16 --h_dim 512 --dim 2 --lr 3e-4 --attn_det_path --attn_latent_path --exp shift_unseen


