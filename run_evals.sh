# unseen amp, D=10
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp amp_seen   --ckpt logs/np/3rnwi657/checkpoints/epoch\=4935-valid_elbo_epoch\=3.982.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp amp_unseen --ckpt logs/np/tmf8puzz/checkpoints/epoch\=4649-valid_elbo_epoch\=3.572.ckpt

# unseen amp, D=5
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5 --exp amp_seen   --ckpt logs/np/1fyg7ee4/checkpoints/epoch\=4441-valid_elbo_epoch\=4.042.ckpt
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5 --exp amp_unseen --ckpt logs/np/qayiflz0/checkpoints/epoch\=4828-valid_elbo_epoch\=3.610.ckpt

# unseen amp, D=2
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2 --exp amp_seen   --ckpt logs/np/391yasyw/checkpoints/epoch\=4959-valid_elbo_epoch\=3.997.ckpt
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2 --exp amp_unseen --ckpt logs/np/2wbyq1ui/checkpoints/epoch\=4754-valid_elbo_epoch\=3.765.ckpt

##################

# unseen shift, D=10
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp shift_seen   --ckpt logs/np/onlo2jcn/checkpoints/epoch\=1905-valid_elbo_epoch\=3.549.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp shift_unseen --ckpt logs/np/1hl5hurx/checkpoints/epoch\=1318-valid_elbo_epoch\=1.092.ckpt 

# unseen shift, D=5
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5 --exp shift_seen    --ckpt logs/np/18tha5uw/checkpoints/epoch\=1936-valid_elbo_epoch\=3.893.ckpt
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5 --exp shift_unseen  --ckpt logs/np/okqoeyrn/checkpoints/epoch\=1884-valid_elbo_epoch\=1.899.ckpt

# unseen shift, D=2
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2 --exp shift_seen    --ckpt logs/np/1xbwho0w/checkpoints/epoch\=1816-valid_elbo_epoch\=3.825.ckpt
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2 --exp shift_unseen  --ckpt logs/np/1autywfx/checkpoints/epoch\=1866-valid_elbo_epoch\=1.844.ckpt



################## scaling pre-training dataset size

################## scaling the number of random functions
# D=10
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_5000_pts_100    --ckpt logs/np/gkvaa62k/checkpoints/epoch\=1944-valid_elbo_epoch\=3.747.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_500_pts_100     --ckpt logs/np/1bdxeok8/checkpoints/epoch\=19519-valid_elbo_epoch\=3.425.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_250_pts_100     --ckpt logs/np/2duvy4rz/checkpoints/epoch\=19287-valid_elbo_epoch\=3.463.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_100_pts_100     --ckpt logs/np/1fkl79jw/checkpoints/epoch\=19209-valid_elbo_epoch\=2.888.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_50_pts_100      --ckpt logs/np/b21igdfn/checkpoints/epoch\=554-valid_elbo_epoch\=-0.215.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_25_pts_100      --ckpt logs/np/2kk8704c/checkpoints/epoch\=212-valid_elbo_epoch\=-1.149.ckpt

# D=5
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 5 --exp fns_5000_pts_100     --ckpt logs/np/4jqlz95v/checkpoints/epoch\=1936-valid_elbo_epoch\=3.714.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 5 --exp fns_500_pts_100      --ckpt logs/np/bqykof17/checkpoints/epoch\=19322-valid_elbo_epoch\=3.702.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 5 --exp fns_250_pts_100      --ckpt logs/np/3t982cqb/checkpoints/epoch\=18387-valid_elbo_epoch\=3.435.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 5 --exp fns_100_pts_100      --ckpt logs/np/31i4ry1a/checkpoints/epoch\=18801-valid_elbo_epoch\=3.536.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 5 --exp fns_50_pts_100       --ckpt logs/np/2bw1j5ka/checkpoints/epoch\=19807-valid_elbo_epoch\=3.034.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 5 --exp fns_25_pts_100       --ckpt logs/np/3mtsb48a/checkpoints/epoch\=6404-valid_elbo_epoch\=1.322.ckpt

# D=2
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 2 --exp fns_5000_pts_100     --ckpt logs/np/3v3v5c15/checkpoints/epoch\=1823-valid_elbo_epoch\=3.626.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 2 --exp fns_500_pts_100      --ckpt logs/np/2xaqwwgd/checkpoints/epoch\=19082-valid_elbo_epoch\=3.801.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 2 --exp fns_250_pts_100      --ckpt logs/np/12sde6g8/checkpoints/epoch\=19732-valid_elbo_epoch\=3.583.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 2 --exp fns_100_pts_100      --ckpt logs/np/32ytpgcl/checkpoints/epoch\=19463-valid_elbo_epoch\=3.309.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 2 --exp fns_50_pts_100       --ckpt logs/np/ernu8d73/checkpoints/epoch\=14793-valid_elbo_epoch\=2.874.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 2 --exp fns_25_pts_100       --ckpt logs/np/2sdamd5j/checkpoints/epoch\=7601-valid_elbo_epoch\=1.967.ckpt

################## scaling the number examples per each function
# D=10
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_5000_pts_200    --ckpt logs/np/2sfq7m92/checkpoints/epoch\=1926-valid_elbo_epoch\=3.688.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_5000_pts_100    --ckpt logs/np/gkvaa62k/checkpoints/epoch\=1944-valid_elbo_epoch\=3.747.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_5000_pts_50     --ckpt logs/np/2oszounl/checkpoints/epoch\=1971-valid_elbo_epoch\=3.614.ckpt
CUDA_VISIBLE_DEVICES=0 python npc/scripts/evaluate_np.py --dim 10 --exp fns_5000_pts_25     --ckpt logs/np/30g6qrmz/checkpoints/epoch\=1758-valid_elbo_epoch\=3.907.ckpt

# D=5
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5  --exp fns_5000_pts_200    --ckpt logs/np/2cgun6et/checkpoints/epoch\=1975-valid_elbo_epoch\=3.596.ckpt
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5  --exp fns_5000_pts_100    --ckpt logs/np/4jqlz95v/checkpoints/epoch\=1936-valid_elbo_epoch\=3.714.ckpt
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5  --exp fns_5000_pts_50     --ckpt logs/np/191mb65q/checkpoints/epoch\=1913-valid_elbo_epoch\=3.602.ckpt
CUDA_VISIBLE_DEVICES=1 python npc/scripts/evaluate_np.py --dim 5  --exp fns_5000_pts_25     --ckpt logs/np/2iet6cmw/checkpoints/epoch\=1988-valid_elbo_epoch\=3.961.ckpt

# D=2
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2  --exp fns_5000_pts_200    --ckpt logs/np/p5rboh1u/checkpoints/epoch\=1973-valid_elbo_epoch\=3.810.ckpt
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2  --exp fns_5000_pts_100    --ckpt logs/np/3v3v5c15/checkpoints/epoch\=1823-valid_elbo_epoch\=3.626.ckpt
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2  --exp fns_5000_pts_50     --ckpt logs/np/248eype8/checkpoints/epoch\=1941-valid_elbo_epoch\=3.797.ckpt 
CUDA_VISIBLE_DEVICES=2 python npc/scripts/evaluate_np.py --dim 2  --exp fns_5000_pts_25     --ckpt logs/np/9hrden6m/checkpoints/epoch\=1797-valid_elbo_epoch\=3.486.ckpt