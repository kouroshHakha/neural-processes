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

