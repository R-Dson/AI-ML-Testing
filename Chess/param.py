import torch._dynamo
import time

torch._dynamo.config.suppress_errors = True
out_dir = 'out'


wandb_log = True # override via command line if you like
wandb_project = 'chess'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'moves'
gradient_accumulation_steps = 4
batch_size = gradient_accumulation_steps * gradient_accumulation_steps
block_size = 64 # context of up to 256 previous characters

n_layer = 16
n_head = 8
n_embd = 384
dropout = 0.2

eval_interval = 1000
eval_iters = 200
log_interval = 50

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 600000
lr_decay_iters = 600000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
#beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay = 1e-1

warmup_iters = 500 # not super necessary potentially
