_base_ = ['../PixArt_xl2_internal.py']

image_size = 256

root = f"/lus/grand/projects/GeomicVar/srikar/pan_cancer"

data = dict(
    type="PanCancerDataLowRes",
    root = root,
    resolution = image_size,
)

# model setting
model = "PixArt_XL_2_UNI"
model_max_length = 1

mixed_precision = 'fp16'  
fp32_attention = True

resume_from = None

vae_pretrained = f"{root}/pretrained_models/sd-3.5-vae"
pe_interpolation = 0.5

# training setting
num_workers = 16
train_batch_size = 128
num_epochs = 10  
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)
auto_lr = None

log_interval = 20
save_model_epochs = 1
save_model_steps = 2000
work_dir = f"{root}/checkpoints/pixart_20x_{image_size}_sd3_vae"

# vae
scale_factor = 1.5305
shift_factor = 0.0609
class_dropout_prob = 0.1

seed = 24