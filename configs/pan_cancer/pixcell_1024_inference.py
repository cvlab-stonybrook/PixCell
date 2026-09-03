# Ready-to-use inference config for the released PixCell-1024 checkpoint.
#
# The sampling scripts (tools/sample_1024.py, tools/sample_4k.py) and
# tools/utils.py:build_model_new read a `config.py` located inside the
# `--workdir`. Copy this file into your workdir as `config.py`, then edit the
# two paths below (`vae_pretrained` and `data["root"]`).
#
# These values match the released `pixcell_1024.ckpt` (see the Model Zoo).

# Generated image resolution.
image_size = 1024

# Model: PixArt-XL/2 backbone conditioned on UNI2-h (1536-d) embeddings, 16-ch
# SD-3.5 VAE latents.
model = "PixArt_XL_2_UNI"

# PixCell-1024 uses a 4x4 grid of 16 UNI tokens, with conditioning positional
# embeddings.
model_max_length = 16
use_cond_pos_embed = True
pe_interpolation = 2.0

# Attention / KV-compression settings (disabled for PixCell).
qk_norm = False
kv_compress = False
kv_compress_config = dict(sampling=None, scale_factor=1, kv_compress_layer=[])

# VAE: the SD-3.5 VAE. Either a local copy of the `vae` subfolder of
# `stabilityai/stable-diffusion-3.5-large`, or a directory you downloaded it to.
vae_pretrained = "/path/to/sd-3.5-vae"

# Dataset providing UNI conditioning embeddings. Point `root` at the sample
# dataset (see the README) or your own pre-extracted features.
data = dict(
    type="PanCancerData",
    root="/path/to/sample_dataset",
    resolution=image_size,
)

class_dropout_prob = 0.1

# Directory where the model writes its init log; "." keeps it in the cwd.
work_dir = "."
