# Run with:
# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1
# train_lora.py --dataset MIST --root_dir /path/to/MIST/ 
# --split train --stain HER2 --train_batch_size 2 --num_epochs 1 --gradient_accumulation_steps 1

import torch
import einops

from peft import LoraConfig
from pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA
from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from tqdm.auto import tqdm
import os

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train LoRA on IHC-stained images")
    parser.add_argument("--dataset", type=str, choices=['MIST', 'HER2Match'])
    parser.add_argument("--root_dir", type=str, default='/path/to/data/')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--stain", type=str, default='')
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./training_stuff")
    parser.add_argument("--uncond_prob", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./")
    args = parser.parse_args()

    # Load UNI
    timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval()

    # Load VAE and scheduler -- same as PixCell-1024
    sd3_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("StonyBrook-CVLab/PixCell-1024", subfolder="scheduler")

    # Create transformer
    config = {
        "_class_name": "PixCellTransformer2DModel",
        "_diffusers_version": "0.32.2",
        "_name_or_path": "pixart_1024/transformer",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 72,
        "attention_type": "default",
        "caption_channels": 1536,
        "caption_num_tokens": 16,
        "cross_attention_dim": 1152,
        "dropout": 0.0,
        "in_channels": 16,
        "interpolation_scale": 2,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "norm_num_groups": 32,
        "norm_type": "ada_norm_single",
        "num_attention_heads": 16,
        "num_embeds_ada_norm": 1000,
        "num_layers": 28,
        "out_channels": 32,
        "patch_size": 2,
        "sample_size": 128,
        "upcast_attention": False,
        "use_additional_conditions": False,
    }
    lora_transformer = PixCellTransformer2DModelLoRA(
        **config
    )
    
    # Load base PixCell-1024 weights from the huggingface repo
    ckpt_path = hf_hub_download(
        repo_id="StonyBrook-CVLab/PixCell-1024",
        filename="transformer/diffusion_pytorch_model.safetensors",
        local_dir="downloads/",

    )
    lora_transformer.load_state_dict(load_file(ckpt_path), strict=False)

    # Add LoRA to cross-attention layers
    target_modules = [
        "attn2.add_k_proj",
        "attn2.add_q_proj",
        "attn2.add_v_proj",
        "attn2.to_add_out",
        "attn2.to_k",
        "attn2.to_out.0",
        "attn2.to_q",
        "attn2.to_v",
    ]
    rank = 4
    lora_dropout = 0.0
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    lora_transformer.add_adapter(transformer_lora_config)

    # Training configuration
    vae_scale = sd3_vae.config.scaling_factor
    vae_shift = getattr(sd3_vae.config, "shift_factor", 0)

    # Create dataloader
    if args.dataset == 'MIST':
        from mist_dataset import MISTDataset

        dataset = MISTDataset(
            root_dir=args.root_dir,
            split=args.split,
            stain=args.stain,
        )
        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    elif args.dataset == 'HER2Match':
        from her2match_dataset import HER2MatchDataset

        dataset = HER2MatchDataset(
            root_dir=args.root_dir,
            split=args.split,
        )
        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
        project_dir=os.path.join(args.output_dir, "logs"),
        kwargs_handlers=[ddp_kwargs],
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    vae = sd3_vae
    noise_scheduler = scheduler

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.requires_grad_(False)

    # Trainable parameters
    lora_parameters = list(filter(lambda p: p.requires_grad, lora_transformer.parameters()))
    optimizer = torch.optim.AdamW([
        {'params': lora_parameters, 'lr': args.learning_rate},
    ])

    # Sanity check for cross-attention LoRA
    assert lora_transformer.transformer_blocks[0].attn2.to_q.base_layer.weight.requires_grad == False
    assert lora_transformer.transformer_blocks[0].attn2.to_q.lora_A.default.weight.requires_grad == True
        
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=None,
        num_cycles=1,
        power=0,
    )

    lora_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_transformer, optimizer, train_dataloader, lr_scheduler
    )
    vae = accelerator.prepare_model(vae, evaluation_mode=True)
    uni_model = accelerator.prepare_model(uni_model, evaluation_mode=True)

    global_step = 0
    # Now you train the model
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_transformer):
                he, ihc = batch
                bs = ihc.shape[0]

                # Extract UNI features from IHC
                # Rearrange 1024x1024 image into 16 256x256 patches
                uni_patches = einops.rearrange(ihc, 'b c (d1 h) (d2 w) -> (b d1 d2) c h w', d1=4, d2=4)
                uni_input = uni_transform(uni_patches)
                with torch.inference_mode():
                    uni_emb_ihc = uni_model(uni_input.to(lora_transformer.device))
                uni_emb_ihc = uni_emb_ihc.unsqueeze(0).reshape(bs,16,-1)

                # Encode IHC images
                ihc = (2*(ihc-0.5)).to(dtype=vae.dtype)
                ihc_latents = vae.encode(ihc.to(lora_transformer.device)).latent_dist.sample()
                ihc_latents = (ihc_latents-vae_shift)*vae_scale

                # Add noise to IHC latents
                t = torch.randint(0, 1000, (bs,), device='cpu', dtype=torch.int64)
                atbar = noise_scheduler.alphas_cumprod[t].view(bs,1,1,1).to(lora_transformer.device)
                epsilon = torch.randn_like(ihc_latents)
                noisy_latents = torch.sqrt(atbar)*ihc_latents + torch.sqrt(1-atbar)*epsilon

                current_timestep = t.clone().to(lora_transformer.device)

                # Randomly drop UNI embeddings
                if args.uncond_prob > 0:
                    uncond = lora_transformer.caption_projection.uncond_embedding.clone().tile(uni_emb_ihc.shape[0],1,1)
                    mask = (torch.rand((bs,1,1), device=lora_transformer.device) < args.uncond_prob).float()
                    uni_emb_ihc = (1-mask)*uni_emb_ihc.to(lora_transformer.device) + mask*uncond

                # Pass noisy IHC through denoiser
                epsilon_pred = lora_transformer(
                    noisy_latents,
                    encoder_hidden_states=uni_emb_ihc.to(lora_transformer.device),
                    timestep=current_timestep,
                    return_dict=False,
                )[0]

                # Compute denoising loss
                loss = ((epsilon_pred[:,:16,:,:] - epsilon)**2).mean()

                # Update weights
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            

        # Save models
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            lora_transformer_unwrapped = accelerator.unwrap_model(lora_transformer)
            torch.save(lora_transformer_unwrapped.state_dict(), os.path.join(args.save_dir, f"{args.dataset}_{args.stain}_lora_{epoch+1}.pth"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
