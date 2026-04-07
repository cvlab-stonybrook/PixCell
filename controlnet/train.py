from dataclasses import dataclass
import torch
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup

from pixcell_transformer_2d import PixCellTransformer2DModel
from pixcell_controlnet import PixCellControlNet
from pixcell_controlnet_transformer import PixCellTransformer2DModelControlNet

from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import os

from accelerate import Accelerator


@dataclass
class TrainingConfig:
    train_batch_size = 1
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    output_dir = "./training_controlnet"
    vae_scale = 0
    vae_shift = 0


def main():
    # Load PixCell-1024
    sd3_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    pipeline = DiffusionPipeline.from_pretrained(
        "StonyBrook-CVLab/PixCell-1024",
        vae=sd3_vae,
        custom_pipeline="StonyBrook-CVLab/PixCell-pipeline",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        device_map=None,
    )

    # Create ControlNet transformer
    # Copy base transformer config
    transformer_config = {}
    for k in pipeline.transformer.config.keys():
        # Remove unused parameters
        if k in ['double_self_attention', 'num_vector_embeds', 'only_cross_attention', 'use_linear_projection']:
            continue
        transformer_config[k] = pipeline.transformer.config[k]
    controlnet = PixCellControlNet(
        base_transformer=PixCellTransformer2DModel(**transformer_config),
        n_blocks=27,
    )
    # Load input projection from base model
    # This is frozen during ControlNet training
    controlnet.transformer.pos_embed.load_state_dict(pipeline.transformer.pos_embed.state_dict())

    # Replace PixCell transformer with PixCell+ControlNet
    controlnet_transformer = PixCellTransformer2DModelControlNet(
        **transformer_config
    )
    controlnet_transformer.load_state_dict(pipeline.transformer.state_dict())
    pipeline.transformer = controlnet_transformer

    # Training configuration
    config = TrainingConfig()
    config.vae_scale = pipeline.vae.config.scaling_factor
    config.vae_shift = getattr(pipeline.vae.config, "shift_factor", 0)

    ## REPLACE WITH YOUR DATASET ##
    dataset = YOURDATASET(
    )
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # Training parameters
    pipeline.vae.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    controlnet.train()
    # Input patch embedding is frozen
    controlnet.transformer.pos_embed.requires_grad_(False)


    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    base_transformer = pipeline.transformer
    controlnet_transformer = controlnet
    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler

    optimizer = torch.optim.AdamW(list(controlnet_transformer.parameters()) + list(uni_mlp.parameters()), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    base_transformer, controlnet_transformer, uni_mlp, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        base_transformer, controlnet_transformer, uni_mlp, vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            img, mask_img, uni_he = batch
            bs = img.shape[0]

            # Encode image
            img = 2*(img-0.5)
            with torch.no_grad():
                img_latents = vae.encode(img.to(base_transformer.device)).latent_dist.mean
                img_latents = (img_latents-config.vae_shift)*config.vae_scale

                mask_img_latents = vae.encode(mask_img.to(base_transformer.device)).latent_dist.mean
                mask_img_latents = (mask_img_latents-config.vae_shift)*config.vae_scale

            # Add noise to IHC latents
            t = torch.randint(0, 1000, (bs,), device='cpu', dtype=torch.int64)
            atbar = noise_scheduler.alphas_cumprod[t].view(-1,1,1,1).to(base_transformer.device)
            epsilon = torch.randn_like(img_latents)
            noisy_latents = torch.sqrt(atbar)*img_latents + torch.sqrt(1-atbar)*epsilon

            with accelerator.accumulate(controlnet_transformer), accelerator.accumulate(uni_mlp):
                current_timestep = t.clone().to(base_transformer.device)

                # Transform H&E UNI to IHC
                pred_uni_ihc = uni_mlp(uni_he.to(base_transformer.device))

                # Pass H&E through ControlNet
                controlnet_outputs = controlnet_transformer(
                    hidden_states=noisy_latents,
                    conditioning=mask_img_latents,
                    encoder_hidden_states=pred_uni_ihc,
                    timestep=current_timestep,
                    return_dict=False,
                )[0]

                # Pass noisy IHC through denoiser
                epsilon_pred = base_transformer(
                    noisy_latents,
                    encoder_hidden_states=pred_uni_ihc,
                    timestep=current_timestep,
                    controlnet_outputs=controlnet_outputs,
                    return_dict=False,
                )[0]

                loss = ((epsilon_pred[:,:16,:,:] - epsilon)**2).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(controlnet_transformer.parameters()) + list(uni_mlp.parameters()), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1


if __name__ == "__main__":
    main()