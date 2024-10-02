from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.fft as fft
import cv2
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from PIL import Image
from typing import List, Dict
from dc.dc_unet import CustomUNet2DConditionModel
from dc.utils.free_lunch import register_free_upblock2d_in, register_free_crossattn_upblock2d_in
import math

@dataclass
class DCConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"

    num_inference_steps: int = 500
    min_step_ratio: float = 0.2
    max_step_ratio: float = 0.9

    src_prompt: str = "a photo of a sks man"
    tgt_prompt: str = "a photo of a Batman"

    log_step: int = 10
    guidance_scale: float = 7.5
    device: torch.device = torch.device("cuda")
    image_guidance_scale: float = 1.5

    psi = 0.075
    chi = math.log(0.1)
    delta = 0.2
    gamma = 0.8

    freeu_b1: float=1.1
    freeu_b2: float=1.1
    freeu_s1: float=0.9
    freeu_s2: float=0.2


class DC(object):
    def __init__(self, config: DCConfig, use_wandb=False):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config.sd_pretrained_model_or_path,
            subfolder="unet"
        ).to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")
    
        self.use_wandb = use_wandb

        self.threshold = 0.2
        self.check = 0
        self.w_s = 1.5
        self.iteration = 0
        self.max_iteration = 3000

        b1 = self.config.freeu_b1
        b2 = self.config.freeu_b2
        s1= self.config.freeu_s1
        s2= self.config.freeu_s2

        register_free_upblock2d_in(self.unet, b1, b2, s1, s2)
        register_free_crossattn_upblock2d_in(self.unet, b1, b2, s1, s2)

        
    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t].to(device)
        alpha_t = self.scheduler.alphas[t].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mean_func = c0 * pred_x0 + c1 * xt
        
        return mean_func, pred_x0

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215
    
    def encode_src_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor.float()
        return self.vae.encode(x)

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)

    def dc_timestep_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)

        idx = torch.full((batch_size,), (max_step-min_step)*((self.max_iteration-self.iteration)/self.max_iteration) + min_step, dtype=torch.long, device="cpu")

        timestep_noralized = idx[0].item() / len(timesteps)
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()
        return t, t_prev, timestep_noralized

    def __call__(
        self,
        tgt_x0,
        src_x0,
        src_emb,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
        step=0,
        current_spot=0,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        t, t_prev, t_normalized = self.dc_timestep_sampling(batch_size)
        
        beta_t = scheduler.betas[t].to(device)
        alpha_t = scheduler.alphas[t].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        
        '''
        beta_t_tau = scheduler.betas[t_tau].to(device)
        alpha_t_tau = scheduler.alphas[t_tau].to(device)
        alpha_bar_t_tau = scheduler.alphas_cumprod[t_tau].to(device)      
        '''

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)
        '''h_t_tau = 0.3 * torch.sqrt(1 - alpha_t_tau) * noise
        with torch.no_grad():
            #DDIM inversion
                latents_noisy = scheduler.add_noise
                src_encoded = src_emb.latent_dist.mode()
                unet_outputs = self.unet.forward(
                    latent_model_input,
                    torch.cat([t] * 3).to(device),
                    encoder_hidden_states=text_embeddings,
                )'''
        
        
        eps = dict()
        pred_x0s = dict()
        noisy_latents = dict()
        
        for latent, cond_text_embedding, name in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"]
        ):
            latents_noisy = scheduler.add_noise(latent, noise, t)
            text_embeddings = torch.cat([cond_text_embedding, uncond_embedding, uncond_embedding], dim=0)
            text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=1)

            src_encoded = src_emb.latent_dist.mode()
            
            uncond_image_latent = torch.zeros_like(src_encoded)
            latent_image = torch.cat([src_encoded, src_encoded, uncond_image_latent], dim=0)
            latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            latent_model_input = torch.cat([latent_model_input, latent_image], dim=1)

            unet_outputs = self.unet.forward(
                latent_model_input,
                torch.cat([t] * 3).to(device),
                encoder_hidden_states=text_embeddings,
            )
            noise_pred = unet_outputs.sample
            unet_feats = unet_outputs.features

            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            if name == "tgt":
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                    self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            else:
                noise_pred = noise_pred_uncond + self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)

            mu, pred_x0 = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)

            eps[name] = noise_pred
            pred_x0s[name] = pred_x0
            noisy_latents[name] = latents_noisy
           
        self.iteration += 1
        
        w_DDS = self.config.delta + self.config.gamma * (t_normalized ** (1/math.e))
        grad = w_DDS * (eps["tgt"] - eps["src"]) + (self.config.psi * (math.exp(t_normalized))) * (tgt_x0 - src_x0)
        grad = torch.nan_to_num(grad)
        
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size 
        
        
        if self.use_wandb:
            import wandb
            wandb.log({
                f"target_prediction_x0_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(pred_x0s["tgt"])), min_size=256), caption=f"{t.item()}"),
                f"source_prediction_x0_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(pred_x0s["src"])), min_size=256), caption=f"{t.item()}"),
                f"target_noise_prediction_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(eps["tgt"])), min_size=256), caption=f"{t.item()}"),
                f"source_noise_prediction_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(eps["src"])), min_size=256), caption=f"{t.item()}"),
                f"target_noisy_latents_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(noisy_latents["tgt"])), min_size=256), caption=f"{t.item()}"),
                f"source_noisy_latents_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(noisy_latents["src"])), min_size=256), caption=f"{t.item()}"),
            }, step=step, commit=False) if step % self.config.log_step == 0 else None
        
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def run_sdedit(self, x0, tgt_prompt=None, num_inference_steps=20, skip=7, eta=0):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        S = num_inference_steps - skip
        t = reversed_timesteps[S - 1]
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        op = timesteps[-S:]

        for t in op:
            xt_input = torch.cat([xt] * 2)
            noise_pred = self.unet.forward(
                xt_input,
                torch.cat([t[None]] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            xt = self.reverse_step(noise_pred, t, xt, eta=eta)

        return xt

    def reverse_step(self, model_output, timestep, sample, eta=0, variance_noise=None):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        variance = self.get_variance(timestep)
        model_output_direction = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z
        return prev_sample

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance


def tensor_to_pil(img):
    if img.ndim == 4:
        img = img[0]
    img = img.cpu().permute(1, 2, 0).detach().numpy()
    
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil_to_tensor(img, device="cpu"):
    device = torch.device(device)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img[None].transpose(0, 3, 1, 2))
    img = img.to(device)
    return img


def resize_image(image, min_size):
    if min(image.size) < min_size:
        image = image.resize((min_size, min_size))
    return image
