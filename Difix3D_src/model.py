import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, DDIMScheduler
p = "Difix3D_src/"
sys.path.append(p)
from lora_vae import AutoencoderKL
from peft import LoraConfig
from einops import rearrange, repeat

class TensorResize:
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation
        
    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)
        resized = F.interpolate(tensor, size=self.size, mode=self.interpolation)
        return resized.squeeze(0)


def process_tensor_batch(input_tensor: torch.Tensor, vae_encoder) -> torch.Tensor:
    """
    Batch processing of tensors: no loop, utilizes VAE batch capability for RGB compression and Mask concatenation.
    
    Args:
        input_tensor: Input tensor, shape (1, 4, 3, h, w)
        vae_encoder: Batch-supported VAE encoder, input (bs*2, 3, H, W), output (bs*2, 16, h', w')
    
    Returns:
        Final tensor, shape (1, 2, 5, h', w')
    """
    rgb_imgs = input_tensor[:, :, :3, ...]  # shape: (1, 2, 3, h, w)
    mask_imgs = input_tensor[:, :, -1:, ...]  # shape: (1, 2, h, w) (single channel)

    # (1, 2, 3, H, W) → (1*2, 3, H, W)
    rgb_batch = rgb_imgs.permute(1, 0, 2, 3, 4).flatten(0, 1)  # shape: (2, 3, h, w)
    vae_feat_batch = vae_encoder(rgb_batch)  # Batch output: (2, 4, h', w')

    # (1,2,1,h,w) rearranged to (2,1,h,w)
    mask_batch = mask_imgs.permute(1, 0, 2, 3, 4).flatten(0, 1)
    # Batch resize to VAE output shape (h',w')
    mask_resized = F.interpolate(
        mask_batch,
        size=vae_feat_batch.shape[-2:],
        mode='bilinear',
        align_corners=True
    )  # shape: (2, 1, h', w')

    # Batch concatenation + dimension restoration: 4D feature + 1D Mask → 5D, then restore to target shape
    combined_batch = torch.cat([vae_feat_batch, mask_resized], dim=1)  # (2, 5, h', w')
    # Restore from (2,5,h',w') to (1,2,5,h',w'), matching the target output format
    return combined_batch.unsqueeze(0).permute(0, 1, 2, 3, 4)

def make_1step_sched():
    repo = "./difix_ref"
    noise_scheduler_1step = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")
    # noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")    
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    
    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)
        
    optimizer.load_state_dict(sd["optimizer"])
    
    return net_difix, optimizer


def save_ckpt(net_difix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_difix.target_modules_vae
    sd["rank_vae"] = net_difix.lora_rank_vae
    sd["state_dict_unet"] = net_difix.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net_difix.vae.state_dict().items() if "lora" in k or "skip" in k}
    
    sd["optimizer"] = optimizer.state_dict()   
    
    torch.save(sd, outf)


class Difix(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4, mv_unet=False, timestep=999, use_gain=False):
        super().__init__()
        repo = "./difix_ref"
        self.use_gain = use_gain
        self.tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder").cuda()        

        # self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()        
        self.sched = make_1step_sched()
        
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", trust_remote_code=True)

        # vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        if repo == "stabilityai/sd-turbo":        
            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        # vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        # vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        # vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        # vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        
        # Initialize skip convs only when loading stabilityai/sd-turbo weights
        if not hasattr(vae.decoder, "skip_conv_1"):
            vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
            vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()

        # vae.decoder.ignore_skip = False
        if not hasattr(vae.decoder, "ignore_skip"):
            vae.decoder.ignore_skip = False    
                
        self.mv_unet = mv_unet
        if mv_unet:
            from mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet")
        # unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        if use_gain:
            ori_conv_in = unet.conv_in

            new_conv_in = torch.nn.Conv2d(
                in_channels=ori_conv_in.in_channels + 5, #uncertain, gate rgb
                out_channels=ori_conv_in.out_channels,
                kernel_size=ori_conv_in.kernel_size,
                stride=ori_conv_in.stride,
                padding=ori_conv_in.padding,
                bias=ori_conv_in.bias is not None
            ).to(ori_conv_in.weight.device)

            new_weight_in = torch.zeros_like(new_conv_in.weight) # gain weight starts from 0
            new_weight_in[:, :ori_conv_in.in_channels, :, :] = ori_conv_in.weight

            with torch.no_grad():
                new_conv_in.weight.copy_(new_weight_in)
                if ori_conv_in.bias is not None:
                    new_conv_in.bias.copy_(ori_conv_in.bias)

            unet.conv_in = new_conv_in

            #conv_out
            ori_conv_out = unet.conv_out
            new_conv_out = torch.nn.Conv2d(
                in_channels=ori_conv_out.in_channels,
                out_channels=ori_conv_out.out_channels + 1,
                kernel_size=ori_conv_out.kernel_size,
                stride=ori_conv_out.stride,
                padding=ori_conv_out.padding,
                bias=ori_conv_out.bias is not None
            ).to(ori_conv_out.weight.device)

            new_weight_out = torch.zeros_like(new_conv_out.weight) # gain weight starts from 0
            new_weight_out[:ori_conv_out.out_channels,: , :, :] = ori_conv_out.weight

            with torch.no_grad():
                new_conv_out.weight.copy_(new_weight_out)
                if ori_conv_out.bias is not None:
                    new_bias = torch.zeros_like(new_conv_out.bias)
                    new_bias[:ori_conv_out.out_channels] = ori_conv_out.bias
                    # gain dimension is still the normal output value
                    new_conv_out.bias.copy_(new_bias)

            unet.conv_out = new_conv_out

        if pretrained_path is not None and os.path.exists(pretrained_path):
            sd = torch.load(pretrained_path, map_location="cpu")
            # vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            # vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            try:
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            except ValueError:
                vae.set_adapter("vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print(f"Initializing model from the pretrained weights of the {repo}")
            target_modules_vae = []

            if repo == "stabilityai/sd-turbo":
                print(f"Initializing model with random weights for the skip connections of the VAE decoder")
                torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
                torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
                torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
                torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
                target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                    "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                    "to_k", "to_q", "to_v", "to_out.0",
                ]
                
                target_modules = []
                for id, (name, param) in enumerate(vae.named_modules()):
                    if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                        target_modules.append(name)
                target_modules_vae = target_modules
                vae.encoder.requires_grad_(False)

            # vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
            #     target_modules=target_modules_vae)
            # vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            # Adapt for stability-sdturbo and nvidia/difx_ref, check if 'vae_skip' layer exists
            if not hasattr(vae, "peft_config") or "vae_skip" not in getattr(vae, "peft_config", {}):
                vae_lora_config = LoraConfig(
                    r=sd["rank_vae"],
                    init_lora_weights="gaussian",
                    target_modules=sd["vae_lora_target_modules"]
                )
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            else:
                try:
                    vae.set_adapter("vae_skip")
                except Exception:
                    pass
                self.lora_rank_vae = lora_rank_vae
                try:
                    self.target_modules_vae = vae.peft_config["vae_skip"].target_modules
                except Exception:
                    self.target_modules_vae = target_modules_vae

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")

        self.unet, self.vae = unet, vae
        # self.vae.decoder.gamma = 1
        if not hasattr(vae.decoder, "gamma"):
            vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
                                
        num_views = x.shape[1]
        if self.mv_unet:
            from mv_unet import BasicTransformerBlock
            BasicTransformerBlock.num_views = num_views

        
        vae_encoder = lambda x:self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        z = rearrange(process_tensor_batch(x, vae_encoder), 'b v c h w -> (b v) c h w')# [2, 5, h', w']
        
        # print(f"nvidia/difix_ref latents shape:{z.shape}")
        # print(f"nvidia/difix_ref latents :{z}")
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        
        unet_input = z
        
        model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
        # print(f"noise_pred shape:{model_pred.shape}")
        # print(f"noise_pred :{model_pred}")

        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
        # print(f"z_denoised shape:{z_denoised[0].shape}")
        # print(f"z_denoised:{z_denoised[0]}")

        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        if self.use_gain:
            output_image = (self.vae.decode(z_denoised[:, :-1] / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        
        return output_image, z_denoised
    
    def sample(self, image, width, height, ref_image=None, gain=None, ref_gain=None, timesteps=None, prompt=None, prompt_tokens=None):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        T_gain = transforms.Compose([
            TensorResize((height, width), interpolation='nearest'),  # Note here the mode is specified using a string
        ])

        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            if gain is not None and ref_gain is not None:
                image_with_gain = torch.cat([T(image), T_gain(gain)], dim=0)
                ref_with_gain = torch.cat([T(ref_image), T_gain(ref_gain)], dim=0) 
                x = torch.stack([image_with_gain, ref_with_gain], dim=0).unsqueeze(0).cuda()
            else:
                x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()

        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[0][:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        
        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        
        sd["optimizer"] = optimizer.state_dict()
        
        torch.save(sd, outf)