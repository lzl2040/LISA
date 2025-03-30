import torch.nn as nn
import torch
from model.diffusion.models import DiT
from model.diffusion import gaussian_diffusion as gd
from model.diffusion import create_diffusion


def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

class DiffSegModel(nn.Module):
    def __init__(self, 
                 model_type, 
                 in_channels, 
                 img_feats_channels,
                 img_size,
                 condition_in_channels,
                 diffusion_steps = 1000,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, 
                                          diffusion_steps=self.diffusion_steps, 
                                          sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        
        self.net = DiT_models[model_type](in_channels=3, img_feats_channels=img_feats_channels,
                                          input_size=img_size, condition_in_channels=condition_in_channels)
    
    def loss(self, x, z, img_feats):
        # x should be noised mask and should has raw image to guide. besides
        B, C = x.shape[:2]
        noise = torch.randn_like(x) # [B, 3, H, W]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise).to(img_feats.dtype)

        # predict noise from x_t
        timestep = timestep.to(img_feats.dtype)
        noise_pred = self.net(x_t, timestep, z, img_feats) # B 6 H W
        # print(noise_pred.shape, noise.shape, x.shape)
        noise_pred, model_var_values = torch.split(noise_pred, C, dim=1)
        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()

        return loss        