import torch
import torch.nn as nn
from diffusers import UNet2DModel  # Lightweight diffusion backbone
from torch.nn import functional as F

class SmallCausalDiffusionTeacher(nn.Module):
    """
    Small causal teacher: Diffusion block for egirl motions (72-dim joints).
    Autoregressive: Denoises step-by-step, conditioned on lang intent (from Qwen3-Omni).
    Params: ~200M (UNet tiny + causal proj); distill to VALM Jacobian.
    """
    def __init__(self, joint_dim=72, intent_dim=512, timesteps=100, hidden_dim=4096):
        super().__init__()
        self.joint_dim = joint_dim
        self.intent_dim = intent_dim
        self.timesteps = timesteps
        
        # Causal conditioner: Lang intent → embedding
        self.intent_proj = nn.Linear(intent_dim, hidden_dim)
        
        # Diffusion UNet (small: 2D for time-joint space; causal via masking)
        self.unet = UNet2DModel(
            sample_size=joint_dim,  # "Height" as joint seq
            in_channels=1,  # Noisy joints (flattened)
            out_channels=1,
            layers_per_block=1,  # Tiny for speed
            block_out_channels=(hidden_dim // 2, hidden_dim),
            down_block_types=('DownBlock2D',),
            up_block_types=('UpBlock2D',),
            num_attn_heads=4,  # Causal self-attn
        )
        
        # Causal mask for autoregressive denoising
        self.register_buffer('causal_mask', torch.tril(torch.ones(timesteps, timesteps)))
        
        # Noise scheduler stub (DDIM for fast sampling)
        self.scheduler = torch.nn.Linear(timesteps, 1)  # Beta schedule
    
    def forward(self, noisy_joints, intent_emb, t):  # t: timestep [B]
        B, T, _ = noisy_joints.shape  # [B, timesteps=100, 72]
        
        # Condition on intent (broadcast)
        cond = self.intent_proj(intent_emb).unsqueeze(1).expand(-1, T, -1)  # [B,T,hidden]
        
        # Flatten for UNet (time as "height")
        noisy_flat = noisy_joints.view(B, 1, -1)  # [B,1,T*joints] causal input
        
        # Causal attn mask in UNet (via custom hook or post-proj)
        pred_noise = self.unet(noisy_flat, timestep=t, encoder_hidden_states=cond).sample
        
        # Reshape back; apply causal mask to prevent future peeking
        pred_noise = pred_noise.view(B, T, self.joint_dim) * self.causal_mask[:T, :T].to(pred_noise.device)
        
        return pred_noise  # Denoised delta for next step
    
    def sample(self, intent_emb, num_samples=1, steps=50):  # Causal generation
        B = intent_emb.shape[0]
        shape = (B, self.timesteps, self.joint_dim)
        x = torch.randn(shape)  # Dummy full-body noise init (relaxed pose)
        
        for i, t in enumerate(self.scheduler(torch.arange(steps)).long()):
            pred = self(x, intent_emb, t.repeat(B))
            x = self.scheduler(pred, x)  # DDIM step (causal: only past cond)
            
        return x.mean(1)  # [B,72] final trajectory (e.g., egirl wave)

# Dummy test: Integrate as teacher for VALM KD
teacher = SmallCausalDiffusionTeacher()
intent = torch.randn(1, 512)  # From Qwen derivation (e.g., "flirty" embed)
trajectory = teacher.sample(intent)  # [1,72] dummy motion
print(f"Generated egirl motion: {trajectory.shape}")  # Ready for Jacobian apply