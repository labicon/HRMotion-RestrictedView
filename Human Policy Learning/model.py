import torch.nn as nn
from diffusers import UNet2DConditionModel

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

class ClassConditionedUnet(nn.Module):
    def __init__(self, location_emb_size=1280, hidden_dim=512, output_size=16):
        super().__init__()

        self.location_emb = MLP(input_dim=9, hidden_dim=hidden_dim, output_dim=location_emb_size)

        self.project_to_high_dim = MLP(3, hidden_dim, 3 * output_size * output_size)
        self.reduce_to_output = nn.Linear(3 * output_size * output_size, 3)

        self.model = UNet2DConditionModel(
            sample_size=output_size,
            in_channels=3, 
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(32, 64, 64, 64),
            mid_block_type="UNetMidBlock2DCrossAttn", 
            cross_attention_dim=location_emb_size,
        )

    def forward(self, noisy_action, t, location):
        output_size = 16
        projected_action = self.project_to_high_dim(noisy_action)
        projected_action = projected_action.view(noisy_action.size(0), 3, output_size, output_size)

        location_emb = self.location_emb(location)
        
        location_emb = location_emb.unsqueeze(1)
        output = self.model(projected_action, timestep=t, encoder_hidden_states=location_emb).sample

        output = output.view(output.size(0), -1)
        output = self.reduce_to_output(output)
        
        return output