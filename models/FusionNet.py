import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 1, padding=1),
            nn.PReLU(mid_channels),
            nn.Conv3d(mid_channels, out_channels, 3, 1, padding=1),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),
                nn.PReLU(out_channels)
            )

    def forward(self, x):
        return self.conv(x)


class FusionNet(nn.Module):

    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv0 = ConvBlock(32, 16)
        self.conv1 = ConvBlock(48, 16)
        self.conv2 = ConvBlock(64, 16)
        self.conv3 = ConvBlock(80, 16)
        self.conv4 = ConvBlock(96, 16)
        self.conv5 = ConvBlock(16, 16)

    def forward(self, feat1, feat2):
        feat0 = torch.cat((feat1, feat2), dim=1)  # 1,32,32,128,128
        feat1 = self.conv0(feat0)  # 1,16,32,128,128
        feat1 = torch.cat((feat1, feat0), dim=1)  # 1,48,32,128,128
        feat2 = self.conv1(feat1)  # 1,16,32,128,128
        feat2 = torch.cat((feat2, feat1), dim=1)  # 1,64,32,128,128
        feat3 = self.conv2(feat2)  # 1,16,32,128,128
        feat3 = torch.cat((feat3, feat2), dim=1)  # 1,80,32,128,128
        feat4 = self.conv3(feat3)  # 1,16,32,128,128
        feat4 = torch.cat((feat4, feat3), dim=1)  # 1,96,32,128,128
        out = self.conv4(feat4)  # 1,16,32,128,128
        out = self.conv5(out)  # 1,16,32,128,128

        return out

class WeightMLPWGN(nn.Module):


    def __init__(self, base_model, theta_f, theta_s2f, theta_f2s,
                 hidden_dim=64, chunk_size=262144):
        super(WeightMLPWGN, self).__init__()

        self.param_names = []
        self.param_shapes = []
        self.num_layers = 0
        self.chunk_size = chunk_size

        for idx, (name, param) in enumerate(base_model.named_parameters()):
            if name not in theta_f:
                raise KeyError(f"{name} not found in theta_f.")
            if name not in theta_s2f:
                raise KeyError(f"{name} not found in theta_s2f.")
            if name not in theta_f2s:
                raise KeyError(f"{name} not found in theta_f2s.")

            self.param_names.append(name)
            self.param_shapes.append(param.shape)

            residual_weight = theta_f[name].detach().clone().to(
                device=param.device,
                dtype=param.dtype
            )

            weight_s2f = theta_s2f[name].detach().clone().to(
                device=param.device,
                dtype=param.dtype
            )

            weight_f2s = theta_f2s[name].detach().clone().to(
                device=param.device,
                dtype=param.dtype
            )

            self.register_buffer(f'theta_f_{idx}', residual_weight)
            self.register_buffer(f'theta_s2f_{idx}', weight_s2f)
            self.register_buffer(f'theta_f2s_{idx}', weight_f2s)

            self.num_layers += 1


        self.weight_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self._zero_init_output()

    def _zero_init_output(self):

        last_layer = self.weight_mlp[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _generate_delta_by_chunks(self, weight_s2f, weight_f2s):

        flat_s2f = weight_s2f.reshape(-1)
        flat_f2s = weight_f2s.reshape(-1)

        numel = flat_s2f.numel()
        delta_list = []

        for start in range(0, numel, self.chunk_size):
            end = min(start + self.chunk_size, numel)

            # [chunk, 2]
            pair_weight = torch.stack(
                [
                    flat_s2f[start:end],
                    flat_f2s[start:end]
                ],
                dim=1
            )

            # [chunk, 1] -> [chunk]
            delta = self.weight_mlp(pair_weight).squeeze(1)

            delta_list.append(delta)

        delta_flat = torch.cat(delta_list, dim=0)

        return delta_flat.reshape_as(weight_s2f)

    def forward(self):
        generated_params = {}

        for idx, name in enumerate(self.param_names):
            theta_f = getattr(self, f'theta_f_{idx}')
            theta_s2f = getattr(self, f'theta_s2f_{idx}')
            theta_f2s = getattr(self, f'theta_f2s_{idx}')
            delta_theta = self._generate_delta_by_chunks(theta_s2f, theta_f2s)


            generated_params[name] = delta_theta

        return generated_params


if __name__ == '__main__':
    feat1 = torch.ones((1, 16, 32, 128, 128))
    feat2 = torch.ones((1, 16, 32, 128, 128))
    Net = FusionNet()
    Net.eval()
    y = Net(feat1, feat2)
    print(Net)
    print(y.shape)
