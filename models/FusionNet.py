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
    """
    using Residual connection
    上一层的输入作为下一层的输入
    """
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
    """
    WGN for Stage III.

    输入：
        theta_f     : Stage I 的 MIF 权重，即融合任务权重 theta_F，作为残差基底
        theta_s2f   : Stage II 的 MIS -> MIF 权重，即 theta_{S->F}
        theta_f2s   : Stage II 的 MIF -> MIS 权重，即 theta_{F->S}

    输出：
        theta_u     : unified weight theta_U

    设计：
        1. 只处理 feature fusion module 的权重，也就是 FusionNet 的权重。
        2. 对每一层参数，flatten 成 1D。
        3. 将 theta_{S->F} 和 theta_{F->S} 的对应位置拼成 2-channel：
              [num_params, 2]
        4. 送入 MLP，输出 [num_params, 1]，相当于 channel-wise 2-to-1。
        5. 输出 reshape 回原来的参数形状。
        6. 加到 theta_F 上作为残差：
              theta_U = theta_F + delta_theta
        7. 最后一层 MLP 全 0 初始化，因此初始 delta_theta = 0，
           即初始 theta_U = theta_F。
    """

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

        # 2-channel -> 1-channel MLP
        # 对每个参数位置独立做 2-to-1 映射
        self.weight_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self._zero_init_output()

    def _zero_init_output(self):
        """
        让 WGN 初始输出全 0。
        这样初始情况下：
            theta_U = theta_F + 0
        """
        last_layer = self.weight_mlp[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _generate_delta_by_chunks(self, weight_s2f, weight_f2s):
        """
        对一层参数生成 delta。
        使用 chunk 是为了避免大层参数一次性 flatten 后显存过高。
        """

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

            # WGN 输出增量
            delta_theta = self._generate_delta_by_chunks(theta_s2f, theta_f2s)

            # 残差连接，残差是融合任务权重 theta_F
            theta_u = theta_f + delta_theta

            generated_params[name] = theta_u

        return generated_params

    def manifold_distance_loss(self, generated_params):

        loss_m = 0.0

        for idx, name in enumerate(self.param_names):
            theta_u = generated_params[name]
            theta_s2f = getattr(self, f'theta_s2f_{idx}')
            theta_f2s = getattr(self, f'theta_f2s_{idx}')

            loss_m = loss_m + torch.mean((theta_u - theta_s2f) ** 2)
            loss_m = loss_m + torch.mean((theta_u - theta_f2s) ** 2)

        loss_m = loss_m / self.num_layers

        return loss_m


if __name__ == '__main__':
    feat1 = torch.ones((1, 16, 32, 128, 128))
    feat2 = torch.ones((1, 16, 32, 128, 128))
    Net = FusionNet()
    Net.eval()
    y = Net(feat1, feat2)
    print(Net)
    print(y.shape)
