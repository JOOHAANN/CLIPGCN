# define the fusion module for combining the three modalities

import torch
import torch.nn as nn

try:
    from . import Pose_RS
except ImportError:
    import Pose_RS

class VideoADLFeatureExtractor(nn.Module):
    """
    Expected input:
        [B, 13, 192, 6, 6]

    Output:
        [B, 13, 25, 6, 6]
    """

    def __init__(
        self,
        in_channels: int = 192,
        out_channels: int = 25,
        temporal_kernel_size: int = 5,
    ):
        super().__init__()

        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size should be odd.")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial convolution:
        # extracts per-frame geometric features
        # and compresses channels 192 -> 25.
        self.conv_xy = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,
        )

        # Temporal depthwise convolution:
        # models temporal changes independently for each channel.
        self.conv_t = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=(1, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0),
            groups=out_channels,
            bias=False,
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W] or [B,C,T,H,W], got {tuple(x.shape)}"
            )

        input_is_btchw = x.shape[2] == self.in_channels
        if input_is_btchw:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected channel dim {self.in_channels} at axis 1 or 2, got {tuple(x.shape)}"
            )

        x = self.conv_xy(x)
        # [B,192,13,6,6] -> [B,25,13,6,6]

        x = self.conv_t(x)
        # [B,25,13,6,6] -> [B,25,13,6,6]

        x = self.bn(x)
        x = self.relu(x)

        return x.permute(0, 2, 1, 3, 4).contiguous()


class ADLFeatureReducer(nn.Module):
    def __init__(
        self,
        in_channels=50,
        spatial_channels=50,
        temporal_channels=50,
        output_dim=512,
        flattened_dim=25200,
        dropout=0.2,
    ):
        super().__init__()

        self.spatial_conv = nn.Conv2d(
            in_channels,
            spatial_channels,
            kernel_size=3,
            padding=1
        )

        self.temporal_conv = nn.Conv1d(
            spatial_channels,
            temporal_channels,
            kernel_size=3,
            padding=1
        )

        self.temporal_bn = nn.BatchNorm1d(temporal_channels)
        self.relu = nn.ReLU(inplace=True)
        self.flattened_dim = flattened_dim
        self.fc = nn.Linear(flattened_dim, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)
        self.fc_act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")

        B, T, C, H, W = x.shape

        # 空间卷积
        x = x.reshape(B * T, C, H, W)
        x = self.spatial_conv(x)

        _, C2, H2, W2 = x.shape
        x = x.reshape(B, T, C2, H2, W2)

        # 调整为 Conv1d 需要的 [batch, channel, time]
        x = x.permute(0, 3, 4, 2, 1)
        x = x.reshape(B * H2 * W2, C2, T)

        # 时间卷积
        x = self.relu(self.temporal_bn(self.temporal_conv(x)))

        x = x.reshape(B, -1)
        if x.shape[1] != self.flattened_dim:
            raise ValueError(
                f"Expected flattened ADL feature length {self.flattened_dim}, got {x.shape[1]}"
            )

        x = self.fc(x)
        x = self.fc_bn(x)
        x = self.fc_act(x)
        x = self.dropout(x)

        return x

class PoseADLFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, grid_size=6):
        super().__init__()
        self.person_attn = nn.Sequential(
            nn.Linear(in_channels, 1)
        )

        self.to_grid = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, grid_size * grid_size),
        )

        self.grid_size = grid_size

    def forward(self, x):
        # x: [B, M, C, T, V] = [B,2,64,13,25]
        B, M, C, T, V = x.shape

        # [B,M,C,T,V] -> [B,T,V,M,C]
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        # 对 person 做 attention
        # attn_logits: [B,T,V,M,1]
        attn_logits = self.person_attn(x)
        attn = torch.softmax(attn_logits, dim=3)

        # 融合 person: [B,T,V,C]
        x = (x * attn).sum(dim=3)

        # 每个 time/joint 的 64 维特征 -> 36
        # [B,T,V,C] -> [B,T,V,36]
        x = self.to_grid(x)

        # [B,T,V,36] -> [B,T,V,6,6]
        x = x.view(B, T, V, self.grid_size, self.grid_size)

        return x
    
class TriModalFusion(nn.Module):
    """
    TriModalFusion

    inputs：
        Video_feature: [B, 13, 192, 6, 6]
        Pose_feature: [B, 2, 64, 13, 25]
        Object_feature: [B, 1, 50, 6, 6] or [B, 50, 6, 6]
        Joint_location: [B, 13, 25, 2]
        
    hidden outputs：
        Video_feature: [B, 13, 25, 6, 6]
        Pose_feature: [B, 13, 25, 6, 6]
        Object_feature: [B, 1, 50, 6, 6]
        Fused_feature: [B, 14, 50, 6, 6]
    
    outputs：        
        ADL_embedding: [B, 512]
    """
    def __init__(self, reducer_dropout: float = 0.2):
        super().__init__()

        # 线性层用于对齐通道数
        self.video_raw_proj = VideoADLFeatureExtractor(in_channels=192, out_channels=25)
        self.pose_raw_proj = PoseADLFeatureExtractor(in_channels=64, grid_size=6)
        self.reducer = ADLFeatureReducer(
            in_channels=50,
            spatial_channels=50,
            temporal_channels=50,
            output_dim=512,
            flattened_dim=25200,
            dropout=reducer_dropout,
        )

    def forward(self, video_feature_raw, pose_feature_raw, object_feature_raw, joint_location_raw):    
        # video_feature_raw: [B, 13, 192, 6, 6]
        # pose_feature_raw: [B, 2, 64, 13, 25]
        # object_feature_raw: [B, 1, 50, 6, 6]
        # joint_location_raw: [B, 13, 25, 2]
        
        video_proj = self.video_raw_proj(video_feature_raw)
        
        pose_proj = self.pose_raw_proj(pose_feature_raw)

        if joint_location_raw.ndim != 4 or joint_location_raw.shape[-1] != 2:
            raise ValueError(
                f"Expected joint_location_raw [B,13,25,2], got {tuple(joint_location_raw.shape)}"
            )

        x_s = joint_location_raw[..., 0]
        y_s = joint_location_raw[..., 1]
        pose_proj = Pose_RS.get_RS_map(
            pose_proj, x_s, y_s
        )
        
        if object_feature_raw.ndim == 4:
            object_feature_raw = object_feature_raw.unsqueeze(1)
        if object_feature_raw.ndim != 5:
            raise ValueError(
                f"Expected object_feature_raw [B,1,50,6,6] or [B,50,6,6], got {tuple(object_feature_raw.shape)}"
            )

        # object feature now has shape [B, 1, 50, 6, 6]
        object_proj = torch.nan_to_num(
            object_feature_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(min=0.0, max=10.0)
        if object_proj.shape[1:] != (1, 50, 6, 6):
            raise ValueError(
                f"Expected object feature [B,1,50,6,6], got {tuple(object_proj.shape)}"
            )


        # Video skip path + pose RS path:
        # concatenate video and pose along the channel dimension.
        vp_feature = torch.cat([video_proj, pose_proj], dim=2)
        # [B, 13, 50, 6, 6]

        # Add object location maps as the extra temporal step.
        fused_feature = torch.cat([vp_feature, object_proj], dim=1)
        # [B, 14, 50, 6, 6]


        # ADL feature reducer: [B,14,50,6,6] -> [B,512]
        fused_feature = self.reducer(fused_feature)

        return fused_feature
