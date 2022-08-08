import torch
import torch.nn as nn

from h2t.models.utils import UpSample2x
from h2t.models.backbone import ConvNeXtExt, ConvNeXtBlock, ResNetExt


class FCN_ConvNext(nn.Module):
    def __init__(
        self,
        num_input_channels=2,
        num_output_channels=2,
        freeze_encoder=True,
        pretrained_encoder=None,
    ):
        super().__init__()

        self.freeze_encoder = freeze_encoder
        self.backbone = ConvNeXtExt.convnext_base(pretrained_encoder)

        # **
        img_list = torch.rand([1, 3, 256, 256])
        out_list = self.backbone(img_list)
        # orderd from lores hires
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        self.conv1x1 = None
        if down_ch_list[0] != down_ch_list[1]:  # channel mapping for shortcut
            self.conv1x1 = nn.Conv2d(
                down_ch_list[0], down_ch_list[1], (1, 1), bias=False
            )

        self.uplist = nn.ModuleList()

        num_blocks = 3

        depths = [3, 3, 27, 3, 3, 3, 3, 3]
        dp_rates_ = [x.item() for x in torch.linspace(0, 0.5, sum(depths))]

        start = 0
        dp_rates = []
        for depth in depths:
            dp_rates.append(dp_rates_[start : start + depth])
            start = start + depth
        dp_rates = dp_rates[-4:]

        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = (
                down_ch_list[ch_idx + 2] if ch_idx + 2 < len(down_ch_list) else ch
            )
            layer_block = [
                ConvNeXtBlock(
                    dim=next_up_ch,
                    drop_path=dp_rates[ch_idx][j],
                    layer_scale_init_value=1.0e-6,
                )
                for j in range(num_blocks)
            ]
            self.uplist.append(
                nn.Sequential(
                    nn.Conv2d(ch, next_up_ch, kernel_size=7, padding=3), *layer_block
                )
            )

        self.clf = nn.Conv2d(next_up_ch, num_output_channels, (1, 1), bias=True)
        self.upsample2x = UpSample2x()
        return

    def forward(self, img_list):
        img_list = img_list / 255.0  # scale to 0-1

        is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not is_freeze):
            # assume output is after each down-sample resolution
            en_list = self.backbone(img_list)

        if self.conv1x1 is not None:
            x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list) + 1):
            y = en_list[-idx]
            x = self.upsample2x(x) + y
            x = self.uplist[idx - 1](x)
        output = self.clf(x)
        return output



class FCN_ResNet(nn.Module):
    def __init__(self, 
        num_input_channels=3,
        num_output_channels=2,
        freeze_encoder=True,
        pretrained_encoder=None,
    ):
        super(FCN_ResNet, self).__init__()
        # Normalize over last dimension
        self.freeze_encoder = freeze_encoder

        self.backbone = ResNetExt.resnet50(num_input_channels, pretrained_encoder)
        img_list = torch.rand([1, 3, 256, 256])
        out_list = self.backbone(img_list)
        # orderd from lores hires
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        self.conv1x1 = None
        if down_ch_list[0] != down_ch_list[1]:  # channel mapping for shortcut
            self.conv1x1 = nn.Conv2d(
                down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        self.uplist = nn.ModuleList()
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = ch
            if ch_idx + 2 < len(down_ch_list):
                next_up_ch = down_ch_list[ch_idx+2]
            self.uplist.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch), nn.ReLU(),
                    nn.Conv2d(ch, next_up_ch, (3, 3), padding=1, bias=False),
                    nn.BatchNorm2d(next_up_ch), nn.ReLU(),
                    nn.Conv2d(next_up_ch, next_up_ch, (3, 3), padding=1, bias=False),
                )
            )

        self.clf = nn.Conv2d(next_up_ch, num_output_channels, (1, 1), bias=True)
        self.upsample2x = UpSample2x()
        return

    def forward(self, img_list):
        img_list = img_list / 255.0  # scale to 0-1

        is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not is_freeze):
            # assume output is after each down-sample resolution
            en_list = self.backbone(img_list)

        if self.conv1x1 is not None:
            x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list)+1):
            y = en_list[-idx]
            x = self.upsample2x(x) + y
            x = self.uplist[idx-1](x)
        output = self.clf(x)
        return output
