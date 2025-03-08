import numpy as np
import torch
import torch.nn as nn

from easydict import EasyDict

from tools.torchie.cnn import constant_init, kaiming_init, xavier_init
from tools.models.utils import Empty, GroupNorm, Sequential
from tools.models.utils import build_norm_layer
from tools.utils.dist import logger


# https://chatgpt.com/c/67a4b6eb-f1c8-8006-bdd4-b98956fb8937
# https://medium.com/lsc-psd/faster-r-cnn%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8Brpn%E3%81%AE%E4%B8%96%E7%95%8C%E4%B8%80%E5%88%86%E3%81%8B%E3%82%8A%E3%82%84%E3%81%99%E3%81%84%E8%A7%A3%E8%AA%AC-dfc0c293cb69
# ROI https://cvml-expertguide.net/terms/cv/roi/ 及び ROI Pooling と ROI Align の違い
# FPN と　RPNの違い
class RPN(nn.Module):
    def __init__(self, cfg: EasyDict):
        super(RPN, self).__init__()
        self._layer_strides = cfg.ds_layer_strides
        self._num_filters = cfg.ds_num_filters
        self._layer_nums = cfg.layer_nums
        self._upsample_strides = cfg.us_layer_strides
        self._num_upsample_filters = cfg.us_num_filters
        self._num_input_features = cfg.num_input_features

        self._norm_cfg = cfg.norm_cfg if not cfg.norm_cfg else dict(type="BN", eps=1e-3, momentum=0.01)

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_strides) - len(self._upsample_strides)
        must_equal_list = []

        # _upsample_strides: [0.5, 1, 2],

        # : [2, 2, 2]
        # _upsample_start_idx: 0
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )
         # must_equal_list: [0.25, 0.25, 0.25]

        # all of value is same in must_equal_list
        for val in must_equal_list:
            assert val == must_equal_list[0]

        # in_:[64, 64, 128]  -> out: [64, 128, 256] so the source code take like this *self._num_filters[:-1]]
        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            # down-sampling
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )

            # up-sampling
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                # us_layer_strides=[0.5, 1, 2],
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                # if true nn.ConvTranspose2d is used, other wise nn.Conv2d is used
                if stride > 1:
                    # 三層目は入力を2倍に
                    print("ConvTranspose2d")
                    print(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride,)
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    # 一層目は画像サイズを入力を半分に、二層目は入力のまま
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)

        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)


    def init_weights(self):
        for m in self.modules():
            # なぜinit_weightsを呼ぶのか
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        # TODO: 普通のSequentialと何が違うのか
        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            # TODO: 普通のnormと何が違うのか
            build_norm_layer(self._norm_cfg, planes)[1],
            nn.ReLU(),
        )

        # shapeを変えずに特徴量を生成
        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self._norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block, planes

    def forward(self, x):
        """
        algo:
            - down-layers[layer1, layer2, layer3]
            - up-layers[layer1, layer2, layer3]
            # the inputs value of each down-layers from previous down-layer output (not up-layer).
            - layer1: downlayer -> uplayer (the output value of the uplayer is appended in the list)
            - layer2: downlayer -> uplayer
            - layer3: downlayer -> uplayer
            - finaly: concat all appended up-layers output in the list:

        """
        ups = []
        for i in range(len(self.blocks)):
            print(f"before block {i} {x.shape}")
            x = self.blocks[i](x)
            print(f"after block {i} {x.shape}")
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            # concat channel-wise
            for i in ups:
                print(f"before concat {i.shape}")
            x = torch.cat(ups, dim=1)
            print(f"after concat {x.shape}")
        return x