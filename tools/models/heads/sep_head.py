import torch.nn as nn

from tools.models.utils import Sequential
# TODO: Understand this method
from tools.torchie.cnn import kaiming_init

class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ) -> None:
        super(SepHead, self).__init__()
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            # TODO: Check this code, as why we don't append the module directly
            fc = Sequential()
            for i in range(num_conv - 1):
                # TODO: Check this code, as what is different between fc.add and fc.add_module
                fc.add(nn.Conv2d(in_channels,
                                 head_conv,
                                 kernel_size=final_kernel,
                                 stride=1,
                                 padding=final_kernel // 2,
                                 bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

                fc.add(nn.Conv2d(head_conv,
                                 classes,
                                 kernel_size=final_kernel,
                                 stride=1,
                                 padding=final_kernel // 2,
                                 bias=True))

                if 'hm' in head:
                    # TODO: what is fill
                    fc[-1].bias.data.fill_(init_bias)
                else:
                    for m in fc.modules:
                        if isinstance(m, nn.Conv2d):
                            kaiming_init(m)
                self.__setattr__(head, fc)


    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict