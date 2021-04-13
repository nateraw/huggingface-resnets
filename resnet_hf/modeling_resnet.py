from huggingface_hub import ModelHubMixin
from torch import nn

from .configuration_resnet import ResnetConfig
from .resnet import BasicBlock, Bottleneck, ResNet

RESNET_BLOCK_TYPE_MAP = {"bottleneck": Bottleneck, "basic": BasicBlock}


class ResnetPreTrainedModel(nn.Module, ModelHubMixin):

    config_class = ResnetConfig
    base_model_prefix = "resnet"

    def __init__(self, *args, **kwargs):
        super().__init__()


class ResnetModel(ResnetPreTrainedModel):
    def __init__(self, config):
        super().__init__()

        self.config = config

        block = RESNET_BLOCK_TYPE_MAP.get(config.block)
        if block is None:
            raise RuntimeError("Block must be either 'bottleneck' or 'basic-block'")

        self.resnet = ResNet(
            block=block,
            layers=config.layers,
            num_labels=config.num_labels,
            zero_init_residual=config.zero_init_residual,
            groups=config.groups,
            width_per_group=config.width_per_group,
            replace_stride_with_dilation=config.replace_stride_with_dilation,
            norm_layer=config.norm_layer,
        )
        self.config.output_size = list(self.resnet.modules())[-4].weight.size(0)

    def forward(self, x):
        return self.resnet(x)
