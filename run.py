from argparse import ArgumentParser
from pathlib import Path

import torch
from torchvision.models.utils import load_state_dict_from_url

from resnet_hf import ResnetConfig, ResnetModel

RESNET_PRETRAINED_TORCHVISION_URL_MAP = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

RESNET_PRETRAINED_TORCHVISION_CONFIG_MAP = {
    "resnet18": ResnetConfig(block="basic", layers=[2, 2, 2, 2]),
    "resnet34": ResnetConfig(block="basic", layers=[3, 4, 6, 3]),
    "resnet50": ResnetConfig(block="bottleneck", layers=[3, 4, 6, 3]),
    "resnet101": ResnetConfig(block="bottleneck", layers=[3, 4, 23, 3]),
    "resnet152": ResnetConfig(block="bottleneck", layers=[3, 8, 36, 3]),
    "resnext50_32x4d": ResnetConfig(block="bottleneck", layers=[3, 4, 6, 3], groups=32, width_per_group=4),
    "resnext101_32x8d": ResnetConfig(block="bottleneck", layers=[3, 4, 23, 3], groups=32, width_per_group=8),
    "wide_resnet50_2": ResnetConfig(block="bottleneck", layers=[3, 4, 6, 3], width_per_group=128),
    "wide_resnet101_2": ResnetConfig(block="bottleneck", layers=[3, 4, 23, 3], width_per_group=128),
}


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./converted_pretrained_models")
    return parser.parse_args(args)


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    for model_name, config in RESNET_PRETRAINED_TORCHVISION_CONFIG_MAP.items():
        model = ResnetModel(config)
        state_dict = load_state_dict_from_url(RESNET_PRETRAINED_TORCHVISION_URL_MAP.get(model_name))
        model.resnet.load_state_dict(state_dict)
        model.save_pretrained(save_dir / model_name)
        config.save_pretrained(save_dir / model_name)


def real_dirty_test(args):
    from torchvision.models import resnet50
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor, Resize, Compose
    from torch.utils.data import DataLoader

    tsfm = Compose([Resize(224), ToTensor()])
    ds = CIFAR10('./', download=False, transform=tsfm)
    loader = DataLoader(ds, batch_size=4)
    x, y = next(iter(loader))

    tv_model = resnet50(pretrained=True)

    hf_config = RESNET_PRETRAINED_TORCHVISION_CONFIG_MAP.get('resnet50')
    hf_model = ResnetModel(hf_config)
    hf_model.load_state_dict(torch.load(Path(args.save_dir) / 'resnet50' / 'pytorch_model.bin'))

    tv_out = tv_model(x)
    hf_out = hf_model(x)

    assert torch.equal(tv_out, hf_out)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    real_dirty_test(args)
