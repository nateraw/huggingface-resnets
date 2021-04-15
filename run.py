from argparse import ArgumentParser
from pathlib import Path

import torch
from torchvision.models.utils import load_state_dict_from_url

from modelz import ResnetConfig, ResnetModel
from modelz.modeling_resnet import RESNET_PRETRAINED_TORCHVISION_CONFIG_MAP, RESNET_PRETRAINED_TORCHVISION_URL_MAP


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


def another_test():
    from torchvision.models import resnet50
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor, Resize, Compose
    from torch.utils.data import DataLoader
    tsfm = Compose([Resize(224), ToTensor()])
    ds = CIFAR10('./', download=False, transform=tsfm)
    loader = DataLoader(ds, batch_size=4)
    x, y = next(iter(loader))

    model = ResnetModel.from_pretrained('nateraw/resnet50')
    return (x, y), model

if __name__ == "__main__":
    args = parse_args()
    # main(args)
    # real_dirty_test(args)
    (x, y), model = another_test()
