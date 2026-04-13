from pathlib import Path
import torch
from cyclegan.model import CycleGAN
import torchvision.transforms as tr
from PIL import Image
import numpy as np

TRANSFORM = tr.Compose([
        tr.Resize((256, 256)),
        tr.ToTensor(),
        tr.Normalize((0.5,), (0.5,))
    ])

def load_model(checkpoint_path: str) -> CycleGAN:
    model = CycleGAN(training=False)
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location="cpu"
    )

    model.generator_A.load_state_dict(checkpoint["generator_A"])
    model.generator_B.load_state_dict(checkpoint["generator_B"])

    model.eval()
    return model

def apply_transform(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image)

def denormalize(img: torch.Tensor) -> np.ndarray:
    return (img.detach().cpu() * 0.5 + 0.5).numpy().transpose((1, 2, 0))
