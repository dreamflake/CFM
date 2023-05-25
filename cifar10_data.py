import torch, torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os, shutil
from pathlib import Path
import random

random.seed(960224)
if __name__ == '__main__':
    for class_idx in range(10):
        dirr = Path("CIFAR10") / str(class_idx)
        desti = Path("CIFAR10_targeted")
        filenames = os.listdir(dirr)
        selected = random.sample(filenames, 100)
        possible_targets = list(range(10))
        possible_targets.remove(class_idx)
        for file in selected:
            target_label = random.choice(possible_targets)
            shutil.copyfile(dirr / file, desti / f"{os.path.splitext(file)[0]}_{class_idx}_{target_label}.png")
