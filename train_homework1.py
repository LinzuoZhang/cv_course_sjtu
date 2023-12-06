# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from tqdm import tqdm

from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=25)
args = parser.parse_args()

cudnn.benchmark = True
plt.ion()  # interactive mode
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            # ),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = "/home/zlz/cv_dl_course/data/data"
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
batch_sizes = {"train": 4, "val": 4}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_sizes[x], shuffle=True, num_workers=4
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(".", flush_secs=1)


model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model.fc = nn.Linear(num_ftrs, 30)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
output_dir = "."
num_epochs = args.epochs
since = time.time()

# Create a temporary directory to save training checkpoints
best_model_params_path = os.path.join(output_dir, "best_model_params.pt")

torch.save(model.state_dict(), best_model_params_path)
best_acc = 0.0
global_step = {"train": 0, "val": 0}
for epoch in range(num_epochs):
    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        running_corrects_top5 = 0

        pbar = tqdm(dataloaders[phase], ncols=100)
        # Iterate over data.
        # for inputs, labels in dataloaders[phase]:
        for i, data in enumerate(pbar):
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                top5_preds = outputs.argsort(axis=1)[:, -5:]
                correct = np.sum(
                    [label in top5 for label, top5 in zip(labels, top5_preds)]
                )
                running_corrects_top5 += correct
                top5_acc = running_corrects_top5 / len(labels) / (i + 1)
                writer.add_scalar(f"top5_acc_{phase}", top5_acc, global_step[phase])
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer_ft.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)
            acc = running_corrects / (i + 1) / inputs.size(0)
            pbar.set_description_str(
                f"{phase}_{epoch:03d} acc:{acc:.4f} loss:{running_loss/(i+1)/inputs.size(0):.4f} "
            )
            writer.add_scalar(f"loss_{phase}", loss, global_step[phase])
            writer.add_scalar(f"acc_top1_{phase}", acc, global_step[phase])

            global_step[phase] = global_step[phase] + 1
        if phase == "train":
            exp_lr_scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        writer.add_scalar(
            f"acc_{phase}",
            epoch_acc,
            epoch,
        )
        # deep copy the model
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_params_path)
time_elapsed = time.time() - since
print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(f"Best val Acc: {best_acc:4f}")
