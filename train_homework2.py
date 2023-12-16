import torch

import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm

from tensorboardX import SummaryWriter
import argparse
import hw2_dataloador
import rcnn_model
import time

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=61)
parser.add_argument("--batches", default=5)
args = parser.parse_args()

cudnn.benchmark = True

transform = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

train_dataset = hw2_dataloador.HW2VOCDataset(
    "/home/zlz/cv_dl_course/Data", "train", transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batches,
    shuffle=True,
    collate_fn=hw2_dataloador.collate_fn,
)
val_dataset = hw2_dataloador.HW2VOCDataset(
    "/home/zlz/cv_dl_course/Data", "val", transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batches,
    shuffle=True,
    collate_fn=hw2_dataloador.collate_fn,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(".", flush_secs=1)

model = rcnn_model.build_torchvision_model(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, [72], 0.1)
num_epochs = args.epochs
step = 1

for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_dataloader)
    running_loss = 0
    for i, (images, targets) in enumerate(pbar):
        images_i = list(image.to(device) for image in images)

        targets_processed = []
        for target in targets:
            target["image_id"] = torch.tensor([i]).to(device)
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)

            targets_processed.append(target)
        loss_dict = model(images_i, targets_processed)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        pbar.set_description(f"Training Epoch: {epoch}. Loss: {running_loss/(i+1):.3f}")
        writer.add_scalar(f"train_loss", running_loss / (i + 1), step)
        writer.add_scalar(
            f"train_loss_classifier", loss_dict["loss_classifier"].cpu(), step
        )
        writer.add_scalar(f"train_loss_box_reg", loss_dict["loss_box_reg"].cpu(), step)
        writer.add_scalar(
            f"train_loss_objectness", loss_dict["loss_objectness"].cpu(), step
        )

        step += 1
    schedular.step()
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"./model_{epoch}.pth")
