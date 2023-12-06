import torch

import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm

from tensorboardX import SummaryWriter
import argparse

import hw2_data_loador
import rcnn_model

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=3)
parser.add_argument("--batches", default=10)
args = parser.parse_args()

cudnn.benchmark = True

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataloader = hw2_data_loador.read_datasets(
    "/home/zlz/cv_dl_course/Dataset/train", "train", transform, args.batches
)
val_dataloader = hw2_data_loador.read_datasets(
    "/home/zlz/cv_dl_course/Dataset/vali", "val", transform, args.batches
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(".", flush_secs=1)

model = rcnn_model.build_torchvision_model(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = args.epochs
step = 1
running_loss = 0
running_loss_val = 0
best_loss = 100000
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_dataloader)
    for i, (images, targets) in enumerate(pbar):
        images_i = list(image.to(device) for image in images)

        targets_processed = []
        for target in targets:
            target = hw2_data_loador.convert_targets(
                target["annotation"]["object"], device
            )
            target["image_id"] = torch.tensor([i]).to(device)
            targets_processed.append(target)
        loss_dict = model(images_i, targets_processed)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        pbar.set_description(f"Training Epoch: {epoch}. Loss: {running_loss/step:.3f}")
        writer.add_scalar(f"train_loss", running_loss / step, step)
        writer.add_scalar(
            f"train_loss_classifier", loss_dict["loss_classifier"].cpu(), step
        )
        writer.add_scalar(f"train_loss_box_reg", loss_dict["loss_box_reg"].cpu(), step)
        writer.add_scalar(
            f"train_loss_objectness", loss_dict["loss_objectness"].cpu(), step
        )

        val_data_iterator = iter(val_dataloader)
        val_batch_data = next(val_data_iterator)
        data_val, tartget = val_batch_data
        images_val = list(image.to(device) for image in data_val)
        for target in targets:
            target = hw2_data_loador.convert_targets(
                target["annotation"]["object"], device
            )
            targets_processed.append(target)
        with torch.no_grad():
            loss_dict = model(images_val, targets_processed)
            running_loss_val = sum(loss for loss in loss_dict.values())
            writer.add_scalar(f"val loss", running_loss_val, step)
            writer.add_scalar(
                f"val_loss_classifier", loss_dict["loss_classifier"].cpu(), step
            )
            writer.add_scalar(
                f"val_loss_box_reg", loss_dict["loss_box_reg"].cpu(), step
            )
            writer.add_scalar(
                f"val_loss_objectness", loss_dict["loss_objectness"].cpu(), step
            )
        step += 1
        if running_loss_val < best_loss:
            best_loss = running_loss_val
            best_dict = model.state_dict()
torch.save(best_dict, "/home/zlz/cv_dl_course/weights/model.pth")
