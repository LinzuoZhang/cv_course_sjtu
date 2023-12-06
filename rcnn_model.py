import torch.nn as nn
import torch

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def build_torchvision_model(device, use_tv_boxpredictor=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 21  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    if use_tv_boxpredictor:
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )
    model.roi_heads.box_predictor = DetectionHead(in_features, num_classes=num_classes)
    model = model.to(device)
    return model


class DetectionHeadChatGPT(nn.Module):
    def __init__(self, in_channels, num_classes, roi_size):
        super(DetectionHead, self).__init__()
        self.roi_size = roi_size
        self.roi_pooling = nn.AdaptiveMaxPool2d(roi_size)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, 4 * num_classes)

    def forward(self, x):
        # RoI pooling

        # Fully connected layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))

        # Classification scores
        cls_score = self.cls_score(x)

        # Bounding box regression predictions
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, 4 * num_classes)

    def forward(self, x):
        # RoI pooling

        # Fully connected layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        # Classification scores
        cls_score = self.cls_score(x)
        # Bounding box regression predictions
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


# 2. 加载预训练的ResNet模型
def load_pretrained_resnet():
    backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


# 3. 构建模型
def build_model(num_classes):
    backbone = load_pretrained_resnet()
    backbone.out_channels = 2048
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        num_classes=num_classes,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = DetectionHead(in_features, num_classes)
    return model
