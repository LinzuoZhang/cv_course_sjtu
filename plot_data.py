import pandas as pd
import matplotlib.pyplot as plt

file_path = "/home/zlz/cv_dl_course/result/tensorboard_data/run117_trainloss.csv"
train_loss_data = pd.read_csv(file_path)

x = train_loss_data["Step"]
train_loss = train_loss_data["Value"]

file_path = "/home/zlz/cv_dl_course/result/tensorboard_data/valilosscsv.csv"
val_loss_data = pd.read_csv(file_path)

x = val_loss_data["Step"]
val_loss = val_loss_data["Value"]

plt.figure(1)
# 绘制曲线
plt.grid(True)
plt.plot(x, train_loss, label="Train Loss")
plt.plot(x, val_loss, label="Val Loss")
plt.xlabel("Step-axis Label")
plt.ylabel("Loss-axis Label")
plt.title("Loss Curve")
plt.legend()
plt.savefig("result/loss_curve.jpg")


file_path = "/home/zlz/cv_dl_course/result/tensorboard_data/boxreg.csv"
reg_loss_data = pd.read_csv(file_path)

x = reg_loss_data["Step"]
reg_loss = reg_loss_data["Value"]

# 绘制曲线
plt.figure(2)
plt.grid(True)
plt.plot(x, reg_loss, label="Box regression loss")
plt.xlabel("Step-axis Label")
plt.ylabel("Loss-axis Label")
plt.title("Loss Curve")
plt.legend()
plt.savefig("result/loss_reg_curve.jpg")

file_path = "/home/zlz/cv_dl_course/result/tensorboard_data/classfier.csv"
classfier_loss_data = pd.read_csv(file_path)

x = classfier_loss_data["Step"]
classfier_loss = classfier_loss_data["Value"]

# 绘制曲线
plt.figure(3)
plt.grid(True)
plt.plot(x, classfier_loss, label="Classifier loss")
plt.xlabel("Step-axis Label")
plt.ylabel("Loss-axis Label")
plt.title("Loss Curve")
plt.legend()
plt.savefig("result/loss_class_curve.jpg")

file_path = "/home/zlz/cv_dl_course/result/tensorboard_data/objectness.csv"
objectness_loss_data = pd.read_csv(file_path)

x = objectness_loss_data["Step"]
objectness_loss = objectness_loss_data["Value"]

# 绘制曲线
plt.figure(4)
plt.grid(True)
plt.plot(x, objectness_loss, label="Objectness loss")
plt.xlabel("Step-axis Label")
plt.ylabel("Loss-axis Label")
plt.title("Loss Curve")
plt.legend()
plt.savefig("result/loss_object_curve.jpg")
