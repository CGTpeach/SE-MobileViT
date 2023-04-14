import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import numpy as np
current_dir = os.path.abspath(os.path.dirname(__file__))
import torch
import torch.nn as nn
import csv
import pandas as pd
# from common_tools import get_mobilenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix,cohen_kappa_score,hamming_loss, jaccard_score,hinge_loss,f1_score
import matplotlib.pyplot as plt
import time
# from model import swin_base_patch4_window7_224 as create_model
# from model import resnet50
# from model_v2 import MobileNetV2
# from model_v3 import MobileNetV3
#from model import efficientnetv2_s as create_model
from model_se import mobile_vit_xx_small as create_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# ASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

batch_size = 64
num_classes = 4
classes =('0','1','2','3') # 为了输出引用
labels = ['0','1','2','3']

img_size = 224
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])}

# load data
# dataset = '/public/home/hk/nurou/data288/train_test'
# dataset = r'C:\Users\Administrator\Desktop\grad\train_test\test'
test_dir = r"/public/home/chenguotao2021/classification/MobileViT/data/4classt_t/test/"
data = {'test': datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])}
test_data_size = len(data['test'])
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)
idx_to_class = {v: k for k, v in data['test'].class_to_idx.items()}  #{0：0，1：1，2：2} 可以将之前名字和数据对应的函数调过来使用

# load model
# model = torch.load('/public/home/hk/nurou/code/vgg16/model/6_3_5.pth')
# model = MobileNetV2(num_classes=3).to(device)
# model.load_state_dict(torch.load(r'D:\学习记录\代码\test_code (2)\model1\m6_3_10.pth',map_location='cpu'))

#model = torch.load(r"/public/home/chenguotao2021/pytorch_classification/MobileViT/weights/best_model.pth",map_location='cpu')
#model.eval()
model = create_model(num_classes=4).to(device)
model_weight_path = "/public/home/chenguotao2021/classification/MobileViT/runs/2022_12_27_00_48_18/best_model.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


def get_filelist(test_dir):
    Filelist = []   #整个路径
    r1 = []   #存放类别名
    r2 = []   #存放图片名
    for home, dirs, files in os.walk(test_dir):   #home到test dirs存0 1 2 files存图片名
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            r1.append(home[-1])
            r2.append(filename)
    return Filelist, r1, r2


def predict(model, test_image_name):
    l1 = []  # 图片名
    l2 = []  # 预测值
    l3 = []  # 预测分数
    l4 = []  # 真实值
    l5 = []  #记录第0类预测分数
    l6 = []  #记录第1类预测分数
    l7 = []  #记录第2类预测分数
    l8 = []  #记录第3类预测分数
    transform = image_transforms['test']
    Filelist, r1 ,r2= get_filelist(test_image_name)
    print(len(Filelist))
    #model = create_model(num_classes=5).to(device)
    #model_weight_path = "/public/home/chenguotao2021/pytorch_classification/MobileViT/weight/mobilevit_xxs.pt"
    #model.load_state_dict(torch.load(model_weight_path, map_location=device))
    #model.eval()
    model =model.to(device)
    timeall = 0
    for file_index,file in enumerate(Filelist):
        test_image_name = file
        print(test_image_name)
        test_image = Image.open(test_image_name)
        test_image_tensor = transform(test_image).unsqueeze(0)
        img_ = test_image_tensor.to(device)

        start = time.time()
        out = model(img_)
        end = time.time()
        timed = end - start
        timeall = timeall + timed
        print(timed * 1000)  # 时间
        _, predicted = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0]
        temp = classes[predicted[0]]

        p = int(r1[file_index])
        print("image_name :", test_image_name, "Prediction : ", temp,
                  ", Score: ", percentage[predicted[0]].item(), "Real :", p)
        l1.append(r2[file_index])  #该图片的序号进行索引
        l2.append(temp)
        l3.append(percentage[predicted[0]].item())
        l4.append(p)
        l5.append(percentage[0].item())
        l6.append(percentage[1].item())
        l7.append(percentage[2].item())
        l8.append(percentage[3].item())

    print("avg time: ", timeall * 1000 / len(Filelist), " ms")
    return l1, l2, l3, l4, l5, l6, l7, l8, r1


#predict
l1, l2, l3, l4, l5, l6, l7, l8, r1 = predict(model, test_dir)

# evaluating indicator
y_pred = l2
y_true = r1
cm = confusion_matrix(y_true, y_pred)
TP = np.diag(cm)
FN = cm.sum(axis=1)-np.diag(cm)
FP = cm.sum(axis=0)-np.diag(cm)
TN = cm.sum() - (FP+FN+TP)
# Sensitivity
TPR = TP/(TP+FN)
# Specificity
TNR = TN/(TN+FP)
# Precision
PPV = TP/(TP+FP)
cls_num = len(labels)
# print(cls_num)
kappa = cohen_kappa_score(y_true,y_pred)  #全是调用的函数
ham_distance = hamming_loss(y_true,y_pred)
acc = cm.trace() / cm.sum()
print('test_data:{},acc:{:.2%},kappa:{:.2%}, ham_distance:{:.2%}'.format(test_data_size,acc,kappa,ham_distance))
for i in range(cls_num):
    print('class:{:<3}, total num:{:<3}, correct num:{:<3}  Recall( Sensitivity): {:.2%} Specificity: {:.2%} Precision: {:.2%} f1-score: {:.2%}'.format(
        labels[i], np.sum(cm[i, :]), cm[i, i],
        TPR[i],
        TNR[i],
        PPV[i],
        (2*TPR[i]*PPV[i])/(TPR[i]+PPV[i])))


# 保存csv
k = {'image_name': l1, 'Prediction': l2, 'Score': l3, 'Real': l4, 'score0': l5, 'score1': l6, 'score2': l7, 'score3':l8}
data = pd.DataFrame(k) #二维表头
data.to_csv('./v2.csv')



# 保存混淆矩阵
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=0)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

tick_marks = np.array(range(len(labels))) + 0.5  # [0.5 1.5 2.5]
np.set_printoptions(precision=2) #控制输出方式 precision：控制输出的小数点个数
plt.figure(figsize=(12, 8), dpi=120)
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array) #生成网格点坐标矩阵 点点对应
for x_val, y_val in zip(x.flatten(), y.flatten()): #填方格中数字，为0就不填
    c = cm[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.0f" % (c,), color='red', fontsize=15, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plot_confusion_matrix(cm, title='test confusion matrix')
# show confusion matrix
plt.savefig('./v2.png', format='png')
plt.show()
