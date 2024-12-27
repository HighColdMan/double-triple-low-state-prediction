import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sn  # 画图模块
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from numpy import interp


def plot_loss(file_name, loss, test_losses, epochs, title, accuracy=None):
    fig, ax1 = plt.subplots()

    # 绘制训练损失和测试损失
    ax1.plot(range(1, epochs + 1), loss, '.-', label='Training Loss', color='tab:blue')  # 训练损失
    ax1.plot(range(1, epochs + 1, 5), test_losses, 'x-', label='Testing Loss', color='tab:orange')  # 测试损失
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建右边的y轴
    ax2 = ax1.twinx()
    
    if accuracy is not None:
        ax2.plot(range(1, epochs + 1, 5), accuracy, 'o-', label='Testing Accuracy', color='tab:green')  # 测试准确率
        ax2.set_ylabel('Accuracy', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

    # 设置标题
    plt.title(title)
    
    # 设置图例，位置调整避免重叠
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 保存图表
    plt.savefig(file_name)

    # 显示图表
    plt.show()

def plot_matrix2(y_true, y_pred, confusion_path, title=None):
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)  # 混淆矩阵
    # annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sn.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g', xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'], cbar=False)
    # sns.heatmap(C1,fmt='g', cmap=name,annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick)
    # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    # xticklabels、yticklabels指定横纵轴标签
    ax.set_title(title)  # 标题
    ax.set_xlabel('Predict')  # x轴
    ax.set_ylabel('True')  # y轴
    plt.savefig(confusion_path)
    # plt.show()


def plot_matrix1(y_true, y_pred):
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)  # 混淆矩阵
    # annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sn.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g', xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'], cbar=False)
    # sns.heatmap(C1,fmt='g', cmap=name,annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick)
    # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    # xticklabels、yticklabels指定横纵轴标签
    ax.set_title('confusion_matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()

def icc(Y, icc_type="icc(3,1)"):
    """
    Args:
        Y: 待计算的数据
        icc_type: 共支持 icc(2,1), icc(2,k), icc(3,1), icc(3,k)四种
    """

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    # MSC = SSC / dfc / n
    MSC = SSC / dfc

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc(2,1)" or icc_type == 'icc(2,k)':
        if icc_type=='icc(2,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    elif icc_type == "icc(3,1)" or icc_type == 'icc(3,k)':
        if icc_type=='icc(3,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC


def getClassICC(y_pred, y_true, icc_type):
    pred_class = getClass(y_pred)
    true_class = getClass(y_true)
    data = [pred_class, true_class]
    iccs = icc(np.array(data), icc_type=icc_type)
    return iccs


def getClass(y):
    label = []
    for i in range(len(y)):
        y_pre = y[i]  # ART_MBP, Primus/MAC, BIS/BIS
        if (y_pre[0]>75) and (y_pre[1]<0.8) and (y_pre[2]<45):
            label.append(0)
        elif(y_pre[0]<75) and (y_pre[1]>0.8) and (y_pre[2]<45):
            label.append(1)
        elif(y_pre[0]<75) and (y_pre[1]<0.8) and (y_pre[2]>45):
            label.append(2)
        elif(y_pre[0]<75) and (y_pre[1]<0.8) and (y_pre[2]<45):
            label.append(3)
        else:
            label.append(4)
    return label


def getAccuracy(pre_y, true_y):
    pred_class = getClass(pre_y)
    true_class = getClass(true_y)
    # 将label 转换为 one-hot
    pred_class = np.eye(5)[pred_class]
    true_class = np.eye(5)[true_class]
    # 计算准确率
    accuracy = np.mean(np.equal(np.argmax(pred_class, axis=1), np.argmax(true_class, axis=1)))
    return accuracy


def boxdata(y_pred, y_true):
    data = np.array(y_pred) - np.array(y_true)
    data1 = data[:, :, 0].reshape(-1)
    data2 = data[:, :, 1].reshape(-1)
    data3 = data[:, :, 2].reshape(-1)
    data = [data1, data2, data3]

    boxplot(data)


def boxplot(data):
    # 箱型图名称
    labels = ["ART_MBP", "MAC", "BIS"]
    # 三个箱型图的颜色 RGB （均为0~1的数据）
    colors = [(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.),
              (137 / 255., 128 / 255., 68 / 255.)]

    # 绘制箱型图
    # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
    bplot = plt.boxplot(data, patch_artist=True, showfliers=False, labels=labels, positions=(1, 1.4, 1.8), widths=0.3)
    # 设置箱型图颜色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(data, patch_artist=True, showfliers=False, labels=labels, positions=(2.5, 2.9, 3.3), widths=0.3)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(data, patch_artist=True, showfliers=False, labels=labels, positions=(4, 4.4, 4.8), widths=0.3)
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    bplot4 = plt.boxplot(data, patch_artist=True, showfliers=False, labels=labels, positions=(5.5, 5.9, 6.3), widths=0.3)
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2.5, 4, 5.5]
    x_position_fmt = ["model1", "model2", "model3", "model4"]
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    plt.ylabel('MAE')
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'], labels, loc='lower right')  # 绘制表示框，右下角绘制
    # plt.savefig(fname="pic.png", figsize=[10, 10])
    plt.show()


def mulboxplot(data1, data2, data3, data4, name, config=None):
    # 箱型图名称
    labels = ["ART_MBP", "MAC", "BIS"]
    # 三个箱型图的颜色 RGB （均为0~1的数据）
    colors = [(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.),
              (137 / 255., 128 / 255., 68 / 255.)]

    # 绘制箱型图
    # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
    bplot = plt.boxplot(data1, patch_artist=True, showfliers=False, labels=labels, positions=(1, 1.4, 1.8), widths=0.3)

    # 设置箱型图颜色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(data2, patch_artist=True, showfliers=False, labels=labels, positions=(2.5, 2.9, 3.3), widths=0.3)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(data3, patch_artist=True, showfliers=False, labels=labels, positions=(4, 4.4, 4.8), widths=0.3)
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    bplot4 = plt.boxplot(data4, patch_artist=True, showfliers=False, labels=labels, positions=(5.5, 5.9, 6.3), widths=0.3)
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2.5, 4, 5.5]
    x_position_fmt = ["model1", "model2", "model3", "model4"]
    
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    plt.ylabel('MAE')
    plt.title(name)
    plt.xlabel('model')
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'], labels, loc='lower right')  # 绘制表示框，右下角绘制
    plt.savefig(fname=os.path.join(config.save_box_path, f'{name}box.png'))
    plt.show()

def get_roc_auc(trues, preds, path, title=None):
    labels = [0, 1, 2, 3, 4]
    nb_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(trues, preds)
    for i in range(nb_classes):
        trues = np.array(trues)
        preds = np.array(preds)
        # trues = trues.astype('int64')
        # trues = np.eye(5)[trues]
        # preds = np.eye(5)[preds]
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(trues.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average: {0:0.2f}'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average: {0:0.2f}'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', "purple", "green", "gray", "magenta"])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Situation {0}: {1:0.2f}'.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(path)
    # plt.show()