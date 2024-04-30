
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os
import numpy as np
import torch
import math
from PIL import Image


def display_cls_confusion_matrix(confusion_mat, labels, test_number,name,method):
    plt.figure(figsize=(9, 7),dpi=300)
    # win_id = self.display_id + 4 if name == 'test' else self.display_id + 5
    color_map = 'Blues' #if name == 'test' else 'Orange'
    confusion_mat = np.array(confusion_mat, dtype=float)
    confusion_mat_number = np.zeros(confusion_mat.shape)
    # test_number = np.array(test_number, dtype=float)
    test_number = np.sum(confusion_mat, axis=1)
    # print('confusion_mat',confusion_mat)
    Expression_count = np.zeros(len(confusion_mat))
    Overall_Accuracy = 0
    
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat[i])):
            # print(confusion_mat[i,j], test_number[i])
            confusion_mat[i,j] = confusion_mat[i,j] / test_number[i]
            # print(confusion_mat[i,j], test_number[i])
        confusion_mat_number[i,:] = confusion_mat[i,:] * test_number[i]/100
        for j in range(len(confusion_mat)):
            if(i == j):
                Expression_count[i] += confusion_mat[i, j]
                Overall_Accuracy += confusion_mat[i, i] * test_number[i]
    Overall_Accuracy = Overall_Accuracy / test_number.sum()
    Overall_Accuracy = np.around(Overall_Accuracy, 2)


    UAR = np.around(sum(Expression_count)/len(confusion_mat), 2)
    WAR = Overall_Accuracy

    # print('Expression_count : {}, Overall_Accuracy: {}'.format(Expression_count,Overall_Accuracy))

    # title = 'Confusion Matrix of {} on {} (Accuracy: {}%)'.format(method,name,Overall_Accuracy)
    save_name = '/home/et23-maixj/mxj/SIRV_baseline/plt/'\
                +"Confusion Matrix of %s on %s UAR %s and WAR %s.jpg" % (method,name,UAR,WAR)
    df_cm = pd.DataFrame(confusion_mat, index = labels, columns = labels)
    # print('df_cm',df_cm)
    # print('test_number',test_number)
    # print('confusion_mat_number',confusion_mat_number)
    #
    # f, ax = plt.subplots(figsize=(9, 6))
    # ax.set_title(title)
    # ax = sn.heatmap(df_cm, annot=True, cmap=color_map,fmt='.2f',annot_kws={'size':9,'weight':'bold', 'color':'red'})
    ax = sn.heatmap(df_cm, annot=True, cmap=color_map,fmt='.2f',annot_kws={'size':10})
    plt.savefig(save_name, bbox_inches='tight')
    # plt.show()


