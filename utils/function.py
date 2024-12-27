import numpy as np
import pandas as pd
import os

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
