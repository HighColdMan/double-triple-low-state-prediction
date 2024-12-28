import pandas as pd
import os
import streamlit as st
from models import mymodel
import numpy as np
import torch
import joblib
from utils import function as fc


st.title("EEEE")
st.title('Deep Learning Models of Double-Triple Low-state Prediction')  # 算法名称 and XXX


def show_paper_result():

    st.header("Results of the classification model based on input Model 1")
    st.subheader("The time window is 5")
    # window 5
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/classificaiton/window_5/Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/classificaiton/window_5/Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/classificaiton/window_5/Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/classificaiton/window_5/Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/classificaiton/window_5/Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/classificaiton/window_5/Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)
    # window 10
    st.subheader("The time window is 10")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/classificaiton/window_10/Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/classificaiton/window_10/Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/classificaiton/window_10/Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/classificaiton/window_10/Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/classificaiton/window_10/Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/classificaiton/window_10/Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)
    # window 15
    st.subheader("The time window is 15")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/classificaiton/window_15/Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/classificaiton/window_15/Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/classificaiton/window_15/Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/classificaiton/window_15/Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/classificaiton/window_15/Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/classificaiton/window_15/Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)
    
    st.subheader("The precision, recall, and F1 score performance of classification model")
    df = pd.read_csv('result/classification.csv', index_col=0)
    st.table(df)

    
    # Regression model result
    st.header("Results of the regression model based on input Model 1")
    st.subheader("winow 5 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/regression/window_5/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/regression/window_5/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/regression/window_5/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)
    st.subheader("winow 10 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/regression/window_10/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/regression/window_10/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/regression/window_10/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)
    st.subheader("winow 15 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/regression/window_15/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/regression/window_15/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/regression/window_15/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("The accuracy, mean absolute error (MAE), and intra class correlation of different regression models")
    df2 = pd.read_csv("result/regression.csv", index_col=0)
    st.table(df2)


def data_input():
    st.subheader("请选择输入连续多少分钟的数据")
    win = st.radio("window", [5, 10, 15])

    columns = ["ART_MBP", "MAC", "BIS"]

    data = {col: [None] * win for col in columns}
    # 使用 DataFrame 保存初始表格
    df3 = pd.DataFrame(data)
    st.subheader("Input Data")
    updated_data = []

    for i in range(win):
        st.write(f"Time: min {i + 1}")
        row_data = []
        cols = st.columns(len(columns)) 
        for j, col in enumerate(columns):
            with cols[j]:  # 在对应列中创建输入框
                value = st.text_input(f"{col} (Time: min {i + 1})", key=f"{col}_{i}")
                row_data.append(value)
        updated_data.append(row_data)
        # st.write(f"Row {i + 1}")
        # for col in columns:
        #     value = st.text_input(f"Enter value for {col} (Row {i + 1})", key=f"{col}_{i}")
        #     row_data.append(value)
        # updated_data.append(row_data)
    
    st.subheader("Filled Table")
    filled_df = pd.DataFrame(updated_data, columns=columns)
    filled_df.index = range(1, len(filled_df) + 1)
    st.write(filled_df)



    return win, filled_df

def process_classification(win, inp):
    min_list = [5, 10, 15]
    train_x = []
    # print(inp)
    # print(inp.iloc[:].values.tolist())
    train_x.append(inp.iloc[:].values.tolist())
    # print(train_x)
    train_x = np.array(train_x, dtype='float32')
    train_x = torch.from_numpy(train_x)

    pred = []
    ckpt_path = os.path.join('checkpoint', f'classification-{win}')
    for min in min_list:
        
        ckpt = torch.load(os.path.join(ckpt_path, f'{min}min', 'best_model_fold_1.pth'), weights_only=True, map_location=torch.device('cpu'))
        net = mymodel.CNN_LSTM_GRU_ResNet_Model(3, 64, 64, 5)   
        net.load_state_dict(ckpt)
        net.eval()
        output = net(train_x)
        # print(f'min {min} output: {output}')
        pred.append(torch.argmax(output))
    # print(f'分类模型输出：{pred}')
    return pred


def process_regression(win, inp):
    min_list = [5, 10, 15]


    pred = []
    ckpt_path = os.path.join('checkpoint', f'regression-{win}')
    for min in min_list:
        train_x = []
        mm = joblib.load(os.path.join('dataset', f'scaler_win{min}.pkl'))
        train_x.append(inp.iloc[:].values.tolist())
        # print(train_x)
        train_x = np.array(train_x, dtype='float32')
        i, j, k = train_x.shape
        train_x = train_x.reshape(-1, k)
        train_x = mm.fit_transform(train_x)
        train_x = train_x.reshape(i, j, k)
        train_x = torch.from_numpy(train_x)

        ckpt = torch.load(os.path.join(ckpt_path, f'{min}min', 'best_model_fold_1.pth'), weights_only=True, map_location=torch.device('cpu'))
        net = mymodel.CNN_LSTM_GRU_ResNet_Model(3, 64, 64, 3)
        net.load_state_dict(ckpt)
        net.eval()
        output = net(train_x).detach().numpy()
        output = mm.inverse_transform(output)[:, :3].reshape(-1)
        output[0] = output[0].astype(int)
        output[2] = output[2].astype(int)
        output[1] = np.around(output[1], 1)
        pred.append(int("".join(map(str, fc.getClass(output)))))
  
    return pred


if __name__ == "__main__":

    
    show_paper_result()

    win, inp = data_input()


    btn_cls = st.button('分类模型预测')
    btn_reg = st.button('回归模型预测')

    if btn_cls:
        p = process_classification(win, inp)
        for i, min in enumerate([5, 10, 15]):
            st.markdown(f"连续{win}分钟输入的分类模型预测{min}分钟后最可能发生的结果为：Situation {p[i]+1}")
    if btn_reg:
        r = process_regression(win, inp)
        for i, min in enumerate([5, 10, 15]):
            # st.markdown(f"连续{win}分钟输入的回归模型预测{min}分钟后最可能发生的结果为：Situation {r[i]+1}")
            st.markdown(f"连续{win}分钟输入的回归模型预测{min}分钟的结果为-ART_MBP:{r[i][0]}-MAC:{r[i][1]:.1f}-BIS:{r[i][2]}")

            
