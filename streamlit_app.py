import pandas as pd
import os
import streamlit as st
from models import mymodel
import numpy as np
import torch
import joblib
from utils import function as fc
import xgboost as xgb
from models.mymodel import MS_CNN_Transformer


st.title('Deep Learning Models of Double-Triple Low-state Prediction')


def show_paper_result():

    st.header("Results of the XGBoost classification Model based on input Model 1")
    st.subheader("winow 5 boxdata")
    # window 5
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_5/xgboost Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_5/xgboost Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_5/xgboost Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_5/xgboost Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_5/xgboost Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_5/xgboost Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)


    # window 10
    st.subheader("The time window is 10")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_10/xgboost Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_10/xgboost Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_10/xgboost Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_10/xgboost Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_10/xgboost Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_10/xgboost Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)


    # window 15
    st.subheader("The time window is 15")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_15/xgboost Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_15/xgboost Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_15/xgboost Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_15/xgboost Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_15/xgboost Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_15/xgboost Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)


    st.subheader("The precision, recall, and F1 score performance of XGBoost model")
    df1 = pd.read_csv('result/xgboost_cls.csv', index_col=0) ###########################
    st.table(df1)

    st.header("Results of the XGBoost regression Model based on input Model 1")

    st.subheader("winow 5 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_5/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_5/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_5/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("winow 10 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_10/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_10/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_10/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("winow 15 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/xgboost/window_15/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/xgboost/window_15/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/xgboost/window_15/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("The accuracy, mean absolute error (MAE), and intra class correlation of different regression models")
    df2 = pd.read_csv("result/xgboost_res.csv", index_col=0) ############
    st.table(df2)

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
    df3 = pd.read_csv('result/classification.csv', index_col=0)
    st.table(df3)

    
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
    df4 = pd.read_csv("result/regression.csv", index_col=0)
    st.table(df4)


    st.header("Results of the transformer classification model based on input Model 1")
    # window 5
    st.subheader("The time window is 5")    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_5/trans Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/trans/window_5/trans Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/trans/window_5/trans Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_5/trans Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/trans/window_5/trans Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/trans/window_5/trans Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)


    # window 10
    st.subheader("The time window is 10")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_10/trans Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/trans/window_10/trans Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/trans/window_10/trans Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_10/trans Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/trans/window_10/trans Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/trans/window_10/trans Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)


    # window 15
    st.subheader("The time window is 15")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_15/trans Model 1 5min confusion_plot.png", caption="Model 1 5min confusion", use_container_width=True)
    with col2:
        st.image("image/trans/window_15/trans Model 1 10min confusion_plot.png", caption="Model 1 10min confusion", use_container_width=True)
    with col3:
        st.image("image/trans/window_15/trans Model 1 15min confusion_plot.png", caption="Model 1 15min confusion", use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_15/trans Model 1 5min roc.png", caption="Model 1 5min Roc", use_container_width=True)
    with col2:
        st.image("image/trans/window_15/trans Model 1 10min roc.png", caption="Model 1 10min roc", use_container_width=True)
    with col3:
        st.image("image/trans/window_15/trans Model 1 15min roc.png", caption="Model 1 15min roc", use_container_width=True)

    st.subheader("The precision, recall, and F1 score performance of transformer model")
    df5 = pd.read_csv('result/trans_cls.csv', index_col=0) ##################################
    st.table(df5)

    st.header("Results of the transformer regression model based on input Model 1")
    st.subheader("winow 5 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_5/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/trans/window_5/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/trans/window_5/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("winow 10 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_10/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/trans/window_10/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/trans/window_10/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("winow 15 boxdata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image/trans/window_15/5min triple-low state predictionbox.png", caption="Model 1 5min boxdata", use_container_width=True)
    with col2:
        st.image("image/trans/window_15/10min triple-low state predictionbox.png", caption="Model 1 10min boxdata", use_container_width=True)
    with col3:
        st.image("image/trans/window_15/15min triple-low state predictionbox.png", caption="Model 1 15min boxdata", use_container_width=True)

    st.subheader("The accuracy, mean absolute error (MAE), and intra class correlation of different regression models")
    df6 = pd.read_csv("result/trans_res.csv", index_col=0)
    st.table(df6)


def data_input():
    st.subheader("Please select how many minutes of continuous data to enter!")
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
            with cols[j]:  # Create an input box in the corresponding column
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
    # print(f'Classification model output：{pred}')
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
        # pred.append(int("".join(map(str, fc.getClass(output)))))
        pred.append(output)
  
    return pred

def process_xgboost_cls(win, inp):
    min_list = [5, 10, 15]
    pred = []
    prob = []
    x = inp.iloc[:].values.astype(np.float32)
    x = x.reshape(1, -1)

    for t in min_list:
        model_path = os.path.join(
            'checkpoint',
            f'xgboost-cls-{win}',
            f'{t}min',
            'xgboost_clsmodel_fold1.json'
        )
    
        cls_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.01,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        cls_model.load_model(model_path)

        y_prob = cls_model.predict_proba(x)      # shape: (1, 5)
        y_pred = np.argmax(y_prob, axis=1)[0]   # scalar

        pred.append(int(y_pred))
        prob.append(y_prob.flatten())

    return pred

def process_xgboost_res(win, inp):
    min_list = [5, 10, 15]
    pred = []
    x = inp.iloc[:].values.astype(np.float32)
    x = x.reshape(1, -1)

    for t in min_list:
        # scaler（和训练一致）
        scaler = joblib.load(
            os.path.join('dataset', f'scaler_win{t}.pkl')
        )

        # 反归一化用
        reg_outputs = []

        for i in range(3):  # MBP / MAC / BIS
            model_path = os.path.join(
                'checkpoint',
                f'xgboost-res-{win}',
                f'{t}min',
                f'xgboost_regmodel{i}_fold1.json'
            )

            reg_model = xgb.XGBRegressor()
            reg_model.load_model(model_path)

            y = reg_model.predict(x)
            reg_outputs.append(y[0])

        reg_outputs = np.array(reg_outputs).reshape(1, -1)
        reg_outputs = scaler.inverse_transform(reg_outputs).reshape(-1)

        # 格式与 process_regression 对齐
        reg_outputs[0] = int(reg_outputs[0])
        reg_outputs[2] = int(reg_outputs[2])
        reg_outputs[1] = np.round(reg_outputs[1], 1)

        pred.append(reg_outputs)

    return pred


def process_trans_cls(win, inp):

    min_list = [5, 10, 15]
    pred = []

    # (seq_len, C) → (1, seq_len, C)
    x = inp.iloc[:].values.astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)

    for t in min_list:
        ckpt_path = os.path.join(
            'checkpoint',
            f'trans-cls-{win}',
            f'{t}min',
            'best_fold1.pth'
        )

        net = MS_CNN_Transformer(
            input_dim=x.shape[2],
            num_outputs_reg=3,
            num_outputs_cls=5
        )

        state = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(state)
        net.eval()

        with torch.no_grad():
            _, cls_out = net(x)
            y_pred = torch.argmax(cls_out, dim=1).item()

        pred.append(y_pred)

    return pred
    
def process_trans_res(win, inp):
    min_list = [5, 10, 15]
    pred = []

    x = inp.iloc[:].values.astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)

    for t in min_list:
        scaler = joblib.load(
            os.path.join('dataset', f'scaler_win{t}.pkl')
        )

        ckpt_path = os.path.join(
            'checkpoint',
            f'trans-res-{win}',
            f'{t}min',
            'best_fold1.pth'
        )

        net = MS_CNN_Transformer(
            input_dim=x.shape[2],
            num_outputs_reg=3,
            num_outputs_cls=5
        )

        state = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(state)
        net.eval()

        with torch.no_grad():
            reg_out, _ = net(x)
            reg_out = reg_out.numpy()

        reg_out = scaler.inverse_transform(reg_out).reshape(-1)

        reg_out[0] = int(reg_out[0])
        reg_out[2] = int(reg_out[2])
        reg_out[1] = np.round(reg_out[1], 1)

        pred.append(reg_out)

    return pred

if __name__ == "__main__":

    
    show_paper_result()

    win, inp = data_input()


    btn_cls = st.button('Classification model prediction')
    btn_reg = st.button('Regression model prediction')

    btn_xgboostcls = st.button('XGBoost Classification model prediction')
    btn_xgboostreg = st.button('XGBoost Regression model prediction')

    btn_transcls = st.button('Transformer Classification model prediction')
    btn_transreg = st.button('Transformer Regression model prediction')


    if btn_cls:
        p = process_classification(win, inp)
        for i, min in enumerate([5, 10, 15]):
            st.markdown(f"The classification model for {win} minutes of continuous input predicts that the most likely outcome in {min} minutes is: Situation{p[i]+1}")
    if btn_reg:
        r = process_regression(win, inp)
        for i, min in enumerate([5, 10, 15]):
            # st.markdown(f"连续{win}分钟输入的回归模型预测{min}分钟后最可能发生的结果为：Situation {r[i]+1}")
            st.markdown(f"The regression model for {win} minutes of continuous input predicts the result for {min} minutes as --ART_MBP:{r[i][0]} --MAC:{r[i][1]:.1f} --BIS:{r[i][2]}")

    if btn_xgboostcls:
        p = process_xgboost_cls(win, inp)
        for i, min in enumerate([5, 10, 15]):
            st.markdown(f"The xgboost classification model for {win} minutes of continuous input predicts that the most likely outcome in {min} minutes is: Situation{p[i]+1}")

    if btn_xgboostreg:
        r = process_xgboost_res(win, inp)
        for i, min in enumerate([5, 10, 15]):
            # st.markdown(f"连续{win}分钟输入的回归模型预测{min}分钟后最可能发生的结果为：Situation {r[i]+1}")
            st.markdown(f"The regression model for {win} minutes of continuous input predicts the result for {min} minutes as --ART_MBP:{r[i][0]} --MAC:{r[i][1]:.1f} --BIS:{r[i][2]}")
                        
    if btn_transcls:
        p = process_trans_cls(win, inp)
        for i, min in enumerate([5, 10, 15]):
            st.markdown(f"The xgboost classification model for {win} minutes of continuous input predicts that the most likely outcome in {min} minutes is: Situation{p[i]+1}")

    if btn_transreg:
        r = process_trans_res(win, inp)
        for i, min in enumerate([5, 10, 15]):
            # st.markdown(f"连续{win}分钟输入的回归模型预测{min}分钟后最可能发生的结果为：Situation {r[i]+1}")
            st.markdown(f"The regression model for {win} minutes of continuous input predicts the result for {min} minutes as --ART_MBP:{r[i][0]} --MAC:{r[i][1]:.1f} --BIS:{r[i][2]}")
               