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

    # # 提交按钮
    # if st.button("Submit"):
    #     st.success("Data submitted successfully!")
    #     st.write("Submitted Data:")
    #     st.write(filled_df)

    # st.title("Fill the Table")
    #     st.subheader(f'请输入{min}分钟的连续数据')
    #     for i in range(min):
    #         st.markdown(f"请输入第{i+1}分钟的数据")
    #         cols1, cols2, cols3 = st.columns(3)
    #         with cols1:
    #             ART_MBP = st.slider("ART_MBP", 0, 500)
    #         with cols2:
    #             MAC = st.slider("MAC", 0, 5)
    #         with cols3:
    #             BIS = st.slider("BIS", 0, 100)
            
    #         inp.append(BIS)
    
    # return inp

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
        output = mm.inverse_transform(output)[:, :3]
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
            st.markdown(f"连续{win}分钟输入的回归模型预测{min}分钟后最可能发生的结果为：Situation {r[i]+1}")


    # do_processing()
    # setup_selectors()

            
