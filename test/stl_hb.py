
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 출력된 다른 이름
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용 시, 마이너스 기호 깨짐 방지

@st.cache_data
def load_data():
    # 데이터 불러오기
    df = pd.read_csv('SN_total.csv')

    df["로그 원수 탁도"] = np.log10(df["원수 탁도"])
    df["로그 응집제 주입률"] = np.log10(df["3단계 1계열 응집제 주입률"])
    
    X = df[
        [
            "로그 원수 탁도",
            "원수 pH",
            "원수 알칼리도",
            "원수 전기전도도",
            "원수 수온",
            "3단계 원수 유입 유량",
            "3단계 침전지 체류시간",
            "로그 응집제 주입률"
        ]
    ]
    # Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X

def run(target, input, max_depth, n_estimators, learning_rate, subsample):

    Xt, Xts, yt, yts = train_test_split(input, target, test_size=0.2, shuffle=False)

    xgb = XGBRegressor(
        random_state=2, 
        n_jobs=-1,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample
    )

    xgb.fit(Xt, yt)
    
    yt_pred = xgb.predict(Xt)
    yts_pred = xgb.predict(Xts)

    mse_train = mean_squared_error(10**yt, 10**yt_pred)
    mse_test = mean_squared_error(10**yts, 10**yts_pred)
    st.write(f"학습 데이터 MSE: {mse_train}")
    st.write(f"테스트 데이터 MSE: {mse_test}")

    r2_train = r2_score(10**yt, 10**yt_pred)
    r2_test = r2_score(10**yts, 10**yts_pred)
    st.write(f"학습 데이터 R2: {r2_train}")
    st.write(f"테스트 데이터 R2: {r2_test}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    print(Xt.columns)
    ax.scatter(Xt.iloc[:, 0], yt, s=3, label="학습 데이터 (실제)")
    ax.scatter(Xt.iloc[:, 0], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    # ax.set_xlabel("로그 원수 탁도")
    # ax.set_ylabel("로그 응집제 주입률")
    
    ax.set_xlabel(input.columns[0])
    ax.set_ylabel(target.columns[0])

    ax.set_title(
        rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
        fontsize=18,
    )

    ax = axes[1]
    # ax.scatter(Xt["로그 원수 탁도"], yt, s=3, label="학습 데이터 (실제)")
    # ax.scatter(Xt["로그 원수 탁도"], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
    
    ax.scatter(Xt.iloc[:, 0], yt, s=3, label="학습 데이터 (실제)")
    ax.scatter(Xt.iloc[:, 0], yt_pred, s=3, label="학습 데이터 (예측)", c="r")

    ax.grid()
    ax.legend(fontsize=13)

    ax.set_xlabel(input.columns[0])
    ax.set_ylabel(target.columns[0])

    ax.set_title(
        rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
        fontsize=18,
    )

    st.pyplot(fig)

def main():

    # load data
    dff = load_data()

    # Select Variable
    st.markdown("## Select Target Variable")
    column = st.selectbox("Target 변수를 선택하세요.", dff.columns.tolist())
    col = dff[[column]]
    st.dataframe(dff[[column]].head())

    st.markdown("## Select Input Variables")
    input_columns = st.multiselect("복수의 컬럼을 선택하세요.", dff.columns.tolist())
    filtered_col = dff[input_columns]

    st.dataframe(dff[input_columns].head())

    



    # Hyperparameters
    max_depth = st.slider("Select max depth", min_value=0, max_value=20, value=3)
    n_estimators = st.slider("n_estimators", min_value=20, max_value=500, value=50)
    learning_rate = st.slider("learning_rate", min_value=0.00, max_value=1.00, step=0.01, value=0.1)
    subsample = st.slider("subsample", min_value=0.00, max_value=1.00, step=0.01, value=0.8)

    if st.button("차트 만들기"):
        run(col, filtered_col, max_depth, n_estimators, learning_rate, subsample)


if __name__ == "__main__":
    main()