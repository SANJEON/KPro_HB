import os
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

def check_to_log(df):
    # 로그 변환 대상 컬럼 리스트 (양수여야 함)
    columns_to_check = ['탁도', '약품주입율 계산']  # 필요시 다른 컬럼 추가

    # 히스토그램 출력
    for col in columns_to_check:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        # 원래 분포
        axs[0].hist(df[col].dropna(), bins=50, color='skyblue')
        axs[0].set_title(f'{col} - 원래 분포')
        
        # 로그 변환 분포
        log_data = df[col][df[col] > 0]  # 0 이하 제외
        log_data = np.log10(log_data)
        axs[1].hist(log_data.dropna(), bins=50, color='lightgreen')
        axs[1].set_title(f'{col} - 로그 변환 분포')

        plt.tight_layout()
        plt.show()


def load_data():
    # 데이터 불러오기
    basedir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(basedir, 'data', 'CA_Water_Quality.CSV')

    print(data_path)
    df = pd.read_csv(data_path, encoding="CP949")
    # df["로그 탁도"] = np.log10(df["탁도"])
    # df["로그 응집제 주입률"] = np.log10(int(df["약품주입율 계산"]))
    # np.log10은 데이터 전처리 후 사용
    
    
    df.rename(columns={'약품주입율 계산': '약품주입율'}, inplace=True)


    df = df[
        [
            "탁도",
            "pH",
            "알칼리도",
            "전기전도도",
            "수온",
            "유입유량",
            "침전탁도",
            "약품주입율"
        ]
    ]

    # 두 컬럼만 숫자로 변환 (숫자가 아닌 값은 NaN으로 처리)
    df['탁도'] = pd.to_numeric(df['탁도'], errors='coerce')
    df['약품주입율'] = pd.to_numeric(df['약품주입율'], errors='coerce')

    # 변환 후 NaN 확인 (비정상값이 있었는지 확인 가능)
    print(df[['탁도', '약품주입율']].isna().sum())

    # 탁도 0개, 약품주입율 11개 데이터가 na로, 행 삭제

    df = df.dropna(subset=['탁도', '약품주입율'])

    # 탁도 분포에 따라 로그 변환 실행
    df['로그 탁도'] = np.log10(df["탁도"])

    return df

def run(target, input, max_depth, n_estimators, learning_rate, subsample):
    # 1. float64로 형변환
    input = input.astype(np.float64)
    target = target.astype(np.float64)

    # 2. target이 DataFrame이면 Series로 변환
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]

    # 3. NaN 및 inf 제거 (input과 target 모두)
    mask = (~input.isna().any(axis=1)) & (~np.isnan(target)) & (~np.isinf(target))
    input = input[mask]
    target = target[mask]

    # 4. train/test split
    Xt, Xts, yt, yts = train_test_split(input, target, test_size=0.2, shuffle=False)

    # 5. 모델 학습
    xgb = XGBRegressor(
        random_state=2, 
        n_jobs=-1,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample
    )
    xgb.fit(Xt, yt)

    # 6. 예측
    yt_pred = xgb.predict(Xt)
    yts_pred = xgb.predict(Xts)

    # 7. 성능 평가 (로그 역변환 없음!)
    mse_train = mean_squared_error(yt, yt_pred)
    mse_test = mean_squared_error(yts, yts_pred)
    r2_train = r2_score(yt, yt_pred)
    r2_test = r2_score(yts, yts_pred)

    st.write(f"학습 데이터 MSE: {mse_train:.4f}")
    st.write(f"테스트 데이터 MSE: {mse_test:.4f}")
    st.write(f"학습 데이터 R²: {r2_train:.4f}")
    st.write(f"테스트 데이터 R²: {r2_test:.4f}")

    # 8. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (x, y, y_pred, title) in enumerate([
        (Xt, yt, yt_pred, "학습 데이터"),
        (Xts, yts, yts_pred, "테스트 데이터")
    ]):
        ax = axes[i]
        ax.scatter(x.iloc[:, 0], y, s=3, label="실제값")
        ax.scatter(x.iloc[:, 0], y_pred, s=3, label="예측값", c="r")
        ax.set_xlabel(input.columns[0])
        ax.set_ylabel(target.name)
        ax.set_title(f"{title} MSE: {round(mean_squared_error(y, y_pred), 4)}, R²: {round(r2_score(y, y_pred), 2)}", fontsize=14)
        ax.grid()
        ax.legend(fontsize=10)

    st.pyplot(fig)


def main():

    # load data
    dff = load_data()

    # Select Variable
    st.markdown("타겟 변수는 침전탁도로 고정")
    column = "침전탁도"
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