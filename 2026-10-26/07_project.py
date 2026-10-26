"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd
from tabulate import tabulate

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.
regression_rows = []
classification_rows = []
predictions = []

# 여러 회귀모델을 생성하여 비교: [Dummy, Linear, KNN15]
# 모델마다 clone(pre)로 독립적인 전처리 객체를 만들어 사용하는 것을 권장
# 참고) clone: 모델의 설정은 복사하고, 학습된 내용은 복사하지 않는 함수
for name, estimator in [("Dummy", DummyRegressor()),
                        ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07r: 회귀모델 학습 및 예측
    # 회귀모델 정답열은 "elapsed_ms"
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)

    # 회귀모델별 성능평가지표 리스트에 담기: [MAE, RMSE]
    regression_rows.append({
        "model": name,
        "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)
    })
    
    predictions.append(pd.DataFrame({
        "request_id": valid.index,
        "task": "regression",
        "model": name,
        "actual": valid["elapsed_ms"].to_numpy(),
        "predicted": prediction
    }))

# 여러 분류모델을 생성하여 비교: [Dummy, Logistic, KNN3]
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07c: 분류모델을 학습 및 예측
    # 분류모델 정답열은 "failed"
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    # 분류모델별 성능평가지표 리스트에 담기: [F1-score, Precision, Recall]
    # 혼동행렬: 행이 실제값, 열이 예측값 / labels[0,1]: 행렬표에서 행,열 배치 순서(이 경우 둘다 0부터 배치) / ravel(): [TN, FP, FN, TP]
    # 이진 분류에서 1=Positive, 0=Negative로 두는 것은 일반적인 관례
    # 찾아내려는 목적을 Positive 로 둠. 이번 실습에선 '실패'를 찾는 것이 목적이므로 실패를 Positive로 둠
    # labels[1,0] 이었다면 tp, fn, fp, tn
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    classification_rows.append({
        "model": name,
        "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0),
        "FP": fp,
        "FN": fn
    })

    # 모델별 혼동행렬을 출력
    data = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    print(f"{name}")
    print(tabulate(data.items(), headers=["지표", "값"]))
    print("")
    
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

# 회귀 모델별 검증 성능 지표를 CSV로 저장
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")

# 분류 모델별 검증 성능 지표를 별도의 CSV로 저장
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")

# 회귀·분류 모델의 검증 데이터별 실제값과 예측값을 행 방향으로 합쳐 CSV로 저장
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

"""이번 실습의 회귀 평가지표 해석"""
# MAE와 RMSE는 낮을수록 좋습니다.
# MAE(일반적인 오차 크기 평가): KNN15가 가장 좋습니다.
# RMSE(큰 오차에 더 민감): Linear가 가장 좋습니다.
# Dummy와 비교한 개선 폭은 크지 않습니다.

"""이번 실습의 분류 평가지표 해석"""
# F1·Precision·Recall은 높을수록, FP·FN은 낮을수록 좋습니다.
# Dummy: 모두 정상으로 예측해 실제 실패 6건을 전부 놓침
# Logistic: 실제 실패 6건 중 2건 탐지, 4건 놓침. 정상 4건을 실패로 오탐
# KNN3: 실제 실패 6건 중 3건 탐지, 3건 놓침. 정상 2건을 실패로 오탐

# KNN3 의 F1-score 가 가장 높으므로 KNN3 살펴봄
# - Precision 0.6: 실패라고 예측한 5건 중 실제 실패가 3건
# - Recall 0.5: 실제 실패 6건 중 3건을 찾아냄
# - F1 0.545: 정밀도와 재현율을 함께 평가한 점수가 모델들 중 가장 높음
# 이번 분류 후보 중에서는 KNN3가 가장 우수하지만, 실제 실패의 절반은 여전히 놓칩니다.
# 검증 데이터가 20건이므로 몇 건의 예측 차이에도 점수가 크게 달라질 수 있습니다.
print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
