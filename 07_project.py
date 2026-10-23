"""07~08 응용: 별도 모듈 없이 이 파일 하나로 완성합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

"""출력 폴더 생성 및 CSV 데이터 불러오기"""
HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
data = pd.read_csv(HERE / "data" / "requests.csv")



"""학습·검증·테스트 데이터 분리"""
# 전체 데이터의 20%를 테스트(test) 데이터로 분리합니다.
# random_state=42: 실행할 때마다 동일하게 분리합니다.
# stratify: 원본 데이터의 성공/실패 비율을 유지합니다.
# 이 실습에서는 테스트 데이터의 개수만 확인하고, 모델 평가는 검증 데이터로 진행합니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])

# 남은 80% 중 25%를 검증(valid) 데이터로 분리합니다.
# 최종 비율: 학습(train) 60%, 검증(valid) 20%, 테스트(test) 20%
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])



"""숫자형·범주형 변수 전처리기 생성 """
# 숫자형 변수의 결측치를 중앙값으로 채운 뒤 표준화합니다.
numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# 범주형 변수의 결측치를 최빈값으로 채운 뒤 원-핫 인코딩합니다.
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# 각 열의 유형에 맞는 전처리기를 연결합니다.
preprocessor = ColumnTransformer([
    ("num", numeric, ["input_chars", "allowed_tools"]),
    ("cat", categorical, ["task_type", "priority"]),
])



"""전처리기와 기준 모델을 연결한 파이프라인 생성"""
# 기준 모델은 단순한 예측 방법으로 성능 비교의 출발점을 만듭니다.
# 여기서는 most_frequent 전략으로, 학습 데이터에서 가장 자주 등장한 클래스로만 예측합니다.
# 이후 만들 모델이 이 단순한 방법보다 잘 예측하는지 확인하는 기준이 됩니다.
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])



"""입력 특성 지정 및 파이프라인 학습"""
# 입력으로 사용할 4개 열을 지정합니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# 학습 데이터로 전처리 기준과 기준 모델을 학습합니다.
pipeline.fit(train[features], train["failed"])



"""검증 데이터 예측 및 결과 저장"""
# TODO A07-1: Pipeline으로 valid의 입력 4열을 사용해 성공/실패를 예측하세요.
prediction = pipeline.predict(valid[features])

# 요청 ID, 실제 정답, 예측값을 하나의 데이터프레임으로 모읍니다.
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction

# 예측 결과를 CSV 파일로 저장합니다.
result.to_csv(OUTPUTS / "valid_predictions.csv", index=False, encoding="utf-8-sig")



"""기준 모델의 정확도 계산 및 결과 출력"""
# TODO A07-2: 실제값과 예측값이 같은 비율(정확도)을 구하세요.
# 비교 결과는 True/False이며, 평균을 계산할 때 각각 1/0으로 처리됩니다.
accuracy = (result["failed"] == result["predicted"]).mean()

# 실제로 실패한 요청 중 실패로 정확히 예측한 요청의 개수를 구합니다.
found = ((result["failed"] == 1) & (result["predicted"] == 1)).sum()

# 결과 출력
print("valid 정확도:", round(accuracy, 2))
print("실패 발견:", int(found), "/", int((result["failed"] == 1).sum()))
print("저장 행 수:", len(result))
print("저장: outputs/valid_predictions.csv / test는 사용하지 않았습니다.")
