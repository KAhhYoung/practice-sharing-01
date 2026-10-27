"""제공 설정: PPT의 후보·seed·CPU 설정. 여기에는 학습된 객체/결과가 없습니다."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, KFold

def new_classifier(name: str):
    if name == 'Dummy': return DummyClassifier(strategy='most_frequent')
    if name == 'Logistic': return LogisticRegression(C=1, max_iter=1000)
    if name == 'KNN3': return KNeighborsClassifier(n_neighbors=3)
    if name == 'Tree':
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)
    if name == 'Forest':
        return RandomForestClassifier(n_estimators=100, max_depth=4,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=1)
    if name == 'XGB':
        # XGBoost가 없어도 실습01~04는 실행 가능합니다. 05부터는 반드시 설치합니다.
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
            min_child_weight=1, subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
            objective='binary:logistic', eval_metric='logloss', tree_method='hist',
            device='cpu', random_state=42, n_jobs=1)
    raise ValueError(f'알 수 없는 분류 모델: {name}')

def new_regressor(name: str):
    if name == 'Dummy': return DummyRegressor(strategy='mean')
    if name == 'Linear': return LinearRegression()
    if name == 'KNN15': return KNeighborsRegressor(n_neighbors=15)
    if name == 'Tree':
        return DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42)
    if name == 'Forest':
        return RandomForestRegressor(n_estimators=100, max_depth=4,
            min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=1)
    if name == 'XGB':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
            min_child_weight=1, subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
            objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
            device='cpu', random_state=42, n_jobs=1)
    raise ValueError(f'알 수 없는 회귀 모델: {name}')



# 학습 데이터 내부에서 5-fold 교차검증용 위치 인덱스 생성
# - 분류: StratifiedKFold로 각 fold의 클래스 비율을 전체와 비슷하게 유지
#         클래스 구성 차이에 따른 검증 결과의 변동을 줄이기 위함
# - 회귀: KFold로 정답값의 분포를 고려하지 않고 행을 분할
# - 공통: 행을 섞되 random_state=42로 분할 결과의 재현성 확보
#
# 반환값: (학습용 위치 배열, 검증용 위치 배열) 튜플 5개를 담은 리스트
# 위치 인덱스는 request_id가 아니며, X.iloc[...] 및 y.iloc[...]에 사용
#
# 반환 예시: 데이터 10개를 가정한 설명용 예시이며 실제 실행 결과와 다를 수 있음
# [
#     (array([1, 2, 3, 4, 6, 7, 8, 9]), array([0, 5])),
#     (array([0, 2, 3, 4, 5, 7, 8, 9]), array([1, 6])),
#     (array([0, 1, 3, 4, 5, 6, 8, 9]), array([2, 7])),
#     (array([0, 1, 2, 4, 5, 6, 7, 9]), array([3, 8])),
#     (array([0, 1, 2, 3, 5, 6, 7, 8]), array([4, 9]))
# ]
def make_folds(X, y, task='classification'):
    """train 내부 위치 인덱스. 위치를 request_id와 혼동하지 않습니다."""
    if not X.index.equals(y.index):
        raise ValueError('X/y 인덱스와 순서가 일치해야 합니다.')
    if task == 'classification':
        if set(y.dropna().unique()) != {0,1} or y.isna().any():
            raise ValueError('분류 CV는 결측 없는 두 클래스 0/1이 모두 필요합니다.')
        if y.value_counts().min() < 5:
            raise ValueError('5-fold에는 각 클래스가 최소 5건 필요합니다.')
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        return list(splitter.split(X, y))
    if task == 'regression':
        return list(KFold(5, shuffle=True, random_state=42).split(X))
    raise ValueError(task)


# 교차 검증의 각 fold에서 어떤 데이터가 학습과 평가에 쓰였는지 기록하는 표를 생성
def fold_table(X, folds, task='classification'):
    """누가 어디에 속했는지 남기는 감사용 표. 점수는 이 함수에서 계산하지 않습니다."""
    rows=[]
    for i,(a,b) in enumerate(folds,1):
        if set(a)&set(b): raise ValueError('fold 학습/평가 중복')
        
        for role,positions in [('fit',a),('evaluate',b)]:   # a: 현재 fold의 학습 데이터 위치 / b: 현재 fold의 검증 데이터 위치
            for p in positions:
                rows.append({'task':task,'fold':i,'role':role,
                             'position_in_train':int(p),'request_id':X.index[p]})
    return pd.DataFrame(rows)
