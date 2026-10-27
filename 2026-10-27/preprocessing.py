"""특성 선택·전처리·모델 생성. 아직 학습하지 않은 Pipeline을 만듭니다."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from course_support import NUM, CAT, FEATURES
from model_config import new_classifier, new_regressor

def make_preprocessor(scale='none'):
    # TODO D7-07-P1: 위 계약과 README를 읽고 구현하세요.
    '''구현계약:
        num: median (+standard 선택), cat: 최빈값 -> dense OneHot(ignore).
    '''
    num=[('imputer',SimpleImputer(strategy='median'))]
    if scale=='standard': num.append(('scaler',StandardScaler()))
    elif scale!='none': raise ValueError('scale: none/standard만 허용')
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('num',Pipeline(num),NUM),('cat',cat,CAT)],remainder='drop')
    # END

def candidates(include_previous=False, task='classification'):
    # TODO D7-07-P2: 위 계약과 README를 읽고 구현하세요.
    '''구현계약:
        기본 분류 Dummy/Tree/Forest/XGB. 응용에 Logistic/KNN3, 회귀는 6후보.
        Dummy와 전날 모델만 scale=standard. 각 후보는 서로 다른 새 객체여야 합니다.
    '''
    if task=='classification':
        names=['Dummy','Logistic','KNN3','Tree','Forest','XGB'] if include_previous else ['Dummy','Tree','Forest','XGB']
        constructor=new_classifier
    elif task=='regression':
        names=['Dummy','Linear','KNN15','Tree','Forest','XGB']
        constructor=new_regressor
    else: raise ValueError(task)
    result={}
    for name in names:
        scale='none' if name in ['Tree','Forest','XGB'] else 'standard'
        result[name]=Pipeline([('pre',make_preprocessor(scale)),('model',constructor(name))])
    return result
    # END
