"""제공 코드: 입력·분할 계약 및 실행 기록. test의 X/y를 만들지 않습니다."""
import numpy as np
from course_support import FEATURES, NUM, check_features, snapshot

# verify_development_data(): 학습·검증·테스트 분할이 정해진 조건을 만족하는지 확인하는 코드
#   1) 데이터 개수 확인: train 60개, valid 20개, test 20개인지 검사
#   2) ID 중복·누락 확인: 분할끼리 ID가 겹치지 않고, 합쳐서 고유 ID가 100개인지 검사
#   3) 입력과 정답 정렬 확인: train/valid 각각 X와 분류·회귀 정답의 ID 및 순서가 같은지 검사
#   4) 입력 특성 재검사: check_features(X)로 열 구성과 값의 유효성 확인
#   5) 검사 결과 반환: 모든 검사를 통과하면 개수와 분할 간 중복 없음(disjoint=True)을 딕셔너리로 반환
def verify_development_data(d):
    if len(d.X_train)!=60 or len(d.X_valid)!=20 or len(d.test_ids)!=20:
        raise ValueError('공통 분할은 60/20/20이어야 합니다.')
    tr=set(d.X_train.index); va=set(d.X_valid.index); te=set(d.test_ids)
    if tr&va or tr&te or va&te or len(tr|va|te)!=100:
        raise ValueError('분할 ID 중복/누락')
    for X,yc,yr in [(d.X_train,d.yc_train,d.yr_train),(d.X_valid,d.yc_valid,d.yr_valid)]:
        if not X.index.equals(yc.index) or not X.index.equals(yr.index): raise ValueError('X/y 정렬 오류')
        check_features(X)
    return {'train_n':60,'valid_n':20,'heldout_test_id_n':20,'disjoint':True}

def boundary_checks(fitted, d):
    """응용·심화 제공 안전 검사: 미지 범주와 결측을 허용, 필수 열 누락은 차단."""
    new=d.X_valid.iloc[:2].copy();new.index=['NEW_A','NEW_B']
    new['task_type']=['translation','translation'];new['priority']=['urgent','urgent']
    new.loc['NEW_A','input_chars']=np.nan
    before=snapshot(fitted['pre'])
    z=fitted['pre'].transform(check_features(new))
    fitted.predict(check_features(new))
    if z.shape!=(2,8) or not np.isfinite(z).all(): raise AssertionError('새 범주/결측 변환 실패')
    if before!=snapshot(fitted['pre']): raise AssertionError('통계 변경')
    try: check_features(new.drop(columns=['priority']))
    except ValueError: blocked=True
    else: blocked=False
    if not blocked: raise AssertionError('필수 열 누락 미차단')
    return {'unknown_category_shape':[2,8],'statistics_unchanged':True,'missing_column_blocked':True}
