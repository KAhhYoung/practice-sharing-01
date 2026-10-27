"""CV·valid·해석. train 내부 CV와 외부 valid의 역할을 분리합니다."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from course_support import cls_metrics, reg_metrics, snapshot, json_ready

# run_candidate():
#   1. 작업에 맞는 정답과 평가 지표를 선택합니다.
#   2. 학습 데이터 안에서 교차검증을 수행합니다.
#   3. 교차검증 점수를 정리합니다.
#   4. 전체 학습 데이터로 별도 모델을 학습합니다.
#       -교차검증에서 사용한 모델들과 별개로, 새 Pipeline을 만들어 전체 학습데이터로 학습합니다.
#       -목적: 학습 데이터 전체를 활용한 모델을 만들어, 별도로 떼어 둔 검증 데이터(X_valid)를 평가하기 위함
#   5. 별도 검증 데이터를 예측하고 전처리 상태를 확인합니다.
#       -목적: 예측 과정에서 전처리기가 다시 학습하거나 학습 통계를 변경하지 않았는지 확인하기 위함
def run_candidate(name, pipe, d, folds, task='classification'):
    is_cls=(task=='classification')
    yt=d.yc_train if is_cls else d.yr_train
    yv=d.yc_valid if is_cls else d.yr_valid
    scorer='f1' if is_cls else 'neg_root_mean_squared_error'
    # TODO D7-07-E1: 위 계약과 README를 읽고 구현하세요.
    '''구현계약:
       행 dict, fold DataFrame, 검증예측 DataFrame, train60으로 fit한 Pipeline 반환.
       CV는 clone으로 fold 안에서 fit; valid는 train60으로 별도 fit한 모델로만 예측.
       보관 test의 입력·정답은 받지 않습니다.
    '''
    cv=cross_validate(pipe,d.X_train,yt,cv=folds,scoring=scorer,n_jobs=1,
                        return_train_score=True,return_estimator=True,error_score='raise')
    # 회귀 scorer는 높을수록 좋게 만들기 위해 음수입니다. 보고서는 양수 ms로 복구.
    scores=cv['test_score'] if is_cls else -cv['test_score']
    train_scores=cv['train_score'] if is_cls else -cv['train_score']
    fitted=clone(pipe).fit(d.X_train,yt)
    before=snapshot(fitted['pre'])
    pred=fitted.predict(d.X_valid)
    if snapshot(fitted['pre'])!=before: raise AssertionError('예측이 학습 통계를 변경했습니다.')
    # END

    metric='F1' if is_cls else 'RMSE_ms'
    valid=cls_metrics(yv,pred) if is_cls else reg_metrics(yv,pred)
    train=cls_metrics(yt,fitted.predict(d.X_train)) if is_cls else reg_metrics(yt,fitted.predict(d.X_train))
    row={'model':name,'task':task,'scaling':'none' if name in ['Tree','Forest','XGB'] else 'standard',
         'CV_mean':float(scores.mean()),'CV_std':float(scores.std(ddof=0)),
         'CV_metric':metric,'train_primary':train[metric],
         **{'valid_'+k:v for k,v in valid.items()},
         'params':json.dumps(json_ready(fitted['model'].get_params()),ensure_ascii=False,sort_keys=True)}
    fold_rows=[]
    for i,((a,b),score,tr,fold_model) in enumerate(zip(folds,scores,train_scores,cv['estimator']),1):
        expected=d.X_train.iloc[a][['input_chars','allowed_tools']].median().to_numpy()
        actual=fold_model['pre'].named_transformers_['num']['imputer'].statistics_
        # 이 검사는 모든 단계에 제공합니다. 06의 통계 감사에서도 같은 원리를 구현합니다.
        if not np.allclose(expected,actual): raise AssertionError('fold外の中央値を使用')
        fold_rows.append({'task':task,'model':name,'fold':i,'fit_n':len(a),'eval_n':len(b),
            'metric':metric,'score':float(score),'train_score':float(tr),'median_audit_passed':True})
    predictions=pd.DataFrame({'request_id':d.X_valid.index,'task':task,'model':name,
                              'actual':yv.to_numpy(),'predicted':pred})
    if is_cls:
        pindex=list(fitted.classes_).index(1)
        predictions['failure_probability']=fitted.predict_proba(d.X_valid)[:,pindex]
    else: predictions['residual_ms']=yv.to_numpy()-pred
    return row,pd.DataFrame(fold_rows),predictions,fitted

def importance_table(forest, d, include_permutation=False):
    # TODO D7-07-E2: 위 계약과 README를 읽고 구현하세요.
    '''구현계약:
        train MDI 8행. 응용이면 valid 원본 4열 순열 진단 4행 추가. 인과/특성선택 아님."""
    '''
    names=forest['pre'].get_feature_names_out()
    values=forest['model'].feature_importances_
    if len(names)!=len(values): raise AssertionError('특성 이름/중요도 불일치')
    rows=[{'feature':n,'method':'MDI','importance':float(v),'repeat_std':None,
            'evaluation_data':'train60','metric':'impurity_decrease'} for n,v in zip(names,values)]
    if include_permutation:
        result=permutation_importance(forest,d.X_valid,d.yc_valid,scoring='f1',
                                        n_repeats=20,random_state=42,n_jobs=1)
        rows += [{'feature':n,'method':'permutation','importance':float(v),'repeat_std':float(s),
                    'evaluation_data':'valid20 (diagnostic)','metric':'F1_drop'}
                    for n,v,s in zip(d.X_valid.columns,result.importances_mean,result.importances_std)]
    return pd.DataFrame(rows)
#   # END

# choose_by_cv(): 교차검증(CV) 평균 점수를 기준으로 가장 좋은 모델의 이름을 반환하는 함수
def choose_by_cv(table, task='classification'):
    # TODO D7-07-E3: 위 계약과 README를 읽고 구현하세요.
    '''구현계약:
        F1 평균 최대 / RMSE 평균 최소. 정확히 동점이면 사전 후보 순서 우선.
        valid 점수는 함수에서 참조하지 않습니다. 선택점수는 최종 성능이 아닙니다.
    '''
    # ascending 파라미터 목적
    #   -회귀: ascending=True → CV 평균 RMSE가 작은 값부터 정렬 → (최종)RMSE 가 가장 낮은 모델 선택
    #   -분류: ascending=False -> CV 평균 F1이 큰 값부터 정렬 → (최종)F1 이 가장 높은 모델 선택
    ascending=(task=='regression')
    ordered=table.sort_values('CV_mean',ascending=ascending,kind='stable')
    return str(ordered.iloc[0]['model'])
    # END
