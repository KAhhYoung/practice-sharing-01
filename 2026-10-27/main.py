"""Day7 앙상블 비교 프로젝트. 모든 상대경로는 이 파일 위치를 기준으로 합니다."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from course_support import load_data, write_json, environment, sha256
from model_config import make_folds, fold_table
from preprocessing import candidates
from evaluate import run_candidate, importance_table, choose_by_cv
from checks import verify_development_data, boundary_checks
ROOT=Path(__file__).resolve().parent
DEFAULT_LEVEL=3

def run_project(data_dir=ROOT/'data', output=ROOT/'outputs', level=DEFAULT_LEVEL):
    if level not in (1,2,3): raise ValueError('level은 1~3')

    # output 디렉토리 생성
    output=Path(output);output.mkdir(parents=True,exist_ok=True)

    # 데이터 불러오기: d는 다음 속성을 가진 DevData 객체
    #    X_train: pd.DataFrame       # 학습 입력 특성
    #    X_valid: pd.DataFrame       # 검증 입력 특성
    #    yc_train: pd.Series         # 학습 분류 정답
    #    yc_valid: pd.Series         # 검증 분류 정답
    #    yr_train: pd.Series         # 학습 회귀 정답
    #    yr_valid: pd.Series         # 검증 회귀 정답
    #    test_ids: list              # test 요청 ID 목록
    #    manifest: pd.DataFrame      # 분할표
    #    source_hashes: dict         # 파일별 해시값
    d=load_data(data_dir);

    # 학습·검증·테스트 분할이 정해진 조건을 만족하는지 확인
    integrity=verify_development_data(d)


    # 학습 데이터 내부에서 5-fold 교차검증용 위치 인덱스 생성 (분류용 분할만 생성)
    cls_folds=make_folds(d.X_train,d.yc_train)

    # 수행할 작업 지정: level 1은 분류만, 그 외에는 분류와 회귀
    tasks=['classification'] if level==1 else ['classification','regression']
    
    comparisons={}; models={}; all_scores=[]; all_pred=[]; all_folds=[]; choices={}
    # TODO D7-07-M1: 위 계약과 README를 읽고 구현하세요.
    # 1) comparison 채우기: 형태는 {task: table}
    # 2) 
    for task in tasks:
        y=d.yc_train if task=='classification' else d.yr_train
        folds=cls_folds if task=='classification' else make_folds(d.X_train,y,task)
        all_folds.append(fold_table(d.X_train,folds,task))      # task=['classification'] 이면 all_folds 에는 1개의 df, ['classificaion', 'regression'] 이면 all_folds 에는 2개의 df 들어감

        rows=[]
        # candidates(): 작업 종류에 맞는 모델 후보들을 만들고, 전처리와 모델을 묶은 Pipeline을 딕셔너리로 반환
        # run_candidate(): 모델 하나를 교차검증으로 평가하고, 전체 학습 데이터로 다시 학습한 뒤 별도 검증 데이터를 예측
        # run_candidate 반환타입: (비교행,fold점수,예측표,학습모델)
        # row 예시
        # row = {
        #   'task': 'classification',
        #   'model': 'Forest',
        #   'cv_mean': 0.85,
        #   'valid_f1': 0.83
        # }
        for name,pipe in candidates(include_previous=(level>=2),task=task).items():
            row,scores,pred,fitted=run_candidate(name,pipe,d,folds,task)
            rows.append(row);all_scores.append(scores);all_pred.append(pred)
            models[(task,name)]=fitted
        table=pd.DataFrame(rows)

        # level == 1: 분류 테이블 1개 생성
        # level >= 2: 분류·회귀 비교표 각각 1개, 총 2개 생성
        # comparisons['classification']  # 분류 비교표
        # comparisons['regression']      # 회귀 비교표
        comparisons[task]=table

        # CV 평균 점수로 최적 모델을 선택하고, 작업별 모델명을 저장 (분류: F1 최대, 회귀: RMSE 최소)
        choices[task]=choose_by_cv(table,task)
    # END


    '''평가 결과를 CSV로 저장'''
    # 작업별 모델 비교표를 저장
    # classification_comparison.csv: 분류 모델 비교표
    # regression_comparison.csv: 회귀 모델 비교표
    for task,table in comparisons.items():
        table.to_csv(output/f'{task}_comparison.csv',index=False,encoding='utf-8-sig')
    
    # 리스트에 모아둔 DataFrame들을 하나로 합쳐 저장
    # |             파일              |             내용               |
    # |------------------------------|-------------------------------|
    # | `cv_scores.csv`              | 작업·모델별 각 fold의 평가 점수   |
    # | `validation_predictions.csv` | 별도 검증 데이터의 실제값과 예측값 |
    # | `fold_manifest.csv`          | 교차검증 데이터 분할 정보         |
    pd.concat(all_scores,ignore_index=True).to_csv(output/'cv_scores.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_pred,ignore_index=True).to_csv(output/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_folds,ignore_index=True).to_csv(output/'fold_manifest.csv',index=False,encoding='utf-8-sig')
    
    '''분류 Forest 모델의 특성 중요도 저장'''
    # 모든 level: MDI 중요도 계산
    # level >= 2: 검증 데이터의 permutation 중요도도 추가
    # 이번 실습에서는 분류 Forest에만 permutation 중요도를 추가 (분류,회귀에 다 사용가능)
    importance=importance_table(models[('classification','Forest')],d,include_permutation=level>=2)
    importance.to_csv(output/'feature_importance.csv',index=False,encoding='utf-8-sig')
    
    '''선택된 분류 모델의 경계 검사'''
    check=boundary_checks(models[('classification',choices['classification'])],d)
    if level>=2:
        import joblib
        # 직접 생성한 파일만 같은 환경에서 복원합니다. 외부 pickle/joblib은 로드하지 않습니다.
        for task,name in choices.items():
            filename=output/f'{task}_pipeline.joblib'
            joblib.dump(models[(task,name)],filename)
            restored=joblib.load(filename)
            a=models[(task,name)].predict(d.X_valid); b=restored.predict(d.X_valid)
            if not np.allclose(a,b): raise AssertionError('저장·복원 예측 불일치')
        write_json(output/'persistence_check.json',{'same_predictions':True,'trusted_local_files_only':True})
    if level>=3:
        # 별도 탐색 실험은 기본 비교표를 변경하지 않습니다.
        from search_extension import run_search
        run_search(d,cls_folds,output)
    summary={'level':level,'choices_by_CV':choices,'tie_rule':'候補宣言順 (完全同点時)',
        'integrity':integrity,'boundary_checks':check,'test_evaluated':False,
        'selection_metric':{'classification':'mean fold F1','regression':'mean fold RMSE'},
        'data_sha256':d.source_hashes,'versions':environment(),
        'warnings':['CVは選択用。validも前日から反復診断され独立最終評価ではない。',
                    '重要度は因果関係ではない。validの重要度で特性を自動削除しない。']}
    write_json(output/'run_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--level',type=int,choices=[1,2,3],default=DEFAULT_LEVEL)
    args=p.parse_args()
    run_project(args.data_dir,args.output,args.level)
if __name__=='__main__': main()
