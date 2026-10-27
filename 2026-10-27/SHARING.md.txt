# 실습 파일 공유

공유일: 2026년 10월 27일
제공된 정답 코드를 보며 코드의 동작을 이해하고, 주요 처리 흐름을 주석으로 정리하는 방식으로 실습하였습니다.


## 파일 구성
- main.py: 전체 실습 실행 및 결과 저장
- checks.py: 데이터 분할과 모델 입력 조건 검사
- course_support.py: 데이터 로딩, 평가 지표 계산, 저장 보조 기능
- evaluate.py: 모델 평가, 특성 중요도 계산, CV 기준 최적 모델 선택
- model_config.py: 모델별 설정 및 교차검증 분할 정의
- preprocessing.py: 전처리 구성 및 모델 Pipeline 생성
- search_extension.py: GridSearchCV를 활용한 결정트리 추가 탐색

- data/: 실행에 필요한 데이터
- README.md: 기존 실습 안내