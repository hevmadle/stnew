# 쿠폰 캠페인 효과와 수익성 분석

이 폴더는 Kaggle의 dunnhumby The Complete Journey 데이터를 이용해 캠페인별 비교집단을 구성하고, 성향점수 매칭으로 캠페인 효과를 추정한 뒤 쿠폰 후보안의 수익성을 비교하는 프로젝트 작업 공간이다.

## 최종 결과물

- 캠페인 번호를 입력받는 재사용 가능한 분석 파이프라인
- 비교집단 품질과 캠페인 효과를 보여주는 Streamlit 앱
- 손익분기점과 사용자가 입력한 쿠폰 후보안을 비교하는 기능

## 작업 방식

1. 이 폴더를 Claude Code의 작업 폴더로 연다.
2. 강사가 제시하는 프롬프트를 한 번에 하나씩 입력한다.
3. 실행 전 예상 결과를 기록하고, 실행 후 터미널의 수치와 생성 파일을 확인한다.
4. `PROJECT_CHECKLIST.md`의 조건을 통과한 뒤 다음 단계로 이동한다.
5. 원본 데이터는 수정하지 않는다.

## 폴더 구조

- `data/raw/`: 분석에 필요한 Kaggle 원본 CSV 7개
- `data/processed/`: 검증된 가구 단위 분석표
- `analysis/`: 데이터 점검 문서와 탐색 코드
- `pipeline/`: 캠페인 입력형 데이터 준비·효과 추정 코드
- `outputs/`: 캠페인별 최소 결과와 최종 검증 보고서
- `app/`: Streamlit 앱과 수익성 계산 코드
- `CLAUDE.md`: Claude Code가 항상 따라야 할 작업 규칙
- `DATA_DICTIONARY.md`: 원본 테이블과 핵심 변수 정의
- `PROJECT_CHECKLIST.md`: 단계별 검증 기준
- `SOURCE_AND_LICENSE.md`: 데이터 출처와 포함 범위

## 실행 환경

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

앱 파일은 실습 과정에서 생성한다. 분석 파이프라인과 검증을 완료하기 전에는 앱 결과를 최종 결과로 사용하지 않는다.
