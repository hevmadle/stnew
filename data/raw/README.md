# 원본 데이터 안내

이 저장소는 용량 문제(`transaction_data.csv` 135MB, GitHub 단일 파일 한도 100MB 초과)로
`data/raw/`의 CSV 원본을 포함하지 않는다(`.gitignore`에서 `data/raw/*.csv` 제외).

로컬 또는 배포 환경에서 실행하려면 다음 7개 파일을 이 폴더에 직접 준비해야 한다
(`SOURCE_AND_LICENSE.md` 참고, Kaggle: dunnhumby - The Complete Journey):

- `campaign_desc.csv`
- `campaign_table.csv`
- `coupon.csv`
- `coupon_redempt.csv`
- `hh_demographic.csv`
- `product.csv`
- `transaction_data.csv`

열 이름과 스키마는 `DATA_DICTIONARY.md`를 참고한다.
