# 데이터 사전

## 원본 테이블

| 파일 | 관측 단위 | 주요 키 | 프로젝트 사용 |
|---|---|---|---|
| `campaign_desc.csv` | 캠페인 | `CAMPAIGN` | 캠페인 유형, 시작일, 종료일 |
| `campaign_table.csv` | 캠페인 수신 가구 | `CAMPAIGN`, `household_key` | 캠페인 노출과 중첩 확인 |
| `coupon.csv` | 캠페인 쿠폰 대상 상품 | `CAMPAIGN`, `COUPON_UPC`, `PRODUCT_ID` | 캠페인 대상 상품 정의 |
| `coupon_redempt.csv` | 가구별 쿠폰 사용 | `household_key`, `CAMPAIGN`, `COUPON_UPC` | 발행 전 쿠폰 사용 이력 점검 |
| `hh_demographic.csv` | 가구 | `household_key` | 인구통계 사전 특성 |
| `product.csv` | 상품 | `PRODUCT_ID` | 상품 분류와 설명 |
| `transaction_data.csv` | 장바구니 내 상품 거래행 | `household_key`, `BASKET_ID`, `PRODUCT_ID`, `DAY` | 발행 전 구매 특성과 캠페인 기간 결과 |

## 주요 원본 변수

| 변수 | 의미 | 주의점 |
|---|---|---|
| `CAMPAIGN` | 캠페인 식별자 | 숫자 순서가 시간 순서를 항상 의미하지는 않음 |
| `START_DAY`, `END_DAY` | 데이터 기준 캠페인 시작·종료일 | 달력 날짜가 아닌 상대 일자 |
| `household_key` | 가구 식별자 | 분석의 기본 관측 단위 |
| `BASKET_ID` | 장바구니 식별자 | 거래행 수를 장바구니 수로 사용하지 않음 |
| `PRODUCT_ID` | 상품 식별자 | `coupon.csv`와 `product.csv` 연결 키 |
| `QUANTITY` | 거래 수량 | 반품·이상값 여부를 점검 |
| `SALES_VALUE` | 거래 매출 | 데이터셋 단위로 표기하며 통화를 임의 추정하지 않음 |
| `COUPON_DISC` | 거래 쿠폰 할인 | 음수 부호와 0의 의미 확인 필요 |

## 분석 테이블 권장 변수

| 구분 | 변수 예시 | 산출 기준 |
|---|---|---|
| 식별·처치 | `household_key`, `campaign_id`, `treatment` | 단독 캠페인 노출 기준 |
| 사전 구매 | `recency`, `pre_baskets`, `pre_sales`, `pre_quantity` | 캠페인 시작 전 설정 기간 |
| 사전 관련성 | `pre_target_purchases`, `pre_coupon_redemptions`, `pre_campaign_count` | 처치 이전 정보만 사용 |
| 인구통계 | 원본 가구 특성 | 결측과 Unknown을 구분 |
| 주요 결과 | `target_purchase`, `target_sales`, `target_quantity` | 캠페인 대상 상품, 캠페인 기간 |
| 보조 결과 | `any_purchase`, `total_sales`, `baskets` | 전체 상품, 캠페인 기간 |

성향점수는 캠페인 수신 가능성을 나타낸다. 결과변수나 캠페인 기간 쿠폰 사용 여부를 성향점수 입력에 포함하지 않는다.
