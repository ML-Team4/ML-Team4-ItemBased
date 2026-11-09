# TuneFit Item-Based CF

ListenBrainz 청취 이력으로 사용자별 Top-10 곡을 추천하는 머신러닝 텀프로젝트입니다. 이 레포는 아이템 기반 협업 필터링 실험을 담당합니다.

## 실행 환경

Python 3.12를 기준으로 가상 환경을 만듭니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import item_based, numpy, pandas, pyarrow, scipy, sklearn"
```

## 데이터

팀에서 추출한 ListenBrainz Parquet 파일을 `data/raw/`에 둡니다. 계획서에 정의된 입력 컬럼은 `user_id`, `listened_at`, `artist_name`, `recording_name`입니다. 아티스트명과 곡명을 정규화한 조합으로 트랙을 식별합니다.

| 구간 | 날짜 (2026년, 양끝 포함) |
| --- | --- |
| 학습 | 8월 17일 ~ 9월 27일 |
| 검증 | 9월 28일 ~ 10월 4일 |
| 테스트 | 10월 5일 ~ 10월 11일 |

원본과 전처리 데이터는 `data/`에, 실험 결과는 `outputs/`에 둡니다. 두 경로의 데이터와 결과는 Git에서 제외합니다.

## 구현 순서

1. Parquet 로딩, 트랙 식별자 정규화 및 시간순 분할
2. 학습 청취 횟수에 `log1p`를 적용한 희소 사용자–트랙 행렬 생성
3. 트랙 간 코사인 유사도 계산, Top-K 이웃 보관 및 학습 이력에 없는 곡 Top-10 추천
4. 학습 구간의 고유 청취자 수로 곡 인기도를 계산하고, 추천 점수를 `인기도 ** gamma`로 나누어 조정
5. 검증 데이터에서 설정을 선택하고 테스트 데이터에서 기본 모델과 인기도 보정 모델의 Recall@10 및 ARP 비교
