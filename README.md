# TuneFit Item-Based CF

ListenBrainz 청취 이력으로 사용자별 Top-10 곡을 추천하는 아이템 기반 협업 필터링 모델입니다. 내장 예제 청취 기록으로 추천을 실행하고 결과를 CSV로 저장할 수 있습니다.

## 모델 구현

1. 학습 데이터의 `(user_id, track_id, listen_count)`를 희소 사용자–트랙 행렬로 변환합니다. 같은 사용자와 곡 조합의 청취 횟수를 합친 뒤 `log1p`를 적용합니다.
2. 곡별 청취 벡터 간 코사인 유사도를 배치 단위로 계산하고, 자기 자신을 제외한 유사한 곡을 최대 Top-K개씩 보관합니다.
3. 사용자가 들은 각 곡의 `log1p(청취 횟수) × 이웃 유사도`를 후보 곡별로 합산합니다. 학습 기간에 이미 들은 곡을 제외하고 최대 10곡을 추천합니다.
4. 곡의 인기도는 학습 기간에 그 곡을 들은 고유 사용자 수입니다. 보정 점수는 `기본 점수 / 인기도 ** gamma`입니다. `gamma=0`이면 기본 모델이고, `gamma`가 커질수록 인기곡의 점수가 더 많이 내려갑니다.

## 실행 방법

레포 루트에서 Python 3.12 기준으로 실행합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m item_based --output outputs/demo_baseline.csv
python -m item_based --gamma 0.5 --output outputs/demo_adjusted.csv
```

명령은 내장된 예제 데이터를 사용합니다. `--top-k`는 곡마다 보관할 이웃 수이며 기본값은 50입니다. `--gamma`는 0부터 1까지 지정할 수 있습니다.

## 실제 데이터 연결

데이터 경로는 프로젝트 루트의 `.env`에 설정합니다. 처음 설정할 때는 아래 명령으로 예제를 복사하고, `DATA_DIR`을 CSV 7개가 들어 있는 폴더로 변경합니다. 상대 경로는 프로젝트 루트 기준입니다.

```bash
cp .env.example .env
```

```dotenv
DATA_DIR=data/
```

`.env`에는 파일 경로만 저장합니다. 실제 데이터는 `data/` 안에 보관하며, `.env`와 데이터 파일은 Git에서 제외됩니다. `.env.example`에는 공유 가능한 예시 경로만 포함합니다. 이미 `.env`가 있다면 덮어쓰지 말고 경로만 확인하세요.

전처리 저장소의 `datasets/listenbrainz-preprocessing.zip`을 풀고 CSV 7개가 있는 폴더를 지정합니다. 원본 재전처리나 시간 분할은 다시 하지 않습니다.

```bash
python -m pip install -r requirements.txt
python -m item_based.data
# 다른 위치를 직접 지정할 수도 있습니다.
python -m item_based.data --data-dir data/
```

경로 우선순위는 `--data-dir` → 실행 환경의 `DATA_DIR` → 프로젝트 루트 `.env`의 `DATA_DIR`입니다. 상대 경로는 프로젝트 루트 기준입니다.

| 파일 | 역할 |
| --- | --- |
| `train_interactions.csv` | `user_idx`, `item_idx`, `listen_count`로 학습 CSR 행렬 생성 |
| `users.csv` | 공통 사용자 인덱스와 원본 ID |
| `items.csv` | 공통 곡 인덱스, 곡명·아티스트명, `train_listener_count` |
| `validation_users.csv`, `validation_targets.csv` | 검증 사용자와 정답 |
| `test_users.csv`, `test_targets.csv` | 테스트 사용자와 정답 |

로더는 UTF-8 BOM과 이름에 포함된 쉼표·개행을 처리하고, `NA`·`NULL` 문자열을 그대로 보존합니다. 공통 인덱스를 유지하며 원시 청취 횟수에 로그 변환을 적용하지 않습니다. 인덱스 범위, 중복 학습·정답 쌍, 학습 인기도, 평가 사용자와 정답 연결, 정답의 학습 이력 중복 및 추천 후보 10곡 조건을 검사합니다.

배포 데이터의 예상 출력은 다음과 같습니다.

```text
학습 행렬: 5,000명 × 1,525,005곡
학습 사용자·곡 쌍: 2,739,292
학습 청취 횟수 합계: 6,623,584
검증: 3,922명 / 정답 193,323쌍
테스트: 3,852명 / 정답 201,203쌍
공통 인덱스·인기도·평가 정답 검증 완료
```

코드에서는 `from item_based.data import load_dataset` 후 `load_dataset()`으로 불러옵니다. 반환값의 `counts`는 사용자 행 × 곡 열의 희소 행렬이고, `users`와 `items`의 행 번호는 공통 인덱스입니다. `validation`·`test`는 각각 `user_indices`와 정답 희소 행렬 `targets`를 제공합니다.

기존 `python -m item_based` 명령은 예제 데이터를 사용합니다. 실제 데이터는 `--real-data`를 지정해야 학습합니다.

### 공통 학습 행렬과 모델 연결

```python
from item_based.data import load_dataset
from item_based.model import ItemBasedCF

dataset = load_dataset()
model = ItemBasedCF(top_k=50, batch_size=128)
model.fit_matrix(dataset.counts)
recommendations = model.recommend_by_index(user_idx=0, top_n=10, gamma=0.0)
for recommendation in recommendations:
    item = dataset.items.iloc[recommendation.item_idx]
    print(recommendation.item_idx, item["recording_name"], recommendation.score)
```

`fit_matrix()`는 공통 인덱스를 그대로 유지하고 원본 CSR을 복사해 float32 `log1p` 가중치를 적용합니다. 원시 청취 횟수 행렬은 바꾸지 않습니다. 인기도는 학습 행렬의 고유 청취자 수이며 로더에서 `items.csv`와 일치함을 검사합니다.

`recommend_by_index()`의 반환값은 `item_idx`, `score`, `popularity`입니다. 양의 점수 내림차순, 동점이면 숫자 `item_idx` 오름차순으로 추천합니다. 학습 청취곡은 제외하고, 양의 점수 후보가 부족하면 나머지 전체 카탈로그의 0점 후보를 같은 인덱스 순서로 채웁니다. 후보가 10개 이상이면 서로 다른 10곡을 반환합니다. 예제용 기존 `fit()`·`recommend()` 인터페이스도 유지합니다.

전체 데이터를 학습하고 사용자 한 명의 추천 CSV를 저장하는 명령은 다음과 같습니다. 실행할 때마다 전체 모델을 새로 학습하며, `--user-idx`는 추천 대상만 선택합니다.

```bash
python -m item_based --real-data --user-idx 0 --output outputs/baseline_user_0.csv
```

출력 컬럼은 `user_idx`, `user_id`, `rank`, `item_idx`, `artist_name`, `recording_name`, `score`, `train_listener_count`입니다. 파일은 UTF-8 BOM CSV이며 이름의 쉼표·따옴표·개행을 CSV 규칙으로 저장합니다.

현재 검증 범위는 작은 데이터의 학습→추천→CSV 저장과 기존 예제 실행입니다. 실제 152만 곡 전체 학습의 메모리·시간 점검은 다음 단계입니다. `--batch-size`는 유사도 계산 블록 크기이며, 낮추면 블록 작업 공간을 줄일 수 있지만 최종 이웃 행렬의 크기는 줄지 않습니다. 전체 평가와 모델 저장·재사용은 후속 작업입니다.

| 구간 | 날짜 (2026년, 양끝 포함) | 용도 |
| --- | --- | --- |
| 학습 | 8월 17일 ~ 9월 27일 | 청취 행렬, 유사도 및 인기도 계산 |
| 검증 | 9월 28일 ~ 10월 4일 | 이웃 수와 인기도 보정 강도 선택 |
| 테스트 | 10월 5일 ~ 10월 11일 | 선택된 설정으로 최종 비교 |

## 결과 처리

예제 실행은 추천 곡 한 개당 한 줄을 CSV에 저장합니다. 두 CSV에서 같은 `track_id`의 `rank`와 `score`를 비교하면 인기도 보정에 따른 변화를 확인할 수 있습니다.

| 컬럼 | 의미 |
| --- | --- |
| `user_id` | 추천 대상 사용자 |
| `rank` | 점수 내림차순의 추천 순위 |
| `track_id` | 추천 곡 ID |
| `score` | 해당 실행의 최종 점수 |
| `popularity` | 학습 기간 고유 청취자 수 |

실제 실험에서는 트랙 ID에 곡명과 아티스트명을 연결하고, 기본 모델과 인기도 보정 모델을 **Recall@10**(검증·테스트 구간의 정답 곡 중 추천에 포함된 비율)과 **ARP**(추천 곡의 평균 학습 인기도)로 비교할 예정입니다. 검증 구간에서 Recall@10 조건을 만족하는 설정 중 ARP가 가장 낮은 설정을 선택하고, 테스트 구간에서 최종 평가합니다. 실제 데이터 연결과 평가 코드는 후속 작업입니다.

원본과 전처리 데이터는 `data/`에, 실험 결과는 `outputs/`에 둡니다. 데이터 파일과 실험 결과는 Git에서 제외합니다.
