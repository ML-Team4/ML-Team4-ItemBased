# TuneFit Item-Based CF

ListenBrainz 청취 이력으로 사용자별 Top-10 곡을 추천하는 아이템 기반 협업 필터링 모델입니다. 내장 예제 청취 기록으로 추천을 실행하고 결과를 CSV로 저장할 수 있습니다.

## 모델 구현

1. 사용자–곡 학습 행렬을 만들고, 반복 청취의 영향을 완화하기 위해 `log1p(청취 횟수)`를 기본 입력으로 사용합니다. `weighting="raw"`를 지정하면 입력값을 변환 없이 사용합니다.
2. 모든 사용자를 축으로 곡별 코사인 유사도를 계산합니다. **추천할 후보 곡을 기준으로** 자기 자신을 제외한 Top-K 이웃을 선택합니다.
3. `sum(이웃 유사도 × 사용자 선호 강도) / sum(이웃 유사도)`의 **가중평균**을 계산합니다. 미관측 이웃 입력은 0이며, 분모는 후보 곡이 선택한 이웃 전체를 사용합니다.
4. 인기도 보정을 위해 `예측 점수 / 고유 학습 청취자 수 ** gamma`를 적용합니다. gamma=0은 보정 전 모델입니다.
5. 사용자의 원래 학습 청취곡을 제외하고 상위 10곡을 추천합니다. 0점 동률과 10곡 채우기는 공통 평가에 필요한 프로젝트 처리 규칙입니다.

## 실행 방법

레포 루트에서 Python 3.12 기준으로 실행합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m item_based --output outputs/demo_baseline.csv
python -m item_based --gamma 0.5 --output outputs/demo_adjusted.csv
```

명령은 내장된 예제 데이터를 사용합니다. `--weighting`의 기본값은 `log1p`이며, 입력값을 그대로 사용할 때는 `raw`를 지정합니다. `--top-k`는 곡마다 보관할 이웃 수이며 기본값은 50입니다. `--gamma`는 0부터 1까지 지정할 수 있습니다.

## 실제 데이터 연결

데이터 경로는 프로젝트 루트의 `.env`에 설정합니다. 처음 설정할 때는 아래 명령으로 예제를 복사하고, `DATA_DIR`을 CSV 7개가 들어 있는 폴더로 변경합니다. 상대 경로는 프로젝트 루트 기준입니다.

```bash
cp .env.example .env
```

```dotenv
DATA_DIR=data/raw
```

`.env`에는 파일 경로만 저장합니다. 실제 데이터는 `data/` 안에 보관하며, `.env`와 데이터 파일은 Git에서 제외됩니다. `.env.example`에는 공유 가능한 예시 경로만 포함합니다. 이미 `.env`가 있다면 덮어쓰지 말고 경로만 확인하세요.

전처리 저장소의 `datasets/listenbrainz-preprocessing.zip`을 풀고 CSV 7개가 있는 폴더를 지정합니다. 원본 재전처리나 시간 분할은 다시 하지 않습니다.

```bash
python -m pip install -r requirements.txt
python -m item_based.data
# 다른 위치를 직접 지정할 수도 있습니다.
python -m item_based.data --data-dir data/raw
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

`fit_matrix()`는 공통 인덱스를 그대로 유지하고 원본 CSR을 복사해 float32로 계산합니다. 기본 `weighting="log1p"`는 로그 변환을 적용하고, `weighting="raw"`는 입력값을 그대로 사용합니다. 원시 청취 횟수 행렬은 바꾸지 않습니다. 인기도는 학습 행렬의 고유 청취자 수이며 로더에서 `items.csv`와 일치함을 검사합니다.

`recommend_by_index()`의 반환값은 `item_idx`, `score`, `popularity`입니다. 양의 점수 내림차순, 동점이면 숫자 `item_idx` 오름차순으로 추천합니다. 학습 청취곡은 제외하고, 양의 점수 후보가 부족하면 나머지 전체 카탈로그의 0점 후보를 같은 인덱스 순서로 채웁니다. 후보가 10개 이상이면 서로 다른 10곡을 반환합니다. 예제용 기존 `fit()`·`recommend()` 인터페이스도 유지합니다.

전체 데이터를 학습하고 사용자 한 명의 추천 CSV를 저장하는 명령은 다음과 같습니다. 실행할 때마다 전체 모델을 새로 학습하며, `--user-idx`는 추천 대상만 선택합니다.

```bash
python -m item_based --real-data --user-idx 0 --output outputs/lecture_user_0.csv
```

출력 컬럼은 `user_idx`, `user_id`, `rank`, `item_idx`, `artist_name`, `recording_name`, `score`, `train_listener_count`입니다. 파일은 UTF-8 BOM CSV이며 이름의 쉼표·따옴표·개행을 CSV 규칙으로 저장합니다.

`--batch-size`는 유사도 계산 블록 크기이며, 낮추면 블록 작업 공간을 줄일 수 있지만 최종 이웃 행렬의 크기는 줄지 않습니다. 최종 테스트 평가와 모델 저장·재사용은 후속 작업입니다.

### 전체 데이터 계산과 메모리

유사도는 곡 블록 단위로 계산하고 각 곡의 Top-K만 저장합니다. 전체 곡×곡 dense 행렬을 만들지 않습니다. 이웃은 Python 숫자 객체 리스트 대신 NumPy 배열로 저장하며, 실제 데이터에서는 점수 float32·인덱스 int32를 사용합니다. K=50의 이웃 저장 버퍼는 최대 약 588 MiB입니다. 입력 데이터·정규화 행렬·임시 유사도 블록은 별도 메모리를 사용합니다.

상위 K개 선택은 점수 경계를 먼저 찾고, 경계에서 동점인 곡은 공통 인덱스 순서로 선택합니다. 마지막에 선택된 곡만 정렬합니다. 이는 전체 정렬과 같은 순위를 유지하며, 예제 데이터의 문자열 ID에도 같은 규칙을 적용합니다. 점수가 모두 같은 경우에도 동점 곡 전체를 점수순으로 정렬하지 않습니다.

실제 데이터 실행 시 약 10초마다 진행률을 출력하고, 학습 종료 후 경과 시간·이웃 수·이웃 CSR 크기를 출력합니다. macOS·Linux에서는 Python 내부의 `resource.getrusage()`로 프로세스 최대 RSS도 측정합니다. 최대 RSS는 데이터 로딩과 작업 공간을 포함하며 이웃 CSR 크기와 다릅니다. 해당 모듈을 제공하지 않는 환경에서는 최대 RSS를 `null`로 저장합니다.

```bash
.venv/bin/python -m item_based --real-data \
  --top-k 50 --batch-size 128 --user-idx 0 \
  --output outputs/lecture_user_0.csv
```

추천 CSV 옆의 `lecture_user_0_metrics.json`에는 데이터 크기, K, 배치 크기, gamma, 학습 시간, 이웃 수, 이웃 CSR 바이트 수, 프로세스 최대 RSS 바이트 수와 추천 개수를 저장합니다. 이 파일도 `outputs/` 아래에서 Git 게시 대상에서 제외됩니다. 정확도 평가는 별도 단계이며 학습 완료나 메모리 측정만으로 추천 품질을 판단하지 않습니다.

회귀 검증은 다음 명령으로 실행합니다. 상위 K의 경계 동률, 작은 행렬의 직접 코사인 계산과의 일치, CSV 저장 및 원본 입력 보존을 검사합니다.

```bash
python -m unittest discover -s tests -v
```

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

실제 실험에서는 기본 모델과 인기도 보정 모델을 **Recall@10**과 **ARP**로 비교합니다. 검증 구간에서 설정을 선택하고, 테스트 구간에서 최종 평가합니다. 현재 전체 검증 사용자 평가까지 구현했으며, 설정 탐색과 최종 테스트는 후속 작업입니다.

### 전체 검증 사용자 평가

```bash
python -m item_based.evaluate --top-k 50 --weighting log1p --gamma 0 \
  --output-dir outputs/validation_k50_gamma0
```

`.env`의 `DATA_DIR` 또는 `--data-dir`을 사용합니다. 매 실행마다 최초 train만으로 모델을 학습하고, `validation_users.csv`의 전체 사용자와 `validation_targets.csv`의 정답을 사용합니다. 테스트 정답은 설정 선택에 사용하지 않습니다. 사용자별 추천 후보는 전체 train 카탈로그에서 해당 사용자의 원래 train 청취곡만 제외한 집합입니다.

- **Recall@10:** 각 사용자의 `추천 10곡과 정답의 교집합 수 / 정답 수`를 계산한 뒤 사용자 평균을 냅니다. 전체 정답 쌍의 적중률과는 다릅니다.
- **ARP:** 각 사용자의 추천 10곡에 대해 `items.csv`의 `train_listener_count` 평균을 계산한 뒤 사용자 평균을 냅니다. 총 재생 횟수는 사용하지 않습니다.
- 추천은 사용자당 서로 다른 10곡이어야 합니다. 0점 추천과 Recall이 0인 사용자도 평가에 포함합니다. 10곡 미달·중복·학습 청취곡 포함·유효하지 않은 점수가 있으면 평가를 중단합니다.

| 결과 파일 | 내용 |
| --- | --- |
| `summary.json` | 알고리즘 버전·입력 가중치, 전체 사용자 수, Recall@10, ARP, K·gamma, 평가 시간, 0점 추천 규모 |
| `users.csv` | 사용자별 정답 수, 적중 수, Recall@10, ARP, 0점 추천 수 |
| `recommendations.csv` | 사용자별 10곡의 공통 곡 인덱스, 순위, 점수와 학습 인기도 |

진행률은 약 10초마다 출력됩니다. 결과는 `outputs/` 아래에 저장하며 Git 게시 대상에서 제외됩니다. 같은 출력 폴더를 지정하면 기존 평가 결과를 갱신하므로, 다른 K·gamma를 비교할 때는 각각 다른 `--output-dir`을 사용하세요. 실행이 실패하면 CSV가 일부만 기록될 수 있고, 성공한 평가에만 `summary.json`을 생성합니다.

원본과 전처리 데이터는 `data/`에, 실험 결과는 `outputs/`에 둡니다. 데이터 파일과 실험 결과는 Git에서 제외합니다.

## 검증 결과

입력에 `log1p(청취 횟수)`를 적용하고, 코사인 유사도로 후보 곡의 Top-K 이웃을 선택한 뒤 가중평균 점수를 계산했습니다. K=50·gamma=0으로 학습하고, 검증 사용자 3,922명에게 각각 10곡을 추천했습니다.

| 항목 | 결과 |
| --- | ---: |
| Recall@10 | 0.014352 |
| ARP | 53.644034 |
| 평가 사용자 | 3,922명 |
| 추천 | 39,220개 |
| 0점 추천 | 275개 |
| 평가 시간 (학습 제외) | 4.3초 |

사용자별 결과와 추천 CSV로 Recall@10·ARP를 독립 재계산해 일치함을 확인했습니다. 결과는 `outputs/validation_k50_gamma0/`에 저장했습니다. gamma=0의 보정 전 결과이며, 이후 검증 데이터로 K·gamma를 비교해 설정을 선택하고 최종 테스트를 진행합니다.
