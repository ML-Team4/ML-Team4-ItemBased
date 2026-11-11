# 강의 알고리즘과 팀 계획서의 연결

기준 자료는 `Machine-Learning-Recommendation.pdf`의 40~59쪽과 TuneFit 팀 계획서 `RcSys-Team4 .pdf`의 3~7쪽이다. 강의는 평점 예측을 설명하고, 팀 계획서는 이 구조를 암묵적 청취 기록의 Top-10 추천에 적용한다.

## 요구사항 대응

| 항목 | 강의 근거 | 팀 계획서 적용 | 구현 |
| --- | --- | --- | --- |
| 입력 | 45~46쪽 사용자–영화 평점, 결측 0 | 5쪽 `log(1+count)` | 기본 `weighting="log1p"` |
| 곡 유사도 | 42·47쪽, 모든 사용자를 축으로 코사인 계산 | 5쪽 트랙 코사인 유사도 | 정규화된 곡 벡터의 내적 |
| 이웃 선택 | 41·48·54~55쪽, 예측 대상 영화의 이웃 | 5쪽 Top-K 유사곡 보관 | 후보 곡별 이웃 선택, 자기 자신 제외 |
| 예측 | 56쪽 유사도 가중평균 | 로그 청취 횟수를 선호 강도로 사용 | 아래 수식의 분자·분모 계산 |
| 최종 추천 | 39쪽 점수 상위 N | 3·5쪽 train 미청취곡 Top-10 | 원래 학습 이력 제외, 상위 10곡 |
| 인기 보정 | 강의 기본식에 없는 프로젝트 확장 | 6쪽 점수 / 인기도의 gamma승 | 기본 gamma=0, 비교 시 0~1 |
| 평가 | 강의의 평점 예측 설명과 구분 | 7쪽 Recall@10·ARP | 전체 공통 검증 사용자 평균 |

## 수식과 행렬 방향

학습 행렬 `X[u, j] = log(1 + listen_count[u, j])`를 만든다. 미관측 좌표는 0으로 표현하며, 이 값이 실제 비선호라는 뜻은 아니다.

각 곡 m의 모든 사용자에 대한 벡터로 코사인 유사도 `S(m,j)`를 계산한다. 자기 자신을 제외한 후보 m의 Top-K 이웃 집합을 `N_K(m)`이라고 하면 강의 56쪽의 예측은 다음과 같다.

```text
prediction(u,m) = sum(S(m,j) * X[u,j] for j in N_K(m))
                  / sum(S(m,j) for j in N_K(m))
```

강의 46·50쪽의 결측 0 처리와 56쪽의 이웃 집합 정의에 맞춰 분모는 후보 m의 선택된 이웃 전체를 기준으로 한다. 사용자가 듣지 않은 이웃의 X는 0이므로 분자에 기여하지 않는다. 사용자마다 관측된 이웃만 골라 분모를 다시 만들지는 않는다. 유사도 합이 0이면 예측은 0으로 처리한다. 저장하지 않은 0 유사도 이웃은 분자와 분모에 모두 영향을 주지 않는다.

이웃 행렬을 `N[m,j] = S(m,j)`로 정의하면 예측 분자는 `X @ N.T`다. Top-K를 자른 뒤에는 N이 대칭이라는 보장이 없으므로 `X @ N`과 같지 않다. 실제 코드는 예측에 사용할 `N.T`를 CSR로 한 번 만들어 보관한다. 후보별 유사도 합도 학습 단계에서 계산해 재사용한다. 전치 시 일시적으로 두 배열이 공존할 수 있으므로 최종 CSR 크기가 전체 최대 메모리를 뜻하지 않는다.

인기도 보정은 이 예측 점수를 계산한 다음 적용한다.

```text
adjusted(u,m) = prediction(u,m) / train_listener_count[m] ** gamma
```

`train_listener_count`는 고유 학습 청취자 수다. gamma=0은 강의 구조에 계획서의 로그 입력을 적용한 기준 모델이다. 로그 변환은 강의의 원래 평점 예제에 있던 단계가 아니라 팀 계획서에서 정한 입력 변환이다.

## 강의 예제 재현

강의 숫자를 검증할 때는 `weighting="raw"`로 평점을 그대로 넣는다.

```python
from scipy.sparse import csr_matrix
from item_based.model import ItemBasedCF

ratings = csr_matrix([[2, 3, 1, 5], [3, 0, 3, 2]])
model = ItemBasedCF(top_k=2, weighting="raw")
model.fit_matrix(ratings)
result = model.recommend_by_index(user_idx=1, top_n=1)
print(result[0].item_idx, result[0].score)
# movie_1, 약 2.373995 → 강의 49쪽의 2.374
```

강의 57쪽에 주어진 거리 0.4034·0.3196을 `similarity=1-distance`로 변환해 평점 2·3을 가중평균하면 2.532811이며, 강의의 2.5328과 일치한다. 이 두 예제와 Top-K 비대칭 방향의 직접 행렬 계산을 테스트로 검증한다.

강의 51~52쪽의 `n_neighbors=3`에는 자기 자신이 포함된다. 프로젝트의 `top_k=2`는 자기 자신을 뺀 이웃 두 개를 의미한다. 대규모 데이터에서 같은 코사인 계산을 희소 블록 연산으로 수행하므로 `NearestNeighbors`를 직접 호출하지 않아도 기본 수식은 같다.

## 실험 구분

프로젝트 기본 명령은 다음과 같다.

```bash
python -m item_based.evaluate --weighting log1p --gamma 0 \
  --output-dir outputs/validation_k50_gamma0
```

결과 JSON은 `algorithm="lecture-item-cf-v1"`, `weighting="log1p"`, `scoring="target-neighborhood weighted average"`를 기록한다. 강의 평점 예제용 raw 모드와 프로젝트의 log1p 모드를 혼동하지 않는다.

이전 구현은 로그 입력·곡 코사인 유사도는 사용했으나, 들었던 곡에서 후보로 점수를 보내고 유사도 합으로 나누지 않았다. 이전 `outputs/validation_baseline/`의 Recall@10=0.007052·ARP=12.066191은 그 구현의 기록이며 수정 후 모델의 성능으로 인용하면 안 된다. 새 결과는 별도 폴더에 저장한다. 기존 데이터 분할·공통 인덱스·평가 사용자와 정답은 유지한다.
