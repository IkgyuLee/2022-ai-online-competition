# Object Detection Task - SKKU Team

## Directory Structure

```
$(USER)
|-- /FINAL_DATA
|-- /submissions
|   |-- leaderboard_best.json
|   |-- leaderboard_weight1.pt
|   |-- leaderboard_weight2.pt
|   |-- leaderboard_weight3.pt
|
|-- /yolov5
|-- README.md
|-- requirements.txt
```

- `/USER` 디렉토리는 위와 같은 구조로 설정
  - `/FINAL_DATA` : `/DATA`의 train 데이터와 augmentation 된 데이터를 담으며, 최종적으로 학습에 사용되는 전체 데이터 폴더
  - `/submissions` : 리더보드 최고점에 해당되는 weight 및 json 파일과, 재현 시 생기는 weight 및 json 파일이 모두 담기는 폴더
  - `/yolov5` : 학습 모델의 전체 코드 폴더

### 1) Source Code

```
$(yolov5)
|-- /data
|-- /models
|-- /runs
|-- /utils
|-- augmentation.py
|-- preprocess.py
|-- test_reproduction.py
|-- test_leaderboard.py
|-- train_all.py
|-- ...
|-- train.py
|-- val.py
```

- `/data` : 하이퍼 파라미터 관련 정보를 담은 폴더
- `/models` : yolov5 모델 정보에 대한 yaml 파일 폴더
- `/runs` : train 결과(경로: `/runs/train`)와 test 결과(경로: `/runs/val`)가 저장되는 폴더 (학습 시 디렉토리 생성됨.)
- `/utils` : train과 test 시에 사용되는 여러 함수들을 모아 둔 코드 폴더
- `augmentation.py` : 학습 전 진행할 데이터 증강을 위한 코드
- `preprocess.py` : 학습 전 진행할 전처리 코드
- `test_reproduction.py` : 재현 학습 결과로 생긴 weight 파일을 활용하여 테스트 후 json 파일을 생성하는 코드
- `test_leaderboard.py` : 리더보드 최고점에 해당되는 json 파일을 생성하는 코드
- `train_all.py` : 모델 학습 코드

### 2) Pre-trained Weight files

```
$(yolov5)
|-- ...
|-- yolov5l6.pt
|-- yolov6x6.pt
```

- `yolov5l6.pt` : 학습 모델 1 가중치 파일 (학습 시 자동으로 다운로드 됨.)
- `yolov5x6.pt` : 학습 모델 2 가중치 파일 (학습 시 자동으로 다운로드 됨.)

---

<br>

## 리더보드 최고점 Json 파일 재현 방법

```bash
$ cd ../USER/yolov5
$ python test_leaderboard.py
```

- `/USER/submissions` 경로에 `leader_reproduction.json` 파일 생성

---

<br>

## 전체 학습 및 테스트 과정 재현 방법

### 0. 전체 실행 순서

```
Install Requirements -> Augmentation -> Preprocess -> Train -> Test
```

- **단, Requirements는 재현 서버에 이미 설치되어 있으므로 서버 환경이 달라지는 경우에만 실행.**

### 1. Install Requirements

```bash
$ cd ../USER
$ pip install -r requirements.txt
```

### 2. Augmentation

```bash
$ cd yolov5
$ python augmentation.py
```

- `augmentation.py` : 학습 전 데이터 증강을 위한 코드
- 재현 서버에서 약 50분 소요
- 최종 학습 이미지 경로 : `/USER/FINAL_DATA/train/images`
- 최종 학습 라벨 경로 : `/USER/FINAL_DATA/train/labels`

### 3. Preprocess

```bash
$ python preprocess.py
```

- `preprocess.py` : 학습 전 진행할 전처리 코드
- 재현 서버에서 약 3초 소요
- `/FINAL_DATA/train` 디렉토리 내에 아래의 파일 생성
  - `data.names`
  - `data.yaml`
  - `train.txt`
  - `val.txt`

### 4. Train

```bash
$ python train_all.py
```

- `train_all.py` : 모델 1과 모델 2를 순서대로 한 번에 실행하는 코드
- `yolovl6` 모델 1 - 재현 서버에서 약 13시간 소요
- `yolovx6` 모델 2 - 재현 서버에서 약 22시간 소요
- 학습 종료 후 `/USER/yolov5/runs/train/yolov5l6-model/weights` 경로와 `/USER/yolov5/runs/train/yolov5x6-model/weights` 경로 안에 각각 학습된 결과 weight 파일(.pt) 생성

### 5. Test

```bash
$ python test_reproduction.py
```

- `test_reproduction.py` : 모델 1과 모델 2 앙상블 테스트 및 json 생성 코드
- 사용할 모델 1과 모델 2 weight 파일들을 `/USER/submissions` 경로로 옮긴 뒤, 이를 이용하여 앙상블 테스트 실행
- 재현 서버에서 약 2시간 소요
- 결과 json 파일 생성 (경로: `/USER/submissions`)

---

<br>

## Submission File

```
$(submissions)
|-- leaderboard_best.json
|-- leaderboard_weight1.pt
|-- leaderboard_weight2.pt
|-- leaderboard_weight3.pt
|-- leaderboard_reproduction.json
|-- reproduction.json
|-- epoch14.pt
|-- epoch15.pt
|-- epoch19.pt
```

- `leaderboard_best.json` : 리더보드 Final 최고점 json 파일
- `leaderboard_weight*.pt` : 리더보드 Final 최고점 weight 파일 1, 2, 3
- `leaderboard_reproduction.json` : 리더보드 최고점 Json 재현 파일 (재현 완료 후 생성됨.)
- `reproduction.json` : 재현 결과 json 파일 (학습 및 테스트 완료 후 생성됨.)
- `epoch14.pt` : yolov5l6 모델 1의 epoch 14에 해당하는 weight 파일 (학습 완료 후 생성됨.)
- `epoch15.pt` : yolov5l6 모델 1의 epoch 15에 해당하는 weight 파일 (학습 완료 후 생성됨.)
- `epoch19.pt` : yolov5x6 모델 1의 epoch 19에 해당하는 weight 파일 (학습 완료 후 생성됨.)
