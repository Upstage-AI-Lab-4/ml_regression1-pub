# 머신러닝 스터디 1조 - House Price Prediction | 아파트 실거래가 예측
## Team

| ![이동호](https://avatars.githubusercontent.com/u/97029997?v=4) | ![김서현](https://avatars.githubusercontent.com/u/177704202?v=4) | ![김이준](https://avatars.githubusercontent.com/u/74906042?v=4) | ![박주연](https://avatars.githubusercontent.com/u/40532035?v=4) | 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [이동호 (팀장)](https://github.com/Horidong)             |            [김서현](https://github.com/tjgusKim)             |            [김이준](https://github.com/yijoon009)             |            [박주연](https://github.com/pbcs0321)             |
| * 외부데이터 추가<br>(기준금리, 부동산 매수심리)<br>* 모델 변경/파라미터 튜닝<br>* 범주형 데이터 라벨링| * LightGBM 하이퍼 파라미터 튜닝<br> * 외부 데이터 추가<br> (인접 초등학교)<br> * 데이터셋 분할<br> * 인기 상위 시공사 라벨링 | * 외부 데이터 추가<br> (지하철, 대장아파트)<br> * 모델 성능 평가<br> * x,y좌표 결측치 추가<br> * 회의 내용 정리 및 요약| * 파생변수 생성<br> (로그 함수화, 스케일링) <br> * 외부 데이터 추가<br> (한강/공원/매매가격지수)|

## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

- 서울시 아파트 실거래가 매매 데이터를 기반으로 아파트 가격을 예측하는 대회

### Timeline

- September 2, 2024 - Start Date
- September 13, 2024 - Final submission deadline

## 2. Components

### Directory

```
├── code
│   ├── baseline_code.ipynb
│   ├── last_code.ipynb
│   └── requirements.txt
├── data
│   ├── budongsan_simli.csv
│   ├── bus_feature.csv
│   ├── elementary_XY.csv
│   ├── koreanbank_rate.csv
│   ├── park.csv
│   ├── rebuilding.csv
│   ├── subway_feature.csv
│   ├── test.csv
│   └── xy_pos.csv
└── outputs
    └── output_name.csv
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board
#### 중간성적 2위🥈 : RMSE: 	15042.1302
![Leaderboard(mid)](https://github.com/user-attachments/assets/5b876442-6ea9-46de-b964-6620209ddd4e)


#### 최종성적 3위🥉 - RMSE: 12518.1396
![Leaderboard(final)](https://github.com/user-attachments/assets/dc81980c-ff28-4142-bdc0-424c11e9b2b0)


### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
