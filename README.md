# 머신러닝 스터디 1조 - House Price Prediction | 아파트 실거래가 예측
## Team

| ![이동호](https://avatars.githubusercontent.com/u/97029997?v=4) | ![김서현](https://avatars.githubusercontent.com/u/177704202?v=4) | ![김이준](https://avatars.githubusercontent.com/u/74906042?v=4) | ![박주연](https://avatars.githubusercontent.com/u/40532035?v=4) | 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [이동호 (팀장)](https://github.com/Horidong)             |            [김서현](https://github.com/tjgusKim)             |            [김이준](https://github.com/yijoon009)             |            [박주연](https://github.com/pbcs0321)             |
| * 외부데이터 추가<br>(기준금리, 부동산 매수심리)<br>* 모델 변경/파라미터 튜닝<br>* 범주형 데이터 라벨링| * LightGBM 하이퍼 파라미터 튜닝<br> * 외부 데이터 추가<br> (인접 초등학교)<br> * 데이터셋 분할<br> * 인기 상위 시공사 라벨링 | * 외부 데이터 추가<br> (지하철, 대장아파트)<br> * 모델 성능 평가<br> * x,y좌표 결측치 추가<br> * 회의 내용 정리 및 요약| * 파생변수 생성<br> (로그 함수화, 스케일링) <br> * 외부 데이터 추가<br> (한강/공원/매매가격지수)|

## 0. Overview
### Environment
- AI Stages Server From Upstage
- Python 3.10.13


### Requirements
- matplotlib==3.7.1
- numpy==1.23.5
- pandas==1.5.3
- scipy==1.11.3
- seaborn==0.12.2
- scikit-learn==1.2.2
- statsmodels==0.14.0
- tqdm==4.66.1


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
├── data                #train.csv 파일(기본 제공 훈련 데이터)은 용량 문제로 git에 업로드하지 않음
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

 ## 기본제공 데이터
 - train.csv : 훈련 데이터
 - test.csv : 테스트 데이터
 - bus_feature.csv : 버스 정류장 정보 데이터
 - subway_feature.csv : 지하철역 정보 데이터

 ## 추가 데이터
 - xy_pos.csv : 주소별 위도/경도 매핑 데이터 (경진대회 게시판 6조 권세진님 공유)
 - budongsan_simli.csv : 월별 부동산 매수 심리 데이터
 - elementary_XY.csv : 초등학교 정보 및 위치 데이터
 - koreanbank_rate.csv : 한국은행 월별 기준금리 데이터
 - park.csv : 서울시 공원 정보 데이터
 - rebuilding.csv : 재건축단지 정보 데이터

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
![Leaderboard_mid](https://github.com/user-attachments/assets/1e5581c4-19a3-4e53-8b60-f415516fd46d)


#### 최종성적 3위🥉 - RMSE: 12518.1396
![Leaderboard_last](https://github.com/user-attachments/assets/76682430-9e23-42ae-bacc-844b0a21d585)


### Presentation

- [[패스트캠퍼스] Upstage AI Lab 4기_1팀_발표자료.pdf](https://github.com/user-attachments/files/17038028/Upstage.AI.Lab.4._1._.pdf)

## etc

### Meeting Log

- Notion (https://www.notion.so/1-18a3949384af474ea54aa80fd9bfc9c3)

### Reference

- _Insert related reference_
