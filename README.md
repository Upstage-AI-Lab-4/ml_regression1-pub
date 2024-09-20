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

- 결측치 확인
<img width="1020" alt="스크린샷 2024-09-20 오전 9 28 29" src="https://github.com/user-attachments/assets/31f88693-3876-4b99-b825-feed607a9811">![image](https://github.com/user-attachments/assets/3792b829-3510-4820-ba6d-c655fd3f4149)

- 이상치 확인
<img width="888" alt="스크린샷 2024-09-19 오후 3 37 48" src="https://github.com/user-attachments/assets/db51316d-bb37-4601-b676-12838fbf906d">![image](https://github.com/user-attachments/assets/9f0e0ce4-c444-4897-a891-10b9bc512c06)



### Data Processing

- 결측치 87만개 이상 column 제거
- x,y 결측치 보충
- x : Robust Scaling, y : Log Scaling
  
## 4. Modeling

### Model descrition

- RandomForest 기반으로 작성된 베이스라인 모델에서 성능 향상을 목표로 LightGBM 모델로 변경하여 사용.
- LightGBM 모델 적용 후 RMSE 점수가 향상됨.
- 대회 중반 이후, XGBoost 및 CatBoost 등 다양한 모델을 테스트했으나, 동일 조건의 테스트 환경에서 RMSE 점수가 LightGBM이 가장 우수한 결과를 보임.
- 최종적으로 LightGBM 모델을 선정하고, 하이퍼파라미터 튜닝을 진행.

### Modeling Process

**교차 검증(Cross Validation)**
- Train 데이터셋과 Valid 데이터셋을 8:2로 구분하여 학습 및 검증을 진행.
	- 베이스라인 코드에서 사용된 방식을 그대로 사용하여 테스트 진행.
- 데이터를 5개로 분할하여 각각 Train 및 Test를 진행하는 K-Fold 방식 적용.
	- 5-Fold 후 전체 모델의 평균값으로 검증.
	- 5-Fold 후 성능이 우수한 상위 3개의 모델 평균값으로 검증.
- 데이터를 시계열에 따라 정렬한 후, K개로 분할하여 순차적으로 Train 및 Test 진행.
	- ‘계약년/월’에 따라 정렬 후 5-Fold 검증 진행.
- 동일한 조건에서 테스트를 진행했으며, Hold-out 방식이 가장 우수한 성능을 보여 최종적으로 Hold-out 방식으로 진행.


**학습 및 평가**
- Train/Valid로 나눈 학습 데이터에 각각 Scaling 적용
	- Log Scaling
	- Robust Scaling
- LightGBM 모델을 사용하여 Scaling 적용된 데이터로 학습 진행
	- n_estimators = 10,000
	- eval_metric = 'rmse'
- 예측값에 대해 역 Scaling 적용
- 최종 검증 RMSE를 통해 모델 성능 확인
	- 모델을 통해 최종 예측한 데이터에서 손실이 발생하는 경우가 종종 있어, output 출력 후 눈으로 직접 확인

<img width="942" alt="image" src="https://github.com/user-attachments/assets/8052a6c1-55cb-4b08-a252-dd76fad468a5">


**Feature Importance**
![image](https://github.com/user-attachments/assets/ab1d3673-c9ab-4c7f-9bdb-f789c92c2b34)


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
