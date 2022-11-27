# Capstone_relvelent-between-literacy-and-creative
기간 : 2022.03~2022.06
데이터분석(SPSS)
참여인원: 4명

## 배경
1. 유기견이 늘어나는 숫자가 점점 늘어나고 있어 부작용으로 보호 환경, 안락사, 개물림 사고 등의 문제가 생기고 있음. 이를 해결하고자 유기견 데이터를 통해 입양 확률을 예측하는 모델링을 하고 보호소 내에 있는 유기견에게 조금 더 나은방향을 제시가기 위해 이 프로젝트를 시작함.

## 설계
참고- kaggle 데이터, 동물보호단체 입양 신처서, 라이프사이클 논문
변수는 유기견, 설문자, 설문자의 라이프스타일 총 3개로 분류해 설문조사지 작성
총 24개의 문항으로 구성
## 분석목포
1. 로지스틱 휘귀분석을 사용해 입양 확률 예측 모델링

2. 24개의 변수 중 입양에 영향을 가장 크게 미치는 변수 분석

## 분석
1. PCA - 변수를 줄일 수 있는 가장 좋은 방법으로 PCA를 실시
PCA는 차원 축소기법.  변수의 갯수를 줄인다기 보다는 변수를 랜덤으로 믹싱하는 것. 데이터의 설명력이 가장 좋은 케이스를 찾았지만 어떠한 속성이 포함되어 있는지는 알수 가 없었음.
```Python
//표준화 
pca_df = pd.DataFrame(data=p_data, columns=['a1','a2','b1','b2','b3','b4','b5','b6','b7', 'c1','c2','c3','c4','c5','c6','c7','c8','c9', 'y'])
#pca_df=전처리 후 데이터 
x = pca_df.drop(['y'], axis=1).values # 독립변인들의 value값만 추출
y = pca_df['y'].values # 종속변인 추출

x = StandardScaler().fit_transform(x) # x객체에 x를 표준화한 데이터를 저장
features = ['인식매체','경험','인식','관심도','지식','가족동의','환경','비용지불의사','정부지원정책','성별','중성화','나이','털길이','품종','색상','크기','성격','공고기간']

pd.DataFrame(x, columns=features).head()
```
```Python
//차원 갯수 정하기
pca = PCA(n_components=18)
pca_array = pca.fit_transform(x)
PCA_df = pd.DataFrame(pca_array, index=pca_df.index,
                      columns=[f"pca{num+1}" for num in range(x.shape[1])])
result = pd.DataFrame({'설명가능한 분산 비율(고윳값)':pca.explained_variance_,
             '기여율':pca.explained_variance_ratio_},
            index=np.array([f"pca{num+1}" for num in range(x.shape[1])]))
result['누적기여율'] = result['기여율'].cumsum()
result
```

2. Random Forest - 예정
Decision Tree를 고민하였으나 이 경우 오버피팅에 으로 인한 일반화 부족의 문제로 이를 보완해주는 Random Forest를 하기로 함.

## 결론 
아직 분석이 따 끝나지 않아 결론을 말하는 것이 순서에 맞지 않아 기대효과로 대체함.
기대효과
1. 보호소 내 입양자가 나타나기만 해야하는 수동적인 관리 시스템 개선
2. 유기견의 특성에 따라 입양 혹은 특기 준비 등 가능성 제시
3. 유기견증가에 따른 부작용(개물림사고, 로드킬, 예산 및 사회 비용 등) 개선


## 배운점
 주제 선정부터 데이터 수집, 데이터 분석까지 일련의 과정을 거치면서 각 과정에 대한 내용을 몸소 깨닫게 됨. 특히 계획 과정에서 분석에대한 기법을 정할때 신중히 고민하여 정했다고 생각했지만 실제분석에 적용했을 때 의도했던 결과와 달랐음. 데이터 활용을 최대로 하기 위한 분석 새로운 분석 방법에 대해 검색, 특강, 교수님 자문 등을 참고하였으며 팀원들과 다양한 변수 추출 기법에 대해서 공부하게 되었음.  
 머신러닝에 sklearn, logistic regression의 라이브러리를 직접 사용함으로써 머신러닝의 여러 기법과 친해짐.
 한글 폰트 깨짐을 해결하는 과정에서 굉장히 많은 시간을 할애하였는데 덕분에 한글 폰트 깨짐 현상에 대해서 지식을 습득하게 됨
