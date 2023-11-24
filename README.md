# cometext-gpt
GPT 기반 문장맥락 이해형 유사 도서 추천 및 검색 알고리즘

<br>

# <div align="center">cometext-gpt</div>
<div align="center"><b>GPT 기반 문장맥락 이해형 유사 도서 추천 및 검색 알고리즘</b></div>
<br>
<div align="center">본 프로젝트의 유사 도서 추천 및 검색 알고리즘을 구현한 페이지입니다.
    키워드 기반이 아닌, 문장 자체를 이해하여 도서를 추천합니다. 이를 위해, KoGPT 언어모델을 기반으로 Semantic Search를 수행했습니다. 알고리즘은 구조는 KoSenteceBert를 참고하여 개발했습니다.
    실제로 유사 도서 추천을 경험해보고 싶다면, <a href="https://github.com/Hanium-Cometext/cometext-gpt/blob/main/main.ipynb">main.ipynb</a>를 실행하면 됩니다.</div>
    
    원래의 계획은, chatGPT API로 도서 DB를 파인튜닝하는 것을 계획했지만, 주최측에서 chatGPT API를 지원할 수 없다고 전달 받았습니다.
    그래서 KoGPT라는 오픈소스를 이용하기로 했고, 기존 모델이 없어서, KoGPT를 이용한 새로운 네트워크 및 알고리즘을 설계/개발하게 되었습니다.
    
<br>

<div align="center"><b>직접 설계한 네트워크 구조</b></div>
<div align="center"><img width="385" alt="image" src="https://github.com/Hanium-Cometext/cometext-gpt/assets/77441026/aebdf2a0-c193-4e84-a053-cfc72e5417d9"></div>

<br>
<br>


## <div align="center">도서 추천 알고리즘</div>
<b>GPT를 기반으로 한 도서 추천 알고리즘 요약</b>

<b>1. GPT Model</b>
    <ul>도서 추천 알고리즘에서는 GPT 모델을 활용하여 입력 문장과 도서 원문 간의 유사도를 계산합니다. 사용된 GPT 모델은 SKT에서 배포한 KoGPT2입니다.</ul>
    <ul>GPT-2는 Open AI에서 개발한 언어 모델입니다. KoGPT2는 GPT-2의 부족한 한국어 성능을 위해 개발되었습니다. 즉, KoGPT2는 한국어 텍스트에 최적화된 사전 학습 모델입니다.
    KoGPT2의 인코더의 hidden_state를 받아 임베딩 벡터를 생성합니다. 각 도서에 대한 임베딩 벡터는 해당 도서의 의미와 특징을 담고 있습니다.</ul>
<b>2. Semantic Search</b>
    <ul>Semantic Search를 수행할 때, 문장 하나가 아니라, 말뭉치와 같이 여러 문장을 대상으로 진행합니다. 즉, 하나의 임베딩 벡터는 도서의 원문 전체에 대한 것이라고 봐도 무방합니다. (실제로 구현할 때는 도서의 원문이 매우 길기 때문에, 도서 당 여러 말뭉치로 나눴습니다.)</ul>
    <ul>Semantic Search 알고리즘을 짜기 위해, Sentecne Bert의 네트워크 구조를 참고했습니다. KoGPT의 hidden state에서 얻은 임베딩 벡터를 mean pooling을 수행합니다. 그 후, 도서 원문의 임베딩 벡터와 사용자 입력의 임베딩 벡터의 코사인 유사도를 계산합니다.</ul>
    <ul>이때, 임베딩 벡터는 Tensor이고, Pytorch의 코사인 유사도 연산을 이용하여 계산을 최적화했습니다. 코사인 유사도 값을 내림차순으로 정렬하여, K개의 추천 도서 목록을 저장합니다.</ul>
<b>3. Real Time</b>
<ul>[embedding vector caching]
    <dl>실시간 처리를 위해, 도서 원문에 대한 embedding vector를 미리 계산하고 저장합니다. 캐싱(Caching)을 통해 응답시간 단축 및 리소스 효율성을 높였습니다.</dl></ul>
<ul>[Pytorch Tensor]
    <dl>embedding vector를 Pytorch Tensor로 저장하여, GPU를 이용해 유사도 계산 과정을 과속화할 수 있습니다. 그리고 KoGPT는 Pytorch 기반의 모델로, 호환성이 높아 성능을 최적화합니다.</dl></ul>


<br>
<br>

## <div align="center">Inference with 3_prompot.ipynb</div>

<details>
<summary>Data Preparing</summary>
    AI HUB의 '도서자료 요약' 데이터셋을 이용했습니다. 위 데이터는 도서의 메타 정보, 원문, 요약 정보를 제공합니다. 본 프로젝트를 실현하기 위해서는, 도서의 제목과 원문이 필수이기 때문에 '도서자료 요약' 데이터셋을 선택했습니다. 위의 데이터셋은 지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있음을 밝힙니다.

</details>

### 1. 도서 데이터셋 원본(json)을 db에 저장
    - demo/1_dataloader.ipynb 실행
    - 결과: data/books.db 생성
1. 원본 데이터셋은 여러개의 Json 파일로 구성됐습니다. 하나의 Json 파일은 도서의 원문 말뭉치 하나, 요약, 메타 데이터로 구성되어 있고, passage_id를 통해 구별할 수 있습니다.
2. passage_id를 통해, Primary Key로 잡아서, DB를 구축합니다.
3. Json을 읽고, 데이터셋을 구축된 DB에 저장합니다.
<br>

### 2. 도서 원문에 대한 embedding vector를 생성하고, 텐서로 저장
    - demo/2_get_embedding.ipynb
    - 결과: data/embeddings.pt & data/passage_ids.pt 생성
1. KoGPT2 (skt/kogpt2-base-v2)에 대한 토크나이징을 통해 도서 원문을 토큰화하여 GPT 모델에 입력할 수 있는 형태로 변환합니다.
2. 토큰화된 도서 원문을 GPT 모델에 넣어서, 인코딩의 마지막 hidden state를 이용하여 도서 원문의 임베딩 벡터를 생성합니다.
3. 도서의 식별자인 passage_id와 임베딩 벡터를 Pytorch Tensor 형태로 저장합니다.

<br>

### 3. 도서 추천 시스템 (*Inference with 3_prompot.ipynb*)
    - demo/3_prompt.ipynb
    - input_text(사용자의 질문)와 k(도서 추천 개수)를 입력하면, 추천 도서의 결과를 얻을 수 있습니다.

1. 앞선 1과 2 과정을 통해, data 폴더 내 books.db, embeddings.pt, passage_ids.pt를 생성했습니다. 3_prompt는 실제로 Inference를 진행할 수 있는 코드입니다.
2. 사용자가 문장을 입력하면, 미리 계산된 도서 원문의 embedding vector와 유사도를 측정을 합니다. 이때, 사용자는 도서 추천 개수(k)를 지정할 수 있습니다.
3. 이때, 추천 도서의 메타 데이터를 추가로 알고 싶다면, 'get_db_by_id()'를 추가 정보를 얻을 수 있습니다.

<br>
<br>

## <div align="center">이후 고려할 것</div>
- 입력에 원하는 정보 말고 불필요한 정보가 많이 들어있을 때 (책을 추천해줘. 라고 할 때, 책을 추천해줘 까지도 유사한 문장을 비교하면 어떡하지?)
- 유사도가 너무 작을 때.. 임계값을 정해야 (top 5 내에 0.3이 있으면?)
  
      ([('콘텐츠 제작 네트워크 구축 지원방안 연구', 0.7349797487258911),
      ('애니메이션 지원정책의 효율성 제고 방안 연구', 0.7051012516021729),
      ('공존, 지구촌을 살리는 위대한 나눔, 적정기술', 0.7011985778808594),
      ('일본 애니메이션 산업 현황과 한 · 일 공동제작 모델 연구', 0.7010248899459839),
      ('한국영화수익성분석', 0.6903142333030701)])


- 예외처리: 도서 원문이 여러 말뭉치로 나뉘는데, 각 말뭉치의 유사도가 다 높아서 동일한 도서가 추천됐을 때
