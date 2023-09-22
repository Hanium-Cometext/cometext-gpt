# cometext-gpt
GPT 기반 문장맥락 이해형 유사 도서 추천 및 검색 알고리즘

## 이후 고려할 것
- 입력에 원하는 정보 말고 불필요한 정보가 많이 들어있을 때 (책을 추천해줘. 라고 할 때, 책을 추천해줘 까지도 유사한 문장을 비교하면 어떡하지?)
- 유사도가 너무 작을 때.. 임계값을 정해야 (top 5 내에 0.3이 있으면?)
  
      ([('콘텐츠 제작 네트워크 구축 지원방안 연구', 0.7349797487258911),
      ('애니메이션 지원정책의 효율성 제고 방안 연구', 0.7051012516021729),
      ('공존, 지구촌을 살리는 위대한 나눔, 적정기술', 0.7011985778808594),
      ('일본 애니메이션 산업 현황과 한 · 일 공동제작 모델 연구', 0.7010248899459839),
      ('한국영화수익성분석', 0.6903142333030701)])


- 예외처리: 도서 원문이 여러 말뭉치로 나뉘는데, 각 말뭉치의 유사도가 다 높아서 동일한 도서가 추천됐을 때
