import streamlit as st
import pandas as pd
import numpy as np
import fasttext
from sklearn.linear_model import LogisticRegression

# 페이지 기본 설정
st.set_page_config(page_title="AI 뉴스 분류기", layout="wide")

# 1. AI 모델 세팅 및 학습 (앱 구동 시 최초 1회만 실행되도록 캐싱)
@st.cache_resource
def load_and_train_model():
    # 데이터와 미니 모델 로드
    df = pd.read_csv('news_data_processed.csv').dropna(subset=['query', 'processed'])
    ft_model = fasttext.load_model('micro_model.bin')

    # 단어 벡터 평균으로 문서 벡터를 만드는 함수
    def get_document_vector(text, model):
        words = str(text).split()
        if not words:
            return np.zeros(model.get_dimension())
        word_vectors = [model.get_word_vector(w) for w in words]
        return np.mean(word_vectors, axis=0)

    # 전체 데이터 벡터화 및 로지스틱 회귀 모델 학습
    X = np.stack(df['processed'].apply(lambda x: get_document_vector(x, ft_model)).values)
    y = df['query'].values

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    return ft_model, clf, get_document_vector

# 백그라운드에서 조용히 모델을 준비합니다.
ft_model, clf, get_document_vector = load_and_train_model()

# 2. 메인 웹 화면 UI
st.title("📰 AI 실시간 뉴스 카테고리 예측기")
st.markdown("뉴스 기사의 본문이나 제목을 입력하면, AI가 어느 분야의 기사인지 즉시 분류합니다.")

# 사용자 텍스트 입력 창
user_input = st.text_area("분석할 텍스트를 입력하세요:", height=150, 
                          placeholder="여기에 뉴스 내용을 복사해서 붙여넣어 보세요.")

# 3. 예측 실행 및 결과 출력
if st.button("카테고리 예측하기", type="primary"):
    if user_input.strip() == "":
        st.warning("분석할 텍스트를 먼저 입력해 주세요!")
    else:
        with st.spinner("AI가 텍스트 문맥을 분석 중입니다..."):
            # 입력받은 텍스트를 숫자로 변환(벡터화)
            doc_vector = get_document_vector(user_input, ft_model)
            
            # 예측 및 확률 계산 (모델에는 항상 2차원 배열로 넣어야 해서 []로 감싸줍니다)
            prediction = clf.predict([doc_vector])[0]
            probabilities = clf.predict_proba([doc_vector])[0]
            
            # 메인 결과 출력
            st.success(f"이 기사는 **'{prediction}'** 카테고리로 분류되었습니다! 🎉")
            
            # 디테일한 확률 표 출력
            st.subheader("📊 AI의 카테고리별 예측 확률")
            prob_df = pd.DataFrame({
                '카테고리': clf.classes_,
                '확률(%)': np.round(probabilities * 100, 2)
            }).sort_values(by='확률(%)', ascending=False)
            
            # 표 형태로 깔끔하게 표시
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
