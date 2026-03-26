import streamlit as st
import pandas as pd
import numpy as np
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="뉴스 카테고리 분류", layout="wide")

st.title("📊 FastText 뉴스 카테고리 분류 결과")
st.write("데이터를 불러오고 AI 모델을 학습하는 중입니다. 잠시만 기다려주세요...")

# 1. 데이터와 모델 불러오기
df = pd.read_csv('news_data_processed.csv').dropna(subset=['query', 'processed'])
ft_model = fasttext.load_model('micro_model.bin')

# 문서 벡터화 함수
def get_document_vector(text, model):
    words = str(text).split()
    if not words:
        return np.zeros(model.get_dimension())
    word_vectors = [model.get_word_vector(w) for w in words]
    return np.mean(word_vectors, axis=0)

# 2. 벡터화 및 학습
df['doc_vector'] = df['processed'].apply(lambda x: get_document_vector(x, ft_model))
X = np.stack(df['doc_vector'].values)
y = df['query'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.success("학습 완료! 결과를 확인하세요.")

col1, col2 = st.columns(2)

# 3. 결과 화면에 출력 (st.text 사용)
with col1:
    st.subheader("📝 분류 성능 평가 보고서")
    st.text(classification_report(y_test, y_pred))

# 4. 혼동 행렬 시각화 (st.pyplot 사용)
with col2:
    st.subheader("🎯 카테고리 분류 혼동 행렬")
    cm = confusion_matrix(y_test, y_pred)
    classes = clf.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 그래프에 혼동 행렬 그리기
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('실제 카테고리 (True Label)')
    plt.xlabel('예측 카테고리 (Predicted Label)')
    
    # 웹 화면에 그래프 띄우기
    st.pyplot(fig)
