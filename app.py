import pandas as pd
import numpy as np
import fasttext
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import platform

# 1. 한글 폰트 설정 (환경에 맞게 자동 적용)
os_name = platform.system()
if os_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif os_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("데이터와 모델을 불러오는 중입니다...")
    
    # 2. 전처리된 뉴스 데이터 로드
    try:
        df = pd.read_csv('news_data_processed.csv')
        # 결측치 제거
        df = df.dropna(subset=['query', 'processed']) 
    except FileNotFoundError:
        print("에러: 'news_data_processed.csv' 파일을 찾을 수 없습니다.")
        return

    # 3. FastText 모델 로드 (이전에 만든 미니 모델 또는 다운로드 받은 모델)
    try:
        # 모델 파일 이름은 현재 환경에 맞춰 수정해주세요 ('micro_model.bin' 등)
        ft_model = fasttext.load_model('micro_model.bin') 
    except:
        print("에러: FastText 모델 파일(.bin)을 찾을 수 없습니다.")
        return

    # 4. 문서 벡터화 함수 정의 (단어 벡터의 평균)
    def get_document_vector(text, model):
        words = str(text).split()
        if not words: # 텍스트가 비어있을 경우 0으로 채워진 벡터 반환
            return np.zeros(model.get_dimension())
        
        # 문장 내 각 단어의 벡터를 구한 뒤, 세로축(axis=0) 기준으로 평균 계산
        word_vectors = [model.get_word_vector(w) for w in words]
        return np.mean(word_vectors, axis=0)

    print("단어 벡터를 평균 내어 문서 벡터를 생성하고 있습니다...")
    # 5. 모든 뉴스 기사를 벡터로 변환
    df['doc_vector'] = df['processed'].apply(lambda x: get_document_vector(x, ft_model))
    
    # 머신러닝 모델에 넣기 위해 리스트 형태를 Numpy 2차원 배열로 변환
    X = np.stack(df['doc_vector'].values)
    y = df['query'].values

    # 6. 학습용 / 테스트용 데이터 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("로지스틱 회귀 모델을 학습합니다...")
    # 7. 로지스틱 회귀 모델 학습
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # 8. 예측 및 성능 평가 (Classification Report)
    y_pred = clf.predict(X_test)
    print("\n=== 📊 분류 성능 평가 보고서 ===")
    print(classification_report(y_test, y_pred))

    # 9. 혼동 행렬(Confusion Matrix) 시각화
    cm = confusion_matrix(y_test, y_pred)
    classes = clf.classes_ # 예측된 카테고리 이름들

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('뉴스 카테고리 분류 혼동 행렬 (Confusion Matrix)', fontsize=16, pad=20)
    plt.ylabel('실제 카테고리 (True Label)', fontsize=12)
    plt.xlabel('예측 카테고리 (Predicted Label)', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
