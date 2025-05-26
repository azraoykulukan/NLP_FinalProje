import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK veri setleri
nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))

#  Veriler okunuyor
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Etiketleri sayıya çevir (eğer text ise)
label_map = {'positive': 1, 'negative': 0}
if train_df["sentiment"].dtype == object:
    train_df["sentiment"] = train_df["sentiment"].map(label_map)
    test_df["sentiment"] = test_df["sentiment"].map(label_map)

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Temizlenmiş yorumları oluştur
train_df["cleaned_review"] = train_df["review"].apply(clean_text)
test_df["cleaned_review"] = test_df["review"].apply(clean_text)

print(f"✅ Train: {len(train_df)} reviews, Test: {len(test_df)} reviews")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df["cleaned_review"])
X_test = vectorizer.transform(test_df["cleaned_review"])

# Model eğitimi
model = LogisticRegression(max_iter=500)
model.fit(X_train, train_df["sentiment"])

# Değerlendirme
y_pred = model.predict(X_test)
print("\n📊 Accuracy:", accuracy_score(test_df["sentiment"], y_pred))
print(classification_report(test_df["sentiment"], y_pred))

# Confusion Matrix
cm = confusion_matrix(test_df["sentiment"], y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("🔢 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Sınıf Dengesini Görüntüle
plt.figure(figsize=(6,4))
sns.countplot(x=train_df["sentiment"], palette="viridis")
plt.title("📊 Eğitim Verisinde Duygu Dağılımı")
plt.xlabel("Sentiment (0=Negative, 1=Positive)")
plt.ylabel("Yorum Sayısı")
plt.tight_layout()
plt.show()

# Canlı yorum testi
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    print(f"📝 Review: {text}\n📊 Sentiment: {'Positive' if pred == 1 else 'Negative'}\n")

while True:
    user_input = input("💬 Enter a movie review (or 'q' to quit): ")
    if user_input.lower() == "q":
        break
    predict_sentiment(user_input)
