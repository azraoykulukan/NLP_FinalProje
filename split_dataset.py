import pandas as pd
from sklearn.model_selection import train_test_split

# IMDb Dataset dosyasını yükle
df = pd.read_csv("IMDB Dataset.csv")

# Kolon adlarını kontrol et (gerekirse düzelt)
df = df.rename(columns={"review": "review", "sentiment": "sentiment"})

# Eksik veri varsa at
df = df.dropna(subset=["review", "sentiment"])

# Eğitim ve test olarak ayır (80-20 oranında)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])

# CSV olarak kaydet
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✅ IMDb verisi başarıyla train/test olarak ayrıldı.")
