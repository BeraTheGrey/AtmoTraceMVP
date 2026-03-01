<p align="center">
  <h1 align="center">🌍 AtmoTrace</h1>
  <p align="center">
    <b>Kirletici Kaynak Tespit Platformu</b><br>
    Hava kirliliğinin <i>nereden geldiğini</i> tespit eden yapay zekâ destekli karar destek sistemi
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-1.51+-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Gemini_AI-2.5_Flash-4285F4?style=flat&logo=google&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="sklearn">
    <img src="https://img.shields.io/badge/Folium-Maps-77B829?style=flat&logo=leaflet&logoColor=white" alt="Folium">
  </p>
  <p align="center">
    🏆 <b>İklim için Dijital Dönüşüm Ideathon'u 2026</b>
  </p>
</p>

---

## 📌 Problem

Geleneksel hava kalitesi izleme sistemleri yalnızca **"Hava şu an ne kadar kirli?"** sorusuna yanıt verir. Kirletici kaynağın tespiti ise şikayetlere, manuel saha denetimlerine ve uzman tahminlerine dayanır — bu süreç **günler** sürebilir.

## 💡 Çözüm

AtmoTrace, T.C. ÇŞİDB resmi istasyon verilerini meteorolojik rüzgâr vektörleriyle sentezleyerek tersine mühendislikle asıl kritik soruyu yanıtlar:

> **"Bu kirlilik tam olarak nereden geliyor?"**

Çoklu istasyon tersine yörünge kesişim analizi ile muhtemel emisyon kaynağını harita üzerinde **otomatik** olarak işaretler.

---

## 🚀 Özellikler

### 🗺️ Uzamsal Analiz
- **Çoklu İstasyon Tersine Yörünge Analizi** — Rüzgâr vektörlerini geriye izleyerek kaynak tespiti
- **Çoklu Kaynak Tespiti (Multi-Peak)** — Birden fazla yoğunluk tepesi ile eş zamanlı kaynak belirleme
- **IDW Enterpolasyon + Gaussian Yumuşatma** — ~330m çözünürlüklü uzamsal yoğunluk haritası
- **Isı Haritası (Heatmap)** — Yörünge kesişim yoğunluğunun görsel temsili
- **Bilinen Emisyon Kaynakları** — Aliağa rafinerileri, OSB'ler, limanlar ile otomatik doğrulama
- **Google Maps Canlı Trafik Katmanı** — Anlık trafik yoğunluğu görselleştirmesi

### 🤖 Yapay Zekâ & Makine Öğrenmesi (7 Modül)

| Modül | Yöntem | Çıktı |
|-------|--------|-------|
| **Kirlilik Tahmini** | Holt Doğrusal Üstel Düzleştirme | 24 saatlik PM10/NO₂ öngörüsü |
| **Kaynak Parmak İzi** | Kural Tabanlı Karar Ağacı | SO₂/NO₂ ve PM oranlarından kaynak sınıflandırma |
| **Anomali Tespiti** | Isolation Forest | Olağandışı kirlilik örüntüsü gösteren istasyonlar |
| **İstasyon Kümeleme** | K-Means (StandardScaler + k=3) | Benzer profildeki istasyon grupları |
| **Sağlık Risk Skoru** | WHO Limit Ağırlıklı Bileşik Skor | 0-100 arası risk endeksi + öneri sistemi |
| **Korelasyon Analizi** | Pearson Korelasyonu + AI Yorum | Ortak kaynak tespiti için kirletici ilişki matrisi |
| **Üretken AI** | Google Gemini 2.5 Flash (LLM) | Yönetici raporu + sohbet asistanı |

### 📊 İstatistik & Görselleştirme
- **HKİ (Hava Kalitesi İndeksi)** — Türk standartlarına uygun renk skalası
- **Rush Hour NO₂ Analizi** — Trafik zirve saatleri korelasyonu
- **Rüzgâr Gülü** — Yön ve hız frekans dağılımı
- **Kirletici Zaman Serisi** — Normalize karşılaştırmalı çizgi grafik
- **Zaman Animasyonu** — Saatlik veri üzerinde oynatma

### 💬 Gemini AI Chatbot
Anlık analiz verileriyle beslenen soru-cevap asistanı. Örnek:
- *"İzmir'de en kirli bölge neresi?"*
- *"Rüzgâr yönü kaynağı etkiliyor mu?"*
- *"Sağlık riski hakkında bilgi ver"*

---

## 🏗️ Mimari

```
AtmoTraceMVP/
├── app.py                  # Streamlit dashboard (1,723 satır)
├── ai_engine.py            # 7 AI/ML modülü (619 satır)
├── analytics.py            # Tersine yörünge & kaynak tespiti (325 satır)
├── data_engine.py          # Veri yükleme katmanı (103 satır)
├── csb_veri_indirme.py     # CSB + Open-Meteo veri çekme (422 satır)
├── requirements.txt        # Bağımlılıklar
├── .streamlit/
│   └── secrets.toml        # API anahtarları (gitignore)
└── izmir_hava_kalitesi_*.csv  # Saatlik kirlilik verileri
```

### Veri Akışı

```
CSB API (Kirlilik) ─┐
                     ├──→ CSV ──→ data_engine ──→ analytics ──→ app.py (Dashboard)
Open-Meteo (Rüzgâr) ┘                              ↑
                                              ai_engine (7 AI modül)
                                                    ↑
                                              Gemini 2.5 Flash
```

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.10+
- Google Gemini API anahtarı ([ücretsiz al](https://aistudio.google.com/apikey))

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/BeraTheGrey/AtmoTraceMVP.git
cd AtmoTraceMVP

# 2. Sanal ortam oluştur
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS/Linux

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Gemini API anahtarını ayarla
mkdir .streamlit
echo 'GEMINI_API_KEY = "YOUR_API_KEY"' > .streamlit/secrets.toml

# 5. İlk veri çekimi (opsiyonel — uygulama içinden de yapılabilir)
python csb_veri_indirme.py

# 6. Uygulamayı başlat
streamlit run app.py
```

---

## 🌐 Canlı Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atmotracemvp.streamlit.app)

---

## 📐 Metodoloji

1. **Veri Toplama** — 25 istasyondan saatlik PM10, PM2.5, SO₂, NO₂, CO, O₃ + istasyon bazlı rüzgâr vektörleri
2. **Ağırlıklı Kirlilik Skoru (WPS)** — 6 kirletici parametrenin toksisite ağırlıklı bileşik skoru
3. **Tersine Yörünge (Back-Trajectory)** — Haversine formülü ile küresel trigonometrik geri izleme
4. **Uzamsal Yoğunluk** — 0.003° (~330m) grid üzerinde kümülatif yoğunluk + Gaussian bulanıklaştırma
5. **Çoklu Kaynak Tespiti** — `scipy.ndimage.label` ile bağımsız yoğunluk bölgeleri
6. **Doğrulama** — Bilinen emisyon kaynakları ile mesafe eşleştirmesi (5km = yüksek uyum)

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji | Görev |
|--------|-----------|-------|
| **Ön Yüz** | Streamlit + Folium | İnteraktif harita ve dashboard |
| **Veri & API** | CSB + Open-Meteo | Anlık kirlilik ve rüzgâr verisi |
| **Analitik** | NumPy, SciPy, Pandas | Uzamsal hesaplama, Gaussian model |
| **AI/ML** | scikit-learn + Gemini 2.5 Flash | K-Means, Isolation Forest, LLM |
| **Görselleştirme** | Plotly + Folium | Zaman serisi, rüzgâr gülü, ısı haritası |
| **Matematik** | Küresel Trigonometri | Haversine tabanlı yörünge vektörleri |

---

## 👥 Takım

**Ahmet Bera Onar** — Geliştirici
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/BeraTheGrey)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmetberaonar/)

**Fatıma Yaylı**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fatima-yayli/)

**Tuba Köten**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/tubak%C3%B6ten742)

**Atakan Tatar**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/atakan-tatar-600284297/)

**Abdullah Önder Aksu**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullah-%C3%B6nder-aksu/)

---

## 📄 Lisans

Bu proje **İklim için Dijital Dönüşüm Ideathon'u 2026** kapsamında geliştirilmiştir.

---

<p align="center">
  <b>AtmoTrace</b> — Kirliliğin kaynağını bul, iklimi koru. 🌱
</p>
