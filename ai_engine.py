"""
AtmoTrace MVP — Yapay Zekâ Motoru (ai_engine.py)

Dört AI/ML modülü:
  1. forecast_pollution()          — 24 saatlik kirlilik tahmini (Holt üstel düzleştirme)
  2. classify_source()             — Kaynak parmak izi sınıflandırma (kural tabanlı karar ağacı)
  3. detect_anomalies()            — Isolation Forest anomali tespiti
  4. generate_executive_report()   — Şablon tabanlı yönetici raporu (fallback)
  4b. generate_executive_report_gemini() — Google Gemini ile üretken AI yönetici raporu
"""

import numpy as np
import pandas as pd
from datetime import timedelta


# =========================================================================== #
#  1. Zaman Serisi Tahmini — 24 Saatlik Kirlilik Öngörüsü
#     Yöntem: Holt Doğrusal Üstel Düzleştirme (level + trend)
# =========================================================================== #
def forecast_pollution(all_df: pd.DataFrame,
                       pollutant: str = "pm10",
                       horizon: int = 24) -> pd.DataFrame:
    """
    Basit Holt üstel düzleştirme ile ileriye dönük tahmin üretir.

    Args:
        all_df:     Tüm saatlik veriyi içeren DataFrame
        pollutant:  Tahmin edilecek kirletici sütun adı
        horizon:    Kaç saat ileri tahmin (varsayılan 24)

    Returns:
        DataFrame: timestamp, tahmin, alt, üst sütunları
    """
    hourly = (
        all_df.groupby("timestamp")[pollutant]
        .mean()
        .dropna()
        .sort_index()
    )

    if len(hourly) < 12:
        return pd.DataFrame(columns=["timestamp", "tahmin", "alt", "üst"])

    values = hourly.values

    # --- Holt doğrusal üstel düzleştirme ---
    alpha = 0.3   # seviye yumuşatma katsayısı
    beta = 0.1    # trend yumuşatma katsayısı

    level = values[0]
    trend = np.mean(np.diff(values[:min(6, len(values))]))

    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    # --- İleri projeksiyon ---
    last_time = hourly.index[-1]
    std = np.std(values[-24:]) if len(values) >= 24 else np.std(values)

    forecasts = []
    for h in range(1, horizon + 1):
        fc = max(0, level + trend * h)            # kirlilik negatif olamaz
        uncertainty = std * np.sqrt(h) * 0.5       # belirsizlik zamanla büyür

        forecasts.append({
            "timestamp": pd.Timestamp(last_time) + timedelta(hours=h),
            "tahmin": round(fc, 2),
            "alt": round(max(0, fc - uncertainty), 2),
            "üst": round(fc + uncertainty, 2),
        })

    return pd.DataFrame(forecasts)


# =========================================================================== #
#  2. Kaynak Parmak İzi Sınıflandırma
#     Yöntem: Kirletici oranlarına dayalı kural tabanlı karar ağacı
# =========================================================================== #
SOURCE_PROFILES = {
    "Sanayi (Ağır)": {
        "desc": "Yüksek SO₂ ve PM — rafineri, demir-çelik, enerji santralı",
        "icon": "🏭",
        "color": "#D32F2F",
    },
    "Sanayi (Hafif)": {
        "desc": "Orta düzey PM ve VOC — OSB, imalat tesisleri",
        "icon": "🏗️",
        "color": "#F57C00",
    },
    "Trafik": {
        "desc": "Yüksek NO₂ ve CO — karayolu, liman trafiği",
        "icon": "🚗",
        "color": "#5D4037",
    },
    "Doğal / Arka Plan": {
        "desc": "Düşük emisyon — toz, deniz tuzu, biyojenik",
        "icon": "🌿",
        "color": "#388E3C",
    },
    "Karma": {
        "desc": "Birden fazla kaynak türünün karışımı",
        "icon": "🔄",
        "color": "#7B1FA2",
    },
}


def classify_source(df_snapshot: pd.DataFrame) -> dict:
    """
    İstasyon kirlilik profilinden kaynak türünü sınıflandırır.

    Returns:
        dict: label, confidence, desc, icon, color, ratios
    """
    pm10 = df_snapshot["pm10"].mean() if "pm10" in df_snapshot else 0
    pm25 = df_snapshot["pm25"].mean() if "pm25" in df_snapshot else 0
    so2 = df_snapshot["so2"].mean() if "so2" in df_snapshot else 0
    no2 = df_snapshot["no2"].mean() if "no2" in df_snapshot else 0
    co = df_snapshot["co"].mean() if "co" in df_snapshot else 0

    # NaN kontrolü
    pm10 = pm10 if pd.notna(pm10) else 0
    pm25 = pm25 if pd.notna(pm25) else 0
    so2 = so2 if pd.notna(so2) else 0
    no2 = no2 if pd.notna(no2) else 0
    co = co if pd.notna(co) else 0

    # Kirletici oranları
    so2_no2_ratio = so2 / max(no2, 0.1)
    pm25_pm10_ratio = pm25 / max(pm10, 0.1)

    ratios = {
        "SO₂/NO₂": round(so2_no2_ratio, 2),
        "PM2.5/PM10": round(pm25_pm10_ratio, 2),
    }

    # --- Kural tabanlı karar ağacı ---
    if so2_no2_ratio > 2.0 and pm10 > 60:
        label = "Sanayi (Ağır)"
        confidence = min(0.95, 0.60 + so2_no2_ratio * 0.08)
    elif so2_no2_ratio > 1.0 and pm10 > 40:
        label = "Sanayi (Hafif)"
        confidence = min(0.90, 0.50 + so2_no2_ratio * 0.10)
    elif no2 > 30 and co > 500 and so2_no2_ratio < 0.8:
        label = "Trafik"
        confidence = min(0.88, 0.50 + (no2 / 100) * 0.30)
    elif pm10 < 30 and so2 < 10 and no2 < 15:
        label = "Doğal / Arka Plan"
        confidence = 0.70
    else:
        label = "Karma"
        confidence = 0.55

    profile = SOURCE_PROFILES[label]

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "desc": profile["desc"],
        "icon": profile["icon"],
        "color": profile["color"],
        "ratios": ratios,
    }


# =========================================================================== #
#  3. Anomali Tespiti — Isolation Forest
# =========================================================================== #
def detect_anomalies(df_snapshot: pd.DataFrame,
                     contamination: float = 0.1) -> pd.DataFrame:
    """
    Isolation Forest ile anlık istasyon verilerinde anomali tespiti.

    Returns:
        DataFrame: orijinal + 'anomaly' sütunu (-1 = anomali, 1 = normal)
                         + 'anomaly_score' sütunu
    """
    from sklearn.ensemble import IsolationForest

    features = ["pm10", "pm25", "so2", "no2"]
    available = [f for f in features if f in df_snapshot.columns]

    df = df_snapshot.copy()
    subset = df[available].copy()

    # Eksik verileri medyan ile doldur
    for col in available:
        median_val = subset[col].median()
        subset[col] = subset[col].fillna(median_val if pd.notna(median_val) else 0)

    valid_mask = subset.notna().all(axis=1)

    if valid_mask.sum() < 5:
        df["anomaly"] = 1
        df["anomaly_score"] = 0.0
        return df

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )

    X = subset.loc[valid_mask].values
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    df["anomaly"] = 1
    df.loc[valid_mask, "anomaly"] = predictions
    df["anomaly_score"] = 0.0
    df.loc[valid_mask, "anomaly_score"] = scores

    return df


# =========================================================================== #
#  4. Üretken AI Yönetici Raporu (Mock LLM)
# =========================================================================== #
def generate_executive_report(
    selected_hour,
    result: dict,
    df_scored: pd.DataFrame,
    nearest_source: dict | None,
    nearest_dist: float,
    fingerprint: dict,
    anomaly_count: int,
    forecast_df: pd.DataFrame | None = None,
) -> str:
    """
    Mock LLM — şablon tabanlı akıllı yönetici raporu üretir.
    Gerçek analiz verilerini kullanarak doğal dilde metin oluşturur.

    Returns:
        str: Markdown formatında yönetici raporu
    """
    hour_str = selected_hour.strftime("%d.%m.%Y %H:%M")
    n_contributing = result["n_contributing"]
    top_station = df_scored.sort_values("pollution_score", ascending=False).iloc[0]

    # Kirlilik seviyesi değerlendirmesi
    avg_pm10 = df_scored["pm10"].mean()
    avg_pm10 = avg_pm10 if pd.notna(avg_pm10) else 0

    if avg_pm10 > 260:
        severity, severity_desc = "KRİTİK", "sağlıksız seviyede"
    elif avg_pm10 > 100:
        severity, severity_desc = "YÜKSEK", "hassas gruplar için riskli seviyede"
    elif avg_pm10 > 50:
        severity, severity_desc = "ORTA", "orta düzeyde"
    else:
        severity, severity_desc = "DÜŞÜK", "kabul edilebilir seviyede"

    # --- Rapor bölümleri ---
    sections = []

    sections.append("## 🤖 AtmoTrace AI — Yönetici Özet Raporu\n")
    sections.append(
        f"**Analiz Zamanı:** {hour_str}  \n"
        f"**Genel Durum:** {severity} — Hava kalitesi {severity_desc}.\n"
    )

    # §1 - Kaynak tespiti
    sections.append("### 📍 Kaynak Tespiti")
    sections.append(
        f"Yapılan çoklu istasyon tersine yörünge analizi sonucunda, "
        f"**{n_contributing} istasyonun** verileri değerlendirilerek "
        f"muhtemel kirletici kaynağın **{result['source_lat']:.4f}°K, "
        f"{result['source_lon']:.4f}°D** koordinatlarında olduğu tespit edilmiştir. "
        f"Yoğunluk tepe değeri **{result['peak_density']:.4f}** olarak ölçülmüştür.\n"
    )

    # §2 - Parmak izi
    sections.append("### 🔬 Kaynak Parmak İzi")
    sections.append(
        f"Kirletici oranları analiz edildiğinde, baskın kaynak türünün "
        f"**{fingerprint['icon']} {fingerprint['label']}** olduğu görülmektedir "
        f"(güven: %{fingerprint['confidence']*100:.0f}). "
        f"{fingerprint['desc']}. "
        f"SO₂/NO₂ oranı **{fingerprint['ratios']['SO₂/NO₂']}**, "
        f"PM2.5/PM10 oranı **{fingerprint['ratios']['PM2.5/PM10']}** "
        f"olarak hesaplanmıştır.\n"
    )

    # §3 - Doğrulama
    if nearest_source:
        sections.append("### 🏭 Doğrulama")
        if nearest_dist < 5:
            sections.append(
                f"Tespit edilen emisyon noktası, **{nearest_source['name']}** "
                f"({nearest_source['type']}) tesisine yalnızca **{nearest_dist:.1f} km** "
                f"mesafededir. Bu **yüksek uyum**, algoritmanın güvenilirliğini "
                f"desteklemektedir.\n"
            )
        elif nearest_dist < 15:
            sections.append(
                f"En yakın bilinen kaynak **{nearest_source['name']}** "
                f"({nearest_source['type']}) **{nearest_dist:.1f} km** mesafededir. "
                f"Olası bir bağlantı bulunmakla birlikte, ara bölgelerdeki ek kaynaklar "
                f"da etkili olabilir.\n"
            )
        else:
            sections.append(
                f"Bilinen en yakın kaynak **{nearest_source['name']}** "
                f"**{nearest_dist:.1f} km** uzaklıktadır. Bu durum, henüz envantere "
                f"alınmamış bir emisyon kaynağına veya geçici bir olaya işaret edebilir.\n"
            )

    # §4 - Anomali
    if anomaly_count > 0:
        sections.append("### ⚠️ Anomali Uyarısı")
        sections.append(
            f"Isolation Forest algoritması, bu saatte **{anomaly_count} istasyonda** "
            f"olağandışı kirlilik örüntüsü tespit etmiştir. Bu istasyonlar, normal "
            f"dağılımdan sapan ani yükseliş veya atipik kirletici kombinasyonları "
            f"sergilemektedir. İlgili istasyonlarda sensör kontrolü veya bölgesel "
            f"olay araştırması önerilmektedir.\n"
        )

    # §5 - Tahmin
    if forecast_df is not None and not forecast_df.empty:
        fc_max = forecast_df["tahmin"].max()
        fc_min = forecast_df["tahmin"].min()
        fc_trend = (
            "yükseliş" if forecast_df["tahmin"].iloc[-1] > forecast_df["tahmin"].iloc[0]
            else "düşüş"
        )
        sections.append("### 📈 24 Saatlik Tahmin")
        sections.append(
            f"Üstel düzleştirme modeline göre, önümüzdeki 24 saatte kirlilik "
            f"seviyesinin **{fc_min:.0f}–{fc_max:.0f} µg/m³** aralığında seyredeceği "
            f"öngörülmektedir. Genel trend **{fc_trend}** yönündedir.\n"
        )

    # §6 - Öneri
    sections.append("### 💡 Eylem Önerisi")
    if severity in ("KRİTİK", "YÜKSEK"):
        sections.append(
            f"- Tespit edilen kaynak bölgesine **acil saha denetimi** düzenlenmesi "
            f"önerilir.\n"
            f"- Rüzgâr yönü koridorundaki yerleşim alanlarına **halk sağlığı uyarısı** "
            f"yayınlanması değerlendirilmelidir.\n"
            f"- En yüksek kirlilik skoru **{top_station['station_name']}** istasyonunda "
            f"({top_station['pollution_score']:.3f}) kaydedilmiştir."
        )
    else:
        sections.append(
            f"- Mevcut kirlilik seviyeleri {severity_desc}. Rutin izlemeye devam "
            f"edilmesi yeterlidir.\n"
            f"- En yüksek kirlilik skoru **{top_station['station_name']}** istasyonunda "
            f"({top_station['pollution_score']:.3f}) kaydedilmiştir."
        )

    sections.append(
        "\n---\n*Bu rapor AtmoTrace AI modülü tarafından otomatik üretilmiştir.*"
    )

    return "\n\n".join(sections)


# =========================================================================== #
#  4b. Üretken AI Yönetici Raporu — Google Gemini
# =========================================================================== #
def generate_executive_report_gemini(
    selected_hour,
    result: dict,
    df_scored: pd.DataFrame,
    nearest_source: dict | None,
    nearest_dist: float,
    fingerprint: dict,
    anomaly_count: int,
    forecast_df: pd.DataFrame | None = None,
    api_key: str = "",
) -> str:
    """
    Google Gemini Flash ile gerçek üretken AI yönetici raporu üretir.
    Tüm analiz verilerini yapılandırılmış prompt olarak gönderir,
    Türkçe Markdown formatında profesyonel bir özet döndürür.

    Başarısız olursa boş string döner (app.py fallback'e geçer).
    """
    import google.generativeai as genai

    if not api_key:
        return ""

    genai.configure(api_key=api_key)

    # --- Veri özetini hazırla ---
    hour_str = selected_hour.strftime("%d.%m.%Y %H:%M")
    n_contributing = result["n_contributing"]
    top_station = df_scored.sort_values("pollution_score", ascending=False).iloc[0]

    avg_pm10 = df_scored["pm10"].mean()
    avg_pm10 = avg_pm10 if pd.notna(avg_pm10) else 0
    avg_pm25 = df_scored["pm25"].mean() if "pm25" in df_scored else 0
    avg_so2 = df_scored["so2"].mean() if "so2" in df_scored else 0
    avg_no2 = df_scored["no2"].mean() if "no2" in df_scored else 0

    forecast_info = ""
    if forecast_df is not None and not forecast_df.empty:
        fc_max = forecast_df["tahmin"].max()
        fc_min = forecast_df["tahmin"].min()
        fc_trend = (
            "yukselis" if forecast_df["tahmin"].iloc[-1] > forecast_df["tahmin"].iloc[0]
            else "dusus"
        )
        forecast_info = (
            f"24 saatlik tahmin: {fc_min:.0f}-{fc_max:.0f} ug/m3 araliginda, "
            f"trend {fc_trend} yonunde."
        )

    nearest_info = ""
    if nearest_source:
        nearest_info = (
            f"En yakin bilinen kaynak: {nearest_source['name']} "
            f"({nearest_source['type']}), {nearest_dist:.1f} km mesafede."
        )

    prompt = f"""Sen AtmoTrace hava kalitesi analiz platformunun yapay zeka raporlama modulusun.
Asagidaki analiz verilerini kullanarak Turkce, profesyonel bir yonetici ozet raporu yaz.
Rapor Markdown formatinda olmali, basliklar ve emoji icermeli.
Raporun sonunda aksiyon onerileri sun.

## ANALIZ VERILERI:

- Analiz zamani: {hour_str}
- Sehir: Izmir, Turkiye
- Degerlendirien istasyon sayisi: {n_contributing}
- Tespit edilen kaynak koordinatlari: {result['source_lat']:.4f}K, {result['source_lon']:.4f}D
- Yogunluk tepe degeri: {result['peak_density']:.4f}

### Kirlilik Ortalamalari:
- PM10: {avg_pm10:.1f} ug/m3
- PM2.5: {avg_pm25:.1f} ug/m3
- SO2: {avg_so2:.1f} ug/m3
- NO2: {avg_no2:.1f} ug/m3

### Kaynak Parmak Izi:
- Baskin kaynak turu: {fingerprint['label']}
- Guven orani: %{fingerprint['confidence']*100:.0f}
- Aciklama: {fingerprint['desc']}
- SO2/NO2 orani: {fingerprint['ratios']['SO₂/NO₂']}
- PM2.5/PM10 orani: {fingerprint['ratios']['PM2.5/PM10']}

### Anomali:
- Anomali tespit edilen istasyon sayisi: {anomaly_count}

### Dogrulama:
{nearest_info if nearest_info else "Bilinen yakin kaynak bulunamadi."}

### Tahmin:
{forecast_info if forecast_info else "Tahmin verisi mevcut degil."}

### En Kirli Istasyon:
- Ad: {top_station['station_name']}
- Kirlilik skoru: {top_station['pollution_score']:.3f}

## RAPOR FORMATI:
- Baslik: "## 🤖 AtmoTrace AI — Yonetici Ozet Raporu"
- Bolumlere ayir: Genel Durum, Kaynak Tespiti, Parmak Izi Analizi, Dogrulama, Anomali, Tahmin, Eylem Onerileri
- Turkce yaz, profesyonel ve bilimsel uslup kullan
- Raporu "Bu rapor AtmoTrace AI (Gemini) tarafindan otomatik uretilmistir." notu ile bitir
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return ""
