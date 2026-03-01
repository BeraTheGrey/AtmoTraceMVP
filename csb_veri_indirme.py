"""
AtmoTrace MVP - CSB Hava Kalitesi + Open-Meteo Ruzgar Verisi Indirme Scripti

Veri kaynaklari:
  - Kirlilik: T.C. Cevre, Sehircilik ve Iklim Degisikligi Bakanligi (CSB)
  - Ruzgar  : Open-Meteo (ucretsiz, API key gerektirmez)

Kullanim:
    python csb_veri_indirme.py
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import json
import time
import sys
import io
import pandas as pd
from datetime import datetime, timedelta

# Windows konsol encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# =========================================================================== #
#  Sabitler
# =========================================================================== #
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BASE_URL = "https://sim.csb.gov.tr/STN/STN_Report"
CONFIG_URL = f"{BASE_URL}/StationDataDownloadNewDefaults"
PAGE_URL = f"{BASE_URL}/StationDataDownloadNew"
DATA_URL = f"{BASE_URL}/StationDataDownloadNewData"

IZMIR_CITY_ID = "4b6e3556-15bc-410b-99af-627aeb67f05f"

# Indirilecek parametreler
PARAMETERS = ["PM10", "PM25", "SO2", "NO2", "CO", "O3"]

# Tarih araligi: son 7 gun (bugun dahil — API mevcut saatleri dondurur)
END_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=6)

# Tek istekte max gun sayisi (API limiti icin parcala)
CHUNK_DAYS = 7


# =========================================================================== #
#  Yardimci Fonksiyonlar
# =========================================================================== #
def _make_session():
    """Retry mekanizmali requests session olusturur."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session_and_token():
    """CSB sitesinden session cookie ve CSRF token alir."""
    session = _make_session()
    page = session.get(PAGE_URL, timeout=60)
    page.raise_for_status()

    match = re.search(
        r'name="__RequestVerificationToken".*?value="([^"]+)"', page.text
    )
    if not match:
        raise RuntimeError("CSRF token bulunamadi!")
    return session, match.group(1)


def get_izmir_stations():
    """Config endpoint'inden Izmir istasyonlarini ve koordinatlarini ceker."""
    session = _make_session()
    r = session.get(CONFIG_URL, timeout=60)
    r.raise_for_status()
    config = r.json()["Object"]

    stations = []
    for s in config["StationIds"]:
        if s.get("CityId") != IZMIR_CITY_ID:
            continue
        # Koordinatlari parse et: POINT (lon lat)
        loc = s.get("Location", "")
        m = re.match(r"POINT\s*\(([^ ]+)\s+([^ ]+)\)", loc)
        lon = float(m.group(1)) if m else None
        lat = float(m.group(2)) if m else None

        stations.append({
            "station_id": s["id"],
            "station_name": s["Name"],
            "lat": lat,
            "lon": lon,
        })

    return stations


def fetch_hourly_data(session, token, station_ids, parameters, start_dt, end_dt):
    """
    Belirtilen istasyonlar ve parametreler icin saatlik veri ceker.

    Args:
        session:     requests.Session (cookie'li)
        token:       CSRF token
        station_ids: list of station UUID strings
        parameters:  list of parameter codes (PM10, PM25, ...)
        start_dt:    datetime baslangic
        end_dt:      datetime bitis

    Returns:
        list[dict]: API'den donen satir verileri
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }

    # Form payload olustur
    parts = [f"__RequestVerificationToken={token}"]
    for sid in station_ids:
        parts.append(f"StationIds={sid}")
    for param in parameters:
        parts.append(f"Parameters={param}")
    parts.append(f"StartDateTime={start_dt.strftime('%Y-%m-%d')}+00%3A00")
    parts.append(f"EndDateTime={end_dt.strftime('%Y-%m-%d')}+23%3A00")
    parts.append("DataPeriods=8")  # 8 = saatlik

    payload = "&".join(parts)

    r = session.post(DATA_URL, data=payload, headers=headers, timeout=120)
    r.raise_for_status()
    resp = r.json()

    if not resp.get("Result"):
        msg = resp.get("FeedBack", {}).get("message", "Bilinmeyen hata")
        print(f"  UYARI: {msg}")
        return []

    return resp.get("Object", {}).get("Data", []) or []


def fetch_wind_data(stations, start_dt, end_dt):
    """
    Open-Meteo API'sinden tum istasyonlar icin saatlik ruzgar verisi ceker.
    Tek bir istek ile tum koordinatlari toplu sorgular.

    Args:
        stations:  list[dict] — her biri station_id, lat, lon icermeli
        start_dt:  datetime baslangic
        end_dt:    datetime bitis

    Returns:
        pd.DataFrame: station_id, timestamp, wind_speed, wind_dir sutunlari
    """
    lats = ",".join(str(s["lat"]) for s in stations if s["lat"])
    lons = ",".join(str(s["lon"]) for s in stations if s["lon"])

    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "timezone": "Europe/Istanbul",
    }

    session = _make_session()
    r = session.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Tek istasyon ise liste degil dict doner — normalize et
    if isinstance(data, dict):
        data = [data]

    valid_stations = [s for s in stations if s["lat"]]
    rows = []
    for i, station_data in enumerate(data):
        hourly = station_data.get("hourly", {})
        times = hourly.get("time", [])
        ws = hourly.get("wind_speed_10m", [])
        wd = hourly.get("wind_direction_10m", [])

        sid = valid_stations[i]["station_id"]
        for j, t in enumerate(times):
            rows.append({
                "station_id": sid,
                "timestamp": pd.Timestamp(t),
                "wind_speed": ws[j] if j < len(ws) else None,
                "wind_dir": wd[j] if j < len(wd) else None,
            })

    return pd.DataFrame(rows)


# =========================================================================== #
#  Programatik Veri Guncelleme (app.py'den cagrilabilir)
# =========================================================================== #
def download_fresh_data(days=7, progress_callback=None):
    """
    CSB + Open-Meteo'dan guncel veriyi indirir ve CSV'ye kaydeder.

    Args:
        days: Kac gunluk veri cekilecek (varsayilan 7)
        progress_callback: Ilerleme bildirimi icin callable(step, total, message)

    Returns:
        str: Kaydedilen CSV dosya yolu (basarisizsa None)
    """
    import pathlib

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days - 1)

    def _notify(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    try:
        # 1. Istasyonlar
        _notify(1, 5, "İzmir istasyonları alınıyor...")
        stations = get_izmir_stations()
        station_lookup = {s["station_id"]: s for s in stations}
        station_ids = [s["station_id"] for s in stations]

        # 2. Oturum
        _notify(2, 5, "CSB oturumu başlatılıyor...")
        session, token = get_session_and_token()

        # 3. Kirlilik verisi
        _notify(3, 5, f"Kirlilik verisi çekiliyor ({start_date.date()} → {end_date.date()})...")
        all_rows = []
        current_start = start_date
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)
            rows = fetch_hourly_data(
                session, token, station_ids, PARAMETERS,
                current_start, current_end,
            )
            all_rows.extend(rows)
            current_start = current_end + timedelta(days=1)
            time.sleep(1)

        if not all_rows:
            return None

        # 4. Ruzgar verisi
        _notify(4, 5, "Rüzgâr verisi çekiliyor (Open-Meteo)...")
        wind_df = fetch_wind_data(stations, start_date, end_date)

        # 5. Birlestir ve kaydet
        _notify(5, 5, "Veriler birleştiriliyor...")
        df = pd.DataFrame(all_rows)
        df["ReadTime"] = pd.to_datetime(df["ReadTime"])

        df["station_name"] = df["Stationid"].map(
            lambda sid: station_lookup.get(sid, {}).get("station_name", "?")
        )
        df["lat"] = df["Stationid"].map(
            lambda sid: station_lookup.get(sid, {}).get("lat")
        )
        df["lon"] = df["Stationid"].map(
            lambda sid: station_lookup.get(sid, {}).get("lon")
        )

        df = df.rename(columns={
            "ReadTime": "timestamp",
            "Stationid": "station_id",
            "PM25": "pm25", "PM10": "pm10", "SO2": "so2",
            "NO2": "no2", "CO": "co", "O3": "o3",
        })

        df = df.merge(wind_df, on=["station_id", "timestamp"], how="left")

        cols = ["timestamp", "station_id", "station_name", "lat", "lon"]
        for p in ["pm10", "pm25", "so2", "no2", "co", "o3"]:
            if p in df.columns:
                cols.append(p)
        cols.extend(["wind_speed", "wind_dir"])
        df = df[cols]

        # CSV'ye kaydet (Cloud ortamında dosya sistemi salt okunur olabilir)
        try:
            project_dir = pathlib.Path(__file__).resolve().parent
            filename = f"izmir_hava_kalitesi_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = project_dir / filename
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
        except OSError:
            pass  # Cloud ortamında CSV yazılamayabilir — DataFrame bellekte döner

        return df

    except Exception as e:
        if progress_callback:
            progress_callback(-1, 5, str(e))
        return None


# =========================================================================== #
#  Ana Akis (komut satirindan calistirma)
# =========================================================================== #
def main():
    print("=" * 60)
    print("AtmoTrace - CSB Kirlilik + Open-Meteo Ruzgar Verisi")
    print("=" * 60)

    # 1. Istasyonlari al
    print("\n[1/5] Izmir istasyonlari aliniyor...")
    stations = get_izmir_stations()
    print(f"      {len(stations)} istasyon bulundu.")

    station_lookup = {s["station_id"]: s for s in stations}
    station_ids = [s["station_id"] for s in stations]

    # 2. Session ve token
    print("[2/5] Oturum baslatiliyor...")
    session, token = get_session_and_token()
    print("      CSRF token alindi.")

    # 3. CSB kirlilik verisini cek
    print(f"[3/5] CSB kirlilik verisi indiriliyor: {START_DATE.date()} -> {END_DATE.date()}")
    print(f"      Parametreler: {', '.join(PARAMETERS)}")

    all_rows = []
    current_start = START_DATE

    while current_start <= END_DATE:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), END_DATE)
        print(f"      {current_start.date()} - {current_end.date()} ...", end=" ")

        rows = fetch_hourly_data(
            session, token, station_ids, PARAMETERS,
            current_start, current_end,
        )
        print(f"{len(rows)} satir")
        all_rows.extend(rows)

        current_start = current_end + timedelta(days=1)
        time.sleep(1)  # API'yi yormamak icin

    if not all_rows:
        print("\n  HATA: Hic veri indirilemedi!")
        return

    # 4. Open-Meteo'dan ruzgar verisi cek
    print(f"\n[4/5] Open-Meteo ruzgar verisi indiriliyor...")
    wind_df = fetch_wind_data(stations, START_DATE, END_DATE)
    print(f"      {len(wind_df)} satir ruzgar verisi alindi.")

    # 5. Birlestir ve CSV'ye kaydet
    print(f"\n[5/5] {len(all_rows)} satir isleniyor...")

    df = pd.DataFrame(all_rows)
    df["ReadTime"] = pd.to_datetime(df["ReadTime"])

    # Istasyon adi ve koordinat ekle
    df["station_name"] = df["Stationid"].map(
        lambda sid: station_lookup.get(sid, {}).get("station_name", "?")
    )
    df["lat"] = df["Stationid"].map(
        lambda sid: station_lookup.get(sid, {}).get("lat")
    )
    df["lon"] = df["Stationid"].map(
        lambda sid: station_lookup.get(sid, {}).get("lon")
    )

    # Sutun isimlerini duzelt
    df = df.rename(columns={
        "ReadTime": "timestamp",
        "Stationid": "station_id",
        "PM25": "pm25",
        "PM10": "pm10",
        "SO2": "so2",
        "NO2": "no2",
        "CO": "co",
        "O3": "o3",
    })

    # Ruzgar verisini merge et (station_id + timestamp uzerinden)
    df = df.merge(
        wind_df,
        on=["station_id", "timestamp"],
        how="left",
    )

    # Sutun sirasi duzenle
    cols = ["timestamp", "station_id", "station_name", "lat", "lon"]
    for p in ["pm10", "pm25", "so2", "no2", "co", "o3"]:
        if p in df.columns:
            cols.append(p)
    cols.extend(["wind_speed", "wind_dir"])
    df = df[cols]

    # CSV kaydet
    filename = f"izmir_hava_kalitesi_{START_DATE.strftime('%Y%m%d')}_{END_DATE.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"\n  Kaydedildi: {filename}")
    print(f"  Toplam satir : {len(df)}")
    print(f"  Istasyon sayisi: {df['station_name'].nunique()}")
    print(f"  Tarih araligi  : {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"\n  PM10  dolu: {df['pm10'].notna().sum()} / {len(df)}")
    print(f"  PM2.5 dolu: {df['pm25'].notna().sum()} / {len(df)}")
    print(f"  Ruzgar dolu: {df['wind_speed'].notna().sum()} / {len(df)}  (Open-Meteo)")

    print("\n" + "=" * 60)
    print("Tamamlandi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
