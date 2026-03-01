"""
AtmoTrace MVP - Veri Motoru
CSB kirlilik + Open-Meteo ruzgar verilerini icerir.
Tum veriler gercek kaynaklardan indirilmis CSV'den okunur.
"""

import pathlib
import pandas as pd
from datetime import datetime

# =========================================================================== #
#  CSV dosya yolu (proje klasorundeki en guncel CSV)
# =========================================================================== #
_PROJECT_DIR = pathlib.Path(__file__).resolve().parent
CSV_PATTERN = "izmir_hava_kalitesi_*.csv"


def _find_csv() -> pathlib.Path:
    """Proje klasorundeki en guncel hava kalitesi CSV dosyasini bulur."""
    files = sorted(_PROJECT_DIR.glob(CSV_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"CSV dosyasi bulunamadi: {_PROJECT_DIR / CSV_PATTERN}\n"
            "Once csb_veri_indirme.py scriptini calistirin."
        )
    return files[-1]


def load_all_data() -> pd.DataFrame:
    """
    CSV'deki tum saatlik veriyi yukler.

    Returns:
        pd.DataFrame: timestamp, station_id, station_name, lat, lon,
                       pm10, pm25, so2, no2, co, o3, wind_speed, wind_dir
    """
    csv_path = _find_csv()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df


def get_available_hours(df: pd.DataFrame | None = None) -> list[datetime]:
    """CSV'deki benzersiz saat dilimleri (sirali)."""
    if df is None:
        df = load_all_data()
    hours = sorted(df["timestamp"].dropna().unique())
    return [pd.Timestamp(h).to_pydatetime() for h in hours]


def load_station_snapshot(
    target_time: datetime | None = None,
    all_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Belirli bir saat icin istasyon bazinda tek satirlik ozet verir.

    Args:
        target_time: Istenilen saat (None ise en son mevcut saat).
        all_data:    Onceden yuklenmis DataFrame (None ise CSV'den okur).

    Returns:
        pd.DataFrame: Her istasyon icin tek satir:
            - station_id, station_name, lat, lon, timestamp
            - pm10, pm25, so2, no2, co, o3  (CSB gercek verisi)
            - wind_speed (m/s), wind_dir (derece)  (Open-Meteo gercek verisi)
    """
    df = all_data if all_data is not None else load_all_data()

    if target_time is None:
        target_time = df["timestamp"].max()
    else:
        target_time = pd.Timestamp(target_time)

    snapshot = df[df["timestamp"] == target_time].copy()

    if snapshot.empty:
        diffs = (df["timestamp"] - target_time).abs()
        closest_time = df.loc[diffs.idxmin(), "timestamp"]
        snapshot = df[df["timestamp"] == closest_time].copy()

    snapshot = snapshot.drop_duplicates(subset=["station_id"], keep="first")
    snapshot = snapshot.reset_index(drop=True)
    return snapshot


# Geriye uyumluluk
def generate_mock_data() -> pd.DataFrame:
    """Geriye uyumluluk icin — artik gercek veri donduruyor."""
    return load_station_snapshot()


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    df = load_station_snapshot()
    print(f"CSV: {_find_csv().name}")
    print(f"Saat: {df['timestamp'].iloc[0]}")
    print(f"Istasyon sayisi: {len(df)}")
    print()
    cols = ["station_name", "pm10", "pm25", "so2", "no2", "wind_speed", "wind_dir"]
    print(df[cols].to_string(index=False))
