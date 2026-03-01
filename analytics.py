"""
AtmoTrace MVP - Analitik Motor (analytics.py)

Kaynak-odakli analiz: Her istasyondan ruzgarin geldigi yone dogru tersine
yorunge cizgisi olusturulur. Bu cizgilerin uzamsal yogunlastigi nokta
"muhtemel kirletici kaynak" olarak tespit edilir.

Yontem:
    1. Her istasyonun kirlilik skoru hesaplanir.
    2. Kirlilik skoru > esik olan istasyonlardan geriye dogru trajectory cizilir.
    3. Trajectory cizgileri bir grid uzerinde yogunluk haritasina donusturulur
       (her cizgi uzerindeki grid noktalarinin degeri, o istasyonun kirlilik
       skoru ile agirliklandirilir).
    4. Grid uzerindeki en yuksek yogunluk noktasi = "Muhtemel Kirletici Kaynak"

Bu yaklasim, birden fazla istasyonun ruzgar verisini birlestirerek
kirliligin NEREDEN geldigi sorusuna yanit verir.
"""

import math
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
#  Sabitler
# --------------------------------------------------------------------------- #
EARTH_RADIUS_M = 6_371_000
BACKTRACK_TIME_S = 3600          # 1 saat geriye izleme
TRAJECTORY_STEPS = 20            # her trajectory kac adimda orneklenir
MIN_SCORE_THRESHOLD = 0.15       # bu skorun altindaki istasyonlar haric

# Kirlilik skoru agirliklari
WEIGHTS = {
    "pm25": 0.35, "pm10": 0.30, "so2": 0.15,
    "no2":  0.10, "co":   0.05, "o3":  0.05,
}

# Kaynak tespiti grid
GRID_RESOLUTION = 0.003    # ~330 metre
LAT_MIN, LAT_MAX = 38.25, 38.55
LON_MIN, LON_MAX = 26.90, 27.35


# --------------------------------------------------------------------------- #
#  Yardimci: Haversine mesafe
# --------------------------------------------------------------------------- #
def _haversine(lat1, lon1, lat2, lon2):
    """Iki koordinat arasi mesafe (metre). Numpy array destekler."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
#  Yardimci: Kuresel koordinat kaydirma
# --------------------------------------------------------------------------- #
def _destination_point(lat, lon, bearing_deg, distance_m):
    """Bir noktadan belirli yon ve mesafede yeni koordinat hesaplar."""
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    br = math.radians(bearing_deg)
    ad = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ad)
        + math.cos(lat1) * math.sin(ad) * math.cos(br)
    )
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(ad) * math.cos(lat1),
        math.cos(ad) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# --------------------------------------------------------------------------- #
#  1. Istasyon Kirlilik Skoru
# --------------------------------------------------------------------------- #
def compute_station_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Her istasyon icin normalize edilmis kirlilik skoru hesaplar."""
    df = df.copy()
    available = [col for col in WEIGHTS if col in df.columns]

    for col in available:
        cmin, cmax = df[col].min(), df[col].max()
        rng = cmax - cmin if cmax != cmin else 1.0
        df[f"norm_{col}"] = (df[col] - cmin) / rng

    scores = pd.Series(0.0, index=df.index)
    wsums = pd.Series(0.0, index=df.index)
    for col in available:
        valid = df[col].notna()
        scores += valid * WEIGHTS[col] * df[f"norm_{col}"].fillna(0)
        wsums += valid.astype(float) * WEIGHTS[col]

    wsums = wsums.replace(0, np.nan)
    df["pollution_score"] = (scores / wsums).fillna(0)
    return df


# --------------------------------------------------------------------------- #
#  2. Tersine Yorunge Cizgisi Olusturma
# --------------------------------------------------------------------------- #
def _build_trajectory(lat, lon, wind_dir, wind_speed,
                      backtrack_s=BACKTRACK_TIME_S, steps=TRAJECTORY_STEPS):
    """
    Bir istasyondan ruzgarin geldigi yone dogru adim adim geri gider.
    Her adimin koordinatini dondurur.

    Meteorolojik ruzgar yonu = ruzgarin geldigi yon.
    Kaynak o tarafta → bearing = wind_dir (donusturme gerekmez).
    """
    total_dist = wind_speed * backtrack_s
    step_dist = total_dist / steps
    bearing = wind_dir % 360

    points = [(lat, lon)]
    cur_lat, cur_lon = lat, lon
    for _ in range(steps):
        cur_lat, cur_lon = _destination_point(cur_lat, cur_lon, bearing, step_dist)
        points.append((cur_lat, cur_lon))
    return points


# --------------------------------------------------------------------------- #
#  3. Kaynak Tespiti: Trajectory Yogunluk Analizi
# --------------------------------------------------------------------------- #
def find_source(df: pd.DataFrame) -> dict:
    """
    Tum istasyonlarin tersine yorunge cizgilerinin uzamsal yogunlugunu
    analiz ederek muhtemel kirletici kaynak noktasini tespit eder.

    Yontem:
        1. Her istasyon icin kirlilik skoru hesaplanir.
        2. Esik ustu istasyonlardan backward trajectory olusturulur.
        3. Her trajectory noktasi, bir grid hucresine duserek o hucrenin
           agirligini arttirir (agirlik = istasyonun kirlilik skoru).
        4. En yuksek agirlikli grid hucresi = muhtemel kaynak.

    Returns:
        dict: source_lat, source_lon, confidence, trajectories,
              grid_data (heatmap icin), contributing_stations
    """
    df_scored = compute_station_scores(df)

    # Ruzgar verisi olan ve esik ustu istasyonlar
    active = df_scored[
        (df_scored["pollution_score"] >= MIN_SCORE_THRESHOLD)
        & df_scored["wind_speed"].notna()
        & df_scored["wind_dir"].notna()
        & (df_scored["wind_speed"] > 0.3)
    ].copy()

    if active.empty:
        row = df.iloc[0]
        return _empty_result(row)

    # Her istasyondan trajectory olustur
    trajectories = []   # [(station_name, score, [(lat,lon), ...]), ...]

    for _, row in active.iterrows():
        points = _build_trajectory(
            row["lat"], row["lon"],
            row["wind_dir"], row["wind_speed"],
        )
        trajectories.append({
            "station_name": row["station_name"],
            "station_lat": row["lat"],
            "station_lon": row["lon"],
            "score": row["pollution_score"],
            "wind_speed": row["wind_speed"],
            "wind_dir": row["wind_dir"],
            "points": points,
        })

    # Grid olustur
    grid_lats = np.arange(LAT_MIN, LAT_MAX, GRID_RESOLUTION)
    grid_lons = np.arange(LON_MIN, LON_MAX, GRID_RESOLUTION)
    n_lat, n_lon = len(grid_lats), len(grid_lons)
    density = np.zeros((n_lat, n_lon))

    # Her trajectory noktasini grid'e yansit
    for traj in trajectories:
        score = traj["score"]
        for pt_lat, pt_lon in traj["points"]:
            # En yakin grid indeksi
            i = int(round((pt_lat - LAT_MIN) / GRID_RESOLUTION))
            j = int(round((pt_lon - LON_MIN) / GRID_RESOLUTION))
            if 0 <= i < n_lat and 0 <= j < n_lon:
                density[i, j] += score

    # Gaussian blur ile yayilma (yakin hucreleri de etkile)
    from scipy.ndimage import gaussian_filter
    density_smooth = gaussian_filter(density, sigma=2.0)

    # --- Çoklu tepe noktası tespiti (multi-peak) ---
    from scipy.ndimage import label, maximum_position

    global_max = density_smooth.max()
    if global_max == 0:
        row = df.iloc[0]
        return _empty_result(row)

    # Yoğunluk eşiği: global max'ın %20'sinden yüksek bölgeleri bul
    binary = density_smooth >= (global_max * 0.20)
    labeled_array, n_features = label(binary)

    # Her bölgenin tepe noktasını bul
    sources = []
    for region_id in range(1, n_features + 1):
        region_mask = labeled_array == region_id
        region_vals = density_smooth * region_mask
        peak_idx = np.unravel_index(np.argmax(region_vals), region_vals.shape)
        peak_val = float(density_smooth[peak_idx])
        s_lat = LAT_MIN + peak_idx[0] * GRID_RESOLUTION
        s_lon = LON_MIN + peak_idx[1] * GRID_RESOLUTION
        sources.append({
            "lat": round(s_lat, 6),
            "lon": round(s_lon, 6),
            "peak_density": round(peak_val, 4),
            "confidence": round(min(1.0, peak_val / max(global_max, 0.01)), 3),
        })

    # Yoğunluğa göre sırala (en güçlü ilk)
    sources.sort(key=lambda s: s["peak_density"], reverse=True)
    # En fazla 5 kaynak göster
    sources = sources[:5]

    # Birincil kaynak (geriye uyumluluk)
    primary = sources[0]
    source_lat = primary["lat"]
    source_lon = primary["lon"]
    peak_val = primary["peak_density"]
    confidence = primary["confidence"]

    # Katki yapan istasyonlar (trajectory'si birincil kaynaga en yakin olanlar)
    for traj in trajectories:
        traj_points = np.array(traj["points"])
        dists = _haversine(source_lat, source_lon,
                           traj_points[:, 0], traj_points[:, 1])
        traj["min_dist_to_source_km"] = round(float(np.min(dists)) / 1000, 2)

    trajectories.sort(key=lambda t: t["min_dist_to_source_km"])

    # Heatmap verisi
    grid_data = []
    threshold = global_max * 0.05
    for i in range(n_lat):
        for j in range(n_lon):
            val = float(density_smooth[i, j])
            if val > threshold:
                grid_data.append([
                    LAT_MIN + i * GRID_RESOLUTION,
                    LON_MIN + j * GRID_RESOLUTION,
                    val / global_max,  # 0-1 normalize
                ])

    return {
        "source_lat": round(source_lat, 6),
        "source_lon": round(source_lon, 6),
        "confidence": round(confidence, 3),
        "peak_density": round(peak_val, 4),
        "trajectories": trajectories,
        "grid_data": grid_data,
        "n_contributing": len(trajectories),
        "all_sources": sources,
    }


def _empty_result(row):
    return {
        "source_lat": row["lat"],
        "source_lon": row["lon"],
        "confidence": 0,
        "peak_density": 0,
        "trajectories": [],
        "grid_data": [],
        "n_contributing": 0,
        "all_sources": [],
    }


# --------------------------------------------------------------------------- #
#  Geriye uyumluluk
# --------------------------------------------------------------------------- #
def calculate_source(hotspot_lat, hotspot_lon, wind_speed, wind_direction,
                     backtrack_seconds=BACKTRACK_TIME_S):
    """Tek noktadan tersine yorunge (eski arayuz)."""
    dest_lat, dest_lon = _destination_point(
        hotspot_lat, hotspot_lon, wind_direction % 360,
        wind_speed * backtrack_seconds
    )
    return {
        "source_lat": round(dest_lat, 6),
        "source_lon": round(dest_lon, 6),
        "distance_km": round(wind_speed * backtrack_seconds / 1000, 2),
        "bearing_deg": round(wind_direction % 360, 1),
        "backtrack_time_s": backtrack_seconds,
    }


# --------------------------------------------------------------------------- #
#  Test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    from data_engine import load_station_snapshot

    df = load_station_snapshot()
    result = find_source(df)

    print("=== Kaynak Tespiti (Trajectory Kesisim Analizi) ===")
    print(f"  Muhtemel Kaynak : {result['source_lat']:.4f}, {result['source_lon']:.4f}")
    print(f"  Guvenilirlik    : {result['confidence']:.1%}")
    print(f"  Yogunluk        : {result['peak_density']:.4f}")
    print(f"  Kullanilan ist. : {result['n_contributing']}")
    print(f"  Grid noktasi    : {len(result['grid_data'])}")

    print(f"\n  Trajectory'si kaynaga en yakin istasyonlar:")
    for t in result["trajectories"][:7]:
        print(f"    {t['station_name']:30s}  skor={t['score']:.3f}  "
              f"ruzgar={t['wind_speed']}m/s {t['wind_dir']:.0f}deg  "
              f"yakinlik={t['min_dist_to_source_km']} km")
