import math
import sys
from dataclasses import dataclass

import pandas as pd


# ================================
# 1. PARAMÈTRES À ADAPTER
# ================================

# URL ou chemin local du CSV de comptage journalier
# Exemple : un fichier open data "comptage_journalier.csv"
DATA_URL = "comptage_journalier.csv"  # mets ici l'URL http(s) OU un chemin local

# Coordonnées de la zone d'intérêt (ex : ton commerce / centre-ville)
TARGET_LAT = 48.50023   # exemple : près de Saint-Brieuc
TARGET_LON = -2.72461

# Rayon en km autour de la zone pour sélectionner les stations SIR
RAYON_KM = 15.0

# Nom du fichier de sortie
OUTPUT_CSV = "indice_activite_journalier.csv"


# ================================
# 2. PARAMS DE MAPPING DE COLONNES
# (À ADAPTER SUR TON FICHIER)
# ================================

@dataclass
class ColumnMapping:
    station_id: str
    date: str
    count: str
    lat: str
    lon: str


# Exemple generique:
# - 'id_station' : identifiant de la station
# - 'date' : date (format YYYY-MM-DD)
# - 'tmj' : trafic moyen journalier ou comptage du jour
# - 'lat' / 'lon' : coordonnées de la station
COLUMN_MAPPING = ColumnMapping(
    station_id="id_station",
    date="date",
    count="tmj",
    lat="lat",
    lon="lon",
)


# ================================
# 3. FONCTIONS UTILITAIRES
# ================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distance entre deux points en km (formule de Haversine).
    """
    R = 6371.0  # Rayon moyen de la Terre en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def load_data(path_or_url: str) -> pd.DataFrame:
    """
    Charge le CSV via pandas. Si c'est une URL, il la télécharge.
    """
    print(f"📥 Chargement des données depuis {path_or_url} ...")
    try:
        df = pd.read_csv(path_or_url, low_memory=False)
    except Exception as e:
        print(f"❌ Erreur de lecture du CSV : {e}")
        sys.exit(1)

    print(f"✅ Fichier chargé : {len(df):,} lignes, {len(df.columns)} colonnes")
    print("   Colonnes disponibles :", list(df.columns))
    return df


def normalize_columns(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """
    Renomme et convertit les colonnes importantes : station, date, comptage, lat, lon.
    Adapte la logique ici en fonction de la structure réelle de ton fichier.
    """
    # On vérifie la présence des colonnes
    for attr_name, col_name in mapping.__dict__.items():
        if col_name not in df.columns:
            print(f"⚠️ Colonne '{col_name}' non trouvée dans le CSV (attendu pour '{attr_name}').")
            print("   Adapte COLUMN_MAPPING à la structure réelle de ton fichier.")
            sys.exit(1)

    df_norm = df.copy()

    # Renommage standardisé
    df_norm = df_norm.rename(
        columns={
            mapping.station_id: "station_id",
            mapping.date: "date",
            mapping.count: "count",
            mapping.lat: "lat",
            mapping.lon: "lon",
        }
    )

    # Conversion types
    df_norm["date"] = pd.to_datetime(df_norm["date"], errors="coerce")
    df_norm["count"] = pd.to_numeric(df_norm["count"], errors="coerce")
    df_norm["lat"] = pd.to_numeric(df_norm["lat"], errors="coerce")
    df_norm["lon"] = pd.to_numeric(df_norm["lon"], errors="coerce")

    df_norm = df_norm.dropna(subset=["date", "count", "lat", "lon", "station_id"])
    print(f"✅ Après normalisation : {len(df_norm):,} lignes")
    return df_norm


def select_stations_near_point(df: pd.DataFrame, lat: float, lon: float, rayon_km: float):
    """
    Ajoute une colonne distance_km et garde les stations dans le rayon.
    """
    print(f"📍 Calcul des distances par rapport au point ({lat:.5f}, {lon:.5f}) ...")
    df = df.copy()

    # On calcule la distance pour chaque station unique
    stations = (
        df[["station_id", "lat", "lon"]]
        .drop_duplicates(subset=["station_id"])
        .reset_index(drop=True)
    )

    stations["distance_km"] = stations.apply(
        lambda row: haversine_km(lat, lon, row["lat"], row["lon"]),
        axis=1,
    )

    stations_sel = stations[stations["distance_km"] <= rayon_km].copy()

    print(f"✅ Stations dans un rayon de {rayon_km} km : {len(stations_sel)} trouvées")
    if stations_sel.empty:
        print("❌ Aucune station dans le rayon choisi. Essaie d'augmenter RAYON_KM.")
        sys.exit(0)

    # Jointure pour ne garder que les mesures de ces stations
    df_sel = df.merge(stations_sel[["station_id", "distance_km"]], on="station_id", how="inner")

    print(f"✅ Lignes conservées après filtrage par distance : {len(df_sel):,}")
    return df_sel, stations_sel


def build_daily_index(df_sel: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les comptages journaliers pour construire un indice d'activité.
    - Somme journalière sur toutes les stations
    - Normalisation optionnelle
    """
    print("📊 Construction de l'indice journalier d'activité routière ...")

    daily = (
        df_sel.groupby("date", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "count_total"})
        .sort_values("date")
    )

    # On peut créer un indice normalisé base 100 sur la première date
    if not daily.empty:
        base_value = daily["count_total"].iloc[0]
        if base_value > 0:
            daily["indice_base100"] = (daily["count_total"] / base_value) * 100.0
        else:
            daily["indice_base100"] = None

    print(f"✅ Série journalière créée : {len(daily):,} jours")
    return daily


# ================================
# 4. MAIN
# ================================

def main():
    print("=== Indice d'activité routière journalier (script séparé) ===")

    df_raw = load_data(DATA_URL)
    df_norm = normalize_columns(df_raw, COLUMN_MAPPING)

    df_sel, stations_sel = select_stations_near_point(
        df_norm,
        lat=TARGET_LAT,
        lon=TARGET_LON,
        rayon_km=RAYON_KM,
    )

    print("\n📍 Stations utilisées :")
    print(stations_sel[["station_id", "lat", "lon", "distance_km"]].head(20))

    daily_index = build_daily_index(df_sel)

    if daily_index.empty:
        print("❌ Aucune donnée journalière après agrégation.")
        sys.exit(0)

    print("\n📈 Aperçu de la série journalière :")
    print(daily_index.head(15))

    print(f"\n💾 Export du fichier journalier vers : {OUTPUT_CSV}")
    daily_index.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n✅ Terminé. Tu peux maintenant corréler 'indice_base100' ou 'count_total'")
    print("   avec ton CAHT journalier en rejoignant sur la colonne 'date'.")


if __name__ == "__main__":
    main()
