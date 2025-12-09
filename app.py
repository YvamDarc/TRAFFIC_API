import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# 1. Utils géographiques
# =========================

def geocode_address(address: str):
    """Retourne (lat, lon) à partir d'une adresse avec Nominatim (OpenStreetMap)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    resp = requests.get(url, params=params, headers={"User-Agent": "streamlit-footfall-app"})
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None, None
    return float(data[0]["lat"]), float(data[0]["lon"])


def fetch_pois_from_osm(lat: float, lon: float, radius_m: int = 500, max_pois: int = 10):
    """
    Récupère des points d'intérêt significatifs autour d'un point via Overpass.
    On filtre sur quelques types de commerces et lieux très fréquentés.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Tags "importants" (commerce, transports, etc.)
    query = f"""
    [out:json][timeout:25];
    (
      node
        ["shop"~"supermarket|mall|department_store|convenience"]
        (around:{radius_m},{lat},{lon});
      node
        ["amenity"~"cinema|theatre|fast_food|restaurant|pub|bar|cafe|bank"]
        (around:{radius_m},{lat},{lon});
      node
        ["amenity"~"bus_station|ferry_terminal|marketplace"]
        (around:{radius_m},{lat},{lon});
      node
        ["railway"="station"]
        (around:{radius_m},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """

    resp = requests.post(overpass_url, data=query, headers={"User-Agent": "streamlit-footfall-app"})
    resp.raise_for_status()
    data = resp.json()

    elements = data.get("elements", [])
    pois = []
    for el in elements:
        if el.get("type") != "node":
            continue
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        poi_type = tags.get("shop") or tags.get("amenity") or tags.get("railway") or "poi"
        pois.append({
            "id": el["id"],
            "name": name,
            "type": poi_type,
            "lat": el["lat"],
            "lon": el["lon"],
        })

    # On limite le nombre de POI pour rester raisonnable
    df_pois = pd.DataFrame(pois)
    if df_pois.empty:
        return df_pois

    # Petite priorité aux centres commerciaux / gares si présents
    priority_keywords = ["mall", "station", "supermarket", "marketplace", "cinema"]
    df_pois["priority"] = df_pois["type"].apply(
        lambda t: 0 if any(pk in t for pk in priority_keywords) else 1
    )
    df_pois = df_pois.sort_values(["priority", "name"]).head(max_pois).reset_index(drop=True)
    return df_pois


# =========================================
# 2. Fournisseur de données de flux
# =========================================

def simulate_daily_footfall_for_poi(poi_id, start_date, end_date):
    """
    ⚠️ FAUX fournisseur de données.
    Cette fonction génère une série temporelle jour par jour
    pour un POI donné, uniquement pour tester l'app.

    ➜ À REMPLACER par :
      - un appel à ton API Rennes
      - ou un fournisseur commercial (MyTraffic, telco, etc.)
    """
    rng = pd.date_range(start_date, end_date, freq="D")

    # Seed basée sur l'ID pour avoir un profil stable par POI
    np.random.seed(int(poi_id) % 2**32)

    # Base level de fréquentation
    base = np.random.randint(300, 1500)

    # Saisonnalité hebdomadaire (moins de monde le dimanche par ex)
    weekday_effect = np.array([1.1, 1.05, 1.0, 1.0, 1.15, 1.4, 0.7])  # lun→dim

    # Un peu de bruit
    noise = np.random.normal(0, base * 0.1, size=len(rng))

    values = []
    for i, d in enumerate(rng):
        factor = weekday_effect[d.weekday()]
        val = max(0, base * factor + noise[i])
        values.append(val)

    df = pd.DataFrame({"date": rng, "footfall": values})
    df["poi_id"] = poi_id
    return df


def get_daily_footfall_for_poi(poi_row, start_date, end_date):
    """
    Wrapper pour une source de données de flux.

    Pour passer en "vraie prod", il suffit de remplacer le contenu
    par des appels API réels, par ex :

      - fetch_from_rennes_api(poi_row['lat'], poi_row['lon'], start_date, end_date)
      - fetch_from_mytraffic(...)
      - fetch_from_google_popular_times(...)

    En gardant le même format de sortie (date, footfall, poi_id).
    """
    # Ici : on utilise le simulateur.
    return simulate_daily_footfall_for_poi(poi_row["id"], start_date, end_date)


# =========================
# 3. App Streamlit
# =========================

st.set_page_config(
    page_title="Analyse de flux - Multi-zones",
    layout="wide"
)

st.title("📈 Analyse générale de flux de personnes par zone géographique")
st.write(
    """
Appli **généraliste** : tu définis une zone (adresse ou coordonnées),  
on récupère les **points d'intérêt significatifs** (OSM) dans le rayon,  
puis on construit une **série quotidienne de flux** par POI et une **moyenne** sur la zone.
    
⚠️ Pour l'instant, les flux sont **simulés**.  
Il suffira de remplacer la fonction `get_daily_footfall_for_poi` par ta vraie API.
"""
)

# ---- Paramètres de la zone ----
st.sidebar.header("🗺️ Paramètres de la zone")

mode = st.sidebar.radio(
    "Mode de saisie de la zone",
    ["Adresse", "Latitude / Longitude"],
    index=0
)

if mode == "Adresse":
    address = st.sidebar.text_input("Adresse / ville / lieu", "Rennes, France")
    lat = lon = None
else:
    lat = st.sidebar.number_input("Latitude", value=48.1173, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=-1.6778, format="%.6f")
    address = None

radius_m = st.sidebar.slider("Rayon de recherche (mètres)", min_value=200, max_value=3000, value=800, step=100)

# Période d'analyse
today = datetime.today().date()
default_start = today - timedelta(days=90)

start_date = st.sidebar.date_input("Date de début", default_start)
end_date = st.sidebar.date_input("Date de fin", today)

if start_date > end_date:
    st.sidebar.error("La date de début doit être <= à la date de fin.")

max_pois = st.sidebar.slider("Nombre maximum de POI à analyser", 3, 30, 10)

run_button = st.sidebar.button("🚀 Lancer l'analyse")

if run_button and start_date <= end_date:
    # 1) Géocodage
    with st.spinner("Géocodage de la zone…"):
        if mode == "Adresse":
            lat, lon = geocode_address(address)
            if lat is None:
                st.error("Impossible de géocoder cette adresse. Essaie d'être plus précis.")
                st.stop()
        # Sinon lat/lon déjà fournis

    st.success(f"Zone centrée sur lat={lat:.5f}, lon={lon:.5f}")

    # 2) Récupération des POI
    st.subheader("📍 Points d'intérêt identifiés")
    with st.spinner("Recherche des POI significatifs via OpenStreetMap…"):
        df_pois = fetch_pois_from_osm(lat, lon, radius_m=radius_m, max_pois=max_pois)

    if df_pois.empty:
        st.warning("Aucun point d'intérêt significatif trouvé dans ce rayon. Essaie d'augmenter le rayon ou de changer de zone.")
        st.stop()

    st.dataframe(df_pois[["name", "type", "lat", "lon"]])

    # 3) Récupération / simulation des séries journalières
    st.subheader("📊 Séries journalières par POI")

    all_series = []
    progress = st.progress(0)
    total = len(df_pois)

    for i, (_, poi) in enumerate(df_pois.iterrows(), start=1):
        df_ts = get_daily_footfall_for_poi(poi, start_date, end_date)
        df_ts["poi_name"] = poi["name"]
        df_ts["poi_type"] = poi["type"]
        all_series.append(df_ts)
        progress.progress(i / total)

    df_all = pd.concat(all_series, ignore_index=True)

    # 4) Visualisation détaillée
    tab1, tab2 = st.tabs(["Détail par POI", "Moyenne de la zone"])

    with tab1:
        st.markdown("### 📌 Détail des flux par POI (simulés)")
        poi_selected = st.selectbox("Choisir un POI", df_pois["name"].tolist())
        df_one = df_all[df_all["poi_name"] == poi_selected].copy()
        df_one = df_one.sort_values("date")

        st.line_chart(
            df_one.set_index("date")["footfall"],
            height=300
        )
        st.write(df_one[["date", "footfall"]])

    # 5) Agrégation : moyenne de la zone
    with tab2:
        st.markdown("### 📊 Moyenne journalière de flux sur l'ensemble de la zone")

        df_zone = (
            df_all
            .groupby("date", as_index=False)["footfall"]
            .mean()
            .rename(columns={"footfall": "footfall_mean"})
        )

        st.line_chart(
            df_zone.set_index("date")["footfall_mean"],
            height=300
        )
        st.write(df_zone)

        # Export CSV
        csv = df_zone.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Télécharger la moyenne journalière (CSV)",
            data=csv,
            file_name="footfall_zone_daily_mean.csv",
            mime="text/csv"
        )

    st.success("Analyse terminée (données simulées). Tu peux maintenant brancher ta vraie API de flux.")
