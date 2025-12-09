import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import folium
from streamlit_folium import st_folium

# =========================
# 0. Session state
# =========================
if "results_ready" not in st.session_state:
    st.session_state["results_ready"] = False
if "df_pois" not in st.session_state:
    st.session_state["df_pois"] = None
if "df_all" not in st.session_state:
    st.session_state["df_all"] = None
if "zone_center" not in st.session_state:
    st.session_state["zone_center"] = None
if "picked_lat" not in st.session_state:
    st.session_state["picked_lat"] = None
if "picked_lon" not in st.session_state:
    st.session_state["picked_lon"] = None


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
    """
    overpass_url = "https://overpass-api.de/api/interpreter"

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

    df_pois = pd.DataFrame(pois)
    if df_pois.empty:
        return df_pois

    priority_keywords = ["mall", "station", "supermarket", "marketplace", "cinema"]
    df_pois["priority"] = df_pois["type"].apply(
        lambda t: 0 if any(pk in t for pk in priority_keywords) else 1
    )
    df_pois = df_pois.sort_values(["priority", "name"]).head(max_pois).reset_index(drop=True)
    return df_pois


# =========================================
# 2. Fournisseur de données de flux (simulé)
# =========================================

def simulate_daily_footfall_for_poi(poi_id, start_date, end_date):
    """Génère une série quotidienne de flux simulés pour un POI."""
    rng = pd.date_range(start_date, end_date, freq="D")
    np.random.seed(int(poi_id) % 2**32)
    base = np.random.randint(300, 1500)
    weekday_effect = np.array([1.1, 1.05, 1.0, 1.0, 1.15, 1.4, 0.7])  # lun→dim
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
    ➜ À remplacer plus tard par ton appel API réel.
    Doit renvoyer un DataFrame : date, footfall, poi_id.
    """
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
Appli **généraliste** : tu définis une zone (via carte, adresse ou coordonnées),  
on récupère les **points d'intérêt significatifs** (OSM) dans le rayon,  
puis on construit une **série quotidienne de flux** par POI et une **moyenne** sur la zone.

⚠️ Pour l'instant, les flux sont **simulés**.  
Il suffira de remplacer la fonction `get_daily_footfall_for_poi` par ta vraie API.
"""
)

# ---- Paramètres de la zone ----
st.sidebar.header("🗺️ Paramètres de la zone")

mode = st.sidebar.radio(
    "Mode de sélection de la zone",
    ["Carte (clic)", "Adresse", "Latitude / Longitude"],
    index=0
)

# Raccourci villes bretonnes (pour aider en mode Adresse)
bzh_choice = None
if mode == "Adresse":
    bzh_choice = st.sidebar.selectbox(
        "Raccourci villes bretonnes",
        ["(aucune)", "Rennes", "Brest", "Quimper", "Lorient", "Vannes", "Saint-Brieuc"]
    )

# Période d'analyse
today = datetime.today().date()
default_start = today - timedelta(days=90)
start_date = st.sidebar.date_input("Date de début", default_start)
end_date = st.sidebar.date_input("Date de fin", today)

if start_date > end_date:
    st.sidebar.error("La date de début doit être <= à la date de fin.")

radius_m = st.sidebar.slider("Rayon de recherche (mètres)", min_value=200, max_value=3000, value=800, step=100)
max_pois = st.sidebar.slider("Nombre maximum de POI à analyser", 3, 30, 10)

run_button = st.sidebar.button("🚀 Lancer / mettre à jour l'analyse")

# =========================
# 4. Carte interactive (mode Carte)
# =========================

if mode == "Carte (clic)":
    st.subheader("🗺️ Sélectionne un point sur la carte (clic gauche)")
    # Carte centrée sur la Bretagne
    center_bzh = [48.0, -2.8]
    m = folium.Map(location=center_bzh, zoom_start=7)

    # Si un point a déjà été choisi, on l'affiche
    if st.session_state["picked_lat"] is not None and st.session_state["picked_lon"] is not None:
        folium.Marker(
            [st.session_state["picked_lat"], st.session_state["picked_lon"]],
            tooltip="Point sélectionné",
            icon=folium.Icon(color="red")
        ).add_to(m)

    map_data = st_folium(m, height=450, width=900, key="bzh_map")

    # Gestion du clic sur la carte
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        st.session_state["picked_lat"] = clicked_lat
        st.session_state["picked_lon"] = clicked_lon

    if st.session_state["picked_lat"] is not None:
        st.info(
            f"Point sélectionné : lat = {st.session_state['picked_lat']:.5f}, "
            f"lon = {st.session_state['picked_lon']:.5f}"
        )

# =========================
# 5. Lancement / mise à jour de l'analyse
# =========================

if run_button and start_date <= end_date:
    # Détermination du centre de zone
    if mode == "Carte (clic)":
        if st.session_state["picked_lat"] is None or st.session_state["picked_lon"] is None:
            st.error("Clique d'abord sur la carte pour choisir un point.")
            st.stop()
        lat = st.session_state["picked_lat"]
        lon = st.session_state["picked_lon"]

    elif mode == "Adresse":
        if bzh_choice and bzh_choice != "(aucune)":
            default_address = f"{bzh_choice}, Bretagne, France"
        else:
            default_address = "Rennes, France"
        address = st.sidebar.text_input("Adresse / ville / lieu", default_address, key="addr_input_run")
        # ⚠️ Si l'utilisateur ne modifie pas l'adresse ici, on prend la valeur par défaut
        addr_to_geocode = address or default_address

        with st.spinner("Géocodage de l'adresse…"):
            lat, lon = geocode_address(addr_to_geocode)
            if lat is None:
                st.error("Impossible de géocoder cette adresse. Essaie d'être plus précis.")
                st.stop()

    else:  # Latitude / Longitude
        lat = st.sidebar.number_input("Latitude", value=48.1173, format="%.6f", key="lat_run")
        lon = st.sidebar.number_input("Longitude", value=-1.6778, format="%.6f", key="lon_run")

    st.session_state["zone_center"] = (lat, lon)

    # 2) Récupération des POI
    with st.spinner("Recherche des POI significatifs via OpenStreetMap…"):
        df_pois = fetch_pois_from_osm(lat, lon, radius_m=radius_m, max_pois=max_pois)

    if df_pois.empty:
        st.warning("Aucun point d'intérêt significatif trouvé dans ce rayon. Essaie d'augmenter le rayon ou de changer de zone.")
        st.session_state["results_ready"] = False
    else:
        # 3) Séries journalières pour chaque POI
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

        # Stockage en session_state
        st.session_state["df_pois"] = df_pois
        st.session_state["df_all"] = df_all
        st.session_state["results_ready"] = True


# =========================
# 6. Affichage des résultats
# =========================

if st.session_state["results_ready"] and st.session_state["df_pois"] is not None:
    df_pois = st.session_state["df_pois"]
    df_all = st.session_state["df_all"]
    lat, lon = st.session_state["zone_center"]

    st.success(f"Zone analysée centrée sur lat = {lat:.5f}, lon = {lon:.5f}")

    st.subheader("📍 Points d'intérêt identifiés")
    st.dataframe(df_pois[["name", "type", "lat", "lon"]])

    # Carte des POI
    st.markdown("### 🗺️ Carte des POI de la zone")
    df_map = df_pois.rename(columns={"lat": "latitude", "lon": "longitude"})
    st.map(df_map, zoom=13)

    st.subheader("📊 Séries journalières")

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

        csv = df_zone.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Télécharger la moyenne journalière (CSV)",
            data=csv,
            file_name="footfall_zone_daily_mean.csv",
            mime="text/csv"
        )
else:
    st.info("Configure la zone dans la barre latérale puis clique sur **🚀 Lancer / mettre à jour l'analyse**.")
