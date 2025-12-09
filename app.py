import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

import folium
from streamlit_folium import st_folium
from populartimes import get as pt_get


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


def bbox_from_center(lat: float, lon: float, radius_m: int):
    """
    Calcule un rectangle englobant (bbox) autour d'un centre (lat, lon)
    à partir d'un rayon en mètres.

    Retourne (southwest_lat, southwest_lon), (northeast_lat, northeast_lon)
    """
    # 1° de latitude ≈ 111 km
    delta_lat = radius_m / 111_000.0

    # 1° de longitude ≈ 111 km * cos(latitude)
    delta_lon = radius_m / (111_000.0 * math.cos(math.radians(lat)))

    sw = (lat - delta_lat, lon - delta_lon)
    ne = (lat + delta_lat, lon + delta_lon)
    return sw, ne


# =========================================
# 2. Fournisseur de données : Popular Times
# =========================================

GOOGLE_POI_TYPES = [
    "store",
    "shopping_mall",
    "supermarket",
    "grocery_or_supermarket",
    "department_store",
    "clothing_store",
    "bakery",
    "restaurant",
    "cafe",
    "bar",
    "movie_theater"
]


def fetch_places_populartimes(api_key: str, lat: float, lon: float, radius_m: int, max_pois: int):
    """
    Appelle Google Popular Times via la librairie `populartimes.get`
    en utilisant une bbox calculée autour du centre.

    Retourne :
    - places : liste brute renvoyée par populartimes
    - df_pois : DataFrame des points d'intérêt (un par établissement)
    """
    sw, ne = bbox_from_center(lat, lon, radius_m)

    # La lib travaille sur une bbox (southwest, northeast)
    # southwest = (lat_min, lon_min), northeast = (lat_max, lon_max)
    sw_lat, sw_lon = sw
    ne_lat, ne_lon = ne

    # Appel à PopularTimes
    places = pt_get(
        api_key,
        GOOGLE_POI_TYPES,
        (sw_lat, sw_lon),
        (ne_lat, ne_lon)
    )

    if not places:
        return [], pd.DataFrame()

    pois = []
    for p in places:
        coord = p.get("coordinates", {})
        pois.append({
            "place_id": p.get("id"),
            "name": p.get("name"),
            "types": ", ".join(p.get("types", [])),
            "lat": coord.get("lat"),
            "lon": coord.get("lng"),
        })

    df_pois = pd.DataFrame(pois).dropna(subset=["lat", "lon"])

    # On limite le nombre de POI
    if not df_pois.empty and len(df_pois) > max_pois:
        df_pois = df_pois.head(max_pois)

    # On filtre aussi la liste brute `places` pour ne garder que ceux des df_pois
    place_ids_kept = set(df_pois["place_id"].tolist())
    places_filtered = [p for p in places if p.get("id") in place_ids_kept]

    return places_filtered, df_pois


def build_daily_series_from_populartimes(place_data: dict, start_date, end_date):
    """
    Transforme le profil hebdomadaire PopularTimes en série journalière
    sur la période [start_date, end_date].

    PopularTimes renvoie, pour chaque jour de la semaine, 24 valeurs (0–100)
    => on agrège par jour : somme des 24 heures = indice de flux quotidien.
    """
    rng = pd.date_range(start_date, end_date, freq="D")

    pop_week = place_data.get("populartimes", [])
    if not pop_week or len(pop_week) != 7:
        # Si données absentes, on renvoie une série à 0
        return pd.DataFrame({
            "date": rng,
            "footfall": np.zeros(len(rng), dtype=float)
        })

    # pop_week est une liste de 7 dicts : [{"name": "Monday", "data": [...]}, ...]
    # On les range dans l'ordre Monday (0) → Sunday (6)
    # En principe l'ordre est déjà bon, mais on sécurise
    day_name_to_index = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    daily_pattern = [0.0] * 7
    for d in pop_week:
        name = d.get("name")
        data = d.get("data", [])
        if name in day_name_to_index and len(data) == 24:
            idx = day_name_to_index[name]
            # indice de flux quotidien = somme des 24 heures
            daily_pattern[idx] = float(sum(data))

    # Construction de la série
    values = []
    for d in rng:
        idx = d.weekday()  # Monday=0
        values.append(daily_pattern[idx])

    df = pd.DataFrame({"date": rng, "footfall": values})
    return df


# =========================
# 3. App Streamlit
# =========================

st.set_page_config(
    page_title="Analyse de flux - Popular Times",
    layout="wide"
)

st.title("📈 Analyse de flux de personnes par zone – données Google Popular Times")

st.write(
    """
Cette application estime la **fréquentation quotidienne** d'une zone en s'appuyant sur :

- les établissements présents autour d'un point (données Google Maps),
- leurs profils d'**affluence moyenne horaire** (*Popular Times*),
- une agrégation en **indice de flux quotidien** sur la période choisie.

🔍 **Important** : Popular Times ne fournit pas un historique jour par jour,
mais un **profil moyen par jour de semaine**.  
La série produite ici est donc un **profil moyen journalier répété sur la période**,
et non la réalité exacte de chaque date.
"""
)

# ---- Clé API Google ----
st.sidebar.header("🔑 Connexion Google")
google_api_key = st.sidebar.text_input(
    "Clé API Google Maps / Places (obligatoire)",
    type="password",
    help="Clé liée à un projet Google Cloud avec accès à l'API Places."
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
    if not google_api_key:
        st.error("Merci de renseigner une clé API Google valide dans la barre latérale.")
        st.stop()

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

    # 1) Récupération des lieux + Popular Times
    with st.spinner("Récupération des établissements et de leurs profils Popular Times…"):
        try:
            places, df_pois = fetch_places_populartimes(
                google_api_key,
                lat,
                lon,
                radius_m=radius_m,
                max_pois=max_pois
            )
        except Exception as e:
            st.error(f"Erreur lors de l'appel Popular Times : {e}")
            st.session_state["results_ready"] = False
            st.stop()

    if df_pois.empty:
        st.warning("Aucun établissement avec données Popular Times trouvé dans ce rayon.")
        st.session_state["results_ready"] = False
    else:
        # 2) Construction des séries journalières pour chaque établissement
        all_series = []
        progress = st.progress(0)
        total = len(places)

        places_by_id = {p.get("id"): p for p in places}

        for i, (_, poi) in enumerate(df_pois.iterrows(), start=1):
            place_id = poi["place_id"]
            pdata = places_by_id.get(place_id)
            if not pdata:
                continue

            df_ts = build_daily_series_from_populartimes(pdata, start_date, end_date)
            df_ts["poi_name"] = poi["name"]
            df_ts["poi_type"] = poi["types"]
            df_ts["place_id"] = place_id
            all_series.append(df_ts)

            progress.progress(i / total)

        if not all_series:
            st.warning("Impossible de construire des séries à partir des données Popular Times disponibles.")
            st.session_state["results_ready"] = False
        else:
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

    st.subheader("📍 Établissements pris en compte (Google Places)")
    st.dataframe(df_pois[["name", "types", "lat", "lon"]])

    # Carte des POI
    st.markdown("### 🗺️ Carte des établissements de la zone")
    df_map = df_pois.rename(columns={"lat": "latitude", "lon": "longitude"})
    st.map(df_map, zoom=13)

    st.subheader("📊 Séries journalières (indice de flux)")

    tab1, tab2 = st.tabs(["Détail par établissement", "Moyenne de la zone"])

    with tab1:
        st.markdown("### 📌 Détail par établissement (indice basé sur Popular Times)")
        poi_selected = st.selectbox("Choisir un établissement", df_pois["name"].tolist())
        df_one = df_all[df_all["poi_name"] == poi_selected].copy()
        df_one = df_one.sort_values("date")

        st.line_chart(
            df_one.set_index("date")["footfall"],
            height=300
        )
        st.write(df_one[["date", "footfall"]])

    with tab2:
        st.markdown("### 📊 Moyenne journalière de l'indice de flux sur l'ensemble de la zone")

        df_zone = (
            df_all
            .groupby("date", as_index=False)["footfall"]
            .mean()
            .rename(columns={"footfall": "footfall_mean"})
        )

        df_zone = df_zone.sort_values("date")

        # Courbe de moyenne
        st.line_chart(
            df_zone.set_index("date")["footfall_mean"],
            height=300
        )
        st.write(df_zone)

        # 🔹 Préambule sur l'origine et la nature de la donnée
        st.markdown(
            """
            ### ℹ️ Origine et nature de l'indicateur

            - **Origine** : données issues de Google Maps / Popular Times, via un appel API sur les
              établissements présents dans le périmètre étudié.
            - **Ce que compte l'indicateur** :
              - pour chaque établissement, Popular Times fournit un **profil horaire moyen** (0–100)
                par jour de la semaine ;
              - ces profils sont **agrégés par jour** (somme des 24 heures) pour produire un
                **indice quotidien de fréquentation** ;
              - pour la zone, on fait ensuite une **moyenne** de ces indices sur
                l'ensemble des établissements retenus.
            - **Granularité** :
              - 1 point = 1 jour civil,
              - la série est un **profil moyen répété** sur la période, pas un historique réel date par date.
            """
        )

        # 🔹 Bloc statistique de synthèse
        st.markdown("### 📌 Statistiques de synthèse sur la période")

        if len(df_zone) >= 2:
            start_date_series = df_zone["date"].iloc[0]
            end_date_series = df_zone["date"].iloc[-1]
            start_val = float(df_zone["footfall_mean"].iloc[0])
            end_val = float(df_zone["footfall_mean"].iloc[-1])
            avg_val = float(df_zone["footfall_mean"].mean())
            total_flux = float(df_zone["footfall_mean"].sum())
            n_days = int(len(df_zone))

            growth_abs = end_val - start_val
            if start_val > 0:
                growth_pct = (end_val / start_val - 1) * 100
            else:
                growth_pct = None

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Indice moyen quotidien de flux",
                f"{avg_val:,.0f}",
                help="Moyenne de l'indice quotidien de fréquentation (Popular Times agrégé) sur la période."
            )

            col2.metric(
                "Indice cumulé de flux sur la période",
                f"{total_flux:,.0f}",
                help="Somme des indices quotidiens de fréquentation (profil moyen répété)."
            )

            if growth_pct is not None:
                col3.metric(
                    "Croissance apparente sur la période",
                    f"{growth_pct:,.1f} %",
                    delta=f"{growth_abs:,.0f}",
                    help=(
                        "Variation entre le premier et le dernier jour de la période, "
                        "en % et en niveau absolu, sur la base du profil moyen."
                    )
                )
            else:
                col3.metric(
                    "Croissance sur la période",
                    "n.c.",
                    help="Non calculable car la valeur de départ est nulle ou manquante."
                )

            st.caption(
                f"Période analysée : du {start_date_series.date()} au {end_date_series.date()} "
                f"({n_days} jours)."
            )
        else:
            st.info("La période sélectionnée est trop courte pour calculer une croissance (au moins 2 jours nécessaires).")

        # Export CSV
        csv = df_zone.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Télécharger la série moyenne journalière (CSV)",
            data=csv,
            file_name="footfall_zone_daily_mean_populartimes.csv",
            mime="text/csv"
        )
else:
    st.info("Configure la zone + la clé API dans la barre latérale puis clique sur **🚀 Lancer / mettre à jour l'analyse**.")
