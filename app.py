import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
Appli **généraliste** : tu définis une zone (adresse ou coordonnées),  
on récupère les **points d'intérêt significatifs** (OSM) dans le rayon,  
puis on construit une **série quotidienne de flux** par POI et une **moyenne** sur la zone.

⚠️ Pour l'instant, les flux sont **simulés**.  
Il suffira de remplacer la fonction `get_daily_footfall_for_poi` par ta vraie API.
"""
)

# --- Carte "Bretagne" générale en haut ---
st.subheader("🗺️ Carte générale – Bretagne")
bzh_cities = pd.DataFrame(
    [
        {"ville": "Rennes", "lat": 48.1173, "lon": -1.6778},
        {"ville": "Brest", "lat": 48.3904, "lon": -4.4861},
        {"ville": "Quimper", "lat": 47.9959, "lon": -4.1023},
        {"ville": "Lorient", "lat": 47.7486, "lon": -3.3664},
        {"ville": "Vannes", "lat": 47.6582, "lon": -2.7608},
        {"ville": "Saint-Brieuc", "lat": 48
