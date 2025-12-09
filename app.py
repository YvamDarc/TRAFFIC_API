import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import streamlit as st

import folium
from streamlit_folium import st_folium

# =====================================================
# 0. Config générale
# =====================================================

st.set_page_config(
    page_title="Trafic routier & Sentinel - France",
    layout="wide",
)

DATA_GOUV_TMJA_DATASET_ID = "dbec4f42-b5fc-429f-b913-eeb758777383"
DATA_GOUV_TMJA_DATASET_API = f"https://www.data.gouv.fr/api/1/datasets/{DATA_GOUV_TMJA_DATASET_ID}/"

# Pour limiter la taille en Streamlit Cloud : on ne charge que les X CSV les plus récents
MAX_TMJA_CSV = 10  # tu peux monter à 32 si tu veux TOUT l'historique

# =====================================================
# 1. Utilitaires
# =====================================================

@st.cache_data(show_spinner=True)
def load_tmja_from_datagouv(max_csv: int = MAX_TMJA_CSV) -> pd.DataFrame:
    """
    Charge les données TMJA depuis l'API data.gouv.fr pour le dataset national.
    Fusionne plusieurs fichiers CSV (les plus récents) en un seul DataFrame.

    Résultat attendu :
    - colonnes principales : anneeMesureTrafic, TMJA, RatioPL, route, depPr, ...
    """
    resp = requests.get(DATA_GOUV_TMJA_DATASET_API, timeout=30)
    resp.raise_for_status()
    dataset = resp.json()

    resources = dataset.get("resources", [])
    # On garde uniquement les CSV
    csv_resources = [r for r in resources if r.get("format", "").lower() == "csv"]

    # On trie par date de mise à jour décroissante (les plus récents d'abord)
    def _res_sort_key(r):
        return r.get("last_modified") or r.get("created") or ""

    csv_resources = sorted(csv_resources, key=_res_sort_key, reverse=True)
    csv_resources = csv_resources[:max_csv]

    dfs = []
    for r in csv_resources:
        url = r.get("url")
        if not url:
            continue
        try:
            # On laisse pandas deviner le séparateur (souvent ; ou ,)
            df = pd.read_csv(url, low_memory=False)
            # Normalisation de certaines colonnes clés si elles existent
            for col in ["anneeMesureTrafic", "annee_mesure_trafic"]:
                if col in df.columns:
                    df["anneeMesureTrafic"] = df[col]
                    break

            for col in ["TMJA", "tmja"]:
                if col in df.columns:
                    df["TMJA"] = pd.to_numeric(df[col], errors="coerce")
                    break

            for col in ["RatioPL", "ratio_pl"]:
                if col in df.columns:
                    df["RatioPL"] = pd.to_numeric(df[col], errors="coerce")
                    break

            if "anneeMesureTrafic" not in df.columns or "TMJA" not in df.columns:
                continue

            dfs.append(df)
        except Exception as e:
            # On logue dans la console, mais on ne casse pas tout
            print(f"Erreur lecture TMJA {url}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # On force anneeMesureTrafic en string puis en int si possible
    df_all["anneeMesureTrafic"] = pd.to_numeric(df_all["anneeMesureTrafic"], errors="coerce")
    df_all = df_all.dropna(subset=["anneeMesureTrafic", "TMJA"])
    df_all["anneeMesureTrafic"] = df_all["anneeMesureTrafic"].astype(int)

    # Petite normalisation : route et depPr en string
    if "route" in df_all.columns:
        df_all["route"] = df_all["route"].astype(str)
    if "depPr" in df_all.columns:
        df_all["depPr"] = df_all["depPr"].astype(str)

    return df_all


def build_tmja_history(df_tmja: pd.DataFrame, text_filter: str | None = None):
    """
    Construit la série historique TMJA (moyenne sur les sections filtrées)
    par année.
    """
    df = df_tmja.copy()

    if text_filter:
        text_filter = text_filter.strip()
        if text_filter:
            mask = False
            # Route
            if "route" in df.columns:
                mask |= df["route"].str.contains(text_filter, case=False, na=False)
            # Département de PR
            if "depPr" in df.columns:
                mask |= df["depPr"].str.contains(text_filter, case=False, na=False)
            # Autres colonnes texte fréquentes
            for col in df.columns:
                if col.lower().startswith("nom") or col.lower().startswith("lib"):
                    mask |= df[col].astype(str).str.contains(text_filter, case=False, na=False)

            df = df[mask]

    if df.empty:
        return df, pd.DataFrame()

    # Série historique : TMJA moyen par année
    df_hist = (
        df.groupby("anneeMesureTrafic", as_index=False)["TMJA"]
        .mean()
        .rename(columns={"TMJA": "TMJA_moyen"})
        .sort_values("anneeMesureTrafic")
    )

    return df, df_hist


def bbox_from_point(lat: float, lon: float, km: float = 5.0):
    """
    BBOX simple autour d'un point (lat, lon) en degrés pour WMS Sentinel.
    """
    # 1° latitude ~ 111 km
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def build_wms_getmap_url(
    base_url: str,
    layer: str,
    lat: float,
    lon: float,
    km: float,
    time_str: str | None = None,
    width: int = 512,
    height: int = 512,
):
    """
    Construit une URL WMS GetMap pour une image Sentinel-2 (ou autre WMS),
    en EPSG:4326.
    """
    min_lat, min_lon, max_lat, max_lon = bbox_from_point(lat, lon, km=km)

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": layer,
        "CRS": "EPSG:4326",
        "BBOX": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png",
    }
    if time_str:
        params["TIME"] = time_str

    # Construit l'URL proprement
    if "?" in base_url:
        url = base_url.split("?")[0]
    else:
        url = base_url

    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{url}?{query}"


# =====================================================
# 2. UI principale
# =====================================================

st.title("🚗 Trafic routier (Open Data) & 🛰 Sentinel-2")

st.write(
    """
Cette application croise :

- des **données réelles de trafic routier** issues de l’Open Data national  
  (*jeu “Trafic moyen journalier annuel sur le réseau routier national” – TMJA*),
- une **vue satellite Sentinel-2** affichée via un service WMS (Copernicus, Sentinel Hub, etc.).

L’idée : **situer une zone**, explorer l’évolution du trafic routier sur le réseau national,
et visualiser en parallèle l’occupation du sol via Sentinel.
"""
)

# -----------------------------------------------------
# 2.1 Carte cliquable pour le contexte géographique
# -----------------------------------------------------

st.subheader("🗺️ Sélection de la zone (clic sur carte)")

center_france = [47.0, 2.0]
m = folium.Map(location=center_france, zoom_start=6)

# Session state pour le point choisi
if "picked_lat" not in st.session_state:
    st.session_state["picked_lat"] = None
if "picked_lon" not in st.session_state:
    st.session_state["picked_lon"] = None

if st.session_state["picked_lat"] is not None and st.session_state["picked_lon"] is not None:
    folium.Marker(
        [st.session_state["picked_lat"], st.session_state["picked_lon"]],
        tooltip="Point sélectionné",
        icon=folium.Icon(color="red"),
    ).add_to(m)

map_data = st_folium(m, height=450, width=900, key="map_france")

if map_data and map_data.get("last_clicked"):
    st.session_state["picked_lat"] = map_data["last_clicked"]["lat"]
    st.session_state["picked_lon"] = map_data["last_clicked"]["lng"]

if st.session_state["picked_lat"] is not None:
    st.info(
        f"Point sélectionné : lat = {st.session_state['picked_lat']:.5f}, "
        f"lon = {st.session_state['picked_lon']:.5f}"
    )
else:
    st.info("Clique sur la carte pour positionner ta zone d’intérêt (facultatif mais pratique pour Sentinel).")

# -----------------------------------------------------
# 2.2 Chargement données TMJA
# -----------------------------------------------------

st.subheader("📊 Trafic routier – TMJA (Open Data national)")

with st.spinner("Chargement des données TMJA depuis data.gouv.fr..."):
    df_tmja = load_tmja_from_datagouv()

if df_tmja.empty:
    st.error("Impossible de charger les données TMJA depuis data.gouv.fr. Vérifie ta connexion ou réessaie plus tard.")
    st.stop()

st.caption(
    "Source : jeu de données **“Trafic moyen journalier annuel sur le réseau routier national”**, "
    "Ministère de la Transition écologique (licence ouverte Etalab)."
)

# -----------------------------------------------------
# 2.3 Filtres et exploration TMJA
# -----------------------------------------------------

with st.expander("🔍 Filtres sur le trafic (route, département, etc.)", expanded=True):
    # Filtre texte global
    filter_text = st.text_input(
        "Filtre texte (route, département, nom de section, etc.)",
        placeholder="ex : A84, N12, 22, Ille-et-Vilaine...",
    )

    # Liste des années disponibles
    years_available = sorted(df_tmja["anneeMesureTrafic"].dropna().unique())
    year_min, year_max = int(min(years_available)), int(max(years_available))

    year_range = st.slider(
        "Plage d'années à considérer pour l'historique",
        min_value=year_min,
        max_value=year_max,
        value=(max(year_min, year_max - 10), year_max),
        step=1,
    )

    # Application des filtres
    df_filtered, df_hist = build_tmja_history(df_tmja, filter_text)

    # Filtre de plage d'années
    df_filtered = df_filtered[
        (df_filtered["anneeMesureTrafic"] >= year_range[0])
        & (df_filtered["anneeMesureTrafic"] <= year_range[1])
    ]
    if not df_hist.empty:
        df_hist = df_hist[
            (df_hist["anneeMesureTrafic"] >= year_range[0])
            & (df_hist["anneeMesureTrafic"] <= year_range[1])
        ]

# -----------------------------------------------------
# 2.4 Affichage TMJA + stats
# -----------------------------------------------------

col_left, col_right = st.columns([2, 3])

with col_left:
    st.markdown("### 🧾 Aperçu des sections filtrées")

    if df_filtered.empty:
        st.warning("Aucune section ne correspond à ce filtre et à cette plage d'années.")
    else:
        cols_show = []
        for c in ["anneeMesureTrafic", "route", "depPr", "TMJA", "RatioPL"]:
            if c in df_filtered.columns:
                cols_show.append(c)
        if not cols_show:
            cols_show = df_filtered.columns[:8].tolist()

        st.dataframe(df_filtered[cols_show].head(500))

        # Stats simples sur la TMJA
        tmja_vals = df_filtered["TMJA"].dropna()
        if not tmja_vals.empty:
            st.markdown("### 📌 Statistiques TMJA sur la sélection")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("TMJA moyen", f"{tmja_vals.mean():,.0f}")
            s_col2.metric("TMJA médian", f"{tmja_vals.median():,.0f}")
            s_col3.metric("TMJA max", f"{tmja_vals.max():,.0f}")
        else:
            st.info("Pas de valeurs TMJA exploitables sur cette sélection.")

with col_right:
    st.markdown("### 📈 Historique TMJA moyen (zones filtrées)")

    if df_hist.empty:
        st.info("Aucune série historique disponible pour ces filtres.")
    else:
        df_hist_plot = df_hist.set_index("anneeMesureTrafic")
        st.line_chart(df_hist_plot["TMJA_moyen"])
        st.write(df_hist)

        st.markdown("#### ℹ️ Interprétation")
        st.markdown(
            """
            - Chaque point représente le **TMJA moyen** (tous véhicules) sur l'ensemble
              des sections retenues pour l'année considérée.  
            - Ce n'est pas un comptage unique mais **une moyenne spatiale** sur la sélection (route, département...).  
            - Le TMJA correspond au **trafic moyen journalier annuel** : nombre moyen de véhicules/jour sur l'année. :contentReference[oaicite:1]{index=1}
            """
        )

        # Export CSV
        csv_tmja = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Télécharger l'historique TMJA (CSV)",
            data=csv_tmja,
            file_name="tmja_historique_filtre.csv",
            mime="text/csv",
        )

# =====================================================
# 3. Onglet Sentinel-2 (en parallèle)
# =====================================================

st.subheader("🛰 Vue Sentinel-2 (WMS) en parallèle")

st.markdown(
    """
Pour superposer une **vue satellite Sentinel-2** autour de la zone cliquée :

1. Récupère une URL WMS pour Sentinel-2 (par ex. via le **Copernicus Data Space** ou **Sentinel Hub**). :contentReference[oaicite:2]{index=2}  
2. Renseigne l'URL de service et le nom du **layer**.  
3. Choisis un rayon (en km) et une date (si ton WMS gère le paramètre TIME).  
4. L’application construit une requête **WMS GetMap** et affiche l’image.
"""
)

with st.expander("⚙️ Paramètres Sentinel-2 (WMS)", expanded=True):
    sentinel_wms_url = st.text_input(
        "URL du service WMS Sentinel-2",
        value="",
        help="Exemple : une URL WMS fournie par Copernicus Data Space ou Sentinel Hub.",
    )
    sentinel_layer = st.text_input(
        "Nom du layer WMS",
        value="",
        help="Exemples typiques : TRUE_COLOR, NDVI, SENTINEL2_L2A...",
    )
    sentinel_radius_km = st.slider(
        "Rayon de la fenêtre autour du point (km)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    # Date TIME (facultative)
    today = date.today()
    sentinel_date = st.date_input(
        "Date d'observation (TIME WMS – facultatif)",
        value=today,
        help="Ne sera prise en compte que si le service WMS gère le paramètre TIME.",
    )
    use_time = st.checkbox("Inclure le paramètre TIME dans la requête WMS", value=False)

# Affichage image Sentinel si possible
if not sentinel_wms_url or not sentinel_layer:
    st.info(
        "Renseigne au moins l'URL WMS et le nom du layer pour afficher une image Sentinel-2."
    )
else:
    # On choisit le centre : point cliqué si dispo, sinon centre France
    if st.session_state["picked_lat"] is not None:
        s_lat = st.session_state["picked_lat"]
        s_lon = st.session_state["picked_lon"]
    else:
        s_lat, s_lon = center_france

    time_str = sentinel_date.isoformat() if use_time else None
    wms_url = build_wms_getmap_url(
        base_url=sentinel_wms_url,
        layer=sentinel_layer,
        lat=s_lat,
        lon=s_lon,
        km=sentinel_radius_km,
        time_str=time_str,
    )

    st.markdown("### 🛰 Aperçu Sentinel-2 (image WMS)")

    try:
        resp_img = requests.get(wms_url, timeout=30)
        resp_img.raise_for_status()
        st.image(resp_img.content, caption=f"Sentinel-2 (WMS) autour de lat={s_lat:.4f}, lon={s_lon:.4f}")
        with st.expander("URL WMS GetMap générée"):
            st.code(wms_url, language="text")
    except Exception as e:
        st.error(f"Impossible de récupérer l'image WMS : {e}")
        with st.expander("URL WMS générée (pour debug)"):
            st.code(wms_url, language="text")
