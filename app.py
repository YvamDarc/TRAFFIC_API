import math
from datetime import date

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
    page_title="Trafic routier (TMJA) & Sentinel",
    layout="wide",
)

# URL par défaut d'un CSV TMJA national (à adapter si besoin).
# Tu peux la surcharger dans la barre latérale.
TMJA_CSV_URL_DEFAULT = ""

# =====================================================
# 1. Fonctions utilitaires
# =====================================================

@st.cache_data(show_spinner=True)
def load_tmja(tmja_csv_url: str) -> pd.DataFrame:
    """
    Charge les données TMJA depuis un CSV hébergé (data.gouv ou autre).

    - tmja_csv_url : URL directe du CSV à lire
    - essaie ; puis , comme séparateur
    - normalise les colonnes anneeMesureTrafic / TMJA / RatioPL si présentes
    """
    if not tmja_csv_url:
        st.error("Aucune URL CSV TMJA fournie. Renseigne-la dans la barre latérale.")
        return pd.DataFrame()

    last_error = None

    for sep in [";", ","]:
        try:
            df = pd.read_csv(tmja_csv_url, sep=sep, low_memory=False)
        except Exception as e:
            last_error = e
            continue

        # Normalisation des colonnes
        # Année de mesure
        annee_col = None
        for col in df.columns:
            if col.lower() in ["anneemesuretrafic", "annee", "annee_mesure_trafic"]:
                annee_col = col
                break

        if annee_col is None:
            last_error = f"Colonne année non trouvée dans {df.columns.tolist()}"
            continue

        df["anneeMesureTrafic"] = pd.to_numeric(df[annee_col], errors="coerce")

        # TMJA
        tmja_col = None
        for col in df.columns:
            if col.lower() in ["tmja", "tmja_mj"]:
                tmja_col = col
                break

        if tmja_col is None:
            last_error = f"Colonne TMJA non trouvée dans {df.columns.tolist()}"
            continue

        df["TMJA"] = pd.to_numeric(df[tmja_col], errors="coerce")

        # Ratio PL (optionnel)
        ratio_col = None
        for col in df.columns:
            if col.lower() in ["ratiopl", "ratio_pl"]:
                ratio_col = col
                break
        if ratio_col is not None:
            df["RatioPL"] = pd.to_numeric(df[ratio_col], errors="coerce")

        df = df.dropna(subset=["anneeMesureTrafic", "TMJA"])
        df["anneeMesureTrafic"] = df["anneeMesureTrafic"].astype(int)

        # Normalisation route / département si elles existent
        for col in df.columns:
            if col.lower() == "route":
                df["route"] = df[col].astype(str)
            if col.lower() in ["deppr", "departement", "dep"]:
                df["depPr"] = df[col].astype(str)

        return df

    # Si tous les essais échouent
    st.error(f"Impossible de lire le CSV TMJA : {last_error}")
    return pd.DataFrame()


def build_tmja_history(df_tmja: pd.DataFrame, text_filter: str | None = None):
    """
    Applique un filtre texte (route, département, libellés) puis
    calcule l'historique TMJA moyen par année.
    """
    df = df_tmja.copy()

    if text_filter:
        text_filter = text_filter.strip()
        if text_filter:
            mask = False
            # Route
            if "route" in df.columns:
                mask |= df["route"].astype(str).str.contains(text_filter, case=False, na=False)
            # Département
            if "depPr" in df.columns:
                mask |= df["depPr"].astype(str).str.contains(text_filter, case=False, na=False)
            # Libellés éventuels
            for col in df.columns:
                cl = col.lower()
                if cl.startswith("nom") or cl.startswith("lib") or "section" in cl:
                    mask |= df[col].astype(str).str.contains(text_filter, case=False, na=False)

            df = df[mask]

    if df.empty:
        return df, pd.DataFrame()

    df_hist = (
        df.groupby("anneeMesureTrafic", as_index=False)["TMJA"]
        .mean()
        .rename(columns={"TMJA": "TMJA_moyen"})
        .sort_values("anneeMesureTrafic")
    )

    return df, df_hist


def bbox_from_point(lat: float, lon: float, km: float = 5.0):
    """
    BBOX simple autour d'un point (lat, lon) pour une fenêtre WMS.
    """
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
    Construit une URL WMS GetMap pour une image Sentinel-2 (ou autre) en EPSG:4326.
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

    if "?" in base_url:
        url = base_url.split("?")[0]
    else:
        url = base_url

    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{url}?{query}"


# =====================================================
# 2. Interface utilisateur
# =====================================================

st.title("🚗 Trafic routier (TMJA) & 🛰 Sentinel-2")

st.write(
    """
Cette application permet de :

- explorer l'**historique de trafic routier** à partir de données TMJA (trafic moyen journalier annuel),
- visualiser en parallèle une **vue satellite Sentinel-2** autour d’un point cliqué sur la carte.

🔎 TMJA = nombre moyen de véhicules/jour sur une année, sur un tronçon donné.
"""
)

# -----------------------------------------------------
# 2.1 Paramètres globaux (sidebar)
# -----------------------------------------------------

st.sidebar.header("⚙️ Paramètres globaux")

tmja_csv_url = st.sidebar.text_input(
    "URL du CSV TMJA (OpenData / data.gouv)",
    value=TMJA_CSV_URL_DEFAULT,
    help="Colle ici l'URL directe d'un CSV de trafic (TMJA) – par exemple une ressource data.gouv.",
)

# -----------------------------------------------------
# 2.2 Carte cliquable pour le contexte
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
    st.info("Clique sur la carte pour positionner ta zone d’intérêt (utile pour la partie Sentinel).")

# =====================================================
# 3. Trafic routier – TMJA
# =====================================================

st.subheader("📊 Trafic routier – TMJA (Open Data)")

with st.spinner("Chargement des données TMJA depuis le CSV..."):
    df_tmja = load_tmja(tmja_csv_url)

if df_tmja.empty:
    st.warning("Aucune donnée TMJA chargée. Vérifie l'URL CSV dans la barre latérale.")
else:
    st.caption(
        "TMJA = Trafic moyen journalier annuel. Les colonnes exactes dépendent du fichier CSV que tu fournis."
    )

    with st.expander("🔍 Filtres sur le trafic (route, département, etc.)", expanded=True):
        # Filtre texte global
        filter_text = st.text_input(
            "Filtre texte (route, département, libellé de section, etc.)",
            placeholder="ex : A84, N12, 22, Ille-et-Vilaine...",
        )

        # Plage d'années
        years_available = sorted(df_tmja["anneeMesureTrafic"].dropna().unique())
        year_min, year_max = int(min(years_available)), int(max(years_available))

        year_range = st.slider(
            "Plage d'années à considérer pour l'historique",
            min_value=year_min,
            max_value=year_max,
            value=(max(year_min, year_max - 10), year_max),
            step=1,
        )

        df_filtered, df_hist = build_tmja_history(df_tmja, filter_text)

        df_filtered = df_filtered[
            (df_filtered["anneeMesureTrafic"] >= year_range[0])
            & (df_filtered["anneeMesureTrafic"] <= year_range[1])
        ]
        if not df_hist.empty:
            df_hist = df_hist[
                (df_hist["anneeMesureTrafic"] >= year_range[0])
                & (df_hist["anneeMesureTrafic"] <= year_range[1])
            ]

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown("### 🧾 Aperçu des sections filtrées")

        if df_filtered.empty:
            st.warning("Aucune section ne correspond à ce filtre et à cette plage d'années.")
        else:
            # Sélection de quelques colonnes intéressantes si dispo
            possible_cols = ["anneeMesureTrafic", "route", "depPr", "TMJA", "RatioPL"]
            cols_show = [c for c in possible_cols if c in df_filtered.columns]
            if not cols_show:
                cols_show = df_filtered.columns[:8].tolist()

            st.dataframe(df_filtered[cols_show].head(500))

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
                - Chaque point représente le **TMJA moyen** sur l'ensemble des sections retenues pour l'année.  
                - Ce n'est pas un point unique mais une **moyenne spatiale** sur la sélection (route, département, etc.).  
                - TMJA = nombre moyen de véhicules/jour sur l'année.
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
# 4. Sentinel-2 en parallèle
# =====================================================

st.subheader("🛰 Vue Sentinel-2 (WMS) en parallèle")

st.markdown(
    """
Pour superposer une **vue satellite Sentinel-2** autour du point cliqué :

1. Récupère une URL de service **WMS** (Copernicus, Sentinel Hub, etc.).  
2. Renseigne l'URL et le layer.  
3. Choisis un rayon (km) et éventuellement une date (paramètre TIME).  
4. L'app génère une requête WMS GetMap et affiche l'image.
"""
)

with st.expander("⚙️ Paramètres Sentinel-2 (WMS)", expanded=True):
    sentinel_wms_url = st.text_input(
        "URL du service WMS Sentinel-2",
        value="",
        help="Exemple : URL WMS Copernicus / Sentinel Hub. Laisse vide si tu n'en as pas encore.",
    )
    sentinel_layer = st.text_input(
        "Nom du layer WMS",
        value="",
        help="Exemple : TRUE_COLOR, NDVI, SENTINEL2_L2A, etc.",
    )
    sentinel_radius_km = st.slider(
        "Rayon de la fenêtre autour du point (km)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    today = date.today()
    sentinel_date = st.date_input(
        "Date d'observation (TIME WMS – facultatif)",
        value=today,
        help="Utilisé seulement si ton service WMS supporte le paramètre TIME.",
    )
    use_time = st.checkbox("Inclure le paramètre TIME dans la requête WMS", value=False)

if not sentinel_wms_url or not sentinel_layer:
    st.info("Renseigne l'URL WMS et le nom du layer pour afficher une image Sentinel-2.")
else:
    # Centre = point cliqué si dispo, sinon centre France
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
        st.image(resp_img.content, caption=f"Sentinel-2 autour de lat={s_lat:.4f}, lon={s_lon:.4f}")
        with st.expander("URL WMS GetMap générée"):
            st.code(wms_url, language="text")
    except Exception as e:
        st.error(f"Impossible de récupérer l'image WMS : {e}")
        with st.expander("URL WMS générée (pour debug)"):
            st.code(wms_url, language="text")
