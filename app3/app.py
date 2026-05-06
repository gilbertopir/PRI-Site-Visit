import streamlit as st
import ezdxf
import pyproj
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from pathlib import Path
from datetime import datetime
from io import StringIO
import math
import dropbox
from dropbox.files import WriteMode

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="PR304 Site Tool",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Mobile-friendly CSS
# -----------------------------
st.markdown("""
<style>
    /* Larger touch targets */
    .stButton > button {
        height: 3.2rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    /* Bigger inputs on mobile */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        font-size: 1.05rem !important;
        padding: 0.6rem !important;
    }
    /* Selectbox */
    div[data-testid="stSelectbox"] > div {
        font-size: 1.05rem !important;
    }
    /* Metrics bigger */
    div[data-testid="stMetric"] label {
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem !important;
        padding: 0.4rem 0.5rem !important;
    }
    /* Reduce padding on mobile */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Config
# -----------------------------
DXF_PATH = Path(__file__).parent / "PR304.dxf"

DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
DROPBOX_CSV    = "/PRI-SITE-APP/PR304_points.csv"
DROPBOX_PP_CSV = "/PRI-SITE-APP/PR304_passing_places.csv"

FEATURE_TYPES = [
    "Custom / Other",
    "VRS",
    "Sign faces",
    "Sign posts",
    "Lighting columns",
    "Traffic signals",
    "Bollards",
    "Road markings",
    "Kerbs / edging",
    "Gullies",
    "Manholes / chambers",
    "Drainage ditches",
    "Culverts / headwalls",
    "Utility covers",
    "Utility marker posts",
    "Fencing",
    "Gates",
    "Trees",
    "Hedges",
    "Overhead lines / poles",
    "Cabinets / comms boxes",
    "Bus stops / laybys",
    "Accesses / driveways",
    "Junctions",
    "Existing walls",
    "Retaining walls",
    "Watercourses",
    "Embankments",
    "Cuttings",
    "Verge edges",
    "Pavement defects",
    "Edge break-up",
    "Ponding / drainage issues",
]

CSV_COLUMNS = [
    "timestamp", "feature_type", "side",
    "offset_from_edge_m", "distance_along_edge_m", "condition", "notes",
    "chainage_m", "distance_from_alignment_m",
    "latitude", "longitude",
    "easting", "northing"
]

DISTANCE_WARNING_M = 50  # warn if further than this from alignment

PP_COLUMNS = [
    "id", "timestamp",
    "side", "status",
    "mid_chainage_m",
    "mid_latitude", "mid_longitude",
    "mid_easting", "mid_northing",
    "width_m", "length_m",
    "notes"
]

# -----------------------------
# Geometry helpers
# -----------------------------
def extract_xy_from_entity(e):
    points = []
    if e.dxftype() == "LWPOLYLINE":
        for p in e.get_points():
            points.append((float(p[0]), float(p[1])))
    elif e.dxftype() == "POLYLINE":
        for v in e.vertices:
            loc = v.dxf.location
            points.append((float(loc.x), float(loc.y)))
    elif e.dxftype() == "LINE":
        points.append((float(e.dxf.start.x), float(e.dxf.start.y)))
        points.append((float(e.dxf.end.x), float(e.dxf.end.y)))
    clean = []
    for p in points:
        if not clean or math.dist(clean[-1], p) > 0.001:
            clean.append(p)
    return clean


def line_length(points):
    return sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


def cumulative_distances(points):
    d = [0.0]
    for i in range(1, len(points)):
        d.append(d[-1] + math.dist(points[i - 1], points[i]))
    return np.array(d)


def point_at_chainage(points, cum_dist, chainage):
    total = cum_dist[-1]
    target = max(0.0, min(float(chainage), float(total)))
    if target >= total:
        return points[-1]
    idx = int(np.searchsorted(cum_dist, target)) - 1
    idx = max(0, min(idx, len(points) - 2))
    seg_len = cum_dist[idx + 1] - cum_dist[idx]
    if seg_len == 0:
        return points[idx]
    frac = (target - cum_dist[idx]) / seg_len
    x1, y1 = points[idx]
    x2, y2 = points[idx + 1]
    return (x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac)


def chainage_from_xy(points, cum_dist, x, y):
    best = None
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            continue
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        distance = math.dist((x, y), (proj_x, proj_y))
        chainage = float(cum_dist[i]) + t * math.sqrt(seg_len_sq)
        if best is None or distance < best["distance"]:
            best = {
                "chainage": chainage,
                "easting": proj_x,
                "northing": proj_y,
                "distance": distance
            }
    return best


# -----------------------------
# Data loading (cached)
# -----------------------------
@st.cache_data
def load_alignment_from_dxf(dxf_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    candidates = []
    for e in msp:
        if e.dxftype() in ["LWPOLYLINE", "POLYLINE", "LINE"]:
            pts = extract_xy_from_entity(e)
            if len(pts) >= 2:
                candidates.append({
                    "entity": e.dxftype(),
                    "layer": e.dxf.layer,
                    "points": pts,
                    "length": line_length(pts)
                })
    if not candidates:
        return None
    longest = max(candidates, key=lambda x: x["length"])
    return {
        "points": longest["points"],
        "cum_dist": cumulative_distances(longest["points"]),
        "total_length": longest["length"],
        "layer": longest["layer"],
        "entity": longest["entity"]
    }


@st.cache_data
def precompute_gps_line(points_tuple):
    """Pre-compute WGS84 coords for the alignment — done once, cached."""
    t = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    return [t.transform(x, y)[::-1] for x, y in points_tuple]


# -----------------------------
# Map helper
# -----------------------------
def make_map(gps_line, center_lat=None, center_lon=None, zoom=17,
             red_marker=None, green_marker=None, orange_line=None, extra_markers=None):
    if center_lat is not None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    else:
        m = folium.Map(location=gps_line[0], zoom_start=15)

    folium.PolyLine(gps_line, color="#1a6fc4", weight=5, opacity=0.85).add_to(m)

    if red_marker:
        folium.Marker(
            red_marker["pos"],
            popup=red_marker.get("popup", ""),
            icon=folium.Icon(color="red", icon="map-marker")
        ).add_to(m)

    if green_marker:
        folium.Marker(
            green_marker["pos"],
            popup=green_marker.get("popup", ""),
            icon=folium.Icon(color="green", icon="map-marker")
        ).add_to(m)

    if orange_line:
        folium.PolyLine(orange_line, color="orange", weight=3, opacity=0.8, dash_array="6").add_to(m)

    if extra_markers:
        for em in extra_markers:
            folium.Marker(
                [em["lat"], em["lon"]],
                popup=em["popup"],
                icon=folium.Icon(color=em.get("color", "blue"), icon="map-marker")
            ).add_to(m)

    return m


# -----------------------------
# Dropbox helpers
# -----------------------------
def get_dbx():
    return dropbox.Dropbox(DROPBOX_TOKEN)


def save_point_to_dropbox(row: dict):
    import csv, io

    dbx = get_dbx()

    # Try to load existing content
    try:
        _, res = dbx.files_download(DROPBOX_CSV)
        existing = res.content.decode("utf-8")
        write_header = not existing.strip()
    except dropbox.exceptions.ApiError:
        existing = ""
        write_header = True

    # Append new row using csv writer (handles quoting automatically)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()
    writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})

    updated = existing + buf.getvalue()

    dbx.files_upload(
        updated.encode("utf-8"),
        DROPBOX_CSV,
        mode=WriteMode.overwrite
    )


def load_points_from_dropbox():
    """Returns (df, error_message). df is None if load failed."""
    try:
        _, res = get_dbx().files_download(DROPBOX_CSV)
        content = res.content.decode("utf-8")
        df = pd.read_csv(
            StringIO(content),
            on_bad_lines="skip",   # skip any corrupt rows from old format
            quoting=0              # csv.QUOTE_MINIMAL — handles both old and new format
        )
        # Ensure all expected columns exist even if file was written with old schema
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[CSV_COLUMNS], None
    except dropbox.exceptions.AuthError:
        return None, "❌ Dropbox authentication failed — check your token in secrets.toml."
    except dropbox.exceptions.ApiError as e:
        if "not_found" in str(e):
            return pd.DataFrame(columns=CSV_COLUMNS), None
        return None, f"❌ Dropbox API error: {e}"
    except Exception as e:
        return None, f"❌ Unexpected error loading data: {e}"


def reset_dropbox_csv():
    """Overwrite the CSV with just the header row."""
    dbx = get_dbx()
    header = ",".join(CSV_COLUMNS) + "\n"
    dbx.files_upload(
        header.encode("utf-8"),
        DROPBOX_CSV,
        mode=WriteMode.overwrite
    )


# -----------------------------
# Passing Places Dropbox helpers
# -----------------------------
def get_next_pp_id():
    """Read existing PP records and return the next ID e.g. PP004."""
    df, _ = load_pp_from_dropbox()
    if df is None or df.empty or "id" not in df.columns:
        return "PP001"
    existing_ids = df["id"].dropna().astype(str).tolist()
    nums = []
    for id_ in existing_ids:
        try:
            nums.append(int(id_.replace("PP", "")))
        except ValueError:
            pass
    next_num = max(nums) + 1 if nums else 1
    return f"PP{next_num:03d}"


def save_pp_to_dropbox(row: dict):
    import csv, io as _io
    dbx = get_dbx()
    try:
        _, res = dbx.files_download(DROPBOX_PP_CSV)
        existing = res.content.decode("utf-8")
        write_header = not existing.strip()
    except dropbox.exceptions.ApiError:
        existing = ""
        write_header = True

    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=PP_COLUMNS, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()
    writer.writerow({col: row.get(col, "") for col in PP_COLUMNS})

    dbx.files_upload(
        (existing + buf.getvalue()).encode("utf-8"),
        DROPBOX_PP_CSV,
        mode=WriteMode.overwrite
    )


def load_pp_from_dropbox():
    """Returns (df, error_message)."""
    try:
        _, res = get_dbx().files_download(DROPBOX_PP_CSV)
        df = pd.read_csv(StringIO(res.content.decode("utf-8")), on_bad_lines="skip")
        for col in PP_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[PP_COLUMNS], None
    except dropbox.exceptions.AuthError:
        return None, "❌ Dropbox authentication failed."
    except dropbox.exceptions.ApiError as e:
        if "not_found" in str(e):
            return pd.DataFrame(columns=PP_COLUMNS), None
        return None, f"❌ Dropbox API error: {e}"
    except Exception as e:
        return None, f"❌ Unexpected error: {e}"


def reset_pp_csv():
    dbx = get_dbx()
    dbx.files_upload(
        (",".join(PP_COLUMNS) + "\n").encode("utf-8"),
        DROPBOX_PP_CSV,
        mode=WriteMode.overwrite
    )


# -----------------------------
# Distance warning helper
# -----------------------------
def distance_warning(dist_m):
    if dist_m > DISTANCE_WARNING_M:
        st.warning(f"⚠️ You are {dist_m:.1f} m from the alignment — check you're in the right location.")


# -----------------------------
# Startup: load DXF
# -----------------------------
st.title("📍 PR304 Site Tool")

if not DXF_PATH.exists():
    st.error("❌ PR304.dxf not found — place it in the same folder as app.py.")
    st.stop()

alignment = load_alignment_from_dxf(str(DXF_PATH))

if alignment is None:
    st.error("❌ Could not load a valid alignment from PR304.dxf.")
    st.stop()

points   = alignment["points"]
cum_dist = alignment["cum_dist"]
total    = alignment["total_length"]

osgb_to_gps = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
gps_to_osgb = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

gps_line = precompute_gps_line(tuple(points))

# Call geolocation ONCE here — shared across Phone GPS and Capture tabs
phone_location = streamlit_geolocation()

with st.expander("ℹ️ Alignment info", expanded=False):
    st.write(f"**Layer:** `{alignment['layer']}` &nbsp;|&nbsp; **Entity:** `{alignment['entity']}`")
    st.write(f"**Points:** `{len(points)}` &nbsp;|&nbsp; **Length:** `{total:.3f} m`")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📱 Phone GPS",
    "📌 Capture",
    "🔁 Passing Places",
    "📋 View",
    "📐 Ch→GPS",
    "🔄 GPS→Ch",
])


# ==============================
# TAB 5 — Chainage → GPS
# ==============================
with tab5:
    st.subheader("Chainage → GPS")

    ch_val = st.number_input(
        "Chainage (m)",
        min_value=0.0,
        max_value=float(total),
        value=0.0,
        step=1.0,
        format="%.3f",
        key="ch_input"
    )

    if st.button("Get GPS", type="primary", use_container_width=True, key="btn_ch_gps"):
        east, north = point_at_chainage(points, cum_dist, ch_val)
        lon, lat = osgb_to_gps.transform(east, north)
        st.session_state["ch_result"] = {
            "Chainage (m)": round(ch_val, 3),
            "Easting":       round(east, 3),
            "Northing":      round(north, 3),
            "Latitude":      round(lat, 8),
            "Longitude":     round(lon, 8),
        }

    if "ch_result" in st.session_state:
        r = st.session_state["ch_result"]
        c1, c2 = st.columns(2)
        c1.metric("Latitude",  r["Latitude"])
        c2.metric("Longitude", r["Longitude"])
        c1.metric("Easting",   r["Easting"])
        c2.metric("Northing",  r["Northing"])

        st.link_button(
            "📍 Open in Google Maps",
            f"https://www.google.com/maps/search/?api=1&query={r['Latitude']},{r['Longitude']}",
            use_container_width=True
        )

        m = make_map(
            gps_line,
            center_lat=r["Latitude"], center_lon=r["Longitude"],
            red_marker={"pos": [r["Latitude"], r["Longitude"]], "popup": f"Ch: {r['Chainage (m)']} m"}
        )
        st_folium(m, width=None, height=400, returned_objects=[])


# ==============================
# TAB 6 — GPS → Chainage
# ==============================
with tab6:
    st.subheader("GPS → Chainage")

    lat_in = st.number_input("Latitude",  value=53.00000000, format="%.8f", key="lat_in")
    lon_in = st.number_input("Longitude", value=-2.00000000, format="%.8f", key="lon_in")

    if st.button("Get Chainage", type="primary", use_container_width=True, key="btn_gps_ch"):
        ex, ny = gps_to_osgb.transform(lon_in, lat_in)
        res = chainage_from_xy(points, cum_dist, ex, ny)
        if res:
            plon, plat = osgb_to_gps.transform(res["easting"], res["northing"])
            st.session_state["gps_result"] = {
                "Chainage (m)":             round(res["chainage"], 3),
                "Distance from Alignment":  round(res["distance"], 3),
                "Projected Latitude":       round(plat, 8),
                "Projected Longitude":      round(plon, 8),
                "Easting":                  round(res["easting"], 3),
                "Northing":                 round(res["northing"], 3),
                "Input Latitude":           round(lat_in, 8),
                "Input Longitude":          round(lon_in, 8),
            }

    if "gps_result" in st.session_state:
        r = st.session_state["gps_result"]
        distance_warning(r["Distance from Alignment"])
        c1, c2 = st.columns(2)
        c1.metric("Chainage",              f"{r['Chainage (m)']} m")
        c2.metric("Distance from Alignment", f"{r['Distance from Alignment']} m")
        c1.metric("Easting",  r["Easting"])
        c2.metric("Northing", r["Northing"])

        st.link_button(
            "📍 Open Projected Point in Google Maps",
            f"https://www.google.com/maps/search/?api=1&query={r['Projected Latitude']},{r['Projected Longitude']}",
            use_container_width=True
        )

        m = make_map(
            gps_line,
            center_lat=r["Projected Latitude"], center_lon=r["Projected Longitude"],
            red_marker={"pos":   [r["Projected Latitude"], r["Projected Longitude"]], "popup": f"Projected Ch: {r['Chainage (m)']} m"},
            green_marker={"pos": [r["Input Latitude"],     r["Input Longitude"]],     "popup": "Input GPS"},
            orange_line=[[r["Input Latitude"], r["Input Longitude"]],
                         [r["Projected Latitude"], r["Projected Longitude"]]]
        )
        st_folium(m, width=None, height=400, returned_objects=[])


# ==============================
# TAB 1 — Phone GPS
# ==============================
with tab1:
    st.subheader("📱 Phone GPS → Chainage")
    st.info("Allow location access when prompted — the button appears at the top of the page.")

    loc = phone_location

    if loc and loc.get("latitude") is not None:
        lat_p = float(loc["latitude"])
        lon_p = float(loc["longitude"])

        ex, ny = gps_to_osgb.transform(lon_p, lat_p)
        res = chainage_from_xy(points, cum_dist, ex, ny)

        if res:
            plon, plat = osgb_to_gps.transform(res["easting"], res["northing"])
            dist_p = round(res["distance"], 3)

            distance_warning(dist_p)

            c1, c2 = st.columns(2)
            c1.metric("Chainage",              f"{round(res['chainage'], 3)} m")
            c2.metric("Distance from Alignment", f"{dist_p} m")
            c1.metric("Easting",  round(ex, 3))
            c2.metric("Northing", round(ny, 3))
            c1.metric("Latitude",  round(lat_p, 8))
            c2.metric("Longitude", round(lon_p, 8))

            st.link_button(
                "📍 Open in Google Maps",
                f"https://www.google.com/maps/search/?api=1&query={lat_p},{lon_p}",
                use_container_width=True
            )

            m = make_map(
                gps_line,
                center_lat=plat, center_lon=plon,
                red_marker={"pos":   [plat, plon],     "popup": f"Projected Ch: {round(res['chainage'],3)} m"},
                green_marker={"pos": [lat_p, lon_p],   "popup": "Your location"},
                orange_line=[[lat_p, lon_p], [plat, plon]]
            )
            st_folium(m, width=None, height=400, returned_objects=[])
    else:
        st.warning("No GPS received yet — allow location permission when prompted.")


# ==============================
# TAB 2 — Capture Point
# ==============================
with tab2:
    st.subheader("📌 Capture Feature")
    st.info("Stand at the feature — location is captured from the top of the page.")

    loc_cap = phone_location

    if loc_cap and loc_cap.get("latitude") is not None:
        lat_c = float(loc_cap["latitude"])
        lon_c = float(loc_cap["longitude"])

        ex_c, ny_c = gps_to_osgb.transform(lon_c, lat_c)
        res_c = chainage_from_xy(points, cum_dist, ex_c, ny_c)

        if res_c:
            dist_c = round(res_c["distance"], 3)
            ch_c   = round(res_c["chainage"], 3)

            distance_warning(dist_c)

            c1, c2 = st.columns(2)
            c1.metric("Chainage",              f"{ch_c} m")
            c2.metric("Distance from Alignment", f"{dist_c} m")

            st.divider()

            feature = st.selectbox("Feature Type", FEATURE_TYPES, key="cap_feature")

            if feature == "Custom / Other":
                custom = st.text_input("Describe the feature", key="cap_custom")
                feature_label = custom.strip() if custom.strip() else "Custom / Other"
            else:
                feature_label = feature

            side = st.radio(
                "Side of Road",
                options=["LHS", "RHS", "BOTH", "N/A"],
                format_func=lambda x: {
                    "LHS":  "LHS — Left Hand Side",
                    "RHS":  "RHS — Right Hand Side",
                    "BOTH": "BOTH — Both Sides",
                    "N/A":  "N/A — Not Applicable",
                }[x],
                horizontal=True,
                key="cap_side"
            )

            dist_edge = st.number_input(
                "Offset from Edge (m)",
                min_value=0.0,
                max_value=999.9,
                value=0.0,
                step=0.1,
                format="%.1f",
                key="cap_dist_edge",
                help="Perpendicular distance from the road edge to the feature"
            )

            dist_along_edge = st.number_input(
                "Distance Along Edge (m)",
                min_value=0.0,
                max_value=99999.9,
                value=0.0,
                step=0.1,
                format="%.1f",
                key="cap_dist_along_edge",
                help="Distance measured along the road edge to the feature"
            )

            condition = st.radio(
                "Condition",
                options=["GOOD", "FAIR", "POOR", "DAMAGED"],
                horizontal=True,
                key="cap_condition"
            )

            notes = st.text_area("Notes (optional)", key="cap_notes", height=80,
                                 placeholder="e.g. damaged, obscured, requires action...")

            st.divider()

            if st.button("📌 Capture Point", type="primary", use_container_width=True, key="btn_capture"):
                row = {
                    "timestamp":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "feature_type":             feature_label,
                    "side":                     side,
                    "offset_from_edge_m":       round(dist_edge, 1),
                    "distance_along_edge_m":    round(dist_along_edge, 1),
                    "condition":                condition,
                    "notes":                    notes.strip(),
                    "chainage_m":               ch_c,
                    "distance_from_alignment_m": dist_c,
                    "latitude":                 round(lat_c, 8),
                    "longitude":                round(lon_c, 8),
                    "easting":                  round(ex_c, 3),
                    "northing":                 round(ny_c, 3),
                }
                try:
                    save_point_to_dropbox(row)
                    st.success(f"✅ Saved: **{feature_label}** at chainage **{ch_c} m**")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Failed to save to Dropbox: {e}")
    else:
        st.warning("No GPS received yet — allow location permission when prompted.")


# ==============================
# TAB 3 — Passing Places Schedule
# ==============================
with tab3:
    st.subheader("🔁 Passing Places Schedule")

    loc_pp = phone_location

    if loc_pp and loc_pp.get("latitude") is not None:
        lat_pp = float(loc_pp["latitude"])
        lon_pp = float(loc_pp["longitude"])

        ex_pp, ny_pp = gps_to_osgb.transform(lon_pp, lat_pp)
        res_pp = chainage_from_xy(points, cum_dist, ex_pp, ny_pp)

        if res_pp:
            ch_pp   = round(res_pp["chainage"], 3)
            dist_pp = round(res_pp["distance"], 3)

            distance_warning(dist_pp)

            # Generate next ID on page load (not on every rerun)
            if "pp_next_id" not in st.session_state:
                st.session_state["pp_next_id"] = get_next_pp_id()

            st.markdown(f"### ID: `{st.session_state['pp_next_id']}`")

            c1, c2 = st.columns(2)
            c1.metric("Mid Chainage", f"{ch_pp} m")
            c2.metric("Distance from Alignment", f"{dist_pp} m")
            c1.metric("Easting",  round(ex_pp, 3))
            c2.metric("Northing", round(ny_pp, 3))

            st.divider()

            pp_side = st.radio(
                "Side of Road",
                options=["LHS", "RHS"],
                format_func=lambda x: {
                    "LHS": "LHS — Left Hand Side",
                    "RHS": "RHS — Right Hand Side",
                }[x],
                horizontal=True,
                key="pp_side"
            )

            pp_status = st.radio(
                "Status",
                options=["Existing", "New"],
                horizontal=True,
                key="pp_status"
            )

            c1, c2 = st.columns(2)
            pp_width = c1.number_input(
                "Width (m)",
                min_value=0.0,
                max_value=99.9,
                value=0.0,
                step=0.1,
                format="%.1f",
                key="pp_width"
            )
            pp_length = c2.number_input(
                "Length (m)",
                min_value=0.0,
                max_value=9999.9,
                value=0.0,
                step=0.1,
                format="%.1f",
                key="pp_length"
            )

            pp_notes = st.text_area(
                "Notes (optional)",
                key="pp_notes",
                height=80,
                placeholder="e.g. surfaced, grass verge, visibility issues..."
            )

            st.divider()

            if st.button("💾 Save Passing Place", type="primary", use_container_width=True, key="btn_save_pp"):
                pp_row = {
                    "id":               st.session_state["pp_next_id"],
                    "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "side":             pp_side,
                    "status":           pp_status,
                    "mid_chainage_m":   ch_pp,
                    "mid_latitude":     round(lat_pp, 8),
                    "mid_longitude":    round(lon_pp, 8),
                    "mid_easting":      round(ex_pp, 3),
                    "mid_northing":     round(ny_pp, 3),
                    "width_m":          round(pp_width, 1),
                    "length_m":         round(pp_length, 1),
                    "notes":            pp_notes.strip(),
                }
                try:
                    save_pp_to_dropbox(pp_row)
                    st.success(f"✅ Saved: **{st.session_state['pp_next_id']}** at chainage **{ch_pp} m**")
                    del st.session_state["pp_next_id"]  # force new ID on next load
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
    else:
        st.warning("No GPS received yet — allow location permission when prompted.")

    # View saved passing places
    st.divider()
    st.subheader("Saved Passing Places")

    c_ref, c_rst = st.columns(2)
    if c_ref.button("🔄 Refresh", use_container_width=True, key="btn_pp_refresh"):
        if "pp_next_id" in st.session_state:
            del st.session_state["pp_next_id"]
        st.rerun()

    if c_rst.button("🗑️ Reset", use_container_width=True, key="btn_pp_reset"):
        st.session_state["confirm_pp_reset"] = True

    if st.session_state.get("confirm_pp_reset"):
        st.warning("⚠️ This will delete all passing place records. Are you sure?")
        cc1, cc2 = st.columns(2)
        if cc1.button("Yes, delete all", type="primary", use_container_width=True, key="btn_pp_confirm_reset"):
            try:
                reset_pp_csv()
                st.session_state["confirm_pp_reset"] = False
                if "pp_next_id" in st.session_state:
                    del st.session_state["pp_next_id"]
                st.success("Passing places CSV reset.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
        if cc2.button("Cancel", use_container_width=True, key="btn_pp_cancel_reset"):
            st.session_state["confirm_pp_reset"] = False
            st.rerun()

    pp_df, pp_err = load_pp_from_dropbox()
    if pp_err:
        st.error(pp_err)
    elif pp_df is None or pp_df.empty:
        st.info("No passing places recorded yet.")
    else:
        st.write(f"**{len(pp_df)} passing place(s) recorded**")
        st.dataframe(pp_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV",
            pp_df.to_csv(index=False),
            file_name=f"PR304_passing_places_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_pp_download"
        )


# ==============================
# TAB 4 — View Captured Points
# ==============================
with tab4:
    st.subheader("📋 Captured Points")

    col_r, col_d, col_x = st.columns(3)

    with col_r:
        if st.button("🔄 Refresh", use_container_width=True, key="btn_refresh"):
            st.rerun()

    with col_x:
        if st.button("🗑️ Reset CSV", use_container_width=True, key="btn_reset"):
            st.session_state["confirm_reset"] = True

    if st.session_state.get("confirm_reset"):
        st.warning("⚠️ This will delete all captured points. Are you sure?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete all", type="primary", use_container_width=True, key="btn_confirm_reset"):
            try:
                reset_dropbox_csv()
                st.session_state["confirm_reset"] = False
                st.success("CSV reset successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
        if c2.button("Cancel", use_container_width=True, key="btn_cancel_reset"):
            st.session_state["confirm_reset"] = False
            st.rerun()

    df, load_error = load_points_from_dropbox()

    if load_error:
        st.error(load_error)
    elif df is None or df.empty:
        st.info("No points captured yet.")
    else:
        st.write(f"**{len(df)} point(s) captured**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        with col_d:
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False),
                file_name=f"PR304_points_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download"
            )

        valid = df.dropna(subset=["latitude", "longitude"])
        if not valid.empty:
            extra = [
                {
                    "lat":   row["latitude"],
                    "lon":   row["longitude"],
                    "popup": f"{row['feature_type']} | Ch: {row['chainage_m']} m",
                    "color": "red"
                }
                for _, row in valid.iterrows()
            ]
            m = make_map(gps_line, extra_markers=extra)
            st_folium(m, width=None, height=500, returned_objects=[])

st.caption("v1.0 • PR304 Site Tool • Chainage ↔ GPS • Point Capture • Dropbox Sync")
