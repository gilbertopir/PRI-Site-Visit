import streamlit as st
import ezdxf
import pyproj
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pathlib import Path
import math

st.set_page_config(
    page_title="PR304 Chainage Finder",
    layout="centered"
)

st.title("📍 PR304 Chainage Finder")
st.caption("Chainage ↔ GPS converter using PR304.dxf")

DXF_PATH = Path(__file__).parent / "PR304.dxf"


# -----------------------------
# Helpers
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
    return sum(
        math.dist(points[i - 1], points[i])
        for i in range(1, len(points))
    )


def cumulative_distances(points):
    d = [0.0]
    for i in range(1, len(points)):
        d.append(d[-1] + math.dist(points[i - 1], points[i]))
    return d


def point_at_chainage(points, cum_dist, chainage):
    total = cum_dist[-1]
    target = max(0.0, min(chainage, total))

    if target >= total:
        return points[-1]

    idx = np.searchsorted(cum_dist, target) - 1
    idx = max(0, min(idx, len(points) - 2))

    seg_len = cum_dist[idx + 1] - cum_dist[idx]

    if seg_len == 0:
        return points[idx]

    frac = (target - cum_dist[idx]) / seg_len

    x1, y1 = points[idx]
    x2, y2 = points[idx + 1]

    x = x1 + (x2 - x1) * frac
    y = y1 + (y2 - y1) * frac

    return x, y


def chainage_from_xy(points, cum_dist, x, y):
    best = None

    px, py = x, y

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        dx = x2 - x1
        dy = y2 - y1

        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq == 0:
            continue

        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        distance = math.dist((px, py), (proj_x, proj_y))
        seg_len = math.sqrt(seg_len_sq)
        chainage = cum_dist[i] + t * seg_len

        if best is None or distance < best["distance"]:
            best = {
                "chainage": chainage,
                "easting": proj_x,
                "northing": proj_y,
                "distance": distance
            }

    return best


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


# -----------------------------
# Load alignment at startup
# -----------------------------
if not DXF_PATH.exists():
    st.error("PR304.dxf not found. Put it in the same folder as app.py.")
    st.stop()

alignment = load_alignment_from_dxf(str(DXF_PATH))

if alignment is None:
    st.error("Could not load a valid alignment from PR304.dxf.")
    st.stop()

points = alignment["points"]
cum_dist = alignment["cum_dist"]
total = alignment["total_length"]

osgb_to_gps = pyproj.Transformer.from_crs(
    "EPSG:27700",
    "EPSG:4326",
    always_xy=True
)

gps_to_osgb = pyproj.Transformer.from_crs(
    "EPSG:4326",
    "EPSG:27700",
    always_xy=True
)


# -----------------------------
# Alignment info
# -----------------------------
with st.expander("Alignment info", expanded=False):
    st.write("File: `PR304.dxf`")
    st.write(f"Layer: `{alignment['layer']}`")
    st.write(f"Entity: `{alignment['entity']}`")
    st.write(f"Points: `{len(points)}`")
    st.write(f"Length: `{total:.3f} m`")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs([
    "Chainage → GPS",
    "GPS → Chainage"
])


# -----------------------------
# Chainage to GPS
# -----------------------------
with tab1:
    st.subheader("Chainage → GPS")

    chainage = st.number_input(
        "Enter chainage in metres",
        min_value=0.0,
        max_value=float(total),
        value=0.0,
        step=1.0,
        format="%.3f",
        key="chainage_input"
    )

    if st.button(
        "Get GPS Location",
        type="primary",
        use_container_width=True
    ):
        east, north = point_at_chainage(points, cum_dist, chainage)
        lon, lat = osgb_to_gps.transform(east, north)

        st.session_state["last_result"] = {
            "Chainage": round(chainage, 3),
            "Easting": round(east, 3),
            "Northing": round(north, 3),
            "Latitude": round(lat, 8),
            "Longitude": round(lon, 8)
        }

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]

        st.metric("Latitude", res["Latitude"])
        st.metric("Longitude", res["Longitude"])

        google_maps_url = (
            f"https://www.google.com/maps/search/?api=1"
            f"&query={res['Latitude']},{res['Longitude']}"
        )

        st.link_button(
            "Open in Google Maps",
            google_maps_url,
            use_container_width=True
        )

        df = pd.DataFrame([res])
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            f"chainage_{res['Chainage']}.csv",
            "text/csv",
            use_container_width=True
        )

        st.subheader("Map View")

        m = folium.Map(
            location=[res["Latitude"], res["Longitude"]],
            zoom_start=17
        )

        line_coords = [
            osgb_to_gps.transform(x, y)[::-1]
            for x, y in points
        ]

        folium.PolyLine(
            line_coords,
            color="blue",
            weight=5,
            opacity=0.8
        ).add_to(m)

        folium.Marker(
            [res["Latitude"], res["Longitude"]],
            popup=f"Chainage: {res['Chainage']} m",
            icon=folium.Icon(color="red")
        ).add_to(m)

        st_folium(m, width=None, height=500)


# -----------------------------
# GPS to Chainage
# -----------------------------
with tab2:
    st.subheader("GPS → Chainage")

    lat_input = st.number_input(
        "Latitude",
        value=53.00000000,
        format="%.8f",
        key="lat_input"
    )

    lon_input = st.number_input(
        "Longitude",
        value=-2.00000000,
        format="%.8f",
        key="lon_input"
    )

    if st.button(
        "Get Chainage from GPS",
        type="primary",
        use_container_width=True
    ):
        x, y = gps_to_osgb.transform(lon_input, lat_input)

        reverse_result = chainage_from_xy(points, cum_dist, x, y)

        if reverse_result:
            proj_lon, proj_lat = osgb_to_gps.transform(
                reverse_result["easting"],
                reverse_result["northing"]
            )

            st.session_state["reverse_result"] = {
                "Input Latitude": round(lat_input, 8),
                "Input Longitude": round(lon_input, 8),
                "Chainage": round(reverse_result["chainage"], 3),
                "Projected Latitude": round(proj_lat, 8),
                "Projected Longitude": round(proj_lon, 8),
                "Projected Easting": round(reverse_result["easting"], 3),
                "Projected Northing": round(reverse_result["northing"], 3),
                "Distance from Alignment": round(reverse_result["distance"], 3)
            }

    if "reverse_result" in st.session_state:
        rev = st.session_state["reverse_result"]

        st.metric("Chainage", f"{rev['Chainage']} m")
        st.metric("Distance from Alignment", f"{rev['Distance from Alignment']} m")

        projected_google_maps_url = (
            f"https://www.google.com/maps/search/?api=1"
            f"&query={rev['Projected Latitude']},{rev['Projected Longitude']}"
        )

        st.link_button(
            "Open Projected Point in Google Maps",
            projected_google_maps_url,
            use_container_width=True
        )

        reverse_df = pd.DataFrame([rev])
        st.dataframe(reverse_df, use_container_width=True, hide_index=True)

        csv = reverse_df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            f"gps_to_chainage_{rev['Chainage']}.csv",
            "text/csv",
            use_container_width=True
        )

        st.subheader("Map View")

        m = folium.Map(
            location=[rev["Projected Latitude"], rev["Projected Longitude"]],
            zoom_start=17
        )

        line_coords = [
            osgb_to_gps.transform(x, y)[::-1]
            for x, y in points
        ]

        folium.PolyLine(
            line_coords,
            color="blue",
            weight=5,
            opacity=0.8
        ).add_to(m)

        folium.Marker(
            [rev["Input Latitude"], rev["Input Longitude"]],
            popup="Input GPS point",
            icon=folium.Icon(color="green")
        ).add_to(m)

        folium.Marker(
            [rev["Projected Latitude"], rev["Projected Longitude"]],
            popup=f"Projected chainage: {rev['Chainage']} m",
            icon=folium.Icon(color="red")
        ).add_to(m)

        folium.PolyLine(
            [
                [rev["Input Latitude"], rev["Input Longitude"]],
                [rev["Projected Latitude"], rev["Projected Longitude"]]
            ],
            color="orange",
            weight=3,
            opacity=0.8
        ).add_to(m)

        st_folium(m, width=None, height=500)


st.caption("v0.7 • Auto-loads PR304.dxf • Chainage ↔ GPS")