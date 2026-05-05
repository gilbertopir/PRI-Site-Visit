import streamlit as st
import ezdxf
import pyproj
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
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
    return sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


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

    return (
        x1 + (x2 - x1) * frac,
        y1 + (y2 - y1) * frac
    )


def chainage_from_xy(points, cum_dist, x, y):
    best = None

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq == 0:
            continue

        t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        distance = math.dist((x, y), (proj_x, proj_y))
        chainage = cum_dist[i] + t * math.sqrt(seg_len_sq)

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


def make_alignment_map(points, osgb_to_gps, marker_lat=None, marker_lon=None, popup="Point"):
    if marker_lat is not None and marker_lon is not None:
        m = folium.Map(location=[marker_lat, marker_lon], zoom_start=17)
    else:
        first_lon, first_lat = osgb_to_gps.transform(points[0][0], points[0][1])
        m = folium.Map(location=[first_lat, first_lon], zoom_start=15)

    line_coords = [osgb_to_gps.transform(x, y)[::-1] for x, y in points]

    folium.PolyLine(
        line_coords,
        color="blue",
        weight=5,
        opacity=0.8
    ).add_to(m)

    if marker_lat is not None and marker_lon is not None:
        folium.Marker(
            [marker_lat, marker_lon],
            popup=popup,
            icon=folium.Icon(color="red")
        ).add_to(m)

    return m


# -----------------------------
# Load alignment
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


with st.expander("Alignment info", expanded=False):
    st.write("File: `PR304.dxf`")
    st.write(f"Layer: `{alignment['layer']}`")
    st.write(f"Entity: `{alignment['entity']}`")
    st.write(f"Points: `{len(points)}`")
    st.write(f"Length: `{total:.3f} m`")


tab1, tab2, tab3 = st.tabs([
    "Chainage → GPS",
    "GPS → Chainage",
    "Use Phone GPS"
])


# -----------------------------
# Tab 1: Chainage to GPS
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

    if st.button("Get GPS Location", type="primary", use_container_width=True):
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

        st.link_button(
            "Open in Google Maps",
            f"https://www.google.com/maps/search/?api=1&query={res['Latitude']},{res['Longitude']}",
            use_container_width=True
        )

        st.dataframe(pd.DataFrame([res]), use_container_width=True, hide_index=True)

        m = make_alignment_map(
            points,
            osgb_to_gps,
            res["Latitude"],
            res["Longitude"],
            f"Chainage: {res['Chainage']} m"
        )
        st_folium(m, width=None, height=500)


# -----------------------------
# Tab 2: Manual GPS to Chainage
# -----------------------------
with tab2:
    st.subheader("GPS → Chainage")

    lat_input = st.number_input("Latitude", value=53.00000000, format="%.8f", key="lat_input")
    lon_input = st.number_input("Longitude", value=-2.00000000, format="%.8f", key="lon_input")

    if st.button("Get Chainage from GPS", type="primary", use_container_width=True):
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

        st.link_button(
            "Open Projected Point in Google Maps",
            f"https://www.google.com/maps/search/?api=1&query={rev['Projected Latitude']},{rev['Projected Longitude']}",
            use_container_width=True
        )

        st.dataframe(pd.DataFrame([rev]), use_container_width=True, hide_index=True)

        m = make_alignment_map(
            points,
            osgb_to_gps,
            rev["Projected Latitude"],
            rev["Projected Longitude"],
            f"Projected chainage: {rev['Chainage']} m"
        )

        folium.Marker(
            [rev["Input Latitude"], rev["Input Longitude"]],
            popup="Input GPS point",
            icon=folium.Icon(color="green")
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


# -----------------------------
# Tab 3: Use phone GPS
# -----------------------------
with tab3:
    st.subheader("Use Phone GPS")
    st.info("Press the button below and allow location access in your phone browser.")

    location = streamlit_geolocation()

    if location and location.get("latitude") is not None and location.get("longitude") is not None:
        lat = float(location["latitude"])
        lon = float(location["longitude"])

        x, y = gps_to_osgb.transform(lon, lat)
        gps_result = chainage_from_xy(points, cum_dist, x, y)

        if gps_result:
            proj_lon, proj_lat = osgb_to_gps.transform(
                gps_result["easting"],
                gps_result["northing"]
            )

            result = {
                "Phone Latitude": round(lat, 8),
                "Phone Longitude": round(lon, 8),
                "Chainage": round(gps_result["chainage"], 3),
                "Projected Latitude": round(proj_lat, 8),
                "Projected Longitude": round(proj_lon, 8),
                "Projected Easting": round(gps_result["easting"], 3),
                "Projected Northing": round(gps_result["northing"], 3),
                "Distance from Alignment": round(gps_result["distance"], 3)
            }

            st.metric("Nearest Chainage", f"{result['Chainage']} m")
            st.metric("Distance from Alignment", f"{result['Distance from Alignment']} m")

            st.link_button(
                "Open Projected Point in Google Maps",
                f"https://www.google.com/maps/search/?api=1&query={result['Projected Latitude']},{result['Projected Longitude']}",
                use_container_width=True
            )

            st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)

            m = make_alignment_map(
                points,
                osgb_to_gps,
                result["Projected Latitude"],
                result["Projected Longitude"],
                f"Projected chainage: {result['Chainage']} m"
            )

            folium.Marker(
                [result["Phone Latitude"], result["Phone Longitude"]],
                popup="Phone GPS point",
                icon=folium.Icon(color="green")
            ).add_to(m)

            folium.PolyLine(
                [
                    [result["Phone Latitude"], result["Phone Longitude"]],
                    [result["Projected Latitude"], result["Projected Longitude"]]
                ],
                color="orange",
                weight=3,
                opacity=0.8
            ).add_to(m)

            st_folium(m, width=None, height=500)

    else:
        st.warning("No GPS location received yet. On mobile, allow location permission when prompted.")

st.caption("v0.8 • Auto-loads PR304.dxf • Chainage ↔ GPS • Phone GPS")