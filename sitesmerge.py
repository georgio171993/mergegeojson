#!/usr/bin/env python3
"""
GeoJSON MultiPolygon Grouper & Splitter (Orbify Rule) - with Dedupe + Ocean Check + Statistics
==============================================================================================

Purpose
-------
- Ingest one or more GeoJSON files.
- **Track Statistics:** Count inputs by type (Point, Polygon, etc.) and outputs.
- Apply Orbify-style spatial efficiency **only to MultiPolygon** features.
- **Ignore MultiPoint** for this rule.
- Optionally **cluster nearby Polygon sites**.
- **De-duplicate** geometries.
- **Ocean check (optional)**.
- Export a clean GeoJSON FeatureCollection.

Usage
-----
python sitesmerge.py --inputs data/a.geojson --output out/merged.geojson ...
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

from shapely.geometry import (
    shape,
    mapping,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    Point,
    LineString,
    MultiPoint,
    MultiLineString
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Geod, CRS, Transformer

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("orbify_sites")

# --- Geodesic helper ---------------------------------------------------------
WGS84_GEOD = Geod(ellps="WGS84")


def geodesic_area_ha(geom: BaseGeometry) -> float:
    """Return geodesic area (hectares) of Polygon or MultiPolygon using WGS84 spheroid."""
    if geom.is_empty:
        return 0.0
    total_m2 = 0.0
    if isinstance(geom, Polygon):
        total_m2 += _polygon_geodesic_area_m2(geom)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            total_m2 += _polygon_geodesic_area_m2(poly)
    else:
        return 0.0
    return abs(total_m2) / 10_000.0  # m² → ha


def _polygon_geodesic_area_m2(poly: Polygon) -> float:
    if poly.is_empty:
        return 0.0
    lon, lat = poly.exterior.coords.xy
    area, _ = WGS84_GEOD.polygon_area_perimeter(lon, lat)
    total = area
    for interior in poly.interiors:
        lon, lat = interior.coords.xy
        hole_area, _ = WGS84_GEOD.polygon_area_perimeter(lon, lat)
        total -= abs(hole_area)
    return total


def bbox_polygon(geom: BaseGeometry) -> Polygon:
    minx, miny, maxx, maxy = geom.bounds
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


# --- Projection helpers ------------------------------------------------------

def make_local_aeqd_crs(center_lon: float, center_lat: float) -> CRS:
    proj4 = (
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs"
    )
    return CRS.from_proj4(proj4)


@dataclass
class Feature:
    geom: BaseGeometry
    props: Dict


def _dataset_center(features: List[Feature]) -> Tuple[float, float]:
    lats, lons = [], []
    for f in features:
        if f.geom.is_empty:
            continue
        c = f.geom.centroid
        lons.append(c.x)
        lats.append(c.y)
    if not lats:
        return 0.0, 0.0
    return sum(lats) / len(lats), sum(lons) / len(lons)


def shapely_transform_to_proj(geom: BaseGeometry, transformer: Transformer) -> BaseGeometry:
    def _tx(x, y, z=None):
        X, Y = transformer.transform(x, y)
        return (X, Y) if z is None else (X, Y, z)
    return shp_transform(_tx, geom)


# --- Stats Helper -----------------------------------------------------------

def get_feature_counts(features: List[Feature]) -> Dict[str, int]:
    """Count occurrences of each geometry type."""
    counts = {
        "Polygon": 0,
        "MultiPolygon": 0,
        "Point": 0,
        "MultiPoint": 0,
        "LineString": 0,
        "MultiLineString": 0,
        "GeometryCollection": 0,
        "Total": len(features)
    }
    for f in features:
        gt = f.geom.geom_type
        # If the geometry type isn't standard, default to just counting total
        if gt in counts:
            counts[gt] += 1
    return counts


# --- IO ---------------------------------------------------------------------

def load_features(paths: List[Path]) -> List[Feature]:
    feats: List[Feature] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("type") == "FeatureCollection":
                for feat in data.get("features", []):
                    g = shape(feat.get("geometry")) if feat.get("geometry") else GeometryCollection()
                    props = feat.get("properties", {}) or {}
                    feats.append(Feature(g, props))
            elif data.get("type") == "Feature":
                g = shape(data.get("geometry")) if data.get("geometry") else GeometryCollection()
                props = data.get("properties", {}) or {}
                feats.append(Feature(g, props))
            else:
                logger.warning("Unsupported top-level type in %s: %s", p, data.get("type"))
        except Exception as e:
            logger.error(f"Error loading {p}: {e}")
            
    logger.info("Loaded %d feature(s) from %d file(s)", len(feats), len(paths))
    return feats


def save_features(features: List[Feature], out_path: Path) -> None:
    out = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(f.geom), "properties": f.props}
            for f in features
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    logger.info("Wrote %d feature(s) → %s", len(features), out_path)


# --- Orbify MultiPolygon rule ------------------------------------------------

def apply_orbify_rule(features: List[Feature], ratio_threshold: float) -> List[Feature]:
    result: List[Feature] = []
    for idx, feat in enumerate(features):
        geom = feat.geom
        gtype = geom.geom_type

        if gtype == "MultiPoint":
            new_props = dict(feat.props)
            new_props["notes"] = (new_props.get("notes", "") + "; MultiPoint ignored for ratio").strip("; ")
            result.append(Feature(geom, new_props))
            continue

        if gtype == "MultiPolygon":
            polys: List[Polygon] = list(geom.geoms)
            polys = _dedupe_polygons_in_list(polys)
            mp_clean = MultiPolygon(tuple(polys)) if len(polys) > 1 else polys[0]

            total_area_ha = geodesic_area_ha(mp_clean if isinstance(mp_clean, MultiPolygon) else mp_clean)
            bbox_area_ha = geodesic_area_ha(bbox_polygon(mp_clean))
            ratio = (total_area_ha / bbox_area_ha) if bbox_area_ha > 0 else 0.0

            if ratio >= ratio_threshold:
                props = dict(feat.props)
                props["orbify_ratio"] = round(ratio, 6)
                props["orbify_decision"] = "single"
                result.append(Feature(mp_clean if isinstance(mp_clean, MultiPolygon) else mp_clean, props))
            else:
                for i, poly in enumerate(polys):
                    props = dict(feat.props)
                    props["orbify_ratio"] = round(ratio, 6)
                    props["orbify_decision"] = "split"
                    props["split_index"] = i
                    result.append(Feature(poly, props))
            continue

        result.append(feat)
    return result


# --- Proximity clustering into MultiPolygons --------------------------------

def cluster_polygons(features: List[Feature], cluster_distance_m: float) -> List[Feature]:
    if cluster_distance_m <= 0:
        return features

    polys: List[Feature] = []
    others: List[Feature] = []
    for f in features:
        if isinstance(f.geom, Polygon):
            polys.append(f)
        else:
            others.append(f)

    if not polys:
        return features

    lat0, lon0 = _dataset_center(polys)
    aeqd = make_local_aeqd_crs(lon0, lat0)
    transformer = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)

    proj_geoms = [shapely_transform_to_proj(f.geom, transformer) for f in polys]

    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            d = proj_geoms[i].distance(proj_geoms[j])
            if d <= cluster_distance_m:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(polys)):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    out: List[Feature] = others.copy()
    for cid, idxs in clusters.items():
        geoms = [polys[i].geom for i in idxs]

        # --- PROPERTY MERGE FIX (Safe for Lists) ---
        props_merged: Dict = {}
        if idxs:
            common_keys = set(polys[idxs[0]].props.keys())
            for i in idxs[1:]:
                common_keys.intersection_update(polys[i].props.keys())
            
            for k in common_keys:
                val0 = polys[idxs[0]].props[k]
                is_consistent = True
                for i in idxs[1:]:
                    if polys[i].props[k] != val0:
                        is_consistent = False
                        break
                if is_consistent:
                    props_merged[k] = val0
        # -------------------------------------------

        props_merged["cluster_id"] = int(cid)
        if len(idxs) == 1:
            out.append(Feature(geoms[0], {**polys[idxs[0]].props, **props_merged}))
        else:
            mp = MultiPolygon(tuple(g for g in geoms))
            out.append(Feature(mp, props_merged))

    logger.info("Clustering: %d polygons → %d clustered site(s)", len(polys), len(clusters))
    return out


# --- DEDUPLICATION -----------------------------------------------------------

def _quantize_projected(geom: BaseGeometry, transformer_to_proj: Transformer, transformer_to_wgs: Transformer, grid_m: float) -> BaseGeometry:
    if geom.is_empty:
        return geom
    g_proj = shapely_transform_to_proj(geom, transformer_to_proj)

    def snap(val: float) -> float:
        return round(val / grid_m) * grid_m

    def _snapper(x, y, z=None):
        return (snap(x), snap(y)) if z is None else (snap(x), snap(y), z)

    g_snapped = shp_transform(_snapper, g_proj)
    def _inv(x, y, z=None):
        X, Y = transformer_to_wgs.transform(x, y)
        return (X, Y) if z is None else (X, Y, z)

    return shp_transform(_inv, g_snapped)


def _polygon_key(poly: Polygon) -> bytes:
    return poly.buffer(0).wkb


def _dedupe_polygons_in_list(polys: List[Polygon]) -> List[Polygon]:
    seen: Set[bytes] = set()
    out: List[Polygon] = []
    for p in polys:
        k = _polygon_key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def dedupe_features(features: List[Feature], tolerance_m: float) -> List[Feature]:
    if tolerance_m < 0:
        tolerance_m = 0.0

    lat0, lon0 = _dataset_center(features)
    aeqd = make_local_aeqd_crs(lon0, lat0)
    to_proj = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    to_wgs = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)

    normed: List[Feature] = []
    for f in features:
        g = f.geom
        if tolerance_m > 0 and isinstance(g, (Polygon, MultiPolygon)):
            g = _quantize_projected(g, to_proj, to_wgs, tolerance_m)
        normed.append(Feature(g.buffer(0) if isinstance(g, (Polygon, MultiPolygon)) else g, dict(f.props)))

    seen_poly: Set[bytes] = set()
    seen_mpoly: Set[Tuple[bytes, ...]] = set()
    out: List[Feature] = []

    for f in normed:
        g = f.geom
        if isinstance(g, Polygon):
            k = _polygon_key(g)
            if k in seen_poly: continue
            if (k,) in seen_mpoly: continue
            seen_poly.add(k)
            out.append(f)
        elif isinstance(g, MultiPolygon):
            parts = _dedupe_polygons_in_list(list(g.geoms))
            if len(parts) == 1:
                k = _polygon_key(parts[0])
                if k in seen_poly: continue
                seen_poly.add(k)
                out.append(Feature(parts[0], f.props))
            else:
                k = tuple(sorted(_polygon_key(p) for p in parts))
                if k in seen_mpoly: continue
                if all(pk in seen_poly for pk in k): continue
                seen_mpoly.add(k)
                out.append(Feature(MultiPolygon(tuple(parts)), f.props))
        else:
            out.append(f)

    logger.info("Dedupe: input=%d → output=%d", len(features), len(out))
    return out


# --- OCEAN / LAND CHECK ------------------------------------------------------

def load_landmask(path: Optional[Path]) -> Optional[BaseGeometry]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        geoms: List[BaseGeometry] = []
        if data.get("type") == "FeatureCollection":
            for feat in data.get("features", []):
                g = shape(feat.get("geometry")) if feat.get("geometry") else GeometryCollection()
                if isinstance(g, (Polygon, MultiPolygon)):
                    geoms.append(g)
        elif data.get("type") == "Feature":
            g = shape(data.get("geometry")) if data.get("geometry") else GeometryCollection()
            if isinstance(g, (Polygon, MultiPolygon)):
                geoms.append(g)
        return unary_union(geoms) if geoms else None
    except Exception as e:
        logger.error("Failed to load landmask: %s", e)
        return None


def filter_ocean(features: List[Feature], landmask: BaseGeometry, min_overlap_ratio: float = 0.95) -> List[Feature]:
    out: List[Feature] = []
    dropped = 0
    for f in features:
        g = f.geom
        try:
            if isinstance(g, (Polygon, MultiPolygon)):
                a = g.area if g.area > 0 else 1.0
                inter = g.intersection(landmask)
                ratio = 0.0 if g.is_empty else (inter.area / a)
                if ratio >= min_overlap_ratio:
                    out.append(f)
                else:
                    dropped += 1
            elif isinstance(g, Point):
                if landmask.contains(g) or landmask.touches(g):
                    out.append(f)
                else:
                    dropped += 1
            elif isinstance(g, LineString):
                if not g.intersection(landmask).is_empty:
                    out.append(f)
                else:
                    dropped += 1
            else:
                out.append(f)
        except Exception:
            dropped += 1
    logger.info("Ocean check: dropped %d feature(s)", dropped)
    return out


# --- Pipeline ----------------------------------------------------------------

def process(
    inputs: List[Path],
    output: Path,
    ratio_threshold: float = 0.0011,
    cluster_distance_m: float = 0.0,
    dedupe_tolerance_m: float = 1.0,
    landmask_path: Optional[Path] = None,
    min_land_overlap: float = 0.95,
) -> Dict[str, Any]:
    
    # 1. Load and Get Initial Stats
    feats = load_features(inputs)
    initial_stats = get_feature_counts(feats)

    # 2. Orbify Rule (Splits/Groups)
    feats = apply_orbify_rule(feats, ratio_threshold)

    # 3. Clustering
    if cluster_distance_m > 0:
        feats = cluster_polygons(feats, cluster_distance_m)

    # 4. Deduplication
    count_before_dedupe = len(feats)
    feats = dedupe_features(feats, dedupe_tolerance_m)
    dropped_dedupe = count_before_dedupe - len(feats)

    # 5. Ocean Check
    dropped_ocean = 0
    landmask = load_landmask(landmask_path)
    if landmask is not None:
        count_before_ocean = len(feats)
        feats = filter_ocean(feats, landmask, min_land_overlap)
        dropped_ocean = count_before_ocean - len(feats)

    # 6. Final Prep & Stats
    final = []
    for i, f in enumerate(feats):
        props = dict(f.props)
        props.setdefault("site_id", f"site_{i+1}")
        if isinstance(f.geom, (Polygon, MultiPolygon)):
            props["area_ha"] = round(geodesic_area_ha(f.geom), 4)
        final.append(Feature(f.geom, props))

    final_stats = get_feature_counts(final)
    save_features(final, output)

    return {
        "initial_counts": initial_stats,
        "final_counts": final_stats,
        "dropped_dedupe": dropped_dedupe,
        "dropped_ocean": dropped_ocean,
        "dropped_total": dropped_dedupe + dropped_ocean
    }


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Orbify GeoJSON Grouper with Stats")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input GeoJSON files")
    ap.add_argument("--output", type=Path, required=True, help="Output GeoJSON path")
    ap.add_argument("--ratio-threshold", type=float, default=0.0011)
    ap.add_argument("--cluster-distance-m", type=float, default=0.0)
    ap.add_argument("--dedupe-tolerance-m", type=float, default=1.0)
    ap.add_argument("--landmask", type=Path, default=None)
    ap.add_argument("--min-land-overlap", type=float, default=0.95)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    stats = process(
        inputs=args.inputs,
        output=args.output,
        ratio_threshold=args.ratio_threshold,
        cluster_distance_m=args.cluster_distance_m,
        dedupe_tolerance_m=args.dedupe_tolerance_m,
        landmask_path=args.landmask,
        min_land_overlap=args.min_land_overlap,
    )
    print("Processing Complete.")
    print(f"Initial: {stats['initial_counts']}")
    print(f"Final:   {stats['final_counts']}")
    print(f"Dropped: {stats['dropped_total']} (Dedupe={stats['dropped_dedupe']}, Ocean={stats['dropped_ocean']})")


# --- Streamlit UI ------------------------------------------------------------

def streamlit_app():
    import streamlit as st
    import tempfile
    import pydeck as pdk
    import pandas as pd

    st.set_page_config(page_title="Orbify GeoJSON Grouper", layout="wide")
    st.title("Orbify GeoJSON Grouper & Splitter (with Stats)")

    with st.sidebar:
        st.header("Parameters")
        ratio = st.number_input("Ratio threshold", value=0.0011, format="%.4f")
        cluster_m = st.number_input("Cluster distance (m)", value=2000.0, step=100.0)
        dedupe_m = st.number_input("Dedupe tolerance (m)", value=1.0, step=0.5)
        min_overlap = st.slider("Min land overlap", 0.0, 1.0, 0.95)
        uploaded = st.file_uploader("Upload Inputs", type=["json","geojson"], accept_multiple_files=True)
        landmask_file = st.file_uploader("Upload Landmask (Optional)", type=["json","geojson"])
        run_btn = st.button("Process")

    if run_btn and uploaded:
        tmp_paths = []
        with tempfile.TemporaryDirectory() as td:
            for uf in uploaded:
                p = Path(td) / uf.name
                p.write_bytes(uf.getvalue())
                tmp_paths.append(p)
            
            landmask_path = None
            if landmask_file:
                landmask_path = Path(td) / ("landmask_" + landmask_file.name)
                landmask_path.write_bytes(landmask_file.getvalue())
            
            out_path = Path(td) / "sites_merged.geojson"
            
            try:
                # RUN PIPELINE AND GET STATS
                stats = process(
                    inputs=tmp_paths,
                    output=out_path,
                    ratio_threshold=ratio,
                    cluster_distance_m=cluster_m,
                    dedupe_tolerance_m=dedupe_m,
                    landmask_path=landmask_path,
                    min_land_overlap=min_overlap,
                )
                
                # --- DISPLAY STATS ---
                st.subheader("Processing Statistics")
                
                # Create a comparison table for geometry types
                init_c = stats["initial_counts"]
                final_c = stats["final_counts"]
                
                # Determine all geometry types present
                all_types = sorted(set(init_c.keys()) | set(final_c.keys()) - {"Total"})
                
                data = []
                for t in all_types:
                    data.append({
                        "Geometry Type": t,
                        "Input Count": init_c.get(t, 0),
                        "Output Count": final_c.get(t, 0),
                        "Difference": final_c.get(t, 0) - init_c.get(t, 0)
                    })
                
                # Summary Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Input Features", init_c["Total"])
                c2.metric("Total Output Features", final_c["Total"])
                c3.metric("Total Missed/Dropped", stats["dropped_total"], 
                          help=f"Dedupe: {stats['dropped_dedupe']}, Ocean: {stats['dropped_ocean']}")

                st.table(pd.DataFrame(data).set_index("Geometry Type"))
                
                # --- MAP & DOWNLOAD ---
                out_bytes = out_path.read_bytes()
                out_json = json.loads(out_bytes.decode("utf-8"))
                
                st.download_button("Download Result", out_bytes, "sites_merged.geojson", "application/geo+json")
                
                st.subheader("Map Preview")
                try:
                    layer = pdk.Layer("GeoJsonLayer", data=out_json, pickable=True, stroked=True, filled=True, wireframe=True)
                    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)
                    if out_json["features"]:
                         # Centering logic simplified
                         from shapely.geometry import shape as s_shape
                         c = s_shape(out_json["features"][0]["geometry"]).centroid
                         view_state = pdk.ViewState(latitude=c.y, longitude=c.x, zoom=6)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
                except Exception:
                    st.warning("Map preview unavailable.")

            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Streamlit Error")

def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if __name__ == "__main__":
    if _running_in_streamlit():
        streamlit_app()
    else:
        main()
