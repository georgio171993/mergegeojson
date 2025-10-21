#!/usr/bin/env python3
"""
GeoJSON MultiPolygon Grouper & Splitter (Orbify Rule) - with Dedupe
===================================================================

Purpose
-------
- Ingest one or more GeoJSON files (Polygon / MultiPolygon; others pass through).
- Apply Orbify-style spatial efficiency **only to MultiPolygon** features:
    * total_area_ha / bbox_area_ha >= 0.006  → keep as **single site**
    * total_area_ha / bbox_area_ha <  0.006  → **split** into per-polygon sites
- **Ignore MultiPoint** for this rule (log a warning; no ratio processing).
- Optionally **cluster nearby Polygon sites** (across all inputs) into
  **MultiPolygon** sites when polygons are within a proximity (e.g., 2 km).
- **De-duplicate** geometries so only one copy remains. Deduping works for:
  * Polygons that are exact/near duplicates of other Polygons.
  * MultiPolygons that are duplicates of other MultiPolygons **or** equivalent
    to an existing Polygon set (e.g., a single-Polygon MultiPolygon).
  * Internal duplicates inside a MultiPolygon (drops repeated parts).
- Export a clean GeoJSON FeatureCollection with final **Polygon**/**MultiPolygon** sites.

Notes
-----
- Geodesic area via `pyproj.Geod` (hectares).
- Distances & dedupe tolerance use a local AEQD projection (meters).
- Metadata properties: `site_id`, `orbify_ratio`, `orbify_decision` ("single"/"split"),
  `cluster_id` (if clustering), `area_ha`.

Usage
-----
python orbify_sites.py \
  --inputs data/a.geojson data/b.geojson \
  --output out/sites_merged.geojson \
  --ratio-threshold 0.006 \
  --cluster-distance-m 2000 \
  --dedupe-tolerance-m 1.0

Dependencies
------------
  pip install shapely pyproj

"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
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


# --- IO ---------------------------------------------------------------------

def load_features(paths: List[Path]) -> List[Feature]:
    feats: List[Feature] = []
    for p in paths:
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
    """Apply the ratio rule to MultiPolygon features only; split when sparse.
    MultiPoint features are logged and skipped for this logic (left unchanged).
    Polygons pass through unchanged here (may be clustered later).
    """
    result: List[Feature] = []
    for idx, feat in enumerate(features):
        geom = feat.geom
        gtype = geom.geom_type

        if gtype == "MultiPoint":
            new_props = dict(feat.props)
            new_props.setdefault("notes", "")
            new_props["notes"] = (new_props["notes"] + "; " if new_props["notes"] else "") + \
                                  "MultiPoint ignored for ratio grouping"
            logger.warning("Feature %d: MultiPoint ignored for ratio-based grouping", idx)
            result.append(Feature(geom, new_props))
            continue

        if gtype == "MultiPolygon":
            polys: List[Polygon] = list(geom.geoms)
            # Drop internal duplicates inside this MultiPolygon before ratio
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
                # Split into individual Polygon sites
                for i, poly in enumerate(polys):
                    props = dict(feat.props)
                    props["orbify_ratio"] = round(ratio, 6)
                    props["orbify_decision"] = "split"
                    props["split_index"] = i
                    result.append(Feature(poly, props))
            continue

        # Pass-through
        result.append(feat)

    logger.info("Orbify rule processed: input=%d → output=%d", len(features), len(result))
    return result


# --- Proximity clustering into MultiPolygons --------------------------------

def cluster_polygons(features: List[Feature], cluster_distance_m: float) -> List[Feature]:
    """Cluster Polygon features by proximity and output Polygon/MultiPolygon features.
    Only Polygon features are clustered; MultiPolygons are kept as-is.
    """
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
        props_merged: Dict = {}
        common_keys = set.intersection(*(set(polys[i].props.keys()) for i in idxs)) if len(idxs) > 1 else set(polys[idxs[0]].props.keys())
        for k in common_keys:
            vals = {polys[i].props[k] for i in idxs}
            if len(vals) == 1:
                props_merged[k] = vals.pop()
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
    """Project → snap coordinates to grid → back to WGS84 to normalize small noise for dedupe."""
    if geom.is_empty:
        return geom
    g_proj = shapely_transform_to_proj(geom, transformer_to_proj)

    def snap(val: float) -> float:
        return round(val / grid_m) * grid_m

    # Apply snapping to all coordinates using a transform
    def _snapper(x, y, z=None):
        return (snap(x), snap(y)) if z is None else (snap(x), snap(y), z)

    g_snapped = shp_transform(_snapper, g_proj)
    # Back to WGS84
    def _inv(x, y, z=None):
        X, Y = transformer_to_wgs.transform(x, y)
        return (X, Y) if z is None else (X, Y, z)

    return shp_transform(_inv, g_snapped)


def _polygon_key(poly: Polygon) -> bytes:
    # Normalize orientation & topology
    return poly.buffer(0).wkb


def _multipolygon_key(mp: MultiPolygon) -> Tuple[bytes, ...]:
    parts = sorted((_polygon_key(p) for p in mp.geoms))
    return tuple(parts)


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
    """Drop duplicate Polygon/MultiPolygon features. Tolerance is used to normalize
    nearly-identical coordinates in meters (projected snapping) before comparing.

    Rules:
    - Polygons compared by normalized topology (`buffer(0)` + optional snapping).
    - MultiPolygons compared as sorted tuples of their polygon keys.
    - If a MultiPolygon contains duplicate parts internally, they are removed.
    - If a MultiPolygon reduces to a single polygon equal to an existing Polygon,
      it is considered a duplicate and removed.
    - Mixed duplicates (Polygon vs one-part MultiPolygon) are unified.
    """
    if tolerance_m < 0:
        tolerance_m = 0.0

    # Build projection centered on dataset for snapping
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
            if k in seen_poly:
                continue
            # also if any MultiPolygon key already consists solely of this polygon, skip
            if (k,) in seen_mpoly:
                continue
            seen_poly.add(k)
            out.append(f)
        elif isinstance(g, MultiPolygon):
            # drop internal dupes first
            parts = _dedupe_polygons_in_list(list(g.geoms))
            if len(parts) == 1:
                k = _polygon_key(parts[0])
                if k in seen_poly:
                    continue
                seen_poly.add(k)
                out.append(Feature(parts[0], f.props))
            else:
                k = tuple(sorted(_polygon_key(p) for p in parts))
                if k in seen_mpoly:
                    continue
                # if an equivalent set already exists as polygons individually, treat as duplicate set
                if all(pk in seen_poly for pk in k):
                    continue
                seen_mpoly.add(k)
                out.append(Feature(MultiPolygon(tuple(parts)), f.props))
        else:
            # Other geometry types: leave as-is (not considered for geometric dedupe)
            out.append(f)

    logger.info("Dedupe: input=%d → output=%d", len(features), len(out))
    return out


# --- Pipeline ----------------------------------------------------------------

def process(
    inputs: List[Path],
    output: Path,
    ratio_threshold: float = 0.006,
    cluster_distance_m: float = 0.0,
    dedupe_tolerance_m: float = 1.0,
) -> None:
    feats = load_features(inputs)

    # Apply Orbify rule on MultiPolygons (split vs keep)
    feats = apply_orbify_rule(feats, ratio_threshold)

    # Optional proximity-based clustering across Polygons
    if cluster_distance_m and cluster_distance_m > 0:
        feats = cluster_polygons(feats, cluster_distance_m)

    # Global geometric de-duplication
    feats = dedupe_features(feats, dedupe_tolerance_m)

    # Assign site ids (stable index-based) & store area
    final = []
    for i, f in enumerate(feats):
        props = dict(f.props)
        props.setdefault("site_id", f"site_{i+1}")
        if isinstance(f.geom, (Polygon, MultiPolygon)):
            props["area_ha"] = round(geodesic_area_ha(f.geom), 4)
        final.append(Feature(f.geom, props))

    save_features(final, output)


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Orbify-style MultiPolygon grouping, clustering, and deduplication for GeoJSON.")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="One or more input GeoJSON files")
    ap.add_argument("--output", type=Path, required=True, help="Output GeoJSON path")
    ap.add_argument("--ratio-threshold", type=float, default=0.006, help="Threshold for area/bbox_area (default 0.006)")
    ap.add_argument("--cluster-distance-m", type=float, default=0.0, help="Max distance (meters) to cluster polygons into MultiPolygons across inputs (0=off)")
    ap.add_argument("--dedupe-tolerance-m", type=float, default=1.0, help="Snap grid in meters for near-duplicate removal (0=exact only)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return ap.parse_args()


def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level))
    process(
        inputs=args.inputs,
        output=args.output,
        ratio_threshold=args.ratio_threshold,
        cluster_distance_m=args.cluster_distance_m,
        dedupe_tolerance_m=args.dedupe_tolerance_m,
    )


# --- Streamlit UI ------------------------------------------------------------
# Minimal UI so you can run `streamlit run orbify_sites.py` directly.
# Requires: pip install streamlit pydeck shapely pyproj

def streamlit_app():
    import streamlit as st
    import tempfile
    import io
    import pydeck as pdk

    st.set_page_config(page_title="Orbify GeoJSON Grouper", layout="wide")
    st.title("Orbify GeoJSON Grouper & Splitter (with Dedupe)")

    with st.sidebar:
        st.header("Parameters")
        ratio = st.number_input("Ratio threshold (area/bbox)", value=0.006, min_value=0.0, step=0.001, format="%.3f")
        cluster_m = st.number_input("Cluster distance (m)", value=2000.0, min_value=0.0, step=100.0)
        dedupe_m = st.number_input("Dedupe tolerance (m)", value=1.0, min_value=0.0, step=0.5)
        log_level = st.selectbox("Log level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

        st.markdown("---")
        uploaded = st.file_uploader("Upload one or more GeoJSON files", type=["json","geojson"], accept_multiple_files=True)
        run_btn = st.button("Process")

    logger.setLevel(getattr(logging, log_level))

    if run_btn:
        if not uploaded:
            st.warning("Please upload at least one GeoJSON file.")
            return

        # Write uploads to temp files so we can reuse the existing pipeline
        tmp_paths = []
        with tempfile.TemporaryDirectory() as td:
            for uf in uploaded:
                p = Path(td) / uf.name
                p.write_bytes(uf.getvalue())
                tmp_paths.append(p)

            out_path = Path(td) / "sites_merged.geojson"
            try:
                process(
                    inputs=tmp_paths,
                    output=out_path,
                    ratio_threshold=ratio,
                    cluster_distance_m=cluster_m,
                    dedupe_tolerance_m=dedupe_m,
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                raise

            # Read back result
            out_bytes = out_path.read_bytes()
            out_json = json.loads(out_bytes.decode("utf-8"))

            # Show quick stats
            feats = out_json.get("features", [])
            n_poly = sum(1 for f in feats if (f.get("geometry") or {}).get("type") == "Polygon")
            n_mpoly = sum(1 for f in feats if (f.get("geometry") or {}).get("type") == "MultiPolygon")

            st.success(f"Processed {len(uploaded)} file(s). Output features: {len(feats)} (Polygon={n_poly}, MultiPolygon={n_mpoly}).")

            # Map preview via pydeck GeoJsonLayer
            st.subheader("Preview")
            try:
                layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=out_json,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    wireframe=False,
                )
                view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)
                if feats:
                    # center on first feature's centroid
                    try:
                        from shapely.geometry import shape as _shape
                        first = _shape(feats[0].get("geometry"))
                        c = first.centroid
                        view_state = pdk.ViewState(latitude=c.y, longitude=c.x, zoom=6)
                    except Exception:
                        pass
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None))
            except Exception as _e:
                st.info("Map preview not available (pydeck error). You can still download the output.")

            # Download button
            st.download_button(
                label="Download merged GeoJSON",
                data=out_bytes,
                file_name="sites_merged.geojson",
                mime="application/geo+json",
            )


def _running_in_streamlit() -> bool:
    """Detect if this file is being executed by Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    # If launched by `streamlit run <this_file>`, run the Streamlit UI.
    # Otherwise, behave as a normal CLI.
    if _running_in_streamlit():
        streamlit_app()
    else:
        main()
