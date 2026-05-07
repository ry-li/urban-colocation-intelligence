from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


DEFAULT_FLOOR_HEIGHT_M = 3.2
DEFAULT_BUILDING_HEIGHT_M = 14.0

CLASS_ORDER = [
    "Office/HQ",
    "Mixed-use/Complex",
    "Shopping Mall",
    "Small Retail/Dining",
    "Transit Hub",
    "Infrastructure",
    "Residential",
    "Other",
]

CLASS_COLORS = {
    "Office/HQ": "#6ec5ff",
    "Mixed-use/Complex": "#ffb65c",
    "Shopping Mall": "#f28cff",
    "Small Retail/Dining": "#ff6f91",
    "Transit Hub": "#ffe066",
    "Infrastructure": "#98a2ad",
    "Residential": "#7bd389",
    "Other": "#d6d3ce",
    "Unknown": "#b0bec5",
}

POI_CATEGORY_ORDER = [
    "Work",
    "Food_Drink",
    "Retail_Leisure",
    "Education",
    "Medical",
    "Transportation",
    "Other_POI",
]

POI_CATEGORY_LABELS = {
    "Work": "Work",
    "Food_Drink": "Food & Drink",
    "Retail_Leisure": "Retail & Leisure",
    "Education": "Education",
    "Medical": "Medical",
    "Transportation": "Transportation",
    "Other_POI": "Other POI",
}

POI_CATEGORY_COLORS = {
    "Work": "#64d2ff",
    "Food_Drink": "#ffb86b",
    "Retail_Leisure": "#ff6f91",
    "Education": "#a78bfa",
    "Medical": "#f87171",
    "Transportation": "#fde047",
    "Other_POI": "#94a3b8",
    "Other": "#94a3b8",
}

VISITOR_TYPE_ORDER = [
    "pure workers",
    "weekend visitors",
    "mix users",
    "pure residents",
]

VISITOR_TYPE_LABELS = {
    "pure workers": "Pure Workers",
    "weekend visitors": "Weekend Visitors",
    "mix users": "Mix Users",
    "pure residents": "Pure Residents",
}

VISITOR_TYPE_COLORS = {
    "pure workers": "#6ec5ff",
    "weekend visitors": "#ffb65c",
    "mix users": "#c084fc",
    "pure residents": "#7bd389",
}

NETWORK_EDGE_LOW = np.array([68.0, 211.0, 255.0])
NETWORK_EDGE_HIGH = np.array([255.0, 99.0, 181.0])


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "Website" / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root containing Website/data.")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any, default: str = "") -> str:
    if is_missing(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def clean_name(value: Any, fallback: str) -> str:
    text = clean_text(value)
    if not text or text == "0":
        return fallback
    return text


def clean_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def clean_int(value: Any, default: int = 0) -> int:
    return int(round(clean_float(value, float(default))))


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if is_missing(value):
        return None
    return value


def round_coordinates(value: Any, precision: int = 6) -> Any:
    if isinstance(value, dict):
        return {key: round_coordinates(item, precision=precision) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_coordinates(item, precision=precision) for item in value]
    if isinstance(value, float):
        return round(value, precision)
    return value


def feature_collection_from_gdf(
    gdf: gpd.GeoDataFrame,
    props_builder: Callable[[pd.Series], dict[str, Any]],
    *,
    precision: int = 6,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": round_coordinates(mapping(geometry), precision=precision),
                "properties": sanitize_json(props_builder(row)),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def bounds_are_finite(gdf: gpd.GeoDataFrame) -> bool:
    return bool(np.isfinite(np.asarray(gdf.total_bounds, dtype=float)).all())


def bng_to_wgs84_coord(easting: float, northing: float) -> tuple[float, float]:
    lat0 = math.radians(49.0)
    lon0 = math.radians(-2.0)
    a = 6377563.396
    b = 6356256.909
    f0 = 0.9996012717
    e0 = 400000.0
    n0 = -100000.0
    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)

    lat = lat0
    meridional_arc = 0.0
    while northing - n0 - meridional_arc >= 0.00001:
        lat = (northing - n0 - meridional_arc) / (a * f0) + lat
        ma = (1 + n + 5 / 4 * n**2 + 5 / 4 * n**3) * (lat - lat0)
        mb = (3 * n + 3 * n**2 + 21 / 8 * n**3) * math.sin(lat - lat0) * math.cos(lat + lat0)
        mc = (15 / 8 * n**2 + 15 / 8 * n**3) * math.sin(2 * (lat - lat0)) * math.cos(2 * (lat + lat0))
        md = 35 / 24 * n**3 * math.sin(3 * (lat - lat0)) * math.cos(3 * (lat + lat0))
        meridional_arc = b * f0 * (ma - mb + mc - md)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    nu = a * f0 / math.sqrt(1 - e2 * sin_lat**2)
    rho = a * f0 * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5
    eta2 = nu / rho - 1
    tan_lat = math.tan(lat)
    tan2 = tan_lat**2
    tan4 = tan2**2
    sec_lat = 1 / cos_lat
    d_e = easting - e0

    vii = tan_lat / (2 * rho * nu)
    viii = tan_lat / (24 * rho * nu**3) * (5 + 3 * tan2 + eta2 - 9 * tan2 * eta2)
    ix = tan_lat / (720 * rho * nu**5) * (61 + 90 * tan2 + 45 * tan4)
    x = sec_lat / nu
    xi = sec_lat / (6 * nu**3) * (nu / rho + 2 * tan2)
    xii = sec_lat / (120 * nu**5) * (5 + 28 * tan2 + 24 * tan4)
    xiia = sec_lat / (5040 * nu**7) * (61 + 662 * tan2 + 1320 * tan4 + 720 * tan4 * tan2)

    lat_osgb = lat - vii * d_e**2 + viii * d_e**4 - ix * d_e**6
    lon_osgb = lon0 + x * d_e - xi * d_e**3 + xii * d_e**5 - xiia * d_e**7

    h = 0.0
    sin_lat = math.sin(lat_osgb)
    cos_lat = math.cos(lat_osgb)
    sin_lon = math.sin(lon_osgb)
    cos_lon = math.cos(lon_osgb)
    nu = a / math.sqrt(1 - e2 * sin_lat**2)
    x1 = (nu + h) * cos_lat * cos_lon
    y1 = (nu + h) * cos_lat * sin_lon
    z1 = ((1 - e2) * nu + h) * sin_lat

    tx, ty, tz = 446.448, -125.157, 542.060
    rx = math.radians(0.1502 / 3600)
    ry = math.radians(0.2470 / 3600)
    rz = math.radians(0.8421 / 3600)
    scale = -20.4894e-6
    x2 = tx + (1 + scale) * x1 - rz * y1 + ry * z1
    y2 = ty + rz * x1 + (1 + scale) * y1 - rx * z1
    z2 = tz - ry * x1 + rx * y1 + (1 + scale) * z1

    a_wgs = 6378137.0
    b_wgs = 6356752.3141
    e2_wgs = 1 - (b_wgs * b_wgs) / (a_wgs * a_wgs)
    p = math.hypot(x2, y2)
    lat_wgs = math.atan2(z2, p * (1 - e2_wgs))
    previous = 0.0
    while abs(lat_wgs - previous) > 1e-12:
        previous = lat_wgs
        nu_wgs = a_wgs / math.sqrt(1 - e2_wgs * math.sin(lat_wgs) ** 2)
        lat_wgs = math.atan2(z2 + e2_wgs * nu_wgs * math.sin(lat_wgs), p)
    lon_wgs = math.atan2(y2, x2)
    return math.degrees(lon_wgs), math.degrees(lat_wgs)


def boundary_to_wgs84(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    projected = boundary.to_crs(4326)
    if bounds_are_finite(projected):
        return projected
    epsg = boundary.crs.to_epsg() if boundary.crs is not None else None
    if epsg != 27700:
        raise ValueError("Boundary reprojection returned non-finite coordinates.")
    fallback = boundary.copy()
    fallback["geometry"] = fallback.geometry.apply(
        lambda geometry: shapely_transform(lambda x, y, z=None: bng_to_wgs84_coord(float(x), float(y)), geometry)
    )
    fallback = fallback.set_crs(4326, allow_override=True)
    if not bounds_are_finite(fallback):
        raise ValueError("Boundary manual EPSG:27700 conversion returned non-finite coordinates.")
    return fallback


def height_for(row: pd.Series) -> float:
    height = clean_float(row.get("height"), 0.0)
    if height > 0:
        return round(height, 2)
    floors = clean_float(row.get("num_floors"), 0.0)
    if floors > 0:
        return round(floors * DEFAULT_FLOOR_HEIGHT_M, 2)
    return DEFAULT_BUILDING_HEIGHT_M


def sample_poi_names(raw_value: Any, *, limit: int = 6) -> tuple[list[str], int]:
    text = clean_text(raw_value)
    if not text or text == "0" or text.lower() == "no pois recorded":
        return [], 0
    normalized = re.sub(r"<br\s*/?>", ",", text, flags=re.IGNORECASE)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    names: list[str] = []
    for item in normalized.split(","):
        name = clean_text(item)
        if name and name not in names:
            names.append(name)
    return names[:limit], max(len(names) - limit, 0)


def building_display_name(row: pd.Series) -> str:
    building_name = clean_name(row.get("building_name"), "")
    if building_name:
        return building_name
    poi_names, _ = sample_poi_names(row.get("all_poi_names"), limit=1)
    if poi_names:
        return f"{poi_names[0]} (POI)"
    return "Building (No POI)"


def hex_from_rgb(rgb: Iterable[float]) -> str:
    parts = [int(round(float(value))) for value in rgb]
    return "#" + "".join(f"{max(0, min(255, part)):02x}" for part in parts)


def hex_to_rgb(hex_value: str) -> np.ndarray:
    stripped = hex_value.lstrip("#")
    if len(stripped) != 6:
        return np.array([148.0, 163.0, 184.0])
    return np.array([int(stripped[index : index + 2], 16) for index in (0, 2, 4)], dtype=float)


def interpolate_hex(low: str, high: str, factor: float) -> str:
    clipped = max(0.0, min(1.0, factor))
    return hex_from_rgb(hex_to_rgb(low) + (hex_to_rgb(high) - hex_to_rgb(low)) * clipped)


def network_color(percentile: float) -> str:
    return hex_from_rgb(NETWORK_EDGE_LOW + (NETWORK_EDGE_HIGH - NETWORK_EDGE_LOW) * max(0.0, min(1.0, percentile)))


def impact_color(value: Any, clamp_value: float) -> str:
    magnitude = abs(clean_float(value, 0.0))
    if magnitude <= 0:
        return "#64748b"
    factor = max(0.0, min(1.0, magnitude / max(clamp_value, 1.0)))
    if factor < 0.5:
        return interpolate_hex("#fde68a", "#f97316", factor / 0.5)
    return interpolate_hex("#f97316", "#b91c1c", (factor - 0.5) / 0.5)


def row_has_value(row: dict[str, Any], key: str) -> bool:
    if key not in row or is_missing(row.get(key)):
        return False
    text = str(row.get(key)).strip().lower()
    return text not in {"", "nan", "none", "null"}


def impact_loss_percent(row: dict[str, Any]) -> float:
    if not row:
        return 0.0
    if row_has_value(row, "loss_ratio"):
        ratio = abs(clean_float(row.get("loss_ratio"), 0.0))
        return ratio * 100.0 if ratio <= 1.0 else ratio
    if row_has_value(row, "total_impact_percent"):
        return abs(clean_float(row.get("total_impact_percent"), 0.0))
    if row_has_value(row, "loss_percent"):
        return abs(clean_float(row.get("loss_percent"), 0.0))
    if row_has_value(row, "impact_percent"):
        return abs(clean_float(row.get("impact_percent"), 0.0))
    return 0.0


def curve_between(
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    edge_index: int,
    steps: int = 24,
) -> list[list[float]]:
    lon1, lat1 = source
    lon2, lat2 = target
    dx = lon2 - lon1
    dy = lat2 - lat1
    distance = math.hypot(dx, dy)
    if distance < 1e-8:
        radius = 0.00013 + (edge_index % 4) * 0.000025
        return [
            [
                round(lon1 + math.cos(theta) * radius, 6),
                round(lat1 + math.sin(theta) * radius * 0.62, 6),
            ]
            for theta in np.linspace(0.0, math.tau, steps)
        ]

    normal_x = -dy / distance
    normal_y = dx / distance
    side = -1.0 if edge_index % 2 else 1.0
    bend = distance * (0.18 + (edge_index % 5) * 0.015) * side
    control_lon = (lon1 + lon2) / 2 + normal_x * bend
    control_lat = (lat1 + lat2) / 2 + normal_y * bend
    coords: list[list[float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t**2 * lon2
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t**2 * lat2
        coords.append([round(float(lon), 6), round(float(lat), 6)])
    return coords


def percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    ranks = pd.Series(values, dtype="float64").rank(method="average", pct=True)
    return [float(value) if not pd.isna(value) else 0.0 for value in ranks.tolist()]


def format_percent(value: Any) -> str:
    return f"{clean_float(value):.1f}%"


def active_color_items(values: Iterable[str], color_map: dict[str, str], order: list[str]) -> list[dict[str, str]]:
    observed = {clean_text(value, "Unknown") for value in values}
    ordered = [item for item in order if item in observed]
    ordered.extend(sorted(item for item in observed if item not in ordered))
    return [{"key": item, "label": item, "color": color_map.get(item, "#b0bec5")} for item in ordered]


def build_payload(project_root: Path) -> dict[str, Any]:
    data_dir = project_root / "Website" / "data"
    final_dir = data_dir / "final_data"

    boundary = gpd.read_file(data_dir / "boundary" / "canary_wharf.geojson")
    buildings = gpd.read_file(final_dir / "2.1_canary_wharf_LSOA_buildings_processed.geojson")
    pois = gpd.read_file(final_dir / "2.2_canary_wharf_LSOA_pois_processed.geojson")
    network = json.loads((final_dir / "0_workers_dependency_network.json").read_text(encoding="utf-8"))
    visitors = read_json_lines(final_dir / "1.1_user_ids_visitortype_optimized.json")
    impact_rows = json.loads((final_dir / "3_counterfactual_simulation_results.json").read_text(encoding="utf-8"))

    for name, gdf in [("boundary", boundary), ("buildings", buildings), ("pois", pois)]:
        if gdf.crs is None:
            raise ValueError(f"{name} is missing CRS metadata.")
    boundary = boundary_to_wgs84(boundary)
    buildings = buildings.to_crs(4326)
    pois = pois.to_crs(4326)

    impact_lookup = {clean_text(row.get("id")): row for row in impact_rows}
    impact_values = [impact_loss_percent(row) for row in impact_rows]
    impact_clamp = float(pd.Series(impact_values).quantile(0.95)) if impact_values else 1.0
    impact_clamp = max(impact_clamp, 1.0)

    node_lookup = {clean_text(node.get("id")): node for node in network.get("nodes", [])}
    building_anchor_lookup: dict[str, tuple[float, float]] = {}
    building_name_lookup: dict[str, str] = {}
    for _, row in buildings.iterrows():
        building_id = clean_text(row.get("id"))
        if not building_id:
            continue
        point = row.geometry.representative_point()
        building_anchor_lookup[building_id] = (float(point.x), float(point.y))
        building_name_lookup[building_id] = building_display_name(row)

    weights = [clean_float(edge.get("weight"), 0.0) for edge in network.get("edges", [])]
    transitions = [clean_float(edge.get("transitions"), 0.0) for edge in network.get("edges", [])]
    weight_ranks = percentile_ranks(weights)
    transition_ranks = percentile_ranks(transitions)

    network_features: list[dict[str, Any]] = []
    for index, edge in enumerate(network.get("edges", [])):
        source_id = clean_text(edge.get("source"))
        target_id = clean_text(edge.get("target"))
        if source_id not in building_anchor_lookup or target_id not in building_anchor_lookup:
            continue
        weight_rank = weight_ranks[index] if index < len(weight_ranks) else 0.0
        transition_rank = transition_ranks[index] if index < len(transition_ranks) else 0.0
        source_node = node_lookup.get(source_id, {})
        target_node = node_lookup.get(target_id, {})
        source_name = clean_name(source_node.get("name"), building_name_lookup.get(source_id, "Building (No POI)"))
        target_name = clean_name(target_node.get("name"), building_name_lookup.get(target_id, "Building (No POI)"))
        network_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": curve_between(
                        building_anchor_lookup[source_id],
                        building_anchor_lookup[target_id],
                        edge_index=index,
                    ),
                },
                "properties": {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_name": source_name,
                    "target_name": target_name,
                    "weight": round(clean_float(edge.get("weight"), 0.0), 6),
                    "transitions": clean_int(edge.get("transitions"), 0),
                    "weight_percentile": round(weight_rank, 3),
                    "transition_percentile": round(transition_rank, 3),
                    "line_color": network_color(weight_rank),
                    "line_width": round(0.65 + 3.35 * transition_rank, 2),
                },
            }
        )

    def boundary_props(row: pd.Series) -> dict[str, Any]:
        return {
            "lsoa21cd": clean_text(row.get("lsoa21cd")),
            "lsoa21nm": clean_text(row.get("lsoa21nm")),
            "msoa21nm": clean_text(row.get("msoa21nm")),
            "lad22nm": clean_text(row.get("lad22nm")),
        }

    def building_props(row: pd.Series) -> dict[str, Any]:
        building_id = clean_text(row.get("id"))
        impact = impact_lookup.get(building_id, {})
        node = node_lookup.get(building_id, {})
        functional_class = clean_text(row.get("final_functional_class"), "Unknown")
        sample_names, remaining_names = sample_poi_names(row.get("all_poi_names"))
        total_impact = impact_loss_percent(impact)
        return {
            "id": building_id,
            "name": building_display_name(row),
            "building_class": clean_text(row.get("building_class"), "Unknown"),
            "building_subtype": clean_text(row.get("building_subtype"), "Unknown"),
            "new_functional_class": clean_text(row.get("new_functional_class"), "Unknown"),
            "final_functional_class": functional_class,
            "class_color": CLASS_COLORS.get(functional_class, CLASS_COLORS["Unknown"]),
            "height_m": clean_float(row.get("height"), 0.0),
            "num_floors": clean_int(row.get("num_floors"), 0),
            "elevation_m": height_for(row),
            "poi_sum": clean_int(row.get("poi_sum"), 0),
            "dominant_share_pct": round(clean_float(row.get("dominant_share_pct"), 0.0), 1),
            "retail_food_ratio": round(clean_float(row.get("Retail_Food_Ratio"), 0.0), 1),
            "work": clean_int(row.get("Work"), 0),
            "food_drink": clean_int(row.get("Food_Drink"), 0),
            "retail_leisure": clean_int(row.get("Retail_Leisure"), 0),
            "education": clean_int(row.get("Education"), 0),
            "medical": clean_int(row.get("Medical"), 0),
            "transportation": clean_int(row.get("Transportation"), 0),
            "other_poi": clean_int(row.get("Other_POI"), 0),
            "sample_poi_names": sample_names,
            "remaining_poi_name_count": remaining_names,
            "inflow": clean_int(node.get("inflow"), 0),
            "outflow": clean_int(node.get("outflow"), 0),
            "total_flow": clean_int(node.get("inflow"), 0) + clean_int(node.get("outflow"), 0),
            "total_impact_percent": round(total_impact, 2),
            "loss_ratio": round(clean_float(impact.get("loss_ratio"), total_impact / 100.0), 4),
            "loss_amount": round(clean_float(impact.get("loss_amount"), 0.0), 2),
            "original_inflow": clean_int(impact.get("original_inflow"), 0),
            "initial_shock_percent": round(abs(clean_float(impact.get("initial_shock_percent"), 0.0)), 2),
            "retail_specific_impact": round(abs(clean_float(impact.get("retail_specific_impact"), 0.0)), 2),
            "impact_color": impact_color(total_impact, impact_clamp),
        }

    def poi_props(row: pd.Series) -> dict[str, Any]:
        category = clean_text(row.get("analysis_category"), "Other_POI")
        limited_category = clean_text(row.get("analysis_category_limited"), "Other")
        building_id = clean_text(row.get("building_id"))
        return {
            "id": clean_text(row.get("id")),
            "name": clean_name(row.get("primary_name"), "Unnamed POI"),
            "analysis_category": category,
            "analysis_category_label": POI_CATEGORY_LABELS.get(category, category.replace("_", " ")),
            "analysis_category_limited": limited_category,
            "taxonomy_l0": clean_text(row.get("taxonomy_l0"), "unclassified"),
            "taxonomy_primary": clean_text(row.get("taxonomy_primary"), "unclassified"),
            "operating_status": clean_text(row.get("operating_status"), "unknown"),
            "confidence": round(clean_float(row.get("confidence"), 0.0), 3),
            "building_id": building_id,
            "building_match_method": clean_text(row.get("building_match_method"), "unmatched"),
            "poi_color": POI_CATEGORY_COLORS.get(category, POI_CATEGORY_COLORS["Other"]),
        }

    visitor_counts = pd.Series([clean_text(row.get("new_visitor_type"), "unknown") for row in visitors]).value_counts()
    visitor_items: list[dict[str, Any]] = []
    for visitor_type in [*VISITOR_TYPE_ORDER, *sorted(set(visitor_counts.index) - set(VISITOR_TYPE_ORDER))]:
        count = int(visitor_counts.get(visitor_type, 0))
        if count == 0:
            continue
        visitor_items.append(
            {
                "key": visitor_type,
                "label": VISITOR_TYPE_LABELS.get(visitor_type, visitor_type.title()),
                "count": count,
                "share": round(count / max(len(visitors), 1) * 100.0, 1),
                "color": VISITOR_TYPE_COLORS.get(visitor_type, "#94a3b8"),
            }
        )

    class_items = active_color_items(
        buildings["final_functional_class"].fillna("Unknown").astype(str).tolist(),
        CLASS_COLORS,
        CLASS_ORDER,
    )

    poi_category_counts = (
        pois["analysis_category"].fillna("Other_POI").astype(str).value_counts().reindex(POI_CATEGORY_ORDER, fill_value=0)
    )
    poi_category_items = [
        {
            "key": key,
            "label": POI_CATEGORY_LABELS.get(key, key.replace("_", " ")),
            "count": int(poi_category_counts.get(key, 0)),
            "color": POI_CATEGORY_COLORS.get(key, "#94a3b8"),
        }
        for key in POI_CATEGORY_ORDER
        if int(poi_category_counts.get(key, 0)) > 0
    ]

    impact_sorted = sorted(impact_rows, key=impact_loss_percent, reverse=True)
    top_impact = [
        {
            "id": clean_text(row.get("id")),
            "name": clean_name(
                row.get("building_name"),
                building_name_lookup.get(clean_text(row.get("id")), "Building (No POI)"),
            ),
            "impact": round(impact_loss_percent(row), 1),
            "class": clean_text(row.get("final_functional_class"), "Unknown"),
        }
        for row in impact_sorted[:5]
    ]

    flow_items = sorted(
        [
            {
                "id": clean_text(node.get("id")),
                "name": clean_name(node.get("name"), f"Building {clean_text(node.get('id'))[:8]}"),
                "inflow": clean_int(node.get("inflow"), 0),
                "outflow": clean_int(node.get("outflow"), 0),
                "total_flow": clean_int(node.get("inflow"), 0) + clean_int(node.get("outflow"), 0),
            }
            for node in network.get("nodes", [])
        ],
        key=lambda item: item["total_flow"],
        reverse=True,
    )[:5]

    bounds = boundary.total_bounds
    center = [float((bounds[0] + bounds[2]) / 2), float((bounds[1] + bounds[3]) / 2)]
    return {
        "generated_from": {
            "boundary": "Website/data/boundary/canary_wharf.geojson",
            "buildings": "Website/data/final_data/2.1_canary_wharf_LSOA_buildings_processed.geojson",
            "pois": "Website/data/final_data/2.2_canary_wharf_LSOA_pois_processed.geojson",
            "network": "Website/data/final_data/0_workers_dependency_network.json",
            "visitors": "Website/data/final_data/1.1_user_ids_visitortype_optimized.json",
            "impact": "Website/data/final_data/3_counterfactual_simulation_results.json",
        },
        "map": {
            "center": center,
            "bounds": [[float(bounds[0]), float(bounds[1])], [float(bounds[2]), float(bounds[3])]],
        },
        "data": {
            "boundary": feature_collection_from_gdf(boundary, boundary_props),
            "buildings": feature_collection_from_gdf(buildings, building_props),
            "pois": feature_collection_from_gdf(pois, poi_props),
            "network": {"type": "FeatureCollection", "features": network_features},
        },
        "summary": {
            "building_count": int(len(buildings)),
            "poi_count": int(len(pois)),
            "assigned_poi_count": int(buildings["poi_sum"].fillna(0).sum()),
            "visitor_count": int(len(visitors)),
            "network_node_count": int(len(network.get("nodes", []))),
            "network_edge_count": int(len(network_features)),
            "impact_building_count": int(len(impact_rows)),
            "impact_clamp": round(impact_clamp, 2),
            "max_loss_percent": round(max(impact_values), 1) if impact_values else 0.0,
            "max_total_impact_percent": round(max(impact_values), 1) if impact_values else 0.0,
        },
        "legends": {
            "classes": class_items,
            "poi_categories": poi_category_items,
            "visitor_types": visitor_items,
            "impact": [
                {"label": "No modeled loss", "color": "#64748b"},
                {"label": "Lower loss", "color": "#fde68a"},
                {"label": "Moderate loss", "color": "#f97316"},
                {"label": "Higher loss", "color": "#b91c1c"},
            ],
            "network": [
                {"label": "Lower weight percentile", "color": "#44d3ff"},
                {"label": "Higher weight percentile", "color": "#ff63b5"},
            ],
        },
        "rankings": {
            "top_impact": top_impact,
            "top_flow": flow_items,
        },
    }


def html_template(payload_json: str) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Canary Wharf Urban Dependency Atlas</title>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --panel: rgba(9, 15, 27, 0.88);
      --panel-strong: rgba(9, 15, 27, 0.96);
      --border: rgba(226, 232, 240, 0.16);
      --text: #f8fafc;
      --muted: #a7b4c8;
      --accent: #64d2ff;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; }
    body {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #050812;
      color: var(--text);
    }
    #map { position: fixed; inset: 0; }
    .context-panel {
      position: fixed;
      top: 20px;
      left: 20px;
      z-index: 5;
      width: min(380px, calc(100vw - 124px));
      max-height: calc(100vh - 40px);
      overflow: auto;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.38);
      backdrop-filter: blur(14px);
    }
    .eyebrow {
      margin: 0 0 8px 0;
      font-size: 11px;
      letter-spacing: 0;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 760;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.12;
      letter-spacing: 0;
      font-weight: 760;
    }
    .subtitle {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 16px 0;
    }
    .metric {
      min-width: 0;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.62);
    }
    .metric-value {
      display: block;
      font-size: 20px;
      font-weight: 760;
      line-height: 1.05;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric-label {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
    }
    .legend-title, .ranking-title {
      margin: 14px 0 8px;
      font-size: 12px;
      color: #dbeafe;
      font-weight: 760;
    }
    .legend-list, .ranking-list {
      display: grid;
      gap: 7px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .legend-row, .ranking-row {
      display: grid;
      grid-template-columns: 16px minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      color: #dbe4f0;
      font-size: 12px;
      line-height: 1.25;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(255, 255, 255, 0.28);
    }
    .bar-track {
      height: 8px;
      margin-top: 4px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.22);
    }
    .bar-fill { height: 100%; border-radius: 999px; }
    .note {
      margin: 14px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .side-nav {
      position: fixed;
      top: 50%;
      right: 18px;
      z-index: 6;
      transform: translateY(-50%);
      width: 176px;
      display: grid;
      gap: 8px;
      padding: 10px;
      background: var(--panel-strong);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.42);
      backdrop-filter: blur(14px);
    }
    .nav-button {
      width: 100%;
      display: grid;
      grid-template-columns: 30px minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      padding: 10px 9px;
      color: #cbd5e1;
      background: rgba(15, 23, 42, 0.66);
      border: 1px solid rgba(226, 232, 240, 0.12);
      border-radius: 8px;
      cursor: pointer;
      font: inherit;
      text-align: left;
    }
    .nav-button:hover, .nav-button.active {
      color: #ffffff;
      border-color: rgba(100, 210, 255, 0.72);
      background: rgba(30, 41, 59, 0.92);
    }
    .about-button {
      display: block;
      width: 100%;
      padding: 10px 9px;
      color: #07111b;
      background: #8ee7ff;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 8px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 800;
      text-align: center;
    }
    .about-button:hover {
      background: #c7f4ff;
    }
    .nav-index {
      display: grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: rgba(100, 210, 255, 0.12);
      color: #8ee7ff;
      font-size: 11px;
      font-weight: 800;
    }
    .nav-label {
      min-width: 0;
      font-size: 12px;
      font-weight: 720;
      line-height: 1.16;
    }
    .view-controls {
      position: fixed;
      left: 20px;
      bottom: 20px;
      z-index: 5;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      backdrop-filter: blur(14px);
    }
    .control-button {
      min-width: 44px;
      padding: 8px 10px;
      border: 1px solid rgba(226, 232, 240, 0.14);
      border-radius: 8px;
      color: #dbe4f0;
      background: rgba(15, 23, 42, 0.72);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 720;
    }
    .control-button:hover {
      border-color: rgba(100, 210, 255, 0.72);
      color: #ffffff;
    }
    .maplibregl-popup-content {
      max-width: min(360px, calc(100vw - 32px));
      padding: 14px;
      color: #f8fafc;
      background: rgba(9, 15, 27, 0.96);
      border: 1px solid rgba(226, 232, 240, 0.16);
      border-radius: 8px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.38);
      font: 12px/1.4 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .maplibregl-popup-tip { border-top-color: rgba(9, 15, 27, 0.96) !important; }
    .maplibregl-popup-close-button { color: #ffffff; font-size: 18px; padding: 4px 8px; }
    .popup-title { margin: 0 18px 8px 0; font-size: 14px; font-weight: 780; }
    .popup-grid { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 10px; }
    .popup-grid span:nth-child(odd) { color: #9fb0c8; }
    .intro-overlay {
      position: fixed;
      inset: 0;
      z-index: 30;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(2, 6, 15, 0.62);
      backdrop-filter: blur(8px);
      transition: opacity 220ms ease;
    }
    .intro-overlay.hidden {
      opacity: 0;
      pointer-events: none;
    }
    .intro-window {
      position: relative;
      width: min(760px, 100%);
      max-height: min(86vh, 840px);
      overflow: auto;
      padding: 26px;
      background: rgba(9, 15, 27, 0.96);
      border: 1px solid rgba(226, 232, 240, 0.18);
      border-radius: 8px;
      box-shadow: 0 24px 72px rgba(0, 0, 0, 0.5);
    }
    .intro-logos {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
    }
    .logo-card {
      display: grid;
      place-items: center;
      min-height: 58px;
      padding: 10px 14px;
      background: #ffffff;
      border-radius: 8px;
    }
    .logo-card img {
      display: block;
      max-width: 160px;
      max-height: 44px;
      object-fit: contain;
    }
    .logo-card.logo-wide img {
      max-width: 220px;
      max-height: 34px;
    }
    .intro-kicker {
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .intro-title {
      margin: 0;
      max-width: 680px;
      font-size: 28px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .intro-credits {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 8px 14px;
      margin: 18px 0;
      padding: 14px 0;
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      color: #dbe4f0;
      font-size: 13px;
      line-height: 1.35;
    }
    .intro-credits dt {
      color: #8ee7ff;
      font-weight: 800;
    }
    .intro-credits dd {
      margin: 0;
    }
    .intro-copy {
      display: grid;
      gap: 10px;
      color: #cbd5e1;
      font-size: 14px;
      line-height: 1.55;
    }
    .intro-copy p {
      margin: 0;
    }
    .intro-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 20px;
    }
    .intro-close-button {
      padding: 10px 14px;
      color: #07111b;
      background: #8ee7ff;
      border: 0;
      border-radius: 8px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
    }
    .intro-close-button:hover {
      background: #c7f4ff;
    }
    .intro-icon-close {
      position: absolute;
      top: 12px;
      right: 12px;
      width: 34px;
      height: 34px;
      color: #dbe4f0;
      background: rgba(15, 23, 42, 0.86);
      border: 1px solid rgba(226, 232, 240, 0.16);
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      font-size: 18px;
      line-height: 1;
    }
    .intro-icon-close:hover {
      color: #ffffff;
      border-color: rgba(100, 210, 255, 0.72);
    }
    @media (max-width: 760px) {
      .context-panel {
        top: 12px;
        left: 12px;
        width: calc(100vw - 96px);
        max-height: 50vh;
        padding: 14px;
      }
      h1 { font-size: 20px; }
      .metrics { grid-template-columns: 1fr; }
      .side-nav {
        right: 10px;
        width: 74px;
        padding: 8px;
      }
      .nav-button {
        grid-template-columns: 1fr;
        justify-items: center;
        gap: 5px;
        padding: 8px 6px;
        text-align: center;
      }
      .nav-label { font-size: 10px; }
      .view-controls { left: 12px; bottom: 12px; }
      .about-button { font-size: 11px; padding: 8px 6px; }
      .intro-overlay { padding: 12px; }
      .intro-window { padding: 18px; }
      .intro-title { font-size: 22px; }
      .intro-credits { grid-template-columns: 1fr; gap: 4px; }
      .logo-card img { max-width: 132px; }
      .logo-card.logo-wide img { max-width: 178px; }
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="context-panel" aria-live="polite">
    <p class="eyebrow">Canary Wharf Atlas</p>
    <h1 id="panel-title"></h1>
    <p id="panel-subtitle" class="subtitle"></p>
    <div id="metrics" class="metrics"></div>
    <div id="legend"></div>
    <div id="ranking"></div>
    <p id="panel-note" class="note"></p>
  </section>
  <nav class="side-nav" aria-label="Map views">
    <button id="about-button" class="about-button" type="button">About</button>
    <button class="nav-button active" data-view="network" type="button"><span class="nav-index">01</span><span class="nav-label">Dependency Network</span></button>
    <button class="nav-button" data-view="functions" type="button"><span class="nav-index">02</span><span class="nav-label">Buildings + POIs</span></button>
    <button class="nav-button" data-view="impact" type="button"><span class="nav-index">03</span><span class="nav-label">Impact Model</span></button>
    <button class="nav-button" data-view="visitors" type="button"><span class="nav-index">04</span><span class="nav-label">Visitor Mix</span></button>
  </nav>
  <div class="view-controls">
    <button id="reset-view" class="control-button" type="button">Reset</button>
    <button id="clear-selection" class="control-button" type="button">Clear</button>
  </div>
  <div id="intro-overlay" class="intro-overlay" role="dialog" aria-modal="true" aria-labelledby="intro-title">
    <section class="intro-window">
      <button id="intro-icon-close" class="intro-icon-close" type="button" aria-label="Close introduction">x</button>
      <div class="intro-logos" aria-label="Project logos">
        <span class="logo-card"><img src="logo/cusp.png" alt="NYU CUSP logo" /></span>
        <span class="logo-card logo-wide"><img src="logo/foster+partners.png" alt="Foster + Partners logo" /></span>
      </div>
      <p class="intro-kicker">Canary Wharf Urban Dependency Atlas</p>
      <h2 id="intro-title" class="intro-title">Urban Colocation Intelligence: Mapping Amenity Clusters and Dependency Networks Using Open-Source POI Data</h2>
      <dl class="intro-credits">
        <dt>Team</dt>
        <dd>Archy Guo, Kunjal Bhatta, Ruoyu Li</dd>
        <dt>Mentors</dt>
        <dd>Takahiro Yabe, Ph.D.; Vaidehi Raipat (NYU Tandon, CUSP/TMI)</dd>
        <dt>Sponsors</dt>
        <dd>Benjamin Michel, Laurens Versluis, Mateo Neira (Foster + Partners)</dd>
      </dl>
      <div class="intro-copy">
        <p>Canary Wharf is transitioning from a financial hub toward a mixed-use district, making it a living laboratory for how offices, amenities, residents, workers, and visitors depend on one another.</p>
        <p>This atlas combines large-scale mobility records with Overture POIs, building functions, and local dependency networks. It maps visitor profiles, building-level amenity clusters, worker transitions, and counterfactual shocks to reveal how everyday urban synergies support district vitality.</p>
      </div>
      <div class="intro-actions">
        <button id="intro-close" class="intro-close-button" type="button">Enter Atlas</button>
      </div>
    </section>
  </div>
  <script id="atlas-payload" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById('atlas-payload').textContent);
    const summary = payload.summary;
    const legends = payload.legends;
    const rankings = payload.rankings;
    const bounds = payload.map.bounds;
    let activeView = 'network';
    let selectedBuildingId = null;
    const viewCameras = {
      network: { zoom: 15.30, pitch: 56, bearing: -22 },
      functions: { zoom: 15.36, pitch: 55, bearing: -20 },
      impact: { zoom: 15.32, pitch: 54, bearing: -18 },
      visitors: { zoom: 15.24, pitch: 52, bearing: -16 }
    };
    const cameraForView = (viewId) => viewCameras[viewId] || viewCameras.network;

    const rasterStyle = {
      version: 8,
      sources: {
        cartoDark: {
          type: 'raster',
          tiles: [
            'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'
          ],
          tileSize: 256,
          attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
        }
      },
      layers: [{ id: 'cartoDark', type: 'raster', source: 'cartoDark' }]
    };

    const map = new maplibregl.Map({
      container: 'map',
      style: rasterStyle,
      center: payload.map.center,
      zoom: viewCameras.network.zoom,
      pitch: viewCameras.network.pitch,
      bearing: viewCameras.network.bearing,
      antialias: true
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    const layerIds = [
      'boundary-fill', 'boundary-line',
      'buildings-context', 'buildings-functional', 'buildings-impact',
      'building-outline', 'selected-building-outline',
      'poi-halo', 'poi-points',
      'network-glow', 'network-arcs'
    ];

    const views = {
      visitors: {
        title: 'Visitor Mix',
        subtitle: 'AOI-level visitor classification for 18,077 anonymized users, shown with the Canary Wharf study boundary as spatial context.',
        metrics: [
          ['Users', number(summary.visitor_count)],
          ['Dominant Type', dominantVisitorLabel()],
          ['Boundary Units', String(payload.data.boundary.features.length)],
          ['Buildings Context', number(summary.building_count)]
        ],
        legendTitle: 'Visitor types',
        legend: legends.visitor_types,
        rankingTitle: '',
        ranking: [],
        note: 'The visitor file contains user labels rather than per-user coordinates, so this view maps the study area and summarizes the composition at AOI level.',
        visible: ['boundary-fill', 'boundary-line', 'buildings-context']
      },
      functions: {
        title: 'Buildings + POIs',
        subtitle: '3D building footprints share one functional-class palette, with POIs colored by analysis category.',
        metrics: [
          ['Buildings', number(summary.building_count)],
          ['POIs', number(summary.poi_count)],
          ['Assigned POIs', number(summary.assigned_poi_count)],
          ['Max Building POIs', maxBuildingPoi()]
        ],
        legendTitle: 'Functional classes',
        legend: legends.classes,
        rankingTitle: 'POI categories',
        ranking: legends.poi_categories.map(item => ({ label: item.label, value: number(item.count), color: item.color, share: item.count / Math.max(summary.poi_count, 1) * 100 })),
        note: 'Click buildings or POIs for names, categories, heights, and assignment details.',
        visible: ['boundary-line', 'buildings-functional', 'building-outline', 'poi-halo', 'poi-points']
      },
      network: {
        title: 'Worker Dependency Network',
        subtitle: 'Curved links connect source and target buildings. Color follows weight percentile; width follows transitions percentile.',
        metrics: [
          ['Network Nodes', number(summary.network_node_count)],
          ['Visible Links', number(summary.network_edge_count)],
          ['Buildings', number(summary.building_count)],
          ['Top Flow', topFlowValue()]
        ],
        legendTitle: 'Network encoding',
        legend: legends.network,
        rankingTitle: 'Highest total flow',
        ranking: rankings.top_flow.map(item => ({ label: item.name, value: number(item.total_flow), color: '#64d2ff' })),
        note: 'Click a building to keep related links prominent and dim unrelated links. Click Clear to restore the full network.',
        visible: ['boundary-line', 'buildings-context', 'building-outline', 'network-glow', 'network-arcs', 'selected-building-outline']
      },
      impact: {
        title: 'Counterfactual Impact Model',
        subtitle: 'Buildings are colored by modeled loss percent after the counterfactual shock simulation.',
        metrics: [
          ['Modeled Buildings', number(summary.impact_building_count)],
          ['Largest Loss', `${summary.max_loss_percent ?? summary.max_total_impact_percent}%`],
          ['Initial Shock', '-20%'],
          ['Impact Scale', `${summary.impact_clamp}%`]
        ],
        legendTitle: 'Modeled loss percent',
        legend: legends.impact,
        rankingTitle: 'Most affected buildings',
        ranking: rankings.top_impact.map(item => ({ label: item.name, value: `${item.impact}%`, color: '#f97316' })),
        note: 'Darker red indicates larger modeled loss. Buildings with no modeled loss remain cool gray.',
        visible: ['boundary-line', 'buildings-impact', 'building-outline']
      }
    };

    function number(value) {
      return Number(value || 0).toLocaleString('en-US');
    }

    function dominantVisitorLabel() {
      const items = legends.visitor_types || [];
      if (!items.length) return 'n/a';
      const top = [...items].sort((a, b) => b.count - a.count)[0];
      return top.label;
    }

    function topFlowValue() {
      return rankings.top_flow?.length ? number(rankings.top_flow[0].total_flow) : 'n/a';
    }

    function maxBuildingPoi() {
      const values = payload.data.buildings.features.map(feature => Number(feature.properties.poi_sum || 0));
      return number(Math.max(...values, 0));
    }

    function addSourcesAndLayers() {
      map.addSource('boundary', { type: 'geojson', data: payload.data.boundary });
      map.addSource('buildings', { type: 'geojson', data: payload.data.buildings });
      map.addSource('pois', { type: 'geojson', data: payload.data.pois });
      map.addSource('network', { type: 'geojson', data: payload.data.network });

      map.addLayer({
        id: 'boundary-fill',
        type: 'fill',
        source: 'boundary',
        paint: {
          'fill-color': '#0f2537',
          'fill-opacity': 0.28
        }
      });
      map.addLayer({
        id: 'boundary-line',
        type: 'line',
        source: 'boundary',
        paint: {
          'line-color': '#dbeafe',
          'line-width': 1.4,
          'line-opacity': 0.82
        }
      });
      map.addLayer({
        id: 'buildings-context',
        type: 'fill-extrusion',
        source: 'buildings',
        paint: {
          'fill-extrusion-color': '#627086',
          'fill-extrusion-height': ['get', 'elevation_m'],
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': 0.36
        }
      });
      map.addLayer({
        id: 'buildings-functional',
        type: 'fill-extrusion',
        source: 'buildings',
        paint: {
          'fill-extrusion-color': ['get', 'class_color'],
          'fill-extrusion-height': ['get', 'elevation_m'],
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': 0.78
        }
      });
      map.addLayer({
        id: 'buildings-impact',
        type: 'fill-extrusion',
        source: 'buildings',
        paint: {
          'fill-extrusion-color': ['get', 'impact_color'],
          'fill-extrusion-height': ['get', 'elevation_m'],
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': 0.82
        }
      });
      map.addLayer({
        id: 'building-outline',
        type: 'line',
        source: 'buildings',
        paint: {
          'line-color': 'rgba(232, 240, 255, 0.58)',
          'line-width': 0.55,
          'line-opacity': 0.7
        }
      });
      map.addLayer({
        id: 'selected-building-outline',
        type: 'line',
        source: 'buildings',
        filter: ['==', ['get', 'id'], ''],
        paint: {
          'line-color': '#ffffff',
          'line-width': 3,
          'line-opacity': 0.94
        }
      });
      map.addLayer({
        id: 'network-glow',
        type: 'line',
        source: 'network',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['get', 'line_color'],
          'line-width': ['+', ['get', 'line_width'], 3],
          'line-opacity': 0.14
        }
      });
      map.addLayer({
        id: 'network-arcs',
        type: 'line',
        source: 'network',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['get', 'line_color'],
          'line-width': ['get', 'line_width'],
          'line-opacity': 0.78
        }
      });
      map.addLayer({
        id: 'poi-halo',
        type: 'circle',
        source: 'pois',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 13, 3, 16, 8],
          'circle-color': ['get', 'poi_color'],
          'circle-opacity': 0.18,
          'circle-blur': 0.7
        }
      });
      map.addLayer({
        id: 'poi-points',
        type: 'circle',
        source: 'pois',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 13, 2.4, 16, 5.2],
          'circle-color': ['get', 'poi_color'],
          'circle-stroke-color': '#06111f',
          'circle-stroke-width': 0.8,
          'circle-opacity': 0.92
        }
      });

      setupInteractions();
      setView('network');
    }

    function setupInteractions() {
      const buildingLayers = ['buildings-context', 'buildings-functional', 'buildings-impact'];
      buildingLayers.forEach(layerId => {
        map.on('click', layerId, event => {
          const feature = event.features?.[0];
          if (!feature) return;
          selectedBuildingId = String(feature.properties.id || '');
          if (activeView === 'network') applyNetworkSelection();
          showPopup(event.lngLat, buildingPopup(feature.properties));
        });
        setPointer(layerId);
      });
      ['poi-points'].forEach(layerId => {
        map.on('click', layerId, event => {
          const feature = event.features?.[0];
          if (!feature) return;
          showPopup(event.lngLat, poiPopup(feature.properties));
        });
        setPointer(layerId);
      });
      ['network-arcs'].forEach(layerId => {
        map.on('click', layerId, event => {
          const feature = event.features?.[0];
          if (!feature) return;
          showPopup(event.lngLat, networkPopup(feature.properties));
        });
        setPointer(layerId);
      });
      map.on('click', event => {
        const features = map.queryRenderedFeatures(event.point, { layers: ['buildings-context', 'buildings-functional', 'buildings-impact', 'poi-points', 'network-arcs'] });
        if (!features.length) clearSelection();
      });
    }

    function setPointer(layerId) {
      map.on('mouseenter', layerId, () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', layerId, () => { map.getCanvas().style.cursor = ''; });
    }

    function showPopup(lngLat, html) {
      new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '360px' })
        .setLngLat(lngLat)
        .setHTML(html)
        .addTo(map);
    }

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
    }

    function detailGrid(rows) {
      return `<div class="popup-grid">${rows.map(([key, value]) => `<span>${esc(key)}</span><span>${esc(value)}</span>`).join('')}</div>`;
    }

    function buildingPopup(props) {
      const samples = Array.isArray(props.sample_poi_names) && props.sample_poi_names.length
        ? `<p class="note">${props.sample_poi_names.map(esc).join(', ')}${Number(props.remaining_poi_name_count || 0) > 0 ? `, +${props.remaining_poi_name_count} more` : ''}</p>`
        : '';
      return `
        <h2 class="popup-title">${esc(props.name)}</h2>
        ${detailGrid([
          ['Functional class', props.final_functional_class],
          ['Building class', props.building_class],
          ['Height', `${props.elevation_m} m`],
          ['POIs', number(props.poi_sum)],
          ['Worker inflow', number(props.inflow)],
          ['Worker outflow', number(props.outflow)],
          ['Modeled loss', `${props.total_impact_percent}%`]
        ])}
        ${samples}
      `;
    }

    function poiPopup(props) {
      return `
        <h2 class="popup-title">${esc(props.name)}</h2>
        ${detailGrid([
          ['Analysis category', props.analysis_category_label],
          ['Taxonomy', props.taxonomy_primary],
          ['Status', props.operating_status],
          ['Confidence', props.confidence],
          ['Building match', props.building_match_method]
        ])}
      `;
    }

    function networkPopup(props) {
      return `
        <h2 class="popup-title">${esc(props.source_name)} to ${esc(props.target_name)}</h2>
        ${detailGrid([
          ['Weight', Number(props.weight || 0).toFixed(6)],
          ['Weight percentile', `P${Math.round(Number(props.weight_percentile || 0) * 100)}`],
          ['Transitions', number(props.transitions)],
          ['Transition percentile', `P${Math.round(Number(props.transition_percentile || 0) * 100)}`]
        ])}
      `;
    }

    function setView(viewId) {
      activeView = viewId;
      const view = views[viewId];
      layerIds.forEach(layerId => {
        if (!map.getLayer(layerId)) return;
        map.setLayoutProperty(layerId, 'visibility', view.visible.includes(layerId) ? 'visible' : 'none');
      });
      document.querySelectorAll('.nav-button').forEach(button => {
        button.classList.toggle('active', button.dataset.view === viewId);
      });
      renderPanel(view);
      clearSelection(false);
      map.easeTo({ ...cameraForView(viewId), center: payload.map.center, duration: 850 });
    }

    function renderPanel(view) {
      document.getElementById('panel-title').textContent = view.title;
      document.getElementById('panel-subtitle').textContent = view.subtitle;
      document.getElementById('metrics').innerHTML = view.metrics.map(([label, value]) => `
        <div class="metric"><span class="metric-value">${esc(value)}</span><span class="metric-label">${esc(label)}</span></div>
      `).join('');
      document.getElementById('legend').innerHTML = view.legend?.length ? `
        <div class="legend-title">${esc(view.legendTitle)}</div>
        <ul class="legend-list">${view.legend.map(item => legendRow(item)).join('')}</ul>
      ` : '';
      document.getElementById('ranking').innerHTML = view.ranking?.length ? `
        <div class="ranking-title">${esc(view.rankingTitle)}</div>
        <ul class="ranking-list">${view.ranking.map(item => rankingRow(item)).join('')}</ul>
      ` : '';
      document.getElementById('panel-note').textContent = view.note;
    }

    function legendRow(item) {
      const trailing = item.share !== undefined ? `${Number(item.share).toFixed(1)}%` : (item.count !== undefined ? number(item.count) : '');
      const bar = item.share !== undefined ? `<div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, Number(item.share || 0))}%; background:${esc(item.color)}"></div></div>` : '';
      return `<li class="legend-row"><span class="swatch" style="background:${esc(item.color)}"></span><span>${esc(item.label)}${bar}</span><span>${esc(trailing)}</span></li>`;
    }

    function rankingRow(item) {
      return `<li class="ranking-row"><span class="swatch" style="background:${esc(item.color || '#64d2ff')}"></span><span>${esc(item.label)}</span><span>${esc(item.value)}</span></li>`;
    }

    function applyNetworkSelection() {
      if (!map.getLayer('network-arcs')) return;
      map.setFilter('selected-building-outline', selectedBuildingId ? ['==', ['get', 'id'], selectedBuildingId] : ['==', ['get', 'id'], '']);
      const related = selectedBuildingId
        ? ['any', ['==', ['get', 'source_id'], selectedBuildingId], ['==', ['get', 'target_id'], selectedBuildingId]]
        : true;
      map.setPaintProperty('network-arcs', 'line-opacity', selectedBuildingId ? ['case', related, 0.9, 0.12] : 0.78);
      map.setPaintProperty('network-glow', 'line-opacity', selectedBuildingId ? ['case', related, 0.24, 0.04] : 0.14);
      map.setPaintProperty('network-arcs', 'line-width', selectedBuildingId ? ['case', related, ['+', ['get', 'line_width'], 1.2], ['*', ['get', 'line_width'], 0.55]] : ['get', 'line_width']);
    }

    function clearSelection(closePopups = true) {
      selectedBuildingId = null;
      applyNetworkSelection();
      if (closePopups) {
        document.querySelectorAll('.maplibregl-popup').forEach(element => element.remove());
      }
    }

    function resetView() {
      map.easeTo({ ...cameraForView(activeView), center: payload.map.center, duration: 650 });
    }

    function openIntro() {
      document.getElementById('intro-overlay').classList.remove('hidden');
    }

    function closeIntro() {
      document.getElementById('intro-overlay').classList.add('hidden');
    }

    document.querySelectorAll('.nav-button').forEach(button => {
      button.addEventListener('click', () => setView(button.dataset.view));
    });
    document.getElementById('about-button').addEventListener('click', openIntro);
    document.getElementById('intro-close').addEventListener('click', closeIntro);
    document.getElementById('intro-icon-close').addEventListener('click', closeIntro);
    document.getElementById('intro-overlay').addEventListener('click', (event) => {
      if (event.target.id === 'intro-overlay') {
        closeIntro();
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        closeIntro();
      }
    });
    document.getElementById('reset-view').addEventListener('click', resetView);
    document.getElementById('clear-selection').addEventListener('click', () => clearSelection(true));

    map.on('load', addSourcesAndLayers);
  </script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)


def main() -> None:
    project_root = resolve_project_root()
    output_dir = project_root / "Website" / "website"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(project_root)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    index_html = html_template(payload_json)
    output_path = output_dir / "index.html"
    output_path.write_text(index_html, encoding="utf-8")
    print(f"Wrote {output_path.relative_to(project_root)}")
    print(
        json.dumps(
            {
                "buildings": payload["summary"]["building_count"],
                "pois": payload["summary"]["poi_count"],
                "network_edges": payload["summary"]["network_edge_count"],
                "visitors": payload["summary"]["visitor_count"],
                "impact_buildings": payload["summary"]["impact_building_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
