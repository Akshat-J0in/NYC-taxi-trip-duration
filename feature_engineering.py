"""
feature_engineering.py

Mirrors the exact feature-engineering steps from the notebook
(nyc-taxi-feature-engineering-for-prediction.ipynb) so that the Streamlit
app builds features in precisely the same way the model was trained on.

If you change anything in the notebook's feature engineering, mirror the
change here too, or predictions will silently go wrong.
"""

import numpy as np
import pandas as pd

# --- Fixed reference points used in the notebook ---
JFK_LAT, JFK_LON = 40.6413, -73.7781
LGA_LAT, LGA_LON = 40.7769, -73.8740
MANHATTAN_LAT, MANHATTAN_LON = 40.7580, -73.9855

# Column order the model was trained on (must match notebook's feature_cols)
FEATURE_COLS = [
    'vendor_id', 'passenger_count', 'pickup_longitude',
    'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
    'distance_KM', 'manhattan_distance', 'bearing',
    'hours', 'month', 'day_of_week',
    'is_weekend', 'is_rush_hour', 'hour_day_interaction', 'is_jfk',
    'is_lga', 'is_airport', 'pickup_distance_from_center',
    'dropoff_distance_from_center'
]

# Valid ranges used during data cleaning -- reused here to validate user input
LAT_RANGE = (40.4, 41.0)
LON_RANGE = (-74.3, -73.6)


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Identical to the notebook's haversine()."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing in degrees [0, 360). Identical to the notebook's version."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def build_feature_row(pickup_datetime, vendor_id, passenger_count,
                       pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Builds a single-row DataFrame of model-ready features from raw trip inputs,
    in the exact column order the model expects (FEATURE_COLS).
    """
    distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    manhattan_distance = abs(pickup_lat - dropoff_lat) + abs(pickup_lon - dropoff_lon)
    bearing = calculate_bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    hours = pickup_datetime.hour
    month = pickup_datetime.month
    day_of_week = pickup_datetime.weekday()  # Monday=0 ... Sunday=6, matches pandas dt.dayofweek

    is_weekend = int(day_of_week >= 5)
    is_rush_hour = int(8 <= hours <= 17)
    hour_day_interaction = hours * day_of_week

    pickup_dist_jfk = haversine(pickup_lat, pickup_lon, JFK_LAT, JFK_LON)
    dropoff_dist_jfk = haversine(dropoff_lat, dropoff_lon, JFK_LAT, JFK_LON)
    pickup_dist_lga = haversine(pickup_lat, pickup_lon, LGA_LAT, LGA_LON)
    dropoff_dist_lga = haversine(dropoff_lat, dropoff_lon, LGA_LAT, LGA_LON)

    is_jfk = int((pickup_dist_jfk <= 2) or (dropoff_dist_jfk <= 2))
    is_lga = int((pickup_dist_lga <= 2) or (dropoff_dist_lga <= 2))
    is_airport = int(is_jfk or is_lga)

    pickup_distance_from_center = haversine(pickup_lat, pickup_lon, MANHATTAN_LAT, MANHATTAN_LON)
    dropoff_distance_from_center = haversine(dropoff_lat, dropoff_lon, MANHATTAN_LAT, MANHATTAN_LON)

    row = {
        'vendor_id': vendor_id,
        'passenger_count': passenger_count,
        'pickup_longitude': pickup_lon,
        'pickup_latitude': pickup_lat,
        'dropoff_longitude': dropoff_lon,
        'dropoff_latitude': dropoff_lat,
        'distance_KM': distance_km,
        'manhattan_distance': manhattan_distance,
        'bearing': bearing,
        'hours': hours,
        'month': month,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'hour_day_interaction': hour_day_interaction,
        'is_jfk': is_jfk,
        'is_lga': is_lga,
        'is_airport': is_airport,
        'pickup_distance_from_center': pickup_distance_from_center,
        'dropoff_distance_from_center': dropoff_distance_from_center,
    }

    return pd.DataFrame([row])[FEATURE_COLS], {
        'distance_km': distance_km,
        'manhattan_distance_deg': manhattan_distance,
        'bearing': bearing,
        'is_airport': is_airport,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend,
    }
