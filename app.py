"""
app.py — NYC Taxi Trip Duration Predictor (Streamlit frontend)

Run with:
    streamlit run app.py

Expects a trained model file "lgb_trip_duration_model.pkl" (LightGBM regressor
trained on log1p(trip_duration)) in the same folder. See export_model.py /
the README for how to produce that file from your notebook.
"""

import datetime as dt

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from feature_engineering import build_feature_row, LAT_RANGE, LON_RANGE

MODEL_PATH = "lgb_trip_duration_model.pkl"

st.set_page_config(page_title="NYC Taxi Trip Duration Predictor", page_icon="🚕", layout="wide")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def in_range(value, value_range):
    return value_range[0] <= value <= value_range[1]


st.title("🚕 NYC Taxi Trip Duration Predictor")
st.caption(
    "Predicts trip duration using the LightGBM model trained on engineered "
    "time, distance, bearing, and airport features from the NYC taxi dataset."
)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{MODEL_PATH}`. Export your trained `lgb_model` from the "
        "notebook (see export_model.py) and place the .pkl file next to app.py."
    )
    st.stop()

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Trip details")

    c1, c2 = st.columns(2)
    with c1:
        pickup_date = st.date_input("Pickup date", value=dt.date(2016, 6, 15))
    with c2:
        pickup_time = st.time_input("Pickup time", value=dt.time(18, 30))

    c3, c4 = st.columns(2)
    with c3:
        vendor_id = st.selectbox("Vendor ID", options=[1, 2])
    with c4:
        passenger_count = st.slider("Passenger count", min_value=1, max_value=6, value=1)

    st.markdown("**Pickup location**")
    p1, p2 = st.columns(2)
    with p1:
        pickup_lat = st.number_input("Pickup latitude", value=40.7580, format="%.6f")
    with p2:
        pickup_lon = st.number_input("Pickup longitude", value=-73.9855, format="%.6f")

    st.markdown("**Dropoff location**")
    d1, d2 = st.columns(2)
    with d1:
        dropoff_lat = st.number_input("Dropoff latitude", value=40.6413, format="%.6f")
    with d2:
        dropoff_lon = st.number_input("Dropoff longitude", value=-73.7781, format="%.6f")

    st.caption(
        f"Valid NYC range used during training — latitude {LAT_RANGE}, longitude {LON_RANGE}."
    )

    predict_clicked = st.button("Predict trip duration", type="primary", use_container_width=True)

with right:
    st.subheader("Pickup → Dropoff map")

    points_df = pd.DataFrame({
        "lat": [pickup_lat, dropoff_lat],
        "lon": [pickup_lon, dropoff_lon],
        "label": ["Pickup", "Dropoff"],
        "color": [[0, 160, 0], [200, 30, 30]],
    })
    line_df = pd.DataFrame({
        "start": [[pickup_lon, pickup_lat]],
        "end": [[dropoff_lon, dropoff_lat]],
    })

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=150,
        pickable=True,
    )
    line_layer = pdk.Layer(
        "LineLayer",
        data=line_df,
        get_source_position="start",
        get_target_position="end",
        get_color=[30, 90, 200],
        get_width=3,
    )
    view_state = pdk.ViewState(
        latitude=(pickup_lat + dropoff_lat) / 2,
        longitude=(pickup_lon + dropoff_lon) / 2,
        zoom=11,
    )
    st.pydeck_chart(pdk.Deck(layers=[scatter_layer, line_layer], initial_view_state=view_state))

if predict_clicked:
    if not (in_range(pickup_lat, LAT_RANGE) and in_range(dropoff_lat, LAT_RANGE)
            and in_range(pickup_lon, LON_RANGE) and in_range(dropoff_lon, LON_RANGE)):
        st.warning(
            "One or more coordinates fall outside the NYC range the model was "
            "trained on. The prediction may be unreliable."
        )

    pickup_datetime = dt.datetime.combine(pickup_date, pickup_time)
    X, derived = build_feature_row(
        pickup_datetime, vendor_id, passenger_count,
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
    )

    log_pred = model.predict(X)[0]
    duration_seconds = max(float(np.expm1(log_pred)), 0)
    minutes, seconds = divmod(int(round(duration_seconds)), 60)

    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted duration", f"{minutes} min {seconds}s")
    m2.metric("Straight-line distance", f"{derived['distance_km']:.2f} km")
    m3.metric("Airport trip?", "Yes" if derived["is_airport"] else "No")

    with st.expander("Engineered features used for this prediction"):
        st.dataframe(X.T.rename(columns={0: "value"}))
