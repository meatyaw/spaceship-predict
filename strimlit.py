import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

encoders = joblib.load(Path(__file__).parent / "artifacts" / "preprocessor.pkl")
model    = joblib.load(Path(__file__).parent / "artifacts" / "model.pkl")

SPENDING_COLS        = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_FEATURES = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Age_group"]
NUMERICAL_FEATURES   = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "Group_size", "Solo", "Family_size", "TotalSpending",
    "HasSpending", "NoSpending", "Age_missing", "CryoSleep_missing",
]

def build_features(raw: dict) -> pd.DataFrame:
    total = sum([raw["RoomService"], raw["FoodCourt"], raw["ShoppingMall"], raw["Spa"], raw["VRDeck"]])
    age   = raw["Age"]

    row = {
        "HomePlanet"         : raw["HomePlanet"],
        "CryoSleep"          : str(raw["CryoSleep"]),
        "Destination"        : raw["Destination"],
        "VIP"                : str(raw["VIP"]),
        "Deck"               : raw["Deck"],
        "Side"               : raw["Side"],
        "Age_group"          : ("Child" if age<=12 else "Teen" if age<=18 else
                                "Young_Adult" if age<=30 else "Adult" if age<=50 else "Senior"),
        "Age"                : age,
        "RoomService"        : raw["RoomService"],
        "FoodCourt"          : raw["FoodCourt"],
        "ShoppingMall"       : raw["ShoppingMall"],
        "Spa"                : raw["Spa"],
        "VRDeck"             : raw["VRDeck"],
        "Cabin_num"          : raw["Cabin_num"],
        "Group_size"         : raw["Group_size"],
        "Solo"               : int(raw["Group_size"] == 1),
        "Family_size"        : raw["Family_size"],
        "TotalSpending"      : total,
        "HasSpending"        : int(total > 0),
        "NoSpending"         : int(total == 0),
        "Age_missing"        : 0,
        "CryoSleep_missing"  : 0,
        "RoomService_ratio"  : raw["RoomService"]  / (total + 1),
        "FoodCourt_ratio"    : raw["FoodCourt"]    / (total + 1),
        "ShoppingMall_ratio" : raw["ShoppingMall"] / (total + 1),
        "Spa_ratio"          : raw["Spa"]          / (total + 1),
        "VRDeck_ratio"       : raw["VRDeck"]       / (total + 1),
    }

    df = pd.DataFrame([row])
    for col in CATEGORICAL_FEATURES:
        le  = encoders[col]
        val = df[col].astype(str).iloc[0]
        val = val if val in set(le.classes_) else le.classes_[0]
        df[col] = le.transform([val])

    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [f"{c}_ratio" for c in SPENDING_COLS]
    return df[feature_columns]

st.set_page_config(page_title="Spaceship Titanic", layout="centered")
st.title(" ASG 04 MD - Matthew - Spaceship Titanic Model Deployment")
st.caption("Predict whether a passenger was transported to an alternate dimension.")
st.divider()

with st.form("input_form"):
    st.write("### Passenger Data")

    col1, col2 = st.columns(2)

    with col1:
        home_planet  = st.selectbox("Home Planet",    ["Earth", "Europa", "Mars"],                      index=0)
        cryo_sleep   = st.selectbox("CryoSleep",      [False, True],                                    index=0)
        destination  = st.selectbox("Destination",    ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],  index=0)
        age          = st.number_input("Age",          0,   100,   28)
        vip          = st.selectbox("VIP",             [False, True],                                    index=0)
        room_service = st.number_input("Room Service", 0.0, 20000.0,  0.0, step=50.0)
        food_court   = st.number_input("Food Court",   0.0, 30000.0,  0.0, step=50.0)

    with col2:
        shopping_mall = st.number_input("Shopping Mall", 0.0, 20000.0,  0.0, step=50.0)
        spa           = st.number_input("Spa",           0.0, 25000.0,  0.0, step=50.0)
        vr_deck       = st.number_input("VR Deck",       0.0, 25000.0,  0.0, step=50.0)
        deck          = st.selectbox("Deck",  ["A","B","C","D","E","F","G","T"],  index=5)
        cabin_num     = st.number_input("Cabin Number",  0,    2000,   100,  step=1)
        side          = st.selectbox("Side",  ["P", "S"],  index=0)
        group_size    = st.number_input("Group Size",    1,      10,     1,  step=1)
        family_size   = st.number_input("Family Size",   1,      10,     1,  step=1)

    submitted = st.form_submit_button("Predict Transported Status", use_container_width=True)

if submitted:
    raw = {
        "HomePlanet"  : home_planet,
        "CryoSleep"   : cryo_sleep,
        "Destination" : destination,
        "Age"         : age,
        "VIP"         : vip,
        "RoomService" : room_service,
        "FoodCourt"   : food_court,
        "ShoppingMall": shopping_mall,
        "Spa"         : spa,
        "VRDeck"      : vr_deck,
        "Deck"        : deck,
        "Cabin_num"   : float(cabin_num),
        "Side"        : side,
        "Group_size"  : int(group_size),
        "Family_size" : int(family_size),
    }

    X          = build_features(raw)
    prediction = model.predict(X)[0]
    prob       = model.predict_proba(X)[0][1]
    result     = "Transported" if prediction == 1 else "Not Transported"

    st.divider()
    st.write("### Prediction Result")

    if prediction == 1:
        st.success(f"**{result}**")
    else:
        st.error(f"**{result}**")

    st.metric("Probability of Being Transported", f"{prob * 100:.1f}%")
    st.progress(float(prob))
