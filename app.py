import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model, setup, create_model, save_model
import plotly.express as px

MODEL_NAME = 'new_trained_model.pkl'
DATA = 'kwik.csv'
CLUSTER_NAMES_AND_DESCRIPTION = 'welcome_survey_cluster_names_and_descriptions_v2.json'





@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    print(f"Printuj all_DF funkcje: {df_with_clusters}")

    return df_with_clusters


@st.cache_data
def get_cluster_and_names_description():
    with open(CLUSTER_NAMES_AND_DESCRIPTION, "r") as f:
        return json.loads(f.read())
    

with st.sidebar:
    st.header("Powiedz nam cos o sobie:")
    st.markdown("Pomozemy Ci znalezc osoby, ktore maja podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])
    email = st.text_input("Podaj swoj emial w celu kontaktu:")

    person_df = pd.DataFrame(
        [
            {
                'age': age,
                'edu_level': edu_level,
                'fav_animals': fav_animals,
                'fav_place': fav_place,
                'gender': gender,
                'email': email
            }
        ]
    )


model = get_model()
all_df = get_all_participants()
model_names_and_description = get_cluster_and_names_description()

predicted_model_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = model_names_and_description[predicted_model_cluster_id]


st.header(f"Najblizej Ci do profilu znajomych: {predicted_cluster_data['name']}")
st.write(f"{predicted_cluster_data['description']}")

same_cluster_df = all_df[all_df["Cluster"] == predicted_model_cluster_id]
st.header(f"Liczba dopasowanych znajomych: {len(same_cluster_df)}")
st.header('Skontaktuj sie z dopasowaną osobą:')

# Create a new DataFrame with incremented index starting from 1
email_df = same_cluster_df['email'].reset_index(drop=True)
email_df.index += 1  # Increment index to start from 1

st.dataframe(email_df.rename_axis('Nr'))  # Rename the index to 'Index'

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.pie(same_cluster_df, names="edu_level", title="Rozkład wykształcenia w grupie")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.pie(same_cluster_df, names="gender", title="Rozkład płci w grupie")
fig.update_layout(
    title="Rozkład płci w grupie",
)
st.plotly_chart(fig)