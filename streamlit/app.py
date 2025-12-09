import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(
    page_title="Netflix Explorer",
    layout="wide"
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

DATA_PATH = "../netflix_titles.csv"

@st.cache_data
def load_data(uploaded_file=None):
    """Charge et prépare le dataset Netflix."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(DATA_PATH)

    # Dates
    df["date_added"] = pd.to_datetime(df["date_added"], format="mixed", errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    # Durée numérique + type (min / seasons)
    if "duration" in df.columns:
        df["duration_int"] = (
            df["duration"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )
        df["duration_type"] = (
            df["duration"]
            .astype(str)
            .str.extract(r"([A-Za-z]+)$")
        )

    return df


@st.cache_data
def build_genre_table(df: pd.DataFrame) -> pd.DataFrame:
    """Explose la colonne listed_in en lignes (show_id, genre)."""
    if "listed_in" not in df.columns:
        return pd.DataFrame()
    g = df[["show_id", "type", "listed_in"]].dropna().copy()
    g["genre"] = g["listed_in"].str.split(", ")
    g = g.explode("genre")
    g["genre"] = g["genre"].str.strip()
    return g


@st.cache_data
def build_country_table(df: pd.DataFrame) -> pd.DataFrame:
    """Explose la colonne country en lignes (show_id, country)."""
    if "country" not in df.columns:
        return pd.DataFrame()
    c = df[["show_id", "type", "country"]].dropna().copy()
    c["country_name"] = c["country"].str.split(", ")
    c = c.explode("country_name")
    c["country_name"] = c["country_name"].str.strip()
    return c


@st.cache_data
def extract_people(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Explose une colonne liste (cast / director) en une série de noms individuels."""
    if col_name not in df.columns:
        return pd.Series(dtype=str)
    s = (
        df[col_name]
        .dropna()
        .astype(str)
        .str.split(",", expand=False)
        .explode()
        .str.strip()
    )
    return s[s != ""]


st.sidebar.title("Options")

uploaded_file = st.sidebar.file_uploader(
    "Téléverser un autre netflix_titles.csv (optionnel)",
    type=["csv"]
)

# Chargement principal
try:
    df = load_data(uploaded_file)
except FileNotFoundError:
    st.error(
        "Fichier `netflix_titles.csv` introuvable.\n"
        "Place-le dans le même dossier que `app.py` ou utilise le bouton de téléversement."
    )
    st.stop()

# Filtres de base
types_dispo = sorted(df["type"].dropna().unique())
type_filter = st.sidebar.multiselect(
    "Type de contenu",
    options=types_dispo,
    default=types_dispo
)

# Années de sortie
if "release_year" in df.columns:
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())
    year_min, year_max = st.sidebar.slider(
        "Année de sortie",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    year_min, year_max = None, None

# Rating (TV-MA, PG-13, etc.)
if "rating" in df.columns:
    ratings = sorted(df["rating"].dropna().unique())
    rating_filter = st.sidebar.multiselect(
        "Classement (rating)",
        options=ratings,
        default=ratings
    )
else:
    rating_filter = None

# Durée max (pour limiter les outliers)
if "duration_int" in df.columns:
    max_dur = int(df["duration_int"].dropna().max())
    dur_max = st.sidebar.slider(
        "Durée maximale (pour les graphes - minutes / saisons)",
        min_value=10,
        max_value=max_dur,
        value=min(300, max_dur)
    )
else:
    dur_max = None

# Application des filtres
df_filtered = df.copy()
if type_filter:
    df_filtered = df_filtered[df_filtered["type"].isin(type_filter)]
if year_min is not None:
    df_filtered = df_filtered[df_filtered["release_year"].between(year_min, year_max)]
if rating_filter is not None and len(rating_filter) > 0:
    df_filtered = df_filtered[df_filtered["rating"].isin(rating_filter)]
if dur_max is not None and "duration_int" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["duration_int"].fillna(0) <= dur_max]

# Tables auxiliaires
df_genres = build_genre_table(df_filtered)
df_countries = build_country_table(df_filtered)

# =========================
# HEADER GLOBAL
# =========================

st.title("Netflix Explorer – Tableau de bord interactif")

st.markdown(
    """
    Analyse exploratoire du catalogue **Netflix** à partir du dataset `netflix_titles.csv`.  
    Utilise les filtres à gauche pour restreindre par type, années, rating, durée, etc.
    """
)

# =========================
# ONGLET 1 : VUE D’ENSEMBLE
# =========================

tab_overview, tab_genres, tab_people, tab_duration, tab_search = st.tabs(
    ["Vue d'ensemble", "Genres & Pays", "Acteurs & Réalisateurs", "Durées & Ratings", "Recherche avancée"]
)

with tab_overview:
    st.subheader("Statistiques globales")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de titres (filtrés)", len(df_filtered))
    with col2:
        st.metric("Films vs Séries", " / ".join(types_dispo))
    with col3:
        if "release_year" in df_filtered.columns:
            st.metric("Période couverte", f"{int(df_filtered['release_year'].min())} - {int(df_filtered['release_year'].max())}")
    with col4:
        if "country" in df_filtered.columns:
            st.metric("Pays (distincts)", df_filtered["country"].dropna().nunique())

    # Aperçu du tableau
    with st.expander("Aperçu du dataset (10 premières lignes)"):
        st.dataframe(df_filtered.head(10), use_container_width=True)

    # Valeurs manquantes + doublons
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Valeurs manquantes")
        na_counts = df_filtered.isna().sum().sort_values(ascending=False)
        st.dataframe(na_counts.to_frame("Nb NA"))
    with col_b:
        st.markdown("### Doublons")
        st.write(f"Nombre de lignes dupliquées : **{df_filtered.duplicated().sum()}**")

    st.markdown("---")

    # Répartition Movie vs TV Show (countplot + Plotly)
    st.markdown("### Répartition des types de contenus")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Countplot (statique, Seaborn)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df_filtered, x="type", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Nombre de titres")
        ax.set_title("Films vs Séries")
        st.pyplot(fig)

    with col2:
        st.markdown("**Barres interactives (Plotly)**")
        type_counts = (
            df_filtered["type"]
            .value_counts()
            .rename_axis("type")
            .reset_index(name="count")
        )
        fig = px.bar(
            type_counts,
            x="type",
            y="count",
            text="count",
            title="Répartition des contenus par type (interactif)"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Évolution dans le temps
    if "release_year" in df_filtered.columns:
        st.markdown("### Nombre de sorties par année")

        year_type = (
            df_filtered
            .groupby(["release_year", "type"])
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            year_type,
            x="release_year",
            y="count",
            color="type",
            markers=True,
            title="Évolution du nombre de contenus par année de sortie"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Distribution des dates d'ajout
    if "year_added" in df_filtered.columns:
        st.markdown("###Titres ajoutés par année sur Netflix")
        added_counts = (
            df_filtered.dropna(subset=["year_added"])
            .groupby("year_added")
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            added_counts,
            x="year_added",
            y="count",
            title="Nombre de titres ajoutés par année"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_genres:
    st.subheader("Genres & Pays")

    col1, col2 = st.columns(2)

    # GENRES
    with col1:
        st.markdown("### Top genres")

        if not df_genres.empty:
            top_n = st.slider("Nombre de genres à afficher", 5, 30, 15, key="n_genres")
            top_genres = (
                df_genres["genre"]
                .value_counts()
                .head(top_n)
                .rename_axis("genre")
                .reset_index(name="count")
            )

            # Barplot statique
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=top_genres,
                x="count",
                y="genre",
                ax=ax
            )
            ax.set_xlabel("Nombre de titres")
            ax.set_ylabel("Genre")
            st.pyplot(fig)

            # Plotly interactif
            fig = px.bar(
                top_genres,
                x="count",
                y="genre",
                orientation="h",
                title="Genres les plus fréquents (interactif)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de colonne `listed_in` exploitable pour les genres.")

    # PAYS
    with col2:
        st.markdown("### Top pays producteurs")

        if not df_countries.empty:
            top_n_c = st.slider("Nombre de pays à afficher", 5, 30, 15, key="n_countries")
            top_countries = (
                df_countries["country_name"]
                .value_counts()
                .head(top_n_c)
                .rename_axis("country")
                .reset_index(name="count")
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=top_countries,
                x="count",
                y="country",
                ax=ax
            )
            ax.set_xlabel("Nombre de titres")
            ax.set_ylabel("Pays")
            st.pyplot(fig)

            # Carte monde rapide (Plotly utilise les noms de pays)
            st.markdown("###Carte (approx.) par pays")
            fig = px.choropleth(
                top_countries,
                locations="country",
                locationmode="country names",
                color="count",
                color_continuous_scale="Reds",
                title="Nombre de titres par pays (top)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de colonne `country` exploitable.")

    # WORDCLOUD descriptions
    st.markdown("### Nuage de mots des descriptions")
    if "description" in df_filtered.columns:
        text = " ".join(df_filtered["description"].dropna().astype(str).tolist())
        stopwords = set(STOPWORDS)
        stopwords.update(["film", "series", "netflix", "show"])

        wordcloud = WordCloud(
            width=1000,
            height=400,
            background_color="white",
            stopwords=stopwords
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Pas de colonne `description` dans le dataset.")


# =========================
# ONGLET 3 : ACTEURS & RÉALISATEURS
# =========================

with tab_people:
    st.subheader("Acteurs & Réalisateurs")

    top_n_people = st.slider("Taille du Top", 5, 30, 15, key="n_people")

    col1, col2 = st.columns(2)

    # Réalisateurs
    with col1:
        st.markdown("### Réalisateurs les plus prolifiques")
        directors = extract_people(df_filtered, "director")
        if not directors.empty:
            top_directors = (
                directors.value_counts()
                .head(top_n_people)
                .rename_axis("director")
                .reset_index(name="count")
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=top_directors,
                x="count",
                y="director",
                ax=ax
            )
            ax.set_xlabel("Nombre de titres")
            ax.set_ylabel("Réalisateur")
            st.pyplot(fig)
        else:
            st.info("Pas de données réalisateurs exploitables.")

    # Acteurs
    with col2:
        st.markdown("### Acteurs / actrices les plus fréquents")
        actors = extract_people(df_filtered, "cast")
        if not actors.empty:
            top_actors = (
                actors.value_counts()
                .head(top_n_people)
                .rename_axis("actor")
                .reset_index(name="count")
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=top_actors,
                x="count",
                y="actor",
                ax=ax
            )
            ax.set_xlabel("Nombre de titres")
            ax.set_ylabel("Acteur / Actrice")
            st.pyplot(fig)
        else:
            st.info("Pas de données casting exploitables.")

    # Comparaison top 10 réal vs acteurs
    if not directors.empty and not actors.empty:
        st.markdown("### Comparaison Top 10 réalisateurs vs Top 10 acteurs")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        top10_dir = directors.value_counts().head(10)
        axes[0].barh(range(len(top10_dir)), top10_dir.values)
        axes[0].set_yticks(range(len(top10_dir)))
        axes[0].set_yticklabels(top10_dir.index)
        axes[0].invert_yaxis()
        axes[0].set_title("Top 10 réalisateurs")

        top10_act = actors.value_counts().head(10)
        axes[1].barh(range(len(top10_act)), top10_act.values)
        axes[1].set_yticks(range(len(top10_act)))
        axes[1].set_yticklabels(top10_act.index)
        axes[1].invert_yaxis()
        axes[1].set_title("Top 10 acteurs/actrices")

        st.pyplot(fig)


with tab_duration:
    st.subheader("Durées & Ratings")

    col1, col2 = st.columns(2)

    # Histogramme de la durée (obligatoire)
    if "duration_int" in df_filtered.columns:
        with col1:
            st.markdown("### Histogramme des durées (obligatoire)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(
                df_filtered["duration_int"].dropna(),
                bins=30,
                kde=False,
                ax=ax
            )
            ax.set_xlabel("Durée numérique (minutes / saisons)")
            ax.set_ylabel("Nombre de titres")
            st.pyplot(fig)

        with col2:
            st.markdown("### Boxplot de la durée par type (obligatoire)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=df_filtered.dropna(subset=["duration_int"]),
                x="type",
                y="duration_int",
                ax=ax
            )
            ax.set_xlabel("Type")
            ax.set_ylabel("Durée numérique")
            st.pyplot(fig)
    else:
        st.info("Pas de colonne `duration` exploitable pour les durées.")

    st.markdown("---")

    # Countplot des ratings (obligatoire : countplot)
    if "rating" in df_filtered.columns:
        st.markdown("### Countplot des ratings (obligatoire)")
        fig, ax = plt.subplots(figsize=(10, 4))
        order = df_filtered["rating"].value_counts().index
        sns.countplot(
            data=df_filtered,
            x="rating",
            order=order,
            ax=ax
        )
        ax.set_xlabel("Rating")
        ax.set_ylabel("Nombre de titres")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)


with tab_search:
    st.subheader("🔍 Recherche avancée dans le catalogue")

    search_col1, search_col2 = st.columns(2)

    with search_col1:
        query = st.text_input(
            "Mot-clé (titre, description, cast, director, genre, pays)",
            ""
        )
    with search_col2:
        max_results = st.slider("Nombre max de résultats", 5, 100, 20)

    if query:
        q = query.lower()

        # On construit un df de recherche enrichi
        df_search = df_filtered.copy()

        # Pour pouvoir filtrer aussi sur genres et pays explosés
        if not df_genres.empty:
            genre_map = df_genres.groupby("show_id")["genre"].apply(lambda x: ", ".join(sorted(set(x))))
            df_search = df_search.merge(genre_map, on="show_id", how="left")
        else:
            df_search["genre"] = np.nan

        if not df_countries.empty:
            country_map = df_countries.groupby("show_id")["country_name"].apply(lambda x: ", ".join(sorted(set(x))))
            df_search = df_search.merge(country_map, on="show_id", how="left", suffixes=("", "_country"))
        else:
            df_search["country_name"] = np.nan

        mask = (
            df_search["title"].astype(str).str.lower().str.contains(q)
            | df_search["description"].astype(str).str.lower().str.contains(q)
            | df_search["cast"].astype(str).str.lower().str.contains(q)
            | df_search["director"].astype(str).str.lower().str.contains(q)
            | df_search["genre"].astype(str).str.lower().str.contains(q)
            | df_search["country_name"].astype(str).str.lower().str.contains(q)
        )

        results = df_search[mask].head(max_results)

        st.write(f"Résultats trouvés : **{len(results)}** (affichés : {len(results)})")

        if not results.empty:
            # On n’affiche que quelques colonnes clés
            cols_to_show = [
                c for c in [
                    "title", "type", "release_year", "rating",
                    "duration", "country_name", "genre", "description"
                ] if c in results.columns
            ]
            st.dataframe(results[cols_to_show], use_container_width=True)
        else:
            st.info("Aucun résultat ne correspond à la recherche.")
    else:
        st.info("Entre un mot-clé pour lancer une recherche.")
