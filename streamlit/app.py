import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(
    page_title="Analyse Netflix – Tableau de bord",
    layout="wide"
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

DATA_PATH = "../netflix_titles.csv"

@st.cache_data
def load_data(uploaded_file=None, csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Charge le dataset Netflix et calcule :
    - date_added -> datetime
    - year_added, month_added
    - duration_int, duration_type
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(csv_path)

    # Dates d'ajout
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], format="mixed", errors="coerce")
        df["year_added"] = df["date_added"].dt.year
        df["month_added"] = df["date_added"].dt.month
    else:
        df["date_added"] = pd.NaT
        df["year_added"] = pd.NA
        df["month_added"] = pd.NA

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
    else:
        df["duration_int"] = np.nan
        df["duration_type"] = pd.NA

    return df


def add_decade_column(df: pd.DataFrame, year_col: str = "release_year", start_decade: int = 1920) -> pd.DataFrame:
    """
    Ajoute une colonne 'decade' basée sur year_col :
    ex: '1920-1929', '1930-1939', etc.
    """
    df = df.copy()
    if year_col not in df.columns:
        df["decade"] = pd.NA
        return df

    years = pd.to_numeric(df[year_col], errors="coerce")
    valid = years.notna()
    if not valid.any():
        df["decade"] = pd.NA
        return df

    y_int = years[valid].astype(int)
    end_decade = int((y_int.max() // 10) * 10)
    if end_decade < start_decade:
        df["decade"] = pd.NA
        return df

    bins = list(range(start_decade, end_decade + 10, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]

    decade = pd.cut(y_int, bins=bins, labels=labels, right=False)

    df["decade"] = pd.NA
    df.loc[valid, "decade"] = decade

    return df


def build_main_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'main_genre' = premier genre de listed_in.
    """
    df = df.copy()
    if "listed_in" not in df.columns:
        df["main_genre"] = pd.NA
        return df

    df["main_genre"] = (
        df["listed_in"]
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    return df


def build_country_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DF avec une ligne par association titre-pays.
    Colonnes: show_id, type, country_name
    """
    cols = [c for c in ["show_id", "type", "country"] if c in df.columns]
    if "country" not in cols:
        return pd.DataFrame()

    c = df[cols].dropna(subset=["country"]).copy()
    c["country_name"] = c["country"].astype(str).str.split(",")
    c = c.explode("country_name")
    c["country_name"] = c["country_name"].str.strip()
    return c


def explode_people(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Explose cast/director en série de noms individuels.
    """
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
    "Téléverser un CSV Netflix (optionnel)",
    type=["csv"],
    key="file_uploader_main"
)

# Chargement du dataset
try:
    df = load_data(uploaded_file)
except FileNotFoundError:
    st.error(
        "Fichier `netflix_titles.csv` introuvable.\n"
        "Place-le dans le même dossier que `app.py` ou téléverse un CSV."
    )
    st.stop()

# Filtres de base
types_dispo = sorted(df["type"].dropna().unique()) if "type" in df.columns else []
if types_dispo:
    type_filter = st.sidebar.multiselect(
        "Type de contenu",
        options=types_dispo,
        default=types_dispo
    )
else:
    type_filter = []

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

if "rating" in df.columns:
    ratings = sorted(df["rating"].dropna().unique())
    rating_filter = st.sidebar.multiselect(
        "Rating",
        options=ratings,
        default=ratings
    )
else:
    rating_filter = None

if "duration_int" in df.columns:
    max_dur = int(df["duration_int"].dropna().max())
    dur_max = st.sidebar.slider(
        "Durée max affichée (minutes / saisons)",
        min_value=10,
        max_value=max(60, max_dur),
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

# Tables dérivées
df_genres = build_main_genre(df_filtered)
df_genres_dec = add_decade_column(df_genres, year_col="release_year")
df_countries_long = build_country_long(df_filtered)
df_main_country = df_filtered.copy()
if "country" in df_main_country.columns:
    df_main_country["main_country"] = (
        df_main_country["country"]
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
df_main_country = add_decade_column(df_main_country, year_col="release_year")


st.title("Analyse Netflix – Tableau de bord")

tab_overview, tab_genres, tab_countries, tab_people, tab_duration = st.tabs(
    [
        "Vue d'ensemble",
        "Genres & décennies",
        "Pays & temporel",
        "Casting & réalisateurs",
        "Durées & ratings",
    ]
)


with tab_overview:
    st.subheader("Aperçu général")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de titres (filtrés)", len(df_filtered))
    with col2:
        st.metric("Nombre de colonnes", df_filtered.shape[1])
    with col3:
        if "release_year" in df_filtered.columns:
            st.metric(
                "Période de sortie",
                f"{int(df_filtered['release_year'].min())} - {int(df_filtered['release_year'].max())}"
            )

    with st.expander("Aperçu du dataset (10 premières lignes)"):
        st.dataframe(df_filtered.head(10), use_container_width=True)

    st.markdown("### Valeurs manquantes & doublons")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Valeurs manquantes")
        na_counts = df_filtered.isna().sum().sort_values(ascending=False)
        st.dataframe(na_counts.to_frame("Nb NA"))
    with col_b:
        st.markdown("#### Doublons")
        st.write(f"Nombre de lignes dupliquées : **{df_filtered.duplicated().sum()}**")

    st.markdown("---")
    st.subheader("Répartition des types de contenus")

    if "type" in df_filtered.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Countplot Movies / TV Shows")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df_filtered, x="type", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Nombre de titres")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Répartition (Plotly)")
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
                title="Répartition des contenus par type"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        if "release_year" in df_filtered.columns:
            st.markdown("#### Évolution du nombre de contenus par année de sortie et par type")
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
                title="Nombre de contenus par année de sortie"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de colonne `type` dans le dataset.")


with tab_genres:
    st.subheader("Genres principaux")

    if "main_genre" not in df_genres.columns:
        st.info("Pas de colonne `listed_in` pour analyser les genres.")
    else:
        top_n = st.slider("Nombre de genres à afficher", 5, 30, 15, key="top_n_genres")

        top_genres = (
            df_genres["main_genre"]
            .value_counts()
            .head(top_n)
            .rename_axis("main_genre")
            .reset_index(name="count")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top genres (Seaborn)")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                data=top_genres,
                x="count",
                y="main_genre",
                ax=ax
            )
            ax.set_xlabel("Nombre de titres")
            ax.set_ylabel("Genre principal")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Top genres (Plotly)")
            fig = px.bar(
                top_genres,
                x="count",
                y="main_genre",
                orientation="h",
                title="Genres principaux (interactif)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Heatmap : popularité des genres par décennie de sortie")

        genre_decade = (
            df_genres_dec
            .dropna(subset=["decade", "main_genre"])
            .groupby(["decade", "main_genre"])
            .size()
            .reset_index(name="count")
        )

        if genre_decade.empty:
            st.info("Impossible de calculer la heatmap (années/genres manquants).")
        else:
            pivot_genre = (
                genre_decade
                .pivot(index="main_genre", columns="decade", values="count")
                .fillna(0)
            )

            top_genres_heat = (
                pivot_genre
                .sum(axis=1)
                .sort_values(ascending=False)
                .head(top_n)
                .index
            )
            pivot_top = pivot_genre.loc[top_genres_heat]

            fig, ax = plt.subplots(
                figsize=(10, max(4, 0.4 * len(pivot_top)))
            )
            sns.heatmap(
                pivot_top,
                cmap="RdYlGn_r",
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "Nombre de titres"},
                ax=ax
            )
            ax.set_title("Popularité des genres par décennie de sortie")
            ax.set_xlabel("Décennie")
            ax.set_ylabel("Genre principal")
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("Nuage de mots des descriptions")

    if "description" in df_filtered.columns:
        text = " ".join(df_filtered["description"].dropna().tolist())
        stopwords = set(STOPWORDS)
        stopwords.update(["film", "series", "netflix", "s","u"])

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stopwords
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Nuage de mots des descriptions")
        st.pyplot(fig)
    else:
        st.info("Pas de colonne `description` pour générer le nuage de mots.")

with tab_countries:
    st.subheader("Analyse des pays")

    if df_countries_long.empty:
        st.info("Pas de colonne `country` dans le dataset.")
    else:
        # Valeurs manquantes sur country (sur df complet filtré)
        if "country" in df_filtered.columns:
            na_country = df_filtered["country"].isna().sum()
            pct_na_country = df_filtered["country"].isna().mean() * 100
            st.write(
                f"Valeurs manquantes dans `country` : **{na_country}** "
                f"({pct_na_country:.2f}%)"
            )

        # Top pays
        country_counts = (
            df_countries_long["country_name"]
            .value_counts()
            .reset_index()
        )
        country_counts.columns = ["country", "count"]

        st.markdown("### Top 15 pays")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.barplot(
                data=country_counts.head(15),
                x="count",
                y="country",
                ax=ax
            )
            ax.set_xlabel("Nombre de titres")
            ax.set_ylabel("Pays")
            st.pyplot(fig)

        with col2:
            fig = px.bar(
                country_counts.head(15),
                x="count",
                y="country",
                orientation="h",
                title="Top 15 pays (interactif)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Heatmap : pays × type de contenu")

        if "type" in df_countries_long.columns:
            country_type_counts = (
                df_countries_long
                .groupby(["country_name", "type"])
                .size()
                .reset_index(name="count")
            )

            top_n_heat = st.slider(
                "Nombre de pays à afficher dans la heatmap",
                5, 30, 12,
                key="top_n_countries_heat"
            )

            top_countries_heat = (
                country_type_counts
                .groupby("country_name")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n_heat)
                .index
            )

            country_type_top = country_type_counts[
                country_type_counts["country_name"].isin(top_countries_heat)
            ]

            country_type_matrix_top = (
                country_type_top
                .pivot(index="country_name", columns="type", values="count")
                .fillna(0)
                .astype(int)
            )

            fig, ax = plt.subplots(
                figsize=(8, max(4, 0.4 * len(country_type_matrix_top)))
            )
            sns.heatmap(
                country_type_matrix_top,
                annot=True,
                fmt="d",
                cmap="YlOrRd",
                cbar_kws={"label": "Nombre de contenus"},
                linewidths=0.5,
                linecolor="gray",
                ax=ax
            )
            ax.set_title("Répartition des contenus Netflix par pays et type")
            ax.set_xlabel("Type de contenu")
            ax.set_ylabel("Pays")
            st.pyplot(fig)
        else:
            st.info("Pas de colonne `type` pour la heatmap pays × type.")

        st.markdown("---")
        st.subheader("Heatmap : popularité des pays par décennie de sortie")

        country_decade = (
            df_main_country
            .dropna(subset=["decade", "main_country"])
            .groupby(["decade", "main_country"])
            .size()
            .reset_index(name="count")
        )

        if country_decade.empty:
            st.info("Impossible de calculer la heatmap (années/pays manquants).")
        else:
            pivot_country = (
                country_decade
                .pivot(index="main_country", columns="decade", values="count")
                .fillna(0)
            )

            top_countries_dec = (
                pivot_country
                .sum(axis=1)
                .sort_values(ascending=False)
                .head(12)
                .index
            )
            pivot_top = pivot_country.loc[top_countries_dec]

            fig, ax = plt.subplots(
                figsize=(10, max(4, 0.4 * len(pivot_top)))
            )
            sns.heatmap(
                pivot_top,
                cmap="RdYlGn_r",
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "Nombre de titres"},
                ax=ax
            )
            ax.set_title("Popularité des pays par décennie de sortie")
            ax.set_xlabel("Décennie")
            ax.set_ylabel("Pays principal")
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("Heatmap temporelle : ajouts par année et par mois")

    if {"year_added", "month_added"}.issubset(df_filtered.columns):
        pivot = df_filtered.pivot_table(
            index="year_added",
            columns="month_added",
            values="show_id",
            aggfunc="count"
        )

        # On force les colonnes 1 à 12
        month_order = list(range(1, 13))
        pivot = pivot.reindex(columns=month_order)

        month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                        "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            pivot,
            cmap="RdYlGn_r",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Nombre de titres"},
            xticklabels=month_labels,
            ax=ax
        )
        ax.set_title("Heatmap des ajouts par année et par mois")
        ax.set_xlabel("Mois")
        ax.set_ylabel("Année")
        st.pyplot(fig)
    else:
        st.info("Colonnes `year_added` et `month_added` manquantes pour la heatmap temporelle.")

with tab_people:
    st.subheader("Top acteurs / actrices & réalisateurs")

    top_n_people = st.slider("Taille du Top", 5, 30, 15, key="top_n_people")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Réalisateurs les plus prolifiques")
        directors = explode_people(df_filtered, "director")
        if not directors.empty:
            top_directors = (
                directors
                .value_counts()
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

    with col2:
        st.markdown("#### Acteurs / actrices les plus présents")
        actors = explode_people(df_filtered, "cast")
        if not actors.empty:
            top_actors = (
                actors
                .value_counts()
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

    st.markdown("---")
    st.subheader("Comparaison Top 10 réalisateurs vs Top 10 acteurs")

    directors_all = explode_people(df_filtered, "director")
    actors_all = explode_people(df_filtered, "cast")

    if not directors_all.empty and not actors_all.empty:
        top10_dir = directors_all.value_counts().head(10)
        top10_act = actors_all.value_counts().head(10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].barh(range(len(top10_dir)), top10_dir.values)
        axes[0].set_yticks(range(len(top10_dir)))
        axes[0].set_yticklabels(top10_dir.index)
        axes[0].invert_yaxis()
        axes[0].set_title("Top 10 réalisateurs")

        axes[1].barh(range(len(top10_act)), top10_act.values)
        axes[1].set_yticks(range(len(top10_act)))
        axes[1].set_yticklabels(top10_act.index)
        axes[1].invert_yaxis()
        axes[1].set_title("Top 10 acteurs/actrices")

        st.pyplot(fig)
    else:
        st.info("Données insuffisantes pour comparer réalisateurs et acteurs.")


with tab_duration:
    st.subheader("Durées")

    if "duration_int" in df_filtered.columns:
        df_dur = df_filtered.dropna(subset=["duration_int"]).copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Histogramme des durées")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(
                df_dur["duration_int"],
                bins=30,
                kde=False,
                ax=ax
            )
            ax.set_xlabel("Durée numérique (minutes / saisons)")
            ax.set_ylabel("Nombre de titres")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Boxplot des durées par type")
            if "type" in df_dur.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(
                    data=df_dur,
                    x="type",
                    y="duration_int",
                    ax=ax
                )
                ax.set_xlabel("Type")
                ax.set_ylabel("Durée numérique")
                st.pyplot(fig)
            else:
                st.info("Pas de colonne `type` pour le boxplot.")
    else:
        st.info("Pas de colonne `duration` / `duration_int` dans le dataset.")

    st.markdown("---")
    st.subheader("Ratings")

    if "rating" in df_filtered.columns:
        rating_counts = (
            df_filtered["rating"]
            .value_counts()
            .rename_axis("rating")
            .reset_index(name="count")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Countplot des ratings (Seaborn)")
            fig, ax = plt.subplots(figsize=(10, 4))
            order = rating_counts["rating"]
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

        with col2:
            st.markdown("#### Barres interactives (Plotly)")
            fig = px.bar(
                rating_counts,
                x="rating",
                y="count",
                title="Répartition des ratings",
                text="count"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de colonne `rating` dans le dataset.")

