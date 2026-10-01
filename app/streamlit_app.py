"""CritiScope : thèmes des critiques Allociné et leur lien avec le sentiment (démo O6)."""

import streamlit as st
import utils

st.set_page_config(page_title="CritiScope", page_icon="🎬", layout="centered")

EXAMPLES = [
    "Ennuyeux à souhait. Les zombies sont très stupides, les survivants encore plus ... Encore "
    "un film de zombies bas de gamme !",
    "Très grand western spaghetti, avec excelent casting, Lee Van Cleef toujours aussi "
    "remarquable, et une belle musique de Riz Ortolani.",
    "l'histoire de deux enfants juifs confrontés à l'antisémitisme durant la seconde guerre "
    "mondiale,un film émouvant avec des moments forts,la reconstitution de cette époque est "
    "parfaite mais l'atout principal est l'interprétation de qualité avec en tête les deux "
    "frères qui sont formidables",
]


@st.cache_resource(show_spinner="Chargement des données…")
def get_path() -> str:
    return str(utils.artifacts_dir())


@st.cache_data(show_spinner=False)
def get_tables(path: str) -> utils.Tables:
    return utils.load_tables(utils.Path(path))


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_predictor(path: str) -> utils.TopicPredictor:
    return utils.TopicPredictor(utils.Path(path))


def show_review(text: str, label: int, max_chars: int = 300) -> None:
    badge = ":blue-badge[positive]" if label == 1 else ":red-badge[négative]"
    short = text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " […]"
    st.markdown(f"{badge} {short}")


def use_example(text: str) -> None:
    st.session_state.review = text
    st.session_state.run = True


def request_run() -> None:
    st.session_state.run = True


def fr_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def fr_dec(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


path = get_path()
data = get_tables(path)
topics = data.topics
subjects = topics[topics["kind"] == "sujet"]
base = data.summary["sentiment"]["base_share_pos"]

st.title("CritiScope")
st.markdown(
    "Quels sujets reviennent dans les critiques de films Allociné, et lesquels sont les plus "
    f"appréciés ? **{data.summary['n_topics']} thèmes** détectés automatiquement dans "
    f"**{fr_int(data.summary['n_docs'])} critiques**."
)

tab_test, tab_sentiment, tab_topics = st.tabs(
    ["Tester une critique", "Thèmes les mieux et moins bien notés", "Les thèmes"]
)

with tab_test:
    st.markdown(
        "Essayez avec une critique Allociné que le modèle n'a jamais vue, ou écrivez la vôtre."
    )
    for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
        with col.container(border=True, height="stretch"):
            st.caption(f"« {example} »")
            st.button(
                "Analyser",
                key=f"example_{EXAMPLES.index(example)}",
                on_click=use_example,
                args=(example,),
                width="stretch",
            )
    st.session_state.setdefault("review", "")
    st.text_area(
        "Votre critique",
        key="review",
        height=120,
        placeholder="Écrivez une critique de film en français…",
    )
    st.button("Trouver le thème", type="primary", on_click=request_run)
    if st.session_state.pop("run", False):
        text = st.session_state.review
        app_cfg = data.summary["app"]
        if len(text.split()) < app_cfg["min_words"]:
            st.warning(f"Écrivez au moins {app_cfg['min_words']} mots.")
        else:
            res = get_predictor(path).predict(text, 1, 3)
            found = topics.loc[topics["topic"] == res["topic"]].iloc[0]
            st.success(f"**{found['label']}** — {found['description']}")
            st.markdown("**Critiques proches**")
            for i, _ in res["similar"]:
                r = data.reviews.iloc[i]
                show_review(r["review"], int(r["label"]))

with tab_sentiment:
    signif = subjects[subjects["direction"] != "neutre"].sort_values("diff_pp")
    short = signif["label"].str.replace(r"^Critiques (de|d')\s*", "", regex=True).str.lower()
    worst = ", ".join(short.head(3))
    best = ", ".join(short.tail(3).iloc[::-1])
    st.markdown(
        f"Les critiques sont les plus positives pour : **{best}**. "
        f"Les plus sévères portent sur : **{worst}**."
    )
    st.plotly_chart(utils.diverging_chart(subjects, base), width="stretch")
    st.caption(
        "Écart de chaque thème à la moyenne des avis positifs. En gris : différence non "
        "significative. Le jeu de données est équilibré (≈ 50 % d'avis positifs) : on compare "
        "les thèmes entre eux."
    )

with tab_topics:
    options = subjects.sort_values("n", ascending=False)["topic"].tolist()
    selected = st.selectbox(
        "Thème", options, format_func=lambda t: topics.loc[topics.topic == t, "label"].iat[0]
    )
    row = topics.loc[topics["topic"] == selected].iloc[0]
    st.markdown(row["description"])
    st.caption(
        f"{fr_int(row['n'])} critiques · {100 * row['share_pos']:.0f} % positives · "
        f"mots-clés : {', '.join(list(row['words'])[:6])}"
    )
    reps = data.representative[
        (data.representative["topic"] == selected) & (data.representative["rank"] == 1)
    ]
    for r in reps.sort_values("label", ascending=False).itertuples():
        show_review(r.review, r.label)

with st.expander("Méthode"):
    comp = data.comparison.set_index("model")
    bt = comp.loc["BERTopic"]
    lda = comp.loc[comp.index.str.startswith(f"LDA (K={int(bt['n_topics'])},")].iloc[0]
    st.markdown(
        f"- **Données** : {fr_int(data.summary['n_docs'])} critiques du jeu `tblard/allocine`.\n"
        "- **Thèmes** : BERTopic sur des embeddings `multilingual-e5-large-instruct`, étiquetés "
        "hors ligne par un LLM (Gemini).\n"
        "- **Sentiment** : test binomial par thème, corrigé pour les tests multiples.\n"
        f"- **Face à LDA** (même nombre de thèmes) : cohérence C_V de {fr_dec(bt['c_v'])} "
        f"contre {fr_dec(lda['c_v'])}."
    )
