---
title: CritiScope
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Ce que disent les critiques Allociné (BERTopic)
---

# CritiScope

Thèmes des critiques de films Allociné extraits avec BERTopic, et leur lien avec le sentiment
des avis. Les artefacts (modèle, tables, embeddings) sont téléchargés depuis
[Dalfaxy/critiscope-bertopic](https://huggingface.co/Dalfaxy/critiscope-bertopic).

Données : [tblard/allocine](https://huggingface.co/datasets/tblard/allocine).
Aucun appel à un LLM dans l'app : les étiquettes des thèmes sont générées hors ligne.
