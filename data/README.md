# Data

All datasets are **public**. Place files here as indicated. Scripts read from
`$DATA_DIR` (default `./data`).

| File / folder | Source | Notes |
|---|---|---|
| `telegram_2026.csv` | Public Telegram war-reporting channels | columns: `date,text,views,forwards` |
| `climate/Climate_Dataset/Climate_CSV/*.csv` | Harvard Dataverse `doi:10.7910/DVN/NL06IX` | Reddit climate posts; `sep=';'` |
| `mosi_features.npz` | CMU-MOSI (public) | keys: `X_text,X_audio,X_vision,y,split` |
| `kiev1_reactions_expanded.csv` | Public Telegram reactions corpus | used by multimodal (text+behavioural) |

The **Wisconsin Diagnostic Breast Cancer** tabular dataset is loaded directly from
scikit-learn (`sklearn.datasets.load_breast_cancer`) and needs no download.

The **eight text-classification benchmarks** (AG News, DBpedia-14, Amazon/Yelp polarity,
IMDB, Rotten Tomatoes, SST-2, TweetEval) are pulled at runtime by the optional
foundation-model runner via Hugging Face `datasets` and are not stored here.

Large raw files are intentionally **git-ignored** (see `.gitignore`); this folder ships
only with this README and the small derived artifacts needed for figures.
