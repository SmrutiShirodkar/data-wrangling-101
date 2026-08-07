# data-wrangling-101

This repository is where I started. Before any structured machine learning work, these five notebooks were my
first hands-on exercises in pulling real data from the web, cleaning it, and asking simple questions of it. They
are not polished production projects, and I am not presenting them as such. I am keeping them because they mark
the actual starting point of a learning path that later work in this portfolio builds on, and because the honest
gaps in each one (missing baselines, no cross-validation, an uninterpreted error metric) are more useful to show
than to hide.

Each notebook is self-contained. Read them in the order below if you want the rough progression I went through:
starting with scraping and cleaning, moving to regression, then into text preprocessing, then a multi-model
classification comparison, and finally a short cross-validated model comparison exercise.

## Notebooks

### covid_wikipedia_scrape.ipynb
Scrapes a country-level COVID-19 statistics table directly from Wikipedia using `requests` and `BeautifulSoup`,
cleans it into a usable pandas DataFrame, and derives a cases-per-death ratio to compare countries. This was my
first real lesson that most of a data project is spent getting data into a usable shape before any analysis can
happen. No modeling, pure scraping and wrangling.

Limitation: the scrape depends on a specific HTML table id that Wikipedia can regenerate at any time, so the
scrape logic is not guaranteed to run unmodified if the source page changes.

### covid_deaths_regression.ipynb
Two regression attempts on COVID-19 death data pulled from Our World in Data. The first fits a single-feature
linear regression (cases predicting deaths) across the full global time series, evaluated in-sample. The second
narrows to South Korea over a twelve-week window and tries to predict the following week using a much larger
feature set.

Limitation: the global model's error is reported in-sample, not on held-out data, so it should be read as a
correlation check rather than a predictive evaluation. The South Korea model uses roughly 38 features on fewer
than 100 training rows, a classic overfitting setup, and has no naive baseline to compare against. Both gaps are
called out directly in the notebook's closing notes, along with what a fix would look like.

### imdb_sentiment_analysis.ipynb
Text preprocessing and a baseline sentiment classifier on the IMDB labeled review dataset. Covers HTML stripping,
punctuation and emoticon handling, stopword removal, and bag-of-words feature extraction, followed by a Random
Forest classifier.

Result: 100 percent training accuracy against roughly 83 percent validation accuracy, a clear overfitting
signature from an unconstrained Random Forest on a 5000-feature sparse matrix. I chose to report both numbers
rather than only the flattering one, since the gap is the more informative result.

Limitation: no regularization or depth constraint on the Random Forest, no TF-IDF comparison, no baseline
against a majority-class predictor. Noted in the closing section as concrete next steps.

### titanic_survival_prediction.ipynb
A full classification pipeline on the Kaggle Titanic dataset: exploratory analysis, feature engineering (title
extraction, age and fare banding, family size, embarkation port), and a head-to-head comparison of six
classifiers (Logistic Regression, KNN, SVM, Perceptron, XGBoost, Random Forest) plus a small neural network, all
evaluated on an identical 80/20 train/validation split.

Limitation: single train/validation split with no cross-validation, no hyperparameter tuning, no confusion
matrix or precision/recall breakdown despite class imbalance in the target, and the final Kaggle submission uses
the neural network rather than the Random Forest that the feature-importance section discusses. All of this is
detailed in the notebook's closing notes.

### breast_cancer_dt_vs_rf.ipynb
A short comparison of a single Decision Tree against a Random Forest on scikit-learn's built-in breast cancer
dataset, both evaluated with 5-fold cross-validation on an identical fold split. The question was simple: does
the ensemble actually outperform the single tree, and by how much.

Result: the Random Forest's mean cross-validated accuracy came out higher than the single Decision Tree's,
consistent with the standard bias-variance argument for ensembling.

Limitation: accuracy only, no precision/recall breakdown despite moderate class imbalance in the target (roughly
63/37), and neither model's hyperparameters were tuned beyond scikit-learn defaults.

## What this repository is not

These are learning exercises on well-known public datasets (Titanic, IMDB reviews, COVID-19 public data), not
original research or production systems. If you are looking for that kind of work, see the other repositories
linked from my profile. What I think is worth your time here is the process on display: reading data cleaning
and modeling choices out loud, checking assumptions with a plot or a groupby before moving on, and being honest
in the notebook itself about what a result does and does not support.
