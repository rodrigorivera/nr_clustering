# nr_clustering
==============================
## Description

Project for the Bayesian Machine Learning class at Skoltech, 2018

Source code of the project under the 'notebooks' directory.
PDF file and TEX sources under 'project_report'

## Topic
Semi-supervised Time-Series Learning with Deep Generative Models

https://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf

## Goal
The goal of the project is to study how the model described in the paper works with time series.
Specific tasks:
1) Review the model
2) Reproduce results with MNIST
3) Modify model for time-series data
4) Apply the modified model for MWD time-series
5\*) Improve model with "VampPrior" approach http://proceedings.mlr.press/v84/tomczak18a/tomczak18a.pdf 


## Team Members
Nikita Klyuchnikov

Rodrigo Rivera

## How to run
1) Execute ./build_all.sh to build the docker container
2) After it has finished installing, run docker ps to see under which port the notebook server is running
3) If the docker container is built but not running, it can be started with ./start.sh


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (Not in use at the moment)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries (Not in use at the moment)
    │
    ├── notebooks          <- Source code is here. Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── project_report     <- PDF file and TEX files of the report of this project
    │
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials (Not in use at the moment).
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project (Not in use at the moment).
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
