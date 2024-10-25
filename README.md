# smoker-status

Final project for CS 4661 - Introduction to Data Science.

## Getting Started

Install [uv][1] which we use for project management.

This project uses data from Kaggle's Playground Series, [Binary Prediction
of Smoker Status using Bio-Signals][2]. After accepting the competition rules,
download the data and store the .csv files in a `data/raw` directory:
```
└── data
    └── raw            <- The original, immutable data dump.
        ├── sample_submission.csv
        ├── test.csv
        └── train.csv
```

Open a Jupyter notebook in the `notebooks/` directory with:
```sh
uv run jupyter notebook notebooks
```

## Developing

Store Jupyter notebooks in the `notebooks/` directory. Follow the [naming
convention][3] for notebooks used by Cookiecutter Data Science.

You should add a cell at the top of notebooks with the following:
```
%load_ext autoreload
%autoreload 2
```
This should make code from the `smoker_status` module importable.

Before commiting, use `ruff` to format your Python code:
```sh
uvx ruff format
```

[1]: https://docs.astral.sh/uv/getting-started/installation/
[2]: https://www.kaggle.com/competitions/playground-series-s3e24/data
[3]: https://cookiecutter-data-science.drivendata.org/using-the-template/
