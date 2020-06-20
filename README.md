# Multi-Modal Deep Learning for Vehicle Sensor Data Abstraction and Attack Detection

[![DOI][doi-shield]][doi-url]
[![arXiv][arxiv-shield]][arxiv-url]

A state-of-the-art multimodal network, Regnet, is adapted and trained for attack detection. Adaptation includes replacing the input layer and abstracting the multimodal input process. The KITTI dataset is extended to represent two attacks on connected vehicles: inpainting and translation. The end product is a smart multimodal module that abstracts connected-vehicle sensor input and guards against data integrity breaches.

This repo contains the code to recreate [Regnet](https://arxiv.org/abs/1707.03167) and our own Multimodal Attack Detection Network.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- To utilize your GPU in training; install your nvidia driver and CUDA packages.

- Then install the required python packages

    ```bash
    pip install -r requirements.txt
    ```

## Built With

- [Tensorflow](https://www.tensorflow.org/) - The modeling framework used

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc

<!-- MARKDOWN LINKS & IMAGES -->
<!-- arXiv -->
[arxiv-url]: https://img.shields.io/badge/arXiv-Preprint-red
[arxiv-shield]: https://img.shields.io/badge/arXiv-Preprint-red

<!-- doi -->
[doi-url]: https://ieeexplore.ieee.org/document/8906405
[doi-shield]: https://img.shields.io/badge/DOI-10.1109%2FICVES.2019.8906405-blue

## Project Organization

```txt
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make     │
` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── dataset
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── regnet.py      <- Functions to construct the Regnet
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
```
