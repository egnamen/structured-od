# Structured Object Detection

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.2-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
</p>

## Installation

```bash
pip install git+ssh://git@github.com/egnamen/structured-od.git
```


## Development installation

Setup the development environment:

```bash
git clone git@github.com:egnamen/structured-od.git
cd structured-od
conda env create -f env.yaml
conda activate structured-od
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
