# Installation

scTenifoldXct requires **Python 3.10+**.

## From PyPI

```shell
pip install scTenifoldXct
```

## From source

```shell
git clone https://github.com/cailab-tamu/scTenifoldXct.git
cd scTenifoldXct
pip install .
```

## Optional dependency groups

```shell
pip install "scTenifoldXct[dev]"    # pytest + ruff
pip install "scTenifoldXct[docs]"   # mkdocs documentation toolchain
```

## Docker

A container with all dependencies and the bundled databases:

```shell
docker build -t sctenifold .
docker run -it --name xct --shm-size=8gb sctenifold
```
