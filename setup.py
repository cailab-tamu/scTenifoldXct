import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
DESCRIPTION = "scTenifoldXct"
PACKAGES = find_packages(exclude=("tests*",))
# exec(open('scTenifoldXct/version.py').read())

INSTALL_REQUIRES = [
        "anndata",
        "matplotlib",
        "numpy",
        "pandas",
        "python_igraph",
        "ray",
        "scanpy",
        "scipy",
        "statsmodels",
        "torch",
        "tqdm",
    ]

setup(
    name="scTenifoldXct",
    # version=__version__,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/cailab-tamu/scTenifoldXct",
    author="Yongjian Yang, TAMU",
    author_email="yjyang027@tamu.edu",
    license="MIT",
    keywords=[
        "neural network",
        "embedding",
        "manifold-learning",
        "computational-biology",
        "single-cell",
        "cell-cell interaction",
        "gene regulatroy network",
        "visualization"
    ],
    classifiers=[
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    ],
    python_requires='~=3.9.6',
    packages=PACKAGES, #["scTenifoldXct"], 
    # package_dir={"scTenifoldXct": 'scTenifoldXct'},
    # package_data={"scTenifoldXct": ['database/*.csv']},
    include_package_data=True, # MANIFEST
    install_requires=INSTALL_REQUIRES,
)