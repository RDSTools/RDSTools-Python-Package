from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RDSTools",
    version="0.1.0",
    author="Jay Kim",
    author_email="jayhk@umich.edu",
    description="Python tools for Respondent-Driven Sampling (RDS) analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rds-tools",
    py_modules=[
        "data_processing",
        "bootstrap",
        "mean",
        "table",
        "regression",
        "parallel_bootstrap",
        "rds_map",
        "network_graph",
        "load_data"
    ],
    include_package_data=True,
    package_data={
        '': ['*.csv'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "statsmodels>=0.12.0",
        "folium>=0.12.0",
        "matplotlib>=3.3.0",
        "networkx>=2.5",
        "python-igraph>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "viz": [
            "pygraphviz>=1.7",           # Optional for Tree layout
        ],
    },
)