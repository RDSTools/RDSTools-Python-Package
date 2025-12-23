Installation
============

Requirements
------------

* Python ≥ 3.7
* pandas ≥ 1.3.0
* numpy ≥ 1.20.0
* statsmodels ≥ 0.12.0

Install from PyPI
-----------------

Once published, install using pip::

    pip install RDSTools

Install from Source
-------------------

Download the package folder and install locally::

    cd RDSTools
    pip install .

For development (editable install)::

    pip install -e .

Additional Dependencies
-----------------------

For visualization features, install additional dependencies::

    pip install networkx python-igraph matplotlib folium

For tree layouts in network graphs (optional)::

    pip install pygraphviz

Or install all optional dependencies::

    pip install RDSTools[viz]

