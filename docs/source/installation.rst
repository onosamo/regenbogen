Installation
============

Requirements
------------

* Python 3.8 or higher
* uv package manager (recommended) or pip

Installing uv (recommended)
----------------------------

First, install the uv package manager if you haven't already:

.. code-block:: bash

   # Install uv using pip
   pip install uv
   
   # Or install using the official installer (requires internet)
   curl -LsSf https://astral.sh/uv/install.sh | sh

Installing from Source
----------------------

To install 🌈 regenbogen 🌈 from the source code using uv (recommended):

.. code-block:: bash

   git clone git@github.com:onosamo/regenbogen.git
   cd regenbogen
   uv sync --group full

Alternative: Install with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer using pip:

.. code-block:: bash

   git clone git@github.com:onosamo/regenbogen.git
   cd regenbogen
   pip install -e .

Verification
------------

To verify your installation, you can run:

.. code-block:: bash

   uv run python examples/video_processing_demo.py

