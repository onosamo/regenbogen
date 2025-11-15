# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "🌈 regenbogen 🌈"
copyright = "2025, Theo Chervinskii"
author = "Theo Chervinskii"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# Theme options for alabaster
html_theme_options = {
    "github_user": "fedor-chervinskii",
    "github_repo": "regenbogen",
    "github_banner": True,
    "github_type": "star",
    "show_powered_by": False,
    "sidebar_width": "230px",
    "page_width": "1024px",
    "description": "Framework for 3D perception pipelines",
}

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
