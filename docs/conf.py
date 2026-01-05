# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

# Mock imports for modules that may not be available during doc build
autodoc_mock_imports = [
    'torch',
    'numpy',
    'pandas',
    'networkx',
    'scipy',
    'sklearn',
    'tqdm',
    'tensorboard',
    'tabpfn'
]

# -- Project information -----------------------------------------------------
project = 'CausalFM Toolkit'
copyright = '2026, CausalFM Team'
author = 'CausalFM Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # Support Markdown
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    'README_DOCS.md',
    'DOCUMENTATION_SUMMARY.md'
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # Don't require _static directory

# -- MyST parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Don't automatically generate stub pages for autosummary
autosummary_generate = False

# Suppress warnings about duplicate files
suppress_warnings = ['autosummary']

