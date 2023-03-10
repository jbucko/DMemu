# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))

#import sphinx_bootstrap_theme

# -- Project information -----------------------------------------------------

project = 'DMemu'
copyright = '2023, Jozef Bucko'
author = 'Jozef Bucko'

# The full version, including alpha/beta/rc tags
version = release = '1.0'

# Paths
html_static_path = ['static']

# -- General configuration ---------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '2.3'

#pygments_style = 'sphinx'

html_theme_options = {
    # General
    'display_version'           : False,
    'logo_only'                 : False,
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.coverage",
    # "sphinx.ext.doctest",
    # "sphinx.ext.extlinks",
    # "sphinx.ext.ifconfig",
    # "sphinx.ext.napoleon",
    # "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.autosectionlabel",
    # "numpydoc",
    # "nbsphinx",
    'sphinx_copybutton',
    "sphinx_rtd_theme",

    #"IPython.sphinxext.ipython_console_highlighting",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = "sphinx_rtd_theme"
#html_theme = 'bootstrap'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# import sphinx_readable_theme

# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme = 'readable'
html_theme = 'sphinx_rtd_theme'

pygments_style = "trac"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {"**": ["globaltoc.html", "sourcelink.html", "searchbox.html"]}
html_short_title = "%s" % (project)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# mathjax_path = (
#     "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
# )

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "templates",
    #"**.ipynb_checkpoints",
]

html_logo              = html_static_path[0] + '/icon.jpg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
