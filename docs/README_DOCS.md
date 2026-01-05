# CausalFM Toolkit Documentation

This directory contains comprehensive documentation for the CausalFM Toolkit.

## Documentation Structure

```
docs/
├── index.md                          # Main documentation homepage
├── index.rst                         # Sphinx index (alternative)
├── conf.py                           # Sphinx configuration
├── requirements.txt                  # Documentation build requirements
│
├── installation.rst                  # Installation guide
├── quickstart.rst                    # Quick start guide
├── citation.rst                      # Citation information
├── license.rst                       # License information
│
├── user_guide/                       # Detailed user guides
│   ├── data_generation.rst          # Data generation guide
│   ├── models.rst                    # Model usage guide
│   ├── training.rst                  # Training guide
│   └── evaluation.rst                # Evaluation guide
│
├── tutorials/                        # Step-by-step tutorials
│   ├── index.rst                     # Tutorial overview
│   ├── tutorial_01_basics.rst       # Basics of CausalFM
│   ├── tutorial_02_data_generation.rst
│   ├── tutorial_03_training.rst
│   └── tutorial_04_evaluation.rst
│
├── examples/                         # Complete working examples
│   ├── standard_cate.rst             # Standard CATE example
│   ├── instrumental_variables.rst    # IV example
│   └── frontdoor_adjustment.rst      # Front-door example
│
└── api/                              # API reference documentation
    ├── index.rst                     # API overview
    ├── data.rst                      # Data API
    ├── models.rst                    # Models API
    ├── training.rst                  # Training API
    └── evaluation.rst                # Evaluation API
```

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

### View Documentation Locally

```bash
cd docs/_build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Build PDF Documentation (Optional)

```bash
cd docs
make latexpdf
```

Requires LaTeX installation.

## Documentation Content

### Getting Started
- **Installation**: How to install CausalFM Toolkit
- **Quick Start**: 5-minute introduction with code examples
- **Tutorials**: Step-by-step learning path

### User Guides
Detailed guides covering:
- Data generation for different causal settings
- Model architecture and usage
- Training configuration and best practices
- Evaluation metrics and analysis

### Examples
Complete, runnable examples:
- Standard CATE estimation workflow
- Instrumental variables analysis
- Front-door adjustment

### API Reference
Comprehensive API documentation for all modules:
- Data generators and loaders
- Model classes and methods
- Training configuration and trainers
- Evaluation metrics

## Documentation Features

✅ **Comprehensive Coverage**: All features documented with examples
✅ **Code Examples**: Every concept illustrated with working code
✅ **Mathematical Notation**: Proper LaTeX rendering for equations
✅ **Cross-References**: Extensive linking between related topics
✅ **API Autodoc**: Automatic API documentation from docstrings
✅ **Multiple Formats**: HTML, PDF, and ePub support
✅ **Search Functionality**: Full-text search in HTML docs
✅ **Mobile Friendly**: Responsive design for all devices

## Contributing to Documentation

We welcome documentation improvements! To contribute:

1. Edit the relevant `.rst` or `.md` files
2. Build the documentation locally to verify
3. Submit a pull request

### Writing Style Guidelines

- Use clear, concise language
- Provide working code examples
- Include expected output when relevant
- Add cross-references to related topics
- Use consistent formatting

### reStructuredText Tips

```rst
# Headers
========  # H1
--------  # H2
~~~~~~~~  # H3

# Code blocks
.. code-block:: python

   # Your code here
   
# Links
:doc:`other_page`
:class:`package.module.Class`
:func:`package.module.function`

# Math
.. math::

   \tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]
```

## Documentation Hosting

The documentation is automatically built and hosted on Read the Docs:

**URL**: https://causalfm.readthedocs.io

Configuration: `.readthedocs.yaml` in repository root

## Questions?

If you have questions about the documentation:

- Open an issue on GitHub
- Check existing documentation at https://causalfm.readthedocs.io
- Contact the maintainers

## License

Documentation is licensed under CC BY 4.0.
Code examples in documentation follow the project's Apache 2.0 license.

