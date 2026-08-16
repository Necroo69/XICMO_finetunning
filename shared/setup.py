"""Installable package so agents can `from xicmo_shared import ...`.

Install in editable mode from the repo root:

    pip install -e shared
"""

from setuptools import setup

setup(
    name="xicmo_shared",
    version="0.1.0",
    description="Shared data, evaluation, and Hub utilities for Xicmo fine-tuning.",
    py_modules=["data_utils", "eval_utils", "upload_to_hf"],
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "datasets>=2.20.0",
        "huggingface_hub>=0.24.0",
        "python-dotenv>=1.0.0",
        "torch>=2.3.0",
    ],
)
