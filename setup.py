"""
GLYCOPULSE-NET Package Setup
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="glycopulse",
    version="1.0.0",
    author="MUHAMMED AFLAH S",
    author_email="ra2411026050147@srmist.edu.in",
    description="Multi-Task Deep Learning for Type 2 Diabetes Stratification, Microvascular Complications, and Counterfactual Recourse",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aflahs/ANN_Project_Submission",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.10.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black", "flake8"],
    },
)
