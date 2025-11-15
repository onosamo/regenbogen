"""Setup script for regenbogen package."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="regenbogen",
    version="0.1.0",
    author="Theo Chervinskii",
    author_email="",
    description="Framework for 3D perception pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/onosamo/regenbogen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "rerun-sdk>=0.15.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "full": [
            "open3d>=0.15.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "regenbogen-example=examples.complete_pipeline_example:run_complete_pipeline_example",
        ],
    },
)
