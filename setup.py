#!/usr/bin/env python3
"""
Setup script for distributed object detection.
"""
import os
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="distributed_object_detection",
    version="0.1.0",
    description="Distributed real-time object detection using MPI, OpenCV, and CUDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chandan Kumar",
    author_email="chandan181singh@gmail.com",
    url="https://github.com/chandan181singh/distributed_object_detection",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "distributed-detection=distributed_object_detection.src.main:main",
        ],
    },
) 