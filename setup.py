from setuptools import setup, find_packages

setup(
    name="signal-processing-toolkit",
    version="0.1.0",
    author="SharmaSaurabh-git",
    author_email="lastyttry0@gmail.com",
    description="A comprehensive Python library for signal processing operations",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SharmaSaurabh-git/signal-processing-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
)
