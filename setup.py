from setuptools import setup, find_packages

setup(
    name="memorization-analysis",
    version="1.0.0",
    description="Memorization in Neural Networks: An Empirical Analysis",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
