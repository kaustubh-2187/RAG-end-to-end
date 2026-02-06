from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="RAG",
    version="0.1.0",
    author="Kaustubh Kamble",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.11',
)