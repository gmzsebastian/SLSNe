from setuptools import setup

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='slsne',
    version='0.1.0',
    author='Sebastian Gomez',
    author_email='sgomez@stsci.edu',
    description='Catalog of Type-I Superluminous Supernovae',
    url='https://github.com/gmzsebastian/SLSNe',
    license='MIT License',
    python_requires='>=3.6',
    packages=['slsne'],
    package_data={'slsne': ['ref_data/use_names.txt']},
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    include_package_data=True,
)
