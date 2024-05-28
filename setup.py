from setuptools import setup

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
    include_package_data=True,
    package_data={'slsne': ['ref_data/*.txt']},
    install_requires=[
        'numpy',
        'matplotlib',
    ]
)
