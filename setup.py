from setuptools import setup

setup(
    name='slsne',
    version='0.2',
    author='Sebastian Gomez',
    author_email='sgomez@stsci.edu',
    description='Catalog of Type-I Superluminous Supernovae',
    url='https://github.com/gmzsebastian/SLSNe',
    license='MIT License',
    python_requires='>=3.6',
    packages=['slsne'],
    include_package_data=True,
    package_data={'slsne': ['ref_data/*', 'ref_data/supernovae/*/*.txt', 'ref_data/extrabol/*.txt', 'ref_data/filters/*.dat']},
    install_requires=[
        'numpy',
        'matplotlib',
        'dust_extinction',
        'extinction',
        'astropy',
        'scipy',
        'numexpr'
    ]
)
