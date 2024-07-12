# SLSNe

Package designed to work with the data of Type I Superluminous Supernovae (SLSNe).

* Code: [https://github.com/gmzsebastian/SLSNe](https://github.com/gmzsebastian/SLSNe)
* Docs: [https://slsne.readthedocs.io/](https://slsne.readthedocs.io/)
* License: MIT

![Tests](https://github.com/gmzsebastian/SLSNe/actions/workflows/ci_tests.yml/badge.svg)
![License](http://img.shields.io/badge/license-MIT-blue.svg)
[![Coverage Status](https://coveralls.io/repos/github/gmzsebastian/SLSNe/badge.svg?branch=main)](https://coveralls.io/github/gmzsebastian/SLSNe?branch=main)
[![Documentation Status](https://readthedocs.org/projects/slsne/badge/?version=latest)](https://slsne.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12706201.svg)](https://doi.org/10.5281/zenodo.12706201)

The package can be installed via PyPI

```
pip install slsne
```

## Attribution

If you use slsne in your work, please cite [Gomez et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240707946G):

```
@ARTICLE{2024arXiv240707946G,
       author = {{Gomez}, Sebastian and {Nicholl}, Matt and {Berger}, Edo and {Blanchard}, Peter K. and {Villar}, V. Ashley and {Rest}, Sofia and {Hosseinzadeh}, Griffin and {Aamer}, Aysha and {Ajay}, Yukta and {Athukoralalage}, Wasundara and {Coulter}, David C. and {Eftekhari}, Tarraneh and {Fiore}, Achille and {Franz}, Noah and {Fox}, Ori and {Gagliano}, Alexander and {Hiramatsu}, Daichi and {Howell}, D. Andrew and {Hsu}, Brian and {Karmen}, Mitchell and {Siebert}, Matthew R. and {K{\"o}nyves-T{\'o}th}, R{\'e}ka and {Kumar}, Harsh and {McCully}, Curtis and {Pellegrino}, Craig and {Pierel}, Justin and {Rest}, Armin and {Wang}, Qinan},
        title = "{The Type I Superluminous Supernova Catalog I: Light Curve Properties, Models, and Catalog Description}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - High Energy Astrophysical Phenomena},
         year = 2024,
        month = jul,
          eid = {arXiv:2407.07946},
        pages = {arXiv:2407.07946},
archivePrefix = {arXiv},
       eprint = {2407.07946},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240707946G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

We also advise to cite the original sources of data used in this work. We provide a script that prints out the bibtex entries for all these original sources of data, or an optional list of objects if the full sample was not used.

```
from slsne.utils import get_references

# Print all references
get_references()

# Print only some references
get_references(['2018lfe','2015bn'])
```
