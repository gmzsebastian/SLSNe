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



If you use the processed spectra or related data products in your work, please cite [Aamer et al. 2025](https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.2674A/abstract):


```
@ARTICLE{Aamer2025,
       author = {{Aamer}, Aysha and {Nicholl}, Matt and {Gomez}, Sebastian and {Berger}, Edo and {Blanchard}, Peter and {Anderson}, Joseph P. and {Angus}, Charlotte and {Aryan}, Amar and {Ashall}, Chris and {Chen}, Ting-Wan and {Dimitriadis}, Georgios and {Galbany}, Llu{\'\i}s and {Gkini}, Anamaria and {Gromadzki}, Mariusz and {Guti{\'e}rrez}, Claudia P. and {Hiramatsu}, Daichi and {Hosseinzadeh}, Griffin and {Inserra}, Cosimo and {Kumar}, Amit and {Kumar}, Harsh and {Kuncarayakti}, Hanindyo and {Leloudas}, Giorgos and {Mazzali}, Paolo and {Medler}, Kyle and {M{\"u}ller-Bravo}, Tom{\'a}s E. and {Ramirez}, Mauricio and {Sankar K}, Aiswarya and {Schulze}, Steve and {Singh}, Avinash and {Sollerman}, Jesper and {Srivastav}, Shubham and {Terwel}, Jacco H. and {Young}, David R.},
        title = "{The Type I superluminous supernova catalogue {\textendash} II. Spectroscopic evolution in the photospheric phase, velocity measurements, and constraints on diversity}",
      journal = {\mnras},
     keywords = {stars: massive, supernovae: general, transients: supernovae, High Energy Astrophysical Phenomena, Astrophysics of Galaxies, Solar and Stellar Astrophysics},
         year = 2025,
        month = aug,
       volume = {541},
       number = {3},
        pages = {2674-2706},
          doi = {10.1093/mnras/staf1113},
archivePrefix = {arXiv},
       eprint = {2503.21874},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.2674A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

We also advise citing the original sources for the spectra used. These can be found in the appendix of Aamer et al. 2025.
