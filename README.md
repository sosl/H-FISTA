# H-FISTA
Hierarchical FISTA - phase retrieval for pulsar spectroscopy

For a full description please see the published paper.

## Installing dependencies

The main dependencies of our code include the widely used libraries such as [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Astropy](https://www.astropy.org/). For plotting, we use [Matplotlib](https://matplotlib.org/). Thus, it is possible you will be able to run H-FISTA without the need to install any extra libraries. 

However, if you want do need to install dependencies, or wish to ensure the exact same versions of packages are used, we provide an easy way to do so. We recommend using [python poetry](https://python-poetry.org/) for installing the dependencies:

```
poetry install
```

Alternatively, you can use pip:

```
pip -r requirements.txt
```

## Input data

The data used in the publication and in the example notebooks can be obtained from [Zenodo](https://zenodo.org/) under this DOI: [10.5281/zenodo.7007226](https://doi.org/10.5281/zenodo.7007226)

H-FISTA accepts input data in ASCII format (as produced by psrflux, part of [PSRchive](http://psrchive.sourceforge.net/)), a FITS file, or a pickle. If using a notebook, you can use any custom reader as long as the data ends up in a NumPy array.

## Example usage

To see how we used the code to produce the results in the paper, please see the notebooks included in this repostitory:
- [Simulated noise-free case with 12.5 per cent local density](./simulated_12.5_NF.ipynb)
- [Simulated noise-free case with 25.0 per cent local density](./simulated_25_NF.ipynb)
- [Simulated noisy case with 12.5 per cent local density](./simulated_12.5.ipynb)
- [Observation of PSR J0837+0610](./J0837+0610.ipynb)
- [Observation of PSR J1939+2134](./J1939+2134.ipynb)

## Command line usage

It is also possible to use the code on the command line. For example to reproduce the results for J0837+0610:

```
python HFISTA.py --data J0837+0610.fits --striation
```

## Contributing

Contributions are welcome. Please use `black` for linting and, if possible, include tests. 
