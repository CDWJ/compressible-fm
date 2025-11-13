# Fluid Simulation on Compressible Flow Maps

[**Duowen Chen** ](https://cdwj.github.io/)**\***, [**Zhiqi Li** ](https://zhiqili-cg.github.io/)**\***, [Taiyuan Zhang](https://orcid.org/0000-0002-7395-5372), [Jinjin He](https://jinjinhe2001.github.io/), [Junwei Zhou](https://zjw49246.github.io/website/), [Bart G van Bloemen Waanders](https://www.bing.com/ck/a?!&&p=b8db80e3bfb8d21a5eee6a33a29bf04ee4fcbf5dcc0c7ed16cb8c87826e9121dJmltdHM9MTc2MjkwNTYwMA&ptn=3&ver=2&hsh=4&fclid=3940d475-7898-6d61-0063-c11979016c79&psq=Bart+G+van+Bloemen+Waanders&u=a1aHR0cHM6Ly93d3cuc2FuZGlhLmdvdi9jY3Ivc3RhZmYvYmFydC1nLXZhbi1ibG9lbWVuLXdhYW5kZXJzLw), [Bo Zhu](https://www.cs.dartmouth.edu/~bozhu/)

*Joint First Author

[![webpage](https://img.shields.io/badge/Project-Homepage-green)](https://cdwj.github.io/projects/compressible-flowmap-project-page/index.html)
[![paper](https://img.shields.io/badge/Paper-Preprint-red)](https://cdwj.github.io/projects/compressible-flowmap-project-page/static/pdfs/SIG_2025__Compressible_Flow_Map_Upload.pdf)
[![code](https://img.shields.io/badge/Source_Code-Github-blue)](https://github.com/CDWJ/compressible-fm)

This repo stores the source code of our SIGGRAPH 2025 paper **Fluid Simulation on Compressible Flow Maps**

<figure>
  <img src="./teaser.png" align="left" width="100%" style="margin: 0% 5% 2.5% 0%">
  <figcaption> We present diverse phenomena simulated using our compressible flow map method. From small-scale water striders swimming on the surface of a
pond, to large-scale ships on vast oceans, and a Dragon capsule landing on Mars, our simulator captures both compressible and weakly compressible dynamics
spanning various scales of applications with a single unified framework.</figcaption>
</figure>
<br />

## Abstract

This paper presents a unified compressible flow map framework designed to accommodate diverse compressible flow systems, including high-Mach-number flows (e.g., shock waves and supersonic aircraft), weakly compressible systems (e.g., smoke plumes and ink diffusion), and incompressible systems evolving through compressible acoustic quantities (e.g., free-surface shallow water). At the core of our approach is a theoretical foundation for compressible flow maps based on Lagrangian path integrals, a novel advection scheme for the conservative transport of density and energy, and a unified numerical framework for solving compressible flows with varying pressure treatments. We validate our method across three representative compressible flow systems, characterized by varying fluid morphologies, governing equations, and compressibility levels, demonstrating its ability to preserve and evolve spatiotemporal features such as vortical structures and wave interactions governed by different flow physics. Our results highlight a wide range of novel phenomena, from ink torus breakup to delta wing tail vortices and vortex shedding on free surfaces, significantly expanding the range of fluid systems that flow-map methods can handle.

## Usage
For compressible flow
```bash
cd compressible
python preprocess_mesh.py
python run.py
```
For shallow water
```bash
cd swe
python run.py
```
For weakly compressible flow
```bash
cd weakly_compressible
python run.py
```
