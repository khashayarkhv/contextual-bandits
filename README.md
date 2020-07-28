# Introduction
This repository includes Matlab scripts for running contextual bandits algorithms. These codes have been used to generate the results in our paper: 

<a href="https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3605">Mostly Exploration-Free Algorithms for Contextual Bandits</a>, Hamsa Bastani, Mohsen Bayati, and Khashayar Khosravi, Forthcoming in Management Science.

# File Descriptions

The scripts containing the implementation of various algorithms discussed in the paper can be found in the folder /scripts. Under the same folder you can find more information about how each algorithm is executed. There following three main scripts in the root folder that generate the results presented in the paper:

* `simulationsynth.m`: this corresponds to the synthetic simulations with linear rewards described in Section 5.1 of the paper. 

* `simulationslogistic.m`: this corresponds to the synthethic simulations with logistic rewards described in Section 5.1 of the paper.

* `simulationsreal.m`: this corressponds to the simulations on real datasets presented in Section 5.2 of the paper (see below on the information about the datasets).

# Datasets

The datasets used for real data simulations that can be found under the folder /datasets, are publicly available for the download. Please include a proper citation (described in the links below) if you wish to use these datasets:

* <a href="https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State"> EEG</a>.

* <a href="https://www.openml.org/d/1044"> Eye Movement</a>.

* <a href="https://www.openml.org/d/1560"> Cardiotocography<a/> which is derived from <a href="https://archive.ics.uci.edu/ml/datasets/Cardiotocography"> this dataset<a/>. 
 
* Warfarin dataset that can be downloaded from <a href="https://www.pharmgkb.org/downloads"> here<a/>.
  
# Citation

If you use our repository in your work, please cite our paper:

Hamsa Bastani, Mohsen Bayati, and Khashayar Khosravi. **Mostly Exploration-Free Algorithms for Contextual Bandits.**, Forthcoming in Management Science (2020). 

BibTex:

```
@article{bastani2020mostly,
  title={Mostly exploration-free algorithms for contextual bandits},
  author={Bastani, Hamsa and Bayati, Mohsen and Khosravi, Khashayar},
  journal={Management Science},
  publisher={INFORMS},
  pages={to appear}
}
```

Any question about the scripts can be directed to the authors <a href = "mailto: khashayar.khv@gmail.com"> via email</a>.


# References

H. Bastani, M. Bayati, and K. Khosravi.
**Mostly Exploration-Free Algorithms for Contextual Bandits.**
[*Forthcoming in Management Science*](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3605), 2020


J. Salojärvi, K. Puolamäki, J. Simola, L. Kovanen, I. Kojo, and S. Kaski.
**Inferring relevance from eye movements: Feature extraction.**
[*In Workshop at NIPS 2005*](http://research.ics.aalto.fi/events/inips2005/inips2005proceedings.pdf#page=45), 2005.


D. Ayres-de-Campos, J. Bernardes, A. Garrido, J. Marques-de-Sa, and L. Pereira-Leite.
**SisPorto 2.0: a program for automated analysis of cardiotocograms.**
*Journal of Maternal-Fetal Medicine, 9(5), 311-318.*, 2000.


International Warfarin Pharmacogenetics Consortium. 
**Estimation of the warfarin dose with clinical and pharmacogenetic data.** 
*New England Journal of Medicine, 360(8), 753-764.*, 2009. 

D. Dua and C. Graff.
**UCI Machine Learning Repository, http://archive.ics.uci.edu/ml.**
*Irvine, CA: University of California, School of Information and Computer Science*, 2019. 

