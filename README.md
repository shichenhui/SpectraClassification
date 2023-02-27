# Data mining techniques on astronomical spectra data.II Classification Analysis
This is the experiment code of the paper - [https://doi.org/10.1093/mnras/stac3292](https://doi.org/10.1093/mnras/stac3292).

In this paper, we investigate the classification methods used for astronomical spectra data. We introduce the main ideas, advantages, caveats, and applications of classification methods. And data sets are
designed by data characteristics, data qualities, and data volumes.
Besides, we experiment with nine basic algorithms(KNN, SVM, LR,
PIL, CNN, DT, RF, GBDT, XGBoost) on A/F/G/K stars classification, star/galaxy/quasar classification, and rare object identification.
Experiments on data characteristics also include the comparative
experiments on the matching sources from the LAMOST survey and
SDSS survey.

For A/F/G/K stars classification, the accuracy on 1D spectra and
PCA shows little difference while PCA spends less time in the
training stage. Because it reduces the spectra dimensionality. So
PCA is often used to classify large-scale and high dimensional data
sets. Among nine basic methods, CNN performs best on 1D spectra
and PCA, due to its powerful ability for feature selection. For the
classification on line indices, KNN shows superiority among other
methods. The performance of classification on SDSS is better than
that on LAMOST. Because the calibration quality of LAMOST is
undesirable, which is affected by many factors (i.e. fibre-to-fibre
sensitivity variations). In addition, high-quality spectra and a large
number of samples help us to train models. But with the growth of
data volumes, the training time of some models will also increase
greatly. So it is necessary to improve the classification speed on
large-scale data sets.

As for star/galaxy/quasar classification, most performance of
classification on rest wavelength frame spectra is better than that
on original spectra. Because redshift causes feature movement on
original spectra. But for some algorithms (PIL, LR, CNN), the
performance of classification on the original spectra is better than
that on the rest wavelength frame spectra. Because original spectra
have much information. These methods can extract feature well and
are less influenced by redshift. For this task, SVM which is good
at binary classification and CNN with powerful ability for feature
selection perform better than other methods.

It is difficult to identify carbon stars, double stars, and artefacts due
to the unbalanced data distributions. Among these three rare objects,
the performance of identifying carbon stars is better than others due
to their obvious characteristics. The performance of searching for
double stars is the worst. In short, researchers need to find other
methods for rare object identification.

In this paper, we only evaluate the classification performance
of nine basic algorithms on astronomical spectra. Other effective
methods still need to be analysed in the future. And experimental
results in this paper can only provide a reference to researchers. In
practical application scenarios, researchers need to choose appropriate methods according to their data characteristics.

- cite

@article{10.1093/mnras/stac3292,
    author = {Yang, Haifeng and Zhou, Lichan and Cai, Jianghui and Shi, Chenhui and Yang, Yuqing and Zhao, Xujun and Duan, Juncheng and Yin, Xiaona},
    title = "{Data mining techniques on astronomical spectra data – II. Classification analysis}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {518},
    number = {4},
    pages = {5904-5928},
    year = {2022},
    month = {11},
    issn = {0035-8711},
    doi = {10.1093/mnras/stac3292},
    url = {https://doi.org/10.1093/mnras/stac3292},
    eprint = {https://academic.oup.com/mnras/article-pdf/518/4/5904/47852595/stac3292.pdf},
}
