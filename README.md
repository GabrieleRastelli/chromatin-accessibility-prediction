<br />
<div align="center">
  <img src="images/logo.jpg" alt="Logo" height="200" style="border-radius:20%">

  <h1 align="center">Chromatin Accessibility Prediction</h1>

  <p align="center">
    A sequence based deep learning approach based on <a href="https://github.com/kimmo1019/Deopen">Deopen</a> to predict scATAC peak height over multiple cell types.
  </p>
</div>

## About The Project

The significance of single-cell ATAC-seq (scATAC) lies in its ability to provide unparalleled insights into the dynamic landscape of chromatin accessibility at the individual cell level. By capturing the open chromatin regions across diverse cell types with high resolution, scATAC enables the elucidation of regulatory elements, including enhancers, promoters, and transcription factor binding sites, crucial for gene expression regulation and cell identity determination.

Through the analysis of scATAC data, one can uncover key DNA regions. Indeed, by using this sequence-based deep learning model, inspired by Deopen, it is possible to not only forecast scATAC peak heights but also identify and prioritize these important DNA regions and their motifs.

## Getting Started

### Preprocessing data for model training

```
python Gen_data.py -bed=<input bed file> -genome=<genome file in fasta format> -target=<targets in npy format> -l=<sequence length> -out=<output folder>
```

### Run model

```
python main.py
```
  
## References

[1] Liu, Qiao, et al. "Chromatin accessibility prediction via a hybrid deep convolutional neural network." Bioinformatics 34.5 (2017): 732-738.