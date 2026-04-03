
<div align="center">

<h1>GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization</h1>

Zixuan Song<sup>1,3</sup>, 
Jing Zhang<sup>2,3 †</sup>, 
Di Wang<sup>2,3 †</sup>, 
Zidie Zhou<sup>1</sup>, 
Wenbin Liu<sup>1</sup>, 
Haonan Guo<sup>2,3 †</sup>,
En Wang<sup>1 †</sup>,
Bo Du<sup>2,3 †</sup>.

<sup>1</sup> Jilin University,  <sup>2</sup> Wuhan University,  <sup>3</sup> Zhongguancun Academy.

<sup>†</sup> Corresponding author

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-models">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >


## 🔥 Update

**2026.3.26**
- Code is now available.

**2026.2.21**
- The paper is accepted by **CVPR 2026**! 🎉

**2025.12.3**
- The paper is post on arXiv! **([arXiv GeoBridge](http://arxiv.org/abs/2512.02697))** 

## 🌞 Abstract

Cross-view geo-localization infers a location by retrieving geo-tagged reference images that visually correspond to a query image. However, the traditional satellite-centric paradigm limits robustness when high-resolution or up-to-date satellite imagery is unavailable. It further underexploits complementary cues across views (e.g., drone, satellite, and street) and modalities (e.g., language and image). To address these challenges, we propose GeoBridge, a foundation model that performs bidirectional matching across views and supports language-to-image retrieval. Going beyond traditional satellite-centric formulations, GeoBridge builds on a novel semantic-anchor mechanism that bridges multi-view features through textual descriptions for robust, flexible localization. In support of this task, we construct GeoLoc, the first large-scale, cross-modal, and multi-view aligned dataset comprising over 50,000 pairs of drone, street-view panorama, and satellite images as well as their textual descriptions, collected from 36 countries, ensuring both geographic and semantic alignment. We performed broad evaluations across multiple tasks. Experiments confirm that GeoLoc pre-training markedly improves geo-location accuracy for GeoBridge while promoting cross-domain generalization and cross-modal knowledge transfer.

<figure>
<div align="center">
<img src=Figs/intro.png width="100%">
</div>

<div align='center'>
 
**Figure 1. Schematic diagram of GeoBridge.**

</div>
<br>

<div align="center">
<img src=Figs/method.png width="100%">
</div>

<div align='center'>

**Figure 2. Overall workflow.**

</div>

## 📖 Datasets

Coming Soon.

## 🚀 Models

Coming Soon.



## 🔨 Usage

### Data Preparation

Please organize the dataset as follows:

```text
data/
├── train/
│   ├── drone/
│   ├── satellite/
│   └── street/
├── val/
│   ├── drone/
│   ├── satellite/
│   └── street/
└── test/
    ├── drone/
    ├── satellite/
    └── street/
```
### Checkpoints
Please download the pretrained checkpoints and place them under:

```text
checkpoints/
├── opts.yaml
└── best_net.pth
```
### Evaluation
Supported evaluation settings include:

- drone ↔ satellite retrieval
- street ↔ satellite retrieval
- satellite ↔ street retrieval
- text → image retrieval

### Example Tasks
GeoBridge supports the following tasks:

1. **Cross-view geo-localization**  
   Retrieve geographically matched reference images across different views.

2. **Bidirectional image retrieval**  
   Perform retrieval between drone, satellite, and street-view imagery.

3. **Language-to-image retrieval**  
   Use natural language descriptions to retrieve semantically aligned geo-images.

## 🍭 Results


<div align="center">
<img src=Figs/over-result.png width="100%">
</div>


## ⭐ Citation

If you find GeoBridge helpful, please give a ⭐ and cite it as follows:

```
@misc{song2025geobridgesemanticanchoredmultiviewfoundation,
      title={GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization}, 
      author={Zixuan Song and Jing Zhang and Di Wang and Zidie Zhou and Wenbin Liu and Haonan Guo and En Wang and Bo Du},
      year={2025},
      eprint={2512.02697},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.02697}, 
}
```

## 🎺 Statement

For any other questions please contact [Zixuan Song](https://github.com/ssssx) at [jlu.edu.cn](songzx24@mails.jlu.edu.cn) or [gmail.com](estrellaluminous@gmail.com).


