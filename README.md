
## [Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints](https://arxiv.org/abs/1707.09457) ##
[Jieyu Zhao](http://jyzhao.net/), [Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Mark Yatskar](https://homes.cs.washington.edu/~my89/), [Vicente Ordonez](http://www.cs.virginia.edu/~vicente/), [Kai-Wei Chang](http://www.cs.virginia.edu/~kc2wc/). EMNLP 2017

**Please navigate the code through [this jupyter notebook](https://github.com/uclanlp/reducingbias/blob/master/src/fairCRF_gender_ratio.ipynb)**

**For details, please refer to [this paper](http://aclweb.org/anthology/D17-1323.pdf)**


- ### Abstract

Language is increasingly being used to define rich visual recognition problems with supporting image collections sourced from the web. Structured prediction models are used in these tasks to take advantage of correlations between co-occurring labels and visual input but risk inadvertently encoding social biases found in web corpora. For example, in the following image, it is possible to predict  the *place* is the **kitchen**, because it is the common place for the *activity* **cooking**. However, in subfigure 4, the model predicts the agent as a woman even though it is a man, which is caused by the inappropriate correlations between the activity **cooking** and the **female** gender.

| ![bias](img/bias_teaser.png)             |
| ---------------------------------------- |
| *Structure prediction can help the model to build the correlations between different parts. However it will also cause some bias problem.* |

In our work, we study data and models associated with multilabel object classification (MLC) and visual semantic role labeling (vSRL). We find that (a) datasets for these tasks contain significant gender bias and (b) models trained on these datasets further amplify existing bias. For example, the activity **cooking** is over 33% more likely to involve females than males in a training set, and a trained model further amplifies the disparity to 68% at test time. We propose to inject corpus-level constraints for calibrating existing structured prediction models and design an algorithm based on Lagrangian relaxation for collective inference. Our method results in almost no performance loss for the underlying recognition task but decreases the magnitude of bias amplification by 47.5% and 40.5% for multilabel classification and visual semantic role labeling, respectively.


- ### Source Code

We provide our calibration function in file "fairCRF_gender_ratio.ipynb". It is based on the Lagrangian Relaxation algorithm. You need to provide your own inference algorithm and also the algorithm you used to get the accuracy performance. The function also needs you to provide your own constraints. We give detailed description about the parameters in the [jupyter notebook](https://github.com/uclanlp/reducingbias/blob/master/src/fairCRF_gender_ratio.ipynb) and we also provide the running example for both vSRL and MLC tasks. 

> To run the vSRL task, you need to have [caffe](http://caffe.berkeleyvision.org/installation.html) installed in your machine.  If you just want to run with the sampled data, be sure to download the .prototxt files from the data/imSitu/ folder and put them to the folder ("crf\_path" in our case) in the same level where caffe is installed. All the other files are also provided under data/imSitu/. Remember to modify all the path in the config.ini file with absolute path.

- ### Data
(**Update 11/12/2018**) 
For the sampled potentials for imSitu can be found here: [dev](https://drive.google.com/file/d/198zs08SlKh4t_sDmyD_qInp5x4RySPSt/view?usp=sharing) and [test](https://drive.google.com/file/d/1TKDza1PUsAOiOxlO5XQuC5YeCj7mwirV/view?usp=sharing). 

We provide all the potential scores for MS-COCO dataset in data/COCO folder.  For complete imSitu potentials, download at [here](https://s3.amazonaws.com/MY89_Transfer/crf_only.tar).

- ### Reference
  Please cite

 ```
 @InProceedings{zhao-EtAl:2017:EMNLP20173,
  author    = {Zhao, Jieyu  and  Wang, Tianlu  and  Yatskar, Mark  and  Ordonez, Vicente  and  Chang, Kai-Wei},
  title     = {Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  year      = {2017},
  pages     = {2941--2951},
  url       = {https://www.aclweb.org/anthology/D17-1319}
 }
 ```

- ### Note
  The accuracy performance on the MLC tasks is improved. The updated results are:

<table>
    <tr>
        <th colspan="4">Performance (%)</th>
    </tr>
    <tr>
        <th colspan="2">MLC: Development Set</th>
        <th colspan="2">MLC: Test Set</th>
    </tr>
    <tr>
        <td>CRF:</td>
        <td>45.31</td>
        <td>CRF:</td>
        <td>45.46</td>
    </tr>
    <tr>
        <td>CRF+RBA</td>
        <td>45.24</td>
        <td>CRF+RBA</td>
        <td>45.41</td>
    </tr>
</table>

