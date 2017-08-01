
The code will be released soon.
## Reducing gender bias ##
- Paper

Language is increasingly being used to define rich visual recognition problems with supporting image collections sourced from the web. Structured prediction models are used in these tasks to take advantage of correlations between co-occurring labels and visual input but risk inadvertently encoding social biases found in web corpora. For example, in the following image, it is possible to predict  the *place* is the *kitchen*, given the *activity* is *cooking*. However, the model also brings some stereotype, such as in subfigure 4, the model predicts the agent as a woman even though it is a man.
| ![bias](bias_teaser.png) |
|:--:|
| *Structure prediction can help the model to build the correlations between different parts. However it will also cause some bias problem. * |



In our work, we study data and models associated with multilabel object classification and visual semantic role labeling. We find that (a) datasets for these tasks contain significant gender bias and (b) models trained on these datasets further amplify existing bias. For example, the activity cooking is over 33% more likely to involve females than males in a training set, and a trained model further amplifies the disparity to 68% at test time. We propose to inject corpus-level constraints for calibrating existing structured prediction models and design an algorithm based on Lagrangian relaxation for collective inference. Our method results in almost no performance loss for the underlying recognition task but decreases the magnitude of bias amplification by 47.5% and 40.5% for multilabel classification and visual semantic role labeling, respectively.


- Codes
- Data
We have provided all the potential scores for MS-COCO dataset in data/ folder.  Also there is sampled potentials for imSitu dataset in data/ folder. For complete imSitu potentials, download at [here](http://homes.cs.washington.edu/~my89/share/potentials.tar)
