# Mitigating Gender Bias Ampliﬁcation in Distribution by Posterior Regularization

This repository contains the source code to reproduce the experiments in the ACL 2020 paper.
[Mitigating Gender Bias Ampliﬁcation in Distribution by Posterior Regularization](https://arxiv.org/abs/1909.01482) by Shengyu Jia, [Tao Meng](https://mtsomethree.github.io), [Jieyu Zhao](https://jyzhao.net) and [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/).

**This repository is still under construction**

## Abstract
Advanced machine learning techniques have boosted the performance of natural language processing applications. However, recent studies ([Zhao et al., 2017](https://arxiv.org/abs/1707.09457)) show that these techniques can inadvertently capture the societal bias hidden in the corpus and further amplify it. However, their analysis is only conducted on models’ top predictions rather than their predicted probability. This leaves a research question: where does the bias ampliﬁcation come from; does it merely come from the requirement of making hard decisions? In this paper, we investigate the gender bias ampliﬁcation issue from the distribution perspective and demonstrate that the bias is ampliﬁed in the view of posterior distribution. We further propose a bias mitigation approach based on posterior regularization. With little performance loss, our method can almost remove the bias ampliﬁcation in the distribution and decrease the magnitude of bias ampliﬁcation by 30.9% in top predictions. Our study sheds the light on understanding the bias ampliﬁcation.

## Data
**Splits**

*train.json, dev.json, test.json*

These files contain annotations for the train/dev/test set. Each image in the [imSitu dataset](http://imsitu.org/) is annotated with three frames corresponding to one verb. Each annotation contains a noun value from Wordnet, or the empty string, indicating empty, for every role associated with the verb
```
import json
train = json.load(open("train.json"))

train['clinging_250.jpg']
#{u'frames': [{u'agent': u'n01882714', u'clungto': u'n05563770', u'place': u''},
#  {u'agent': u'n01882714', u'clungto': u'n05563770', u'place': u''},
#  {u'agent': u'n01882714', u'clungto': u'n00007846', u'place': u''}],
# u'verb': u'clinging'}
```

**Images**

Images resized to 256x256 here (3.7G):

https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar

Original images can be found here (34G) :

https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar

## Running experiments

**Requirements**

```bash
python == 3.7.3
pytorch == 1.1.0
```

**Train model**

Before following steps, please modify your own path in "constant.py" first.

To train the model, please use this command:
```bash
python baseline_crf_prob.py --command train --encoding true --weights true
```

If you switch "encoding" to false to not use a given encoding file, or switch "weights" fo false to not use a given weight file.
You can also change "command" to "eval" for testing one model or "predict" for predicting results with a certain model, where you must give the encoding file and weight file.

**Run the posterior regularization process**

Before following steps, please modify your own path in "constant.py" first.

When running the posterior regularization process, please use this command:
```bash
python run.py
```

