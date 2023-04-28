# Adaptive Prompt Learning

Official github repository for **Learning to Prompt Adaption for Universal Cross-Domain Retrieval**. Please refer to [paper link](https://openreview.net/pdf?id=QEpMwcDaKX) for detailed infomation.

> Abstract: Universal Cross-Domain Retrieval (UCDR) aims to find relevant images across various unknown domains and categories. It requires a model to handle both domain shift (\ie, unknown domains adaptation) and semantic shift (\ie, unknown categories transferability). Recent studies address the above issues by leveraging expert knowledge from semantic information or pre-trained models as guidance to fine-tune model. However, such a fine-tuning paradigm results in loss of generalization ability due to corruption of original model.In this paper, we present a novel paradigm built upon the recent promising prompt learning to address it. Our method, named Adaptive Prompt Learning (APL) dynamically captures domain and semantic shift and then effectively rectifies them. Specifically, source prompts, which hold base knowledge from source domains and semantics, are obtained by the first stage source prompts learning. Then in the second stage, target prompts that adapted to target domains and semantics are inferred with guidance of the above source prompts. Such a procedure is done adaptively by feeding input and source prompts to a target prompt encoder. Both of two stages are trained with a unified mask-and-align objective to achieve the goal. With the design of APL, the proposed method is evaluated to be effective on three benchmark datasets. Results show that our method significantly outperforms current state-of-the-art by a superior margin for both domain and category generalization, with a $17.92\%$ average performance improvement for mean Average Precision. Moreover, the proposed method also has large performance to conventional prompt tuning under all settings. Our method is publicly available at \url{THE~ANONYMOUS~URL}.

<img src="./main_figure.png"/>

## Requirements

Our implementation is tested on Ubuntu 18.04.5 with Tesla V100-SXM2-32GB. Supports for other platforms and hardwares are possible with no warrant. To install the required packages:

```bash
conda env create -f APL.yaml
conda activate APL
```
## Data Preparation
1. Download three datasets using scripts
   - 1


Please download AlexNet pretrianed model from ()()() and place it into `./data/models/`

### Dataset Preparation

CIFAR-10:

- Training set: 5000
- Query set: 1000
- Gallery: 54000

You should split CIFAR-10 by yourself. [Download it](https://www.cs.toronto.edu/~kriz/cifar.html) and split it as the paper described.

---

NUS-WIDE:

- Training set: 10000
- Query set: 5000
- Gallery: 190834

Please [download the dataset](https://github.com/lhmRyan/deep-supervised-hashing-DSH/issues/8#issuecomment-314314765) and put it into a specific directory. Then you should modify the prefixs of all paths in [/data/nus21](./data/nus21)

---

Imagenet:

- Training set: 10000
- Query set: 5000
- Gallery: 128564

Please download the dataset (ILSVRC >= 2012) and put it into a specific directory. Then you should modify the prefixs of all paths in [/data/imagenet](./data/imagenet)

## Run

Firstly, list all the parameter of scripts:

    python main.py --help

    usage: main.py [-h] [--Dataset DATASET] [--Mode MODE] [--BitLength BITLENGTH]
               [--ClassNum CLASSNUM] [--K K] [--PrintEvery PRINTEVERY]
               [--LearningRate LEARNINGRATE] [--Epoch EPOCH]
               [--BatchSize BATCHSIZE] [--Device DEVICE] [--UseGPU [USEGPU]]
               [--noUseGPU] [--SaveModel [SAVEMODEL]] [--noSaveModel] [--R R]
               [--Lambda LAMBDA] [--Tau TAU] [--Mu MU] [--Nu NU]

    optional arguments:
    -h, --help            show this help message and exit
    --Dataset DATASET     The preferred dataset, 'CIFAR', 'NUS' or 'Imagenet'
    --Mode MODE           'train' or 'eval'
    --BitLength BITLENGTH
                            Binary code length
    --ClassNum CLASSNUM   Label num of dataset
    --K K                 The centroids number of a codebook
    --PrintEvery PRINTEVERY
                            Print every ? iterations
    --LearningRate LEARNINGRATE
                            Init learning rate
    --Epoch EPOCH         Total epoches
    --BatchSize BATCHSIZE
                            Batch size
    --Device DEVICE       GPU device ID
    --UseGPU [USEGPU]     Use /device:GPU or /cpu
    --noUseGPU
    --SaveModel [SAVEMODEL]
                            Save model at every epoch done
    --noSaveModel
    --R R                 mAP@R, -1 for all
    --Lambda LAMBDA       Lambda, decribed in paper
    --Tau TAU             Tau, decribed in paper
    --Mu MU               Mu, decribed in paper
    --Nu NU               Nu, decribed in paper

To perform a training:

    python main.py --Dataset='CIFAR' --ClassNum=10 --LearningRate=0.001 --Device=0

To perform an evaluation:

    python Eval.py --Dataset='CIFAR' --ClassNum=10 --Device=0 --R=-1

## Citations

Please use the following bibtex to cite our papers:

```
@inproceedings{DPQ,
  title={Beyond Product Quantization: Deep Progressive Quantization for Image Retrieval},
  author={Gao, Lianli and Zhu, Xiaosu and Song, Jingkuan and Zhao, Zhou and Shen, Heng Tao},
  booktitle={Proceedings of the 2019 International Joint Conferences on Artifical Intelligence (IJCAI)},
  year={2019}
}
```
```
@inproceedings{DRQ,
  title={Deep Recurrent Quantization for Generating Sequential Binary Codes},
  author={Song, Jingkuan and Zhu, Xiaosu and Gao, Lianli and Xu, Xin-Shun and Liu, Wu and Shen, Heng Tao},
  booktitle={Proceedings of the 2019 International Joint Conferences on Artifical Intelligence (IJCAI)},
  year={2019}
}
```