# ***ConsistTL***: Modeling ***Consist***ency in ***T***ransfer ***L***earning for Low-Resource Neural Machine Translation
Implementation of our paper pubslished in EMNLP 2022

## Brief Introduction
 

<div align="center">
    <img src="/image/cotl-v7-1.png" width="80%" title="Framework of ConsisTL."</img>
    <p class="image-caption">Framework of consistency-based transfer learning. </p>
    <a href=https://github.com/facebookresearch/fairseq/tree/v0.12.0><img alt="Latest Release" src="https://img.shields.io/badge/fairseq-v0.12.0-brightgreen" /></a>
    <a href=https://www.python.org/downloads/release/python-370/><img alt="Latest Release" src="https://img.shields.io/badge/python-v3.7-brightgreen" /></a>
    <a href=https://drive.google.com/file/d/15CXWVj0NIMjDjxEfPCw2WktoYADUuX8O/view?usp=sharing><img alt="Latest Release" src="https://img.shields.io/badge/Parent-High%20Resource-%23307CE5" /></a>
    <a href=https://drive.google.com/file/d/1B23gkfQ3O430KSGVRCqTLyjPO01A5e6L/view?usp=sharing><img alt="Latest Release" src="https://img.shields.io/badge/Child-Low%20Resource-%23E84142" /></a>
</div>


Transfer learning is a simple and powerful method to boost the model performance of low-resource neural machine translation (NMT). Existing transfer learning methods for NMT are static, which simply transfer the knowledge from a parent model to a child model once and for all via parameter initialization. In this paper, we instead propose a novel transfer learning method for NMT, namely ConsistTL, which can continuously transfer parent knowledge during the whole training of the child model. Specifically, for each training instance of the child model, ConsistTL constructs the semantically-equivalent instance for the parent model, and encourages the prediction consistency between the parent and child for this instance, which is equivalent to the child model learning each instance under the guidance of the parent model.

## Preparation 1: Install fairseq
```bash
cd ConsisTL
pip install --editable .
cd ..
# python>=3.7
# We don't need to install pytorch individually. 
```

## Preparation 2: Dowload and binarize data
```bash
# download and preprocess student data
mkdir tr_en
cd tr_en
# donwload tr-en from https://drive.google.com/file/d/1B23gkfQ3O430KSGVRCqTLyjPO01A5e6L/view?usp=sharing
# raw tr-en can be downloaded from https://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-tr.txt.zip
cd ..
fairseq-preprocess -s tr -t en --trainpref tr_en/pack_clean/train --validpref tr_en/pack_clean/valid --testpref tr_en/pack_clean/test --srcdict tr_en/dict.tr.txt --tgtdict dict.en.txt --workers 10 --destdir ${STUDENT_DATA}
# download and preprocess teacher data
mkdir de_en
cd de_en
#donwload de-en from https://drive.google.com/file/d/15CXWVj0NIMjDjxEfPCw2WktoYADUuX8O/view?usp=sharing
cd ..
fairseq-preprocess -s de -t en --trainpref de_en/pack_clean/train --validpref de_en/pack_clean/valid --testpref de_en/pack_clean/test --joined-dictionary --destdir ${TEACHER_DATA} --workers 10
```


## Step 1: Train two parent models
```bash
cd full_process_scripts

# train two parent model
## train for en-de
### path of binarized parent model training data
BIN_TEACHER_DATA=${BIN_TEACHER_DATA}
bash train_parent.sh en de $BIN_TEACHER_DATA
## train for de-en
bash train_parent.sh de en $BIN_TEACHER_DATA
```


## Step 2: Generate semantically-equivalent sentences
```bash
#gen synthetic de-en for tr-en
## English sentences in child data
CHILD_EN=${CHILD_EN}
## path of trained reversed teacher checkpoint
REVERSED_TEACHER_CHECKPOINT=${REVERSED_TEACHER_CHECKPOINT}
## auxiliary source
AUX_SRC_BIN=${AUX_SRC_BIN}
bash gen.sh $CHILD_EN $BIN_TEACHER_DATA $REVERSED_TEACHER_CHECKPOINT $AUX_SRC_BIN
```


## Step 3: Exploit Token Matching (TM) for initialization
```bash
#switch checkpoint
## path of initialized checkpoint
INIT_CHECKPOINT=${INIT_CHECKPOINT}
## path of student data
BIN_STUDENT_DATA=${BIN_STUDENT_DATA}
## path of teacher checkpoint
python ../ConsisTL/preprocessing_scripts/TM.py --checkpoint $TEACHER_CHECKPOINT --output $INIT_CHECKPOINT --parent-dict $BIN_TEACHER_DATA/dict.de.txt --child-dict $BIN_STUDENT_DATA/dict.tr.txt --switch-dict src
```


## Step 4: Train Child Model (s)
```bash
# train for TM-TL
bash train.sh $STUDENT_DATA $INIT_CHECKPOINT

# train for ConsisTL
bash ConsisTL.sh $PREFIX-bin $TEACHER_CHECKPOINT $TEACHER_DATA $STUDENT_DATA $INIT_CHECKPOINT
```


