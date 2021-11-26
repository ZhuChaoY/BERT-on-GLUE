# BERT-on-GLUE
An implementation of BERT finetuning on GLUE and superGLUE(only CB and BoolQ) dataset by tensorflow.

## Reference
(1) **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Code: https://github.com/google-research/bert)   
(2) **GLUE**: [GLUE: A MULTI-TASK BENCHMARK AND ANALYSIS PLATFORM FOR NATURAL LANGUAGE UNDERSTANDING](https://arxiv.org/pdf/1804.07461v2.pdf)   
(3) **superGLUE**: [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/pdf/1905.00537v3.pdf)   

## File
### Pretrained BERT
base/model.ckpt  : download from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip  
large/model.ckpt : download from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip  
### GLUE
QQP/ : upzip the train.zip  
MNLI/ : download the train.txt from https://dl.fbaipublicfiles.com/glue/data/MNLI.zip   
**else**/ : all ready
### superGLUE   
all ready   

## Results 
### GLUE (dev) 
|**Dataset**| **Index** | **base** |**large**|
|     --    |   --   |    --   |    --    | 
| **WNLI**  |   ACC  | 0.563 | 0.563 |
| **RTE**   |   ACC  | 0.671 | 0.747 |
| **MRPC**  | ACC&F1 | 0.903 | 0.887 |
| **STS-B** |   COR  | 0.885 | 0.879 |
| **CoLA**  |   MCC  | 0.605 | 0.641 |
| **SST-2** |   ACC  | 0.922 | 0.928 |
| **QNLI**  |   ACC  | 0.905 | 0.915 |
| **QQP**   | ACC&F1 | 0.895 | 0.900 |
| **MNLI**  |   ACC  | 0.836 | 0.860 |
| **Mean (without WNLI)**  | \ | 0.828 | 0.845 |
| **Mean**  | \ | 0.798 | 0.813|

```
python Run_GLUE.py --model base --task GLUE --dataset [DATASET] --dropout 0.1 --l_r 2e-5 --batch_size 32 --epoches 10 --earlystop 1
```
```
python Run_GLUE.py --model large --task GLUE --dataset [DATASET] --dropout 0.1 --l_r 2e-5 --batch_size 32 --epoches 10 --earlystop 1
```
**[DATASET]** from {WNLI, RTE, MRPC, STS-B, CoLA, SST-2, QNLI, QQP, MNLI}


### superGLUE (dev) 
|**Dataset**| **Index** | **base** |**large**|
|     --    |   --   |    --   |    --    | 
| **CB**    | ACC&F1 | 0.839 | |
| **BoolQ** |   ACC  | 0.715 | |
```
python Run_GLUE.py --model base --task superGLUE --dataset [DATASET] --dropout 0.1 --l_r 2e-5 --batch_size 32 --epoches 10 --earlystop 1
```
```
python Run_GLUE.py --model large --task superGLUE --dataset [DATASET] --dropout 0.1 --l_r 2e-5 --batch_size 32 --epoches 10 --earlystop 1
```
**[DATASET]** from {CB, BoolQ}
