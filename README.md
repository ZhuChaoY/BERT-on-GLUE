# BERT-on-GLUE
An implementation of BERT finetuning on GLUE dataset by tensorflow.

## Reference
(1) **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Code: https://github.com/google-research/bert)   
(2) **GLUE**: [GLUE: A MULTI-TASK BENCHMARK AND ANALYSIS PLATFORM FOR NATURAL LANGUAGE UNDERSTANDING](https://arxiv.org/pdf/1804.07461v2.pdf)

## File
### Pretrained BERT
base/model.ckpt  : download from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip  
large/model.ckpt : download from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip  
### GLUE
QQP/ : upzip the train.zip  
MNLI/ : download the train.txt from https://gluebenchmark.com/tasks  
**else**/ : all ready

## Results (on dev) 
|           | **Index** | **base** |**large**|
|     --    |   --   |    --   |    --    | 
| **WNLI**  |   ACC  | 0.563 | |
| **RTE**   |   ACC  | 0.693 | |
| **MRPC**  | ACC&F1 | 0.885 | |
| **STS-B** |   COR  | 0.879 | |
| **CoLA**  |   MCC  | 0.575 | 0.621 |
| **SST-2** |   ACC  | 0.925 | 0.942 |
| **QNLI**  |   ACC  | 0.891 | 0.918 |
| **QQP**   | ACC&F1 | 0.894 | |
| **MNLI**  |   ACC  | 0.835 | |
| **Mean (without WNLI)**  | \ | 0.822 | |
| **Mean**  | \ | 0.793 | |

```
python Run_GLUE.py --model base --dataset [DATASET] --len_d 128 --dropout 0.1 --l_r 2e-5 --batch_size 32 --epoches 10 --earlystop 1
```
```
python Run_GLUE.py --model large --dataset [DATASET] --len_d 128 --dropout 0.1 --l_r 2e-5 --batch_size 32 --epoches 10 --earlystop 1
```
**[DATASET]** from {WNLI, RTE, MRPC, STS-B, CoLA, SST-2, QNLI, QQP, MNLI}
