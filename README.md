# \#Dialogue-Generation-Torch

 

## \###Dependency

The code has been tested with:

\* python 3.6 

\* pytorch 1.1.0

\* Ubuntu 18.04



You first need to create an environment using `anaconda` or `virtualenv` and then activate the env.

Please follow  (https://github.com/microsoft/unilm/tree/master/unilm-v1) to build the unilm environment, especially apex and nlg-eval.

Other dependencies need

```
pip install -r requirements.txt
```

## \###Model

Download from (https://github.com/microsoft/unilm/tree/master/s2s-ft) to get `[unilm1.2-base-uncased]` model. The model are trained by using the same model configuration and WordPiece vocabulary as BERT Base.

Note that  `unilm_v2_bert_pretrain` folder shoud  contains  three components: `bert_config.json` 、`unilm1.2-base-uncased.bin`  and  `vocab.txt`.



## \###Data

### Train data

Download and unzip from Google Drive (https://drive.google.com/file/d/1bKjHtJMDwxsXRwQD2UQg37TdEfVg6l3k/view?usp=sharing).  Because of the large size of train data , this process will take some time.

### Test data

#### Wizard of Wikipedia 

Download Wizard of Wikipedia dataset from (https://parl.ai/projects/wizard_of_wikipedia/), then preprocess it by 

```
python preprocess/wizard_preprocess.py ${data_path}/test_random_split.json

python preprocess/wizard_preprocess.py ${data_path}/test_topic_split.json
```



#### CMU_DoG

Download CMU_DoG dataset fom (https://github.com/lizekang/ITDD), then preprocess it by

```
python3 preprocess/cmu_dog_preprocess.py ${data_path}/ITDD_data/src-test-tokenized.txt ${data_path}/ITDD_data/tgt-test-tokenized.txt ${data_path}/ITDD_data/knl-test-tokenized.txt
```



The `${data_path}` is location of raw dataset. So you could put the above three raw test datasets under `${data_path}` folder. The processed data will be  placed in the corresponding folder under `test_data` folder .

## \###RUN

For more details about training and test, please refer to ``run.py''

