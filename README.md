# HateMemeDetection

This is the code for our submission for Task A to the Shared Task on Hateful Memes Detection at WOAH 2021. It is built on top of the [winning system](https://github.com/HimariO/HatefulMemesChallenge) of a previous hateful memes binary classification shared task.

## Dataset Download

The original memes dataset could be downloaded from [Facebook](https://www.drivendata.org/competitions/64/hateful-memes).

The augmented images could be downloaded [here](https://uni-duisburg-essen.sciebo.de/s/FfYDT9iLGjLqkQL).

After both folders are downloaded, merge them together in a folder called `data/hateful_memes/img`.

## Training

Use the following command to train:

```bash
bash VL-BERT/run_train.sh
```

CSV and JSON inference results will be located at `checkpoints/vl-bert/{MODEL_NAME}/{CONFIG_NAME}`.

### **Pre-trained VL-BERT Model**

Download the pre-trained VL-BERT model [here](https://drive.google.com/file/d/15IAT7NVCXtTj_9itl7OXtA_jXRwiaVWZ/view).

After downloading, place the models in the folder `pretrain_model`.

### **Training Configurations**

Path to the training config file: `VL-BERT/cfgs/cls`.

Path to the dataset config file: `VL-BERT/cls/data/datasets`.


 VL-BERT Large                    | Training config file    | Dataset config file        | Box annotations  | Train/Val set  |
| ------------------------ |:-------------------|:------------ | :------------- | :------------- |
| +W | large_4x14G_fp32_k8s_v4.yaml | cls_v2.py            | box_annos.json   | train.entity.jsonl |
| +W,RG | large_4x14G_fp32_k8s_v5_race.yaml | cls_v3.py            | box_annos.race.json   | train.entity.jsonl |
| +W,E | large_4x14G_fp32_k8s_v5_emotion.yaml | cls_v5.py            | box_annos.emotion.json   | train.entity.jsonl |
| +W,RG,E | large_4x14G_fp32_k8s_v5_race_emotion.yaml | cls_v4.py            | box_annos.race_emotion.json   | train.entity.jsonl |
| U&#124;+W | large_4x14G_fp32_k8s_v4.yaml | cls_v2.py            | box_annos.json   | train_undersampled.entity.jsonl |
| U&#124;+W,RG | large_4x14G_fp32_k8s_v5_race.yaml | cls_v3.py            | box_annos.race.json   | train_undersampled.entity.jsonl |
| I&#124;+W | large_4x14G_fp32_k8s_v4.yaml | cls_v2.py            | box_annos_imgaug.json   | train_imgaug.entity.jsonl |
| I&#124;+W,RG | large_4x14G_fp32_k8s_v5_race.yaml | cls_v3.py            | box_annos_imgaug.race.json   | train_imgaug.entity.jsonl |
| T&#124;+W | large_4x14G_fp32_k8s_v4.yaml | cls_v2.py            | box_annos.json   | train_textaug.entity.jsonl |
| I&#124;+W,RG | large_4x14G_fp32_k8s_v5_race.yaml | cls_v3.py            | box_annos.race.json   | train_textaug.entity.jsonl |
| IT&#124;+W | large_4x14G_fp32_k8s_v4.yaml | cls_v2.py            | box_annos_imgaug.json   | train_imgtextaug.entity.jsonl |
| I&#124;+W,RG | large_4x14G_fp32_k8s_v5_race.yaml | cls_v3.py            | box_annos_imgaug.race.json   | train_imgtextaug.entity.jsonl |


Note: W = Web Entities, RG = Race and Gender, E = Emotion, U = Undersampling, I = Image Augmentation, IT = Image and Text Augmentation


Results in the paper are tested on `dev_all.entity.jsonl`.


For training, the original train set is split into two sets — `train1` as the training set and `train2` as the validation set. If you want to replicate the results in the paper, use `train1(_<aug>).entity.jsonl` as the training set and `train2(_<aug>).entity.jsonl` as the validation set. For example: For text augmentation, training set: `train1_textaug.entity.jsonl` and validation set: `train2_textaug.entity.jsonl`.


The used box annotations and train/val sets must be specified inside the Python dataset config file `cls_v<version>.py`.


You can specify which split should be used as train, val, and test sets in the training config file. For example, "`train1`" corresponds to any `train1(_<aug>).entity.jsonl`, "val_all" corresponds `dev_all.entity.jsonl`.