import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

import argparse

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import os
import glob


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


def evaluate(scores, labels, CLASSES, system_name):
    """
    Evaluates the predicted classes w.r.t. a gold file.
    """

    vocab_map = dict([(i, v) for i, v in enumerate(CLASSES)])

    mlb = MultiLabelBinarizer()
    mlb.fit([CLASSES])
    # Hack to maintain order
    mlb.classes_ = np.array(CLASSES)
    print(CLASSES)
    gold_label = mlb.transform(labels.tolist())
    print(gold_label[20:26].tolist())
    pred_score = np.matrix(scores.tolist())
    pred_label = (pred_score > 0.5).astype(int)
    print(pred_label[20:26].tolist())

    # visualize roc_auc
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    CLASSES1 = ['none' if x == 'pc_empty' else x for x in CLASSES]
    for i in range(len(CLASSES1)):
        fpr[i], tpr[i], _ = roc_curve(gold_label[:, i], pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    f, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    label_font = {'size': '25'}
    title_font = {'size': '21'}
    plt.rcParams.update({'font.size': 25})
    for i in range(len(CLASSES1)):
        disp = ConfusionMatrixDisplay(confusion_matrix(gold_label[:, i],
                                                       pred_label[:, i]),
                                      display_labels=['False', 'True'])
        disp.plot(ax=axes[i], cmap=plt.cm.Greys, values_format='.8g')
        disp.ax_.set_title(f'%s' % CLASSES1[i])
        disp.ax_.tick_params(axis='both', which='major', labelsize=15)
        if i < 3:
            disp.ax_.set_xlabel(' ', fontdict=label_font)
        if i % 3 != 0:
            disp.ax_.set_ylabel(' ', fontdict=label_font)
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.14, hspace=0.2)
    #f.colorbar(disp.im_, ax=axes)
    plt.savefig('../confmatrix.png', dpi=600)

    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(6, 4))
    area = []
    pattern = ['-+', '-.', ':', '-*', '-,', '--']
    for i in range(len(CLASSES)):
        area.append(roc_auc[i])
        print('%s (area = %0.2f)' % (CLASSES1[i], roc_auc[i]))
        plt.plot(fpr[i], tpr[i], pattern[i], label='%s (AUC = %0.2f)' % (CLASSES1[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(r"$VL-BERT_{LARGE}$ + (R, G and W)")
    #plt.title("%s with micro roc_auc = %0.2f and micro F1 = %0.2f" %(system_name.upper(),
     #           roc_auc_score(gold_label, pred_score, average="micro", multi_class="ovr"),
        #      f1_score(gold_label, pred_label, average="micro")))
    plt.legend(loc="lower right")
    plt.savefig('../roc_curves/%s.png' % system_name, dpi=800)
    print(sum(area)/6)


    roc_auc = roc_auc_score(gold_label, pred_score, average="micro", multi_class="ovr")
    f1 = f1_score(gold_label, pred_label, average="micro")

    #cm = confusion_matrix(gold_label.argmax(axis=1), pred_label.argmax(axis=1))
    #df_cm = pd.DataFrame(cm)
    #sn.heatmap(df_cm, annot=True)



    conf_mat_dict = {}

    for label_col in range(len(CLASSES)):
        y_true_label = gold_label[:, label_col]
        y_pred_label = pred_label[:, label_col]
        conf_mat_dict[CLASSES[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)



    #cm = confusion_matrix(gold_label, pred_label)
    #df_cm = pd.dataframe(cm)
    #sn.heatmap(df_cm, annot=True)
    return f1, roc_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file_path",
        "-g",
        type=str,
        required=True,
        help="Paths to the file with gold annotations.",
    )
    parser.add_argument(
        "--pred_file_path",
        "-p",
        type=str,
        required=False,
        help="Path to the file with predictions",
    )

    args = parser.parse_args()

    for json_file in glob.glob('../dev_jsons/large_4x14G_fp32_k8s_v5_race_cls_val_all.json'):
        with open(args.gold_file_path, encoding="utf-8") as gold_f:
            reference = pd.read_json(gold_f, lines=True)
            reference = reference.drop_duplicates(['id'])


        with open(json_file, encoding="utf-8") as pred_f:
            predictions = pd.read_json(pred_f, lines=True)
            predictions = predictions.drop_duplicates(['id'])

        eval_dataset = predictions.merge(reference, on='id')

        for task in ['pc']:
            classes=eval_dataset[f'gold_{task}'].apply(lambda x:x[0]).unique()
            pred=eval_dataset[f'pred_{task}'].apply(lambda x: [x[k] for k in classes])
            gold=eval_dataset[f'gold_{task}']
            system_name = '_'.join(os.path.basename(json_file).split('_')[5:]).replace('.json', '')
            f1,roc_auc = evaluate(pred,gold,classes, system_name)
            #print(json_file)
            print(f"{task} f1:{f1} roc_auc:{roc_auc}")

