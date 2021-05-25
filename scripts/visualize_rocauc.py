import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

import argparse

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def evaluate(scores, labels, CLASSES):
    """
    Evaluates the predicted classes w.r.t. a gold file.
    """

    vocab_map = dict([(i, v) for i, v in enumerate(CLASSES)])

    mlb = MultiLabelBinarizer()
    mlb.fit([CLASSES])
    # Hack to maintain order
    mlb.classes_ = np.array(CLASSES)

    gold_label = mlb.transform(labels.tolist())
    pred_score = np.matrix(scores.tolist())
    pred_label = (pred_score > 0.5).astype(int)

    # visualize roc_auc
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(gold_label[:, i], pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    plt.figure()
    for i in range(len(CLASSES)):
        plt.plot(fpr[i], tpr[i], label='ROC curve for %s (area = %0.2f)' % (CLASSES[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('VL-BERT_LARGE (Baseline)')
        plt.legend(loc="lower right")
    plt.show()


    #skplt.metrics.plot_roc_curve(gold_label, pred_score)
    #plt.show()

    roc_auc = roc_auc_score(gold_label, pred_score, average="micro", multi_class="ovr")
    f1 = f1_score(gold_label, pred_label, average="micro")
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
        required=True,
        help="Path to the file with predictions",
    )

    args = parser.parse_args()

    with open(args.gold_file_path, encoding="utf-8") as gold_f:
        reference = pd.read_json(gold_f, lines=True)
        reference = reference.drop_duplicates(['id'])

    with open(args.pred_file_path, encoding="utf-8") as pred_f:
        predictions = pd.read_json(pred_f, lines=True)
        predictions = predictions.drop_duplicates(['id'])

    eval_dataset = predictions.merge(reference, on='id')

    for task in ['pc']:
        classes=eval_dataset[f'gold_{task}'].apply(lambda x:x[0]).unique()
        pred=eval_dataset[f'pred_{task}'].apply(lambda x: [x[k] for k in classes])
        gold=eval_dataset[f'gold_{task}']
        f1,roc_auc = evaluate(pred,gold,classes)
        print(f"{task} f1:{f1} roc_auc:{roc_auc}")

