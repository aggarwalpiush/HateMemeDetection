from statsmodels.stats.contingency_tables import mcnemar


import numpy as np

import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.stats.api import SquareTable
import pandas as pd
import codecs
import os
import glob


def mcnemar_test(metrA, metrB, gold):
    metAmetB_correct = 0
    metAmetB_incorrect = 0
    metA_correct_metB_not = 0
    metB_correct_metA_not = 0

    for i, label in enumerate(gold):
        lab = label.tolist()[0]
        metA = metrA[i].tolist()[0]
        metB = metrB[i].tolist()[0]
        if lab == metA and lab == metB:
            metAmetB_correct += 1
        elif lab == metA and not lab == metB:
            metA_correct_metB_not += 1
        elif lab == metB and not lab == metA:
            metB_correct_metA_not += 1
        else:
            metAmetB_incorrect += 1
    table = [[metAmetB_correct, metA_correct_metB_not], [metB_correct_metA_not, metAmetB_incorrect]]
    print(table)
    result = mcnemar(table, exact=False, correction=True)
    return result

def bowker_test(metrA, metrB, gold, classes):
    before = []
    after = []
    for i, label in enumerate(gold):
        metA = metrA[i].tolist()[0]
        metB = metrB[i].tolist()[0]
        #print(i)
        #print(metrA[i])
        before.append(classes[metA.index(1)])
        after.append(classes[metB.index(1)])
    d = {'before': before, 'after': after}
    myDf = pd.DataFrame(d)
    myCross = pd.crosstab(myDf['before'], myDf['after'])
    print(myCross)
    print(SquareTable(myCross, shift_zeros=False).symmetry())
    return 0


def evaluate(scores_m1, scores_m2, labels, CLASSES):
    """
    Evaluates the predicted classes w.r.t. a gold file.
    """

    vocab_map = dict([(i, v) for i, v in enumerate(CLASSES)])

    mlb = MultiLabelBinarizer()
    mlb.fit([CLASSES])
    # Hack to maintain order
    mlb.classes_ = np.array(CLASSES)

    gold_label = mlb.transform(labels.tolist())
    pred_score_m1 = np.matrix(scores_m1.tolist())

    pred_label_m1 = (pred_score_m1 == pred_score_m1.max(1)).astype(int)


    pred_score_m2 = np.matrix(scores_m2.tolist())
    pred_label_m2 = (pred_score_m2  == pred_score_m2.max(1)).astype(int)


    for i, clas in enumerate(classes):
        print(clas)
        res = mcnemar_test(pred_label_m1[:,i], pred_label_m2[:,i], np.transpose(np.matrix(gold_label[:,i])))
        print('statistic=%.3f, p-value=%.3f\n' % (res.statistic, res.pvalue))

    bowker_test(pred_label_m1, pred_label_m2, np.matrix(gold_label), CLASSES)
    return mcnemar_test(pred_label_m1, pred_label_m2, np.matrix(gold_label))


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
        "--pred_file_path1",
        "-p1",
        type=str,
        required=True,
        help="Path to the file with method1 predictions",
    )
    parser.add_argument(
        "--pred_file_path2",
        "-p2",
        type=str,
        required=False,
        help="Path to the file with method2 predictions",
    )

    args = parser.parse_args()

    alpha = 0.05
    for comp_file in glob.glob('../dev_jsons/*'):
        with open(args.gold_file_path, encoding="utf-8") as gold_f:
            reference = pd.read_json(gold_f, lines=True)
            reference = reference.drop_duplicates(['id'])

        with open(args.pred_file_path1, encoding="utf-8") as pred_f1:
            predictions_m1 = pd.read_json(pred_f1, lines=True)
            predictions_m1 = predictions_m1.drop_duplicates(['id'])

        with open(comp_file, encoding="utf-8") as pred_f2:
            predictions_m2 = pd.read_json(pred_f2, lines=True)
            predictions_m2 = predictions_m2.drop_duplicates(['id'])

        eval_dataset1 = predictions_m1.merge(reference, on='id')
        eval_dataset2 = predictions_m2.merge(reference, on='id')

        for task in ['pc']:
            classes=eval_dataset1[f'gold_{task}'].apply(lambda x:x[0]).unique()
            pred1=eval_dataset1[f'pred_{task}'].apply(lambda x: [x[k] for k in classes])
            pred2 = eval_dataset2[f'pred_{task}'].apply(lambda x: [x[k] for k in classes])
            gold=eval_dataset1[f'gold_{task}']
            print('\n\n===================================\n\n')
            print('modelA: %s\n' % os.path.basename(args.pred_file_path1))
            print('ModelB: %s\n' % os.path.basename(comp_file))
            result = evaluate(pred1, pred2, gold, classes)
            with codecs.open('../dev_sig_test/mcnemer_%s_%s' %(os.path.basename(args.pred_file_path1),
                                               os.path.basename(comp_file)), 'w', 'utf-8') as wr_obj:
                wr_obj.write('modelA: %s\n' % os.path.basename(args.pred_file_path1))
                wr_obj.write('ModelB: %s\n' % os.path.basename(comp_file))
                print('statistic=%.3f, p-value=%.3f\n' % (result.statistic, result.pvalue))
                wr_obj.write('statistic=%.3f, p-value=%.3f\n' % (result.statistic, result.pvalue))
                # interpret the p-value
                if result.pvalue > alpha:
                    wr_obj.write('Same proportions of errors (fail to reject H0)\n')
                else:
                    wr_obj.write('Different proportions of errors (reject H0)\n')
                print('===================================\n\n')







