import codecs
import json
import sys


def main():
    with codecs.open(sys.argv[1], 'r', 'utf-8') as json_obj:
        news_jsons = []
        for line in json_obj:
            json_lin_obj = json.loads(line)
            new_json_line = json_lin_obj
            for key,val in json_lin_obj['pred_pc'].items():
                if key == 'pc_empty':
                    new_json_line['pred_pc'][key] = float(1.0)
                else:
                    new_json_line['pred_pc'][key] = float(0.0)
            news_jsons.append(new_json_line)
    with codecs.open("%s_mj_class"%sys.argv[1], 'w', 'utf-8') as jsonw_obj:
        for entry in news_jsons:
            json.dump(entry, jsonw_obj)
            jsonw_obj.write('\n')


if __name__ == '__main__':
    main()

