from typing import TextIO

import train_entity_extraction_refactored
import argparser
import itertools
import collections

import calendar
import os
import time


def hypertune(f: TextIO, rf: TextIO):
    arg = argparser.parse_entity_recognition_arguments()

    my_dict = {#'pretrained_model': ['xlm-roberta-large'],
               'pretrained_model': ['roberta-base', 'roberta-large'],
               # 'lr':[0.00001, 0.000001, 0.000005],
               'lr':[0.000005],
               'batch_size': [16],
               'epoch': [30],
               'dropout': [0.5],
               'fine_tune':[True]}

    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


    counter = 0
    savepath = str(arg.save_path)
    #rf.write("counter,seed,lr,batch_size,epoch,sequence_length,gradient_clip,test_acc,f1,precision,recall,test_acc_avg,f1_avg,precision_avg,recall_avg,save_path\n")
    rf.write(
        "counter,seed,lr,batch_size,epoch,sequence_length,gradient_clip,test_acc,f1,precision,recall,test_acc_avg,save_path\n")
    for oneConfig in permutations_dicts:
        test_acc_list = []
        test_f1_list = []
        test_precision_list = []
        test_recall_list = []

        for seed in range(1, 2):
            gmt = time.gmtime()

            ts = calendar.timegm(gmt)
            record = []

            arg.save_path = os.path.join(savepath, str(ts))
            record.append(str(counter))
            record.append(str(seed))
            arg.__setattr__('seed', seed)
            for oneParamKey, oneParamValue in oneConfig.items():
                record.append(str(oneParamValue))
                arg.__setattr__(oneParamKey, oneParamValue)

            print("\n" * 5)
            print("*" * 40)
            print(" " * 10, "Parameters")
            print(oneConfig)
            print("\n")
            print(arg)
            print("\n" * 2)

            f.write("\n" * 5)
            f.write("*" * 40)
            f.write(" " * 10 + "Parameters")
            f.write(str(oneConfig))
            f.write("\n")
            f.write(str(arg))
            f.write("\n" * 2)
            f.flush()
            os.fsync(f)
            try:
                classifier = train_entity_extraction_refactored.EntityClassification(arg)
                accuracy_list, f1_list, precision_list, recall_list = classifier.train()
                record.append(str(sum(accuracy_list)/len(accuracy_list)))
                
                record.append(str(f1_list))
                record.append(str(precision_list))
                record.append(str(recall_list))
                #record.append(str(sum(f1_list)/len(f1_list)))
                # record.append(str(sum(precision_list) / len(precision_list)))
                # record.append(str(sum(recall_list) / len(recall_list)))

                test_acc_list.append(sum(accuracy_list)/len(accuracy_list))
                #test_f1_list.append(sum(f1_list)/len(f1_list))
                # test_precision_list.append(sum(precision_list) / len(precision_list))
                # test_recall_list.append(sum(recall_list) / len(recall_list))

                record.append(str(sum(test_acc_list) / seed))
                #record.append(str(sum(test_f1_list) / seed))
                # record.append(str(sum(test_precision_list) / seed))
                # record.append(str(sum(test_recall_list) / seed))

                record.append(str(arg.save_path))
                record.append("\n")
                rf.write(",".join(record))
                rf.flush()
                os.fsync(rf)

                f.flush()
                os.fsync(f)
            except Exception as e:
                print("\n" * 2)
                print("Exception occurred for configuration: ", oneConfig)
                print(e)
                print("\n" * 2)

                f.write("\n" * 2)
                f.write("Exception occurred for configuration: " + str(oneConfig))
                f.write(str(e))
                f.write("\n" * 2)
                f.flush()
                os.fsync(f)
            counter = counter + 1

    # print("\n\n")
    # print("*" * 40)
    # print("Results:")
    # print(results)
    # print("\n\n")
    #
    # f.write("\n\n")
    # f.write("*" * 40)
    # f.write("Results:")
    # f.write(str(results))
    # f.write("\n\n")
    #
    # print("*" * 40)
    # f.write("*" * 40)
    # idx_acc, max_test_acc = max(results, key=lambda item: item[1])
    #
    # print('Maximum value:', max_test_acc)
    #
    # print("\n\n")
    # print("*" * 40)
    #
    # f.write('Maximum value:' + str(max_test_acc))
    #
    # f.write("\n\n")
    # f.write("*" * 40)
    # idx_f1, max_f1 = max(results, key=lambda item: item[2])
    #
    # print('Maximum value:', max_f1)
    # f.write('Maximum value:' + str(max_f1))


if __name__ == "__main__":

    gmt = time.gmtime()

    ts = calendar.timegm(gmt)

    path = 'out/logs/hypertune/' + str(ts)
    if not os.path.isdir('out'):
        os.mkdir('out')
        os.mkdir('out/logs')
        os.mkdir('out/logs/hypertune')

    if not os.path.isdir('out/logs'):
        os.mkdir('out/logs')
        os.mkdir('out/logs/hypertune')

    if not os.path.isdir('out/logs/hypertune'):
        os.mkdir('out/logs/hypertune')

    with open(path, 'w') as f, open(path + "entity_extraction_report.csv", 'w') as rf:
        hypertune(f, rf)




