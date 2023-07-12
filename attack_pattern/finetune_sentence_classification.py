
from typing import TextIO

import train_sentence_classification
import argparser
import itertools
import collections


import calendar
import os
import time


def hypertune(f: TextIO, rf: TextIO):

    arg = argparser.parse_bert_sentence_classification_arguments()

    params_dict = {'model': ['bert-base-uncased'],
              'lr':[0.000005],
              'batch_size': [16],
              'epoch': [20],
              'dropout': [0.5],
              'fine_tune':[True]}

    keys, values = zip(*params_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    Hypertuning = collections.namedtuple('Hypertuning', ['Params', 'test_acc', 'f1'])

    counter = 0
    savepath = str(arg.save_path)
    rf.write("counter,seed,lr,batch_size,epoch,dropout,fine_tune,test_acc,f1,test_acc_avg,f1_avg,save_path\n")
    for oneConfig in permutations_dicts:
        test_acc_list = []
        f1_list = []
        for seed in range(1,2):
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


            print("\n"*5)
            print("*"*40)
            print(" "*10,"Parameters")
            print(oneConfig)
            print("\n")
            print(arg)
            print("\n"*2)

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
                classifier = train_sentence_classification.Classification(arg)
                test_acc, f1, _, _ = classifier.run()
                record.append(str(test_acc))
                record.append(str(f1))

                test_acc_list.append(test_acc)
                f1_list.append(f1)

                record.append(str(sum(test_acc_list)/seed))
                record.append(str(sum(f1_list) / seed))

                record.append(str(arg.save_path))
                record.append("\n")
                rf.write(",".join(record))
                rf.flush()
                os.fsync(rf)

                hyp = Hypertuning(oneConfig, test_acc, f1)
                print("Training results: ")
                print(hyp)

                f.write("Training results: ")
                f.write(str(hyp))
                results.append(hyp)
                f.flush()
                os.fsync(f)
            except Exception as e:
                print("\n"*2)
                print("Exception occurred for configuration: " , oneConfig)
                print(e)
                print("\n"*2)

                f.write("\n" * 2)
                f.write("Exception occurred for configuration: " +  str(oneConfig))
                f.write(str(e))
                f.write("\n" * 2)
                f.flush()
                os.fsync(f)
            counter = counter + 1

    print("\n\n")
    print("*" * 40)
    print("Results:")
    print(results)
    print("\n\n")

    f.write("\n\n")
    f.write("*" * 40)
    f.write("Results:")
    f.write(str(results))
    f.write("\n\n")


    print("*" * 40)
    f.write("*" * 40)

    print(results)

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
    
    with open(path, 'w') as f, open(path + "report.csv", 'w') as rf:
        hypertune(f,rf)
