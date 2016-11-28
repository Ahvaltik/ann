from pybrain import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
import pickle
import sys


def read_dataset(_samples_filename, _labels_filename):
    samples_file = open(_samples_filename, mode='r')
    labels_file = open(_labels_filename, mode='r')
    samples = map(lambda x: map(float, x.split(',')), samples_file.readlines())
    labels = map(lambda x: map(float, x.split(',')), labels_file.readlines())
    in_size = len(samples[0])
    for sample in samples:
        if len(sample) != in_size:
            raise Exception('Improper sample file')
    out_size = len(labels[0])
    for label in labels:
        if len(label) != out_size:
            raise Exception('Improper label file')
    dataset = SupervisedDataSet(in_size, out_size)
    map(lambda x: dataset.addSample(x[0], x[1]), zip(samples, labels))
    samples_file.close()
    labels_file.close()
    return dataset, in_size, out_size


if len(sys.argv) is not 4:
    print "Usage: python train.py <samples_filename> <labels_filename> <net_filename>"
else:
    samples_filename = sys.argv[1]
    labels_filename = sys.argv[2]
    net_filename = sys.argv[3]
    try:
        ds, in_size, out_size = read_dataset(samples_filename, labels_filename)
        net = buildNetwork(in_size, 3, out_size, bias=True, hiddenclass=TanhLayer)
        trainer = BackpropTrainer(net, ds)
        trainer.trainUntilConvergence()
        net_file = open(net_filename, mode='w')
        pickle.dump(net, net_file)
        net_file.flush()
        net_file.close()
    except Exception as err:
        print err
