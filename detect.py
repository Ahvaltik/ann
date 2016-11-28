import pickle
import sys


def read_samples(_samples_filename):
    samples_file = open(_samples_filename, mode='r')
    _samples = map(lambda x: map(float, x.split(',')), samples_file.readlines())
    samples_file.close()
    return _samples


if len(sys.argv) is not 4:
    print "Usage: python detect.py <samples_filename> <labels_filename> <net_filename>"
else:
    samples_filename = sys.argv[1]
    labels_filename = sys.argv[2]
    net_filename = sys.argv[3]
    samples = read_samples(samples_filename)

    net_file = open(net_filename, mode='r')
    net = pickle.load(net_file)
    net_file.close()

    labels = open(labels_filename, mode='w')
    for sample in samples:
        activ_str = ', '.join(map(str, net.activate(sample))) + '\n'
        labels.write(activ_str)
    labels.flush()
    labels.close()

