# ann
Simple pybrain wrapper

Requires pybrain and by extension NumPy, SciPy and all dependent libraries.

Usage.
To train network:
  python train.py <samples_filename> <labels_filename> <net_filename>
To execute detection:
  python detect.py <samples_filename> <labels_filename> <net_filename>

Both samples file and labels file should be CSV.
