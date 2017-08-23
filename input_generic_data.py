"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import numpy as np

import numpy


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_features(filename):
    data = np.genfromtxt(filename,delimiter=',')[:,1:]
    return data


def extract_labels(filename):
    Yt = np.genfromtxt(filename, delimiter=',')[:,0]
    labels = np.array([1-Yt, Yt]).T
    return labels


class DataSet(object):

  def __init__(self, features, labels, fake_data=False):
    assert features.shape[0] == labels.shape[0], (
      "features.shape: %s labels.shape: %s" % (features.shape,
                                             labels.shape))
    self._num_examples = features.shape[0]
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]

class SemiDataSet(object):
    def __init__(self, features, labels, n_labeled):
        self.n_labeled = n_labeled

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(features, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        features = features[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(2)[l==1][0] for l in labels])
        n_classes = y.max() + 1
        n_from_each_class = n_labeled / n_classes
        i_labeled = []
        for c in range(n_classes):
            i = indices[y==c][:n_from_each_class]
            i_labeled += list(i)
        l_features = features[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_features, l_labels)

    def next_batch(self, batch_size):
        unlabeled_features, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_features, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_features, labels = self.labeled_ds.next_batch(batch_size)
        features = numpy.vstack([labeled_features, unlabeled_features])
        return features, labels

def read_data_sets(train_dir, n_labeled = 100, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  TRAIN_FEATURES = '/Users/tdawn/Documents/wrk/data/m5.ladder.train.csv'
  TRAIN_LABELS = '/Users/tdawn/Documents/wrk/data/m5.ladder.train.csv'
  TEST_FEATURES = '/Users/tdawn/Documents/wrk/data/m5.ladder.test.csv.small'
  TEST_LABELS = '/Users/tdawn/Documents/wrk/data/m5.ladder.test.csv.small'
  VALIDATION_SIZE = 0

  train_features = extract_features(TRAIN_FEATURES)
  train_labels = extract_labels(TRAIN_LABELS)

  test_features = extract_features(TEST_FEATURES)
  test_labels = extract_labels(TEST_LABELS)

  validation_features = train_features[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_features = train_features[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = SemiDataSet(train_features, train_labels, n_labeled)
  data_sets.validation = DataSet(validation_features, validation_labels)
  data_sets.test = DataSet(test_features, test_labels)

  return data_sets
