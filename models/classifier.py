import numpy as np
from sklearn import svm

class Classifier:
    def __init__(self):
        self.classifier = svm.SVC(C=1, kernel='rbf', break_ties=True, cache_size=8000)
        self.train_vectors = None
        self.train_types = None

    def train(self, vectors, types):
        if self.train_vectors is None:
            self.train_vectors = vectors
            self.train_types = types

        self.train_vectors = np.concatenate((self.train_vectors, vectors), axis=0)
        self.train_types = np.concatenate((self.train_types, types), axis=0)

    def fit(self, N):
        # balanced sub sample
        x, y = self.balanced_subsample(self.train_vectors, self.train_types, float(N) / self.train_vectors.shape[0])
        self.classifier.fit(x, y)

    def validate(self, vectors, types):
        res = self.classifier.score(vectors, types)

        return res

    # https://stackoverflow.com/questions/23455728/scikit-learn-balanced-subsampling
    def balanced_subsample(self, x,y,subsample_size=1.0):

        class_xs = []
        min_elems = None

        for yi in np.unique(y):
            elems = x[(y == yi)]
            class_xs.append((yi, elems))
            if min_elems == None or elems.shape[0] < min_elems:
                min_elems = elems.shape[0]

        use_elems = min_elems
        if subsample_size < 1:
            use_elems = int(min_elems*subsample_size)

        xs = []
        ys = []

        for ci,this_xs in class_xs:
            if len(this_xs) > use_elems:
                np.random.shuffle(this_xs)

            x_ = this_xs[:use_elems]
            y_ = np.empty(use_elems)
            y_.fill(ci)

            xs.append(x_)
            ys.append(y_)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        return xs,ys