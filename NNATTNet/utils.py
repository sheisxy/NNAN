import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator,FixedLocator
import random
import scipy as sp
import scipy.stats
from sklearn.neighbors import NearestNeighbors


class BatchCreate(object):
    def __init__(self,images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        '''
        Disruption in the first epoch
        '''
        if self._epochs_completed ==0 and start ==0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        if start+batch_size>self._num_examples:
            #finished epoch
            self._epochs_completed += 1
            '''
            When the remaining sample number of an epoch is less than batch size,
            the difference between them is calculated.
            '''
            rest_num_examples = self._num_examples-start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            '''Disrupt the data'''
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            '''next epoch'''
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part),axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
def drawHeatMap_multiple_dimensions(features):
    fig, ax = plt.subplots()

    row, col = 50, 100
    features = features[0:row, :]
    features = features[:, 0:col]
    im = ax.imshow(features)

    ax.set_title("features")
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

def drawHeatMap_one_dimension(features):
    fig, ax = plt.subplots()
    features = features.reshape(1, len(features))

    row, col = 1, 20
    features = features[0:row, :]
    features = features[:, 0:col]
    im = ax.imshow(features)

    ax.set_title("features")
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

def draw_weight_line_cahrt(data):

    x = range(len(data))
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    # pl.ylim(-1, 110)  # 限定纵轴的范围

    plt.plot(x, data, marker='.', mec='r', mfc='w')
    # plt.xticks(x, names, rotation=1)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)

    plt.xlabel('features')  # X轴标签
    plt.ylabel("weight")  # Y轴标签
    plt.show()
    # pyplot.yticks([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # plt.title("A simple plot") #标题
    # plt.savefig('D:\\f1.jpg', dpi=900)

def cross_validation1(intMat, seeds, A_sim, B_sim, num=10):
    cv_data = defaultdict(list)
    num_targets, num_drugs = intMat.shape

    # get all the negative and positive index
    pos = []
    neg = []
    for i in range(num_targets):
        for j in range(num_drugs):
            if np.sum(intMat[i, :]) >= 1 and np.sum(intMat[:, j]) >= 1:
                if intMat[i][j] == 1:
                    pos.append(i * num_drugs + j)
                else:
                    neg.append(i * num_drugs + j)
    pos = np.array(pos)
    neg = np.array(neg)
    for seed in seeds:
        prng = np.random.RandomState(seed)
        pos_index = prng.permutation(pos)
        neg_index = prng.permutation(neg)
        pos_step = int(pos_index.size / num)
        neg_step = int(neg_index.size / num)
        for i in range(num):
            if i < num:
                pos_ii = pos_index[i * pos_step:(i + 1) * pos_step]
                neg_ii = neg_index[i * neg_step:(i + 1) * neg_step]
            else:
                pos_ii = pos_index[i * pos_step:]
                neg_ii = neg_index[i * neg_step:]

            pos_diff_set = np.array(list(set(pos).difference(set(pos_ii))))
            neg_diff_set = np.array(list(set(neg).difference(set(neg_ii))))

            pos_xy = np.array([[k / num_drugs, k % num_drugs] for k in pos_ii], dtype=np.int32)
            pos_x, pos_y = pos_xy[:, 0], pos_xy[:, 1]

            temp = np.array(intMat, copy=True)
            W = np.ones(intMat.shape)
            W[pos_x, pos_y] = 0
            temp[pos_x, pos_y] = 0

            neg_ii_new = []
            for l in neg_ii:
                u = int(l / num_drugs)
                v = l % num_drugs
                if np.sum(temp[u, :]) >= 1 and np.sum(temp[:, v]) >= 1:
                    neg_ii_new.append(l)

            test_data = np.vstack((np.array([[k / num_drugs, k % num_drugs] for k in pos_ii], dtype=np.int32),
                                   np.array([[k / num_drugs, k % num_drugs] for k in neg_ii_new], dtype=np.int32)))
            train_data = np.vstack((np.array([[k / num_drugs, k % num_drugs] for k in pos_diff_set], dtype=np.int32),
                                    np.array([[k / num_drugs, k % num_drugs] for k in neg_diff_set], dtype=np.int32)))


            edge_encodeA = embedding(intMat * W, A_sim)
            edge_encodeB = embedding(intMat.T * W.T, B_sim)
            edge_encodeAB = np.concatenate((edge_encodeA, edge_encodeB.transpose(1, 0, 2)), axis=-1)



            test_x, test_y = test_data[:, 0], test_data[:, 1]
            test_data = edge_encodeAB[test_x, test_y]
            test_label0 = intMat[test_x, test_y]
            test_label1 = 1 - intMat[test_x, test_y]
            test_label = np.vstack((test_label0, test_label1)).T


            train_x, train_y = train_data[:, 0], train_data[:, 1]
            train_data = edge_encodeAB[train_x, train_y]
            train_label0 = temp[train_x, train_y]
            train_label1 = 1 - temp[train_x, train_y]
            train_label = np.vstack((train_label0, train_label1)).T

            index1 = np.where(train_label0 == 1)[0]

            s = Smote(train_data[index1, :], N=13)  # smote
            new_data = s.over_sampling()

            train_label0 = np.r_[train_label0, np.ones((new_data.shape[0]))]
            train_label1 = 1 - train_label0
            train_label = np.vstack((train_label0, train_label1)).T

            cv_data[seed].append((train_data, train_label, test_data, test_label))
    return cv_data

def cross_validation23(intMat, seeds , sim, num=10):
    cv_data = defaultdict(list)
    num_targets, num_drugs = intMat.shape
    intMat = remove_positive_edges(intMat, seeds, 0)
    for seed in seeds:
        prng = np.random.RandomState(seed)
        index = prng.permutation(num_drugs)  # shape[num_drugs]
        step = int(index.size / num)
        for i in range(num):
            if i < num:
                ii = index[i * step:(i + 1) * step]
            else:
                ii = index[i * step:]
            diff_set = set(index).difference(set(ii))
            tar = []
            for m in range(num_targets):
                if np.sum(intMat[m, np.array(list(diff_set))] == 1) >= 1:
                    tar.append(m)
            test_data = np.array([[k, j] for k in tar for j in ii], dtype=np.int32)
            train_data = np.array([[k, j] for k in tar for j in diff_set], dtype=np.int32)

            test_x, test_y = test_data[:, 0], test_data[:, 1]


            W = np.ones(intMat.shape)
            W[test_x, test_y] = 0
            A = delete(intMat * W, train_data, seed, 0.8)

            tar = []
            for m in range(num_targets):
                if np.sum(A[m, np.array(list(diff_set))] == 1) >= 1:
                    tar.append(m)
            test_data = np.array([[k, j] for k in tar for j in ii], dtype=np.int32)
            train_data = np.array([[k, j] for k in tar for j in diff_set], dtype=np.int32)

            edge_encode = embedding(A, sim)

            train_x, train_y = train_data[:, 0], train_data[:, 1]
            train_data = edge_encode[train_x, train_y]
            train_label0 = A[train_x, train_y]
            train_label1 = 1 - A[train_x, train_y]
            train_label = np.vstack((train_label0, train_label1)).T

            test_x, test_y = test_data[:, 0], test_data[:, 1]
            test_data = edge_encode[test_x, test_y]
            test_label0 = intMat[test_x, test_y]
            test_label1 = 1 - intMat[test_x, test_y]
            test_label = np.vstack((test_label0, test_label1)).T

            if np.sum(test_label0 == 1) == 0:
                continue
            cv_data[seed].append((train_data, train_label, test_data, test_label))
    return cv_data

# edge encode
def embedding(intMat, sim):
    T_count, D_count = np.shape(intMat)
    edge_encode = []

    for i in range(T_count):
        for j in range(D_count):
            index = np.argsort(sim[j, :])[::-1]

            embedding = np.delete((np.sort(sim[j, :])[::-1] * intMat[i, index]), 0)
            edge_encode.append(embedding)
    edge_encode = np.array(edge_encode, dtype='float')
    edge_encode = edge_encode.reshape(T_count, D_count, -1)

    return edge_encode

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# randomly remove some edges
def remove_positive_edges(A, seed, percent):
    pos_samples_sum = np.sum(A == 1)
    pos_samples = np.where(A == 1)
    step = int(pos_samples_sum * percent)

    pos_samples_zip = np.array(list(zip(pos_samples[0], pos_samples[1])))
    np.random.seed(seed)
    np.random.shuffle(pos_samples_zip)
    if percent == 0:
        return A
    x, y = zip(*pos_samples_zip[0: step])
    A[x, y] = 0
    return A

def delete(intMat, train_data, seed, percent):
    A = np.array(intMat, copy=True)
    train_x, train_y = train_data[:, 0], train_data[:, 1]
    train_label = A[train_x, train_y]

    pos_samples_sum = np.sum(train_label == 1)
    pos_samples = np.where(train_label == 1)[0]
    prng = np.random.RandomState(seed)
    index = prng.permutation(pos_samples)  # shape[num_drugs]
    step = int(pos_samples_sum * percent)
    train_del_x, train_del_y = train_x[index[0:step]], train_y[index[0:step]]
    A[train_del_x, train_del_y] = 0
    return A


def load_data_from_file(dataset, folder):
    def load_rating_file_as_matrix(filename):
        data = []
        stat = []
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.strip().split("\t"), dtype='int')
                    stat.append(sum(arr))
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.strip().split("\t")
                    try:
                        arr = arr[1:]
                        arr = np.array(arr, dtype='int')
                        stat.append(sum(arr))
                        data.append(arr)
                    except:
                        pass
                    line = f.readline()

        # Construct matrix
        mat = np.array(data)
        return mat

    def load_matrix(filename):
        data = []
        # for the situation that data files contain col/row name
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.strip().split("\t"), dtype='float')
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.strip().split("\t")
                    try:
                        arr = arr[1:]
                        arr = np.array(arr, dtype='float')
                        data.append(arr)
                    except:
                        pass
                    line = f.readline()
        mat = np.array(data)
        return mat

    int_array = load_rating_file_as_matrix(os.path.join(folder, dataset + "_int.txt"))

    A_sim = load_matrix(os.path.join(folder, dataset + "_A_sim.txt"))
    B_sim = load_matrix(os.path.join(folder, dataset + "_B_sim.txt"))

    intMat = np.array(int_array, dtype=np.float64)
    A_sim = np.array(A_sim, dtype=np.float64)
    B_sim = np.array(B_sim, dtype=np.float64)

    return intMat, A_sim, B_sim

def get_names(dataset, folder):
    with open(os.path.join(folder, dataset + "_int.txt"), "r") as inf:
        B = next(inf).strip("\n").split('\t')
        A = [line.strip("\n").split('\t')[0] for line in inf]
        if '' in A:
            A.remove('')
        if '' in B:
            B.remove('')
    return A, B

class Smote:
    def __init__(self, samples, N=1, k=5):
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0

    # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N = int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]
            # print nnarray
            self._populate(N, i, nnarray)
        return self.synthetic

    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap * dif
            self.newindex += 1



