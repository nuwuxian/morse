from __future__ import print_function
import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal

import random
import torch
import numpy as np

def solver(noise_rate, imb_type='step', imb_ratio=0.1, max_num=5000, num_class=10):
    from scipy import optimize

    eq_num = num_class * 3
    var_num = num_class ** 2

    M = np.zeros((eq_num, var_num + 1))
    # top 10 sum to 1
    for i in range(num_class):
        st = i * num_class
        for j in range(st, st + num_class):
            M[i, j] = 1
        M[i, var_num] = 1.0

    # 10: diagonal noise_rate
    for i in range(num_class, 2 * num_class):
        j = i % num_class
        M[i, j * (num_class + 1)] = 1
        M[i, var_num] = 1 - noise_rate

    cls_num_list = []
    if imb_type == 'step':
        for i in range(int(num_class / 2)):
            cls_num_list.append(max_num)
        for i in range(int(num_class / 2)):
            cls_num_list.append(int(max_num * imb_ratio))
    else:
        cls_num = num_class
        for cls_idx in range(cls_num):
            num = max_num * (imb_ratio**(cls_idx / (cls_num - 1.0)))
            cls_num_list.append(int(num))

    # 10: imb top 5 1000 top 5 100 
    for i in range(2 * num_class, 3 * num_class):
        # 0-10-20 ... 90
        # 1-11-21 ... 91
        # 9-19-29 ... 99
        st = i % num_class
        for j in range(st, st+var_num, num_class):
            M[i, j] = cls_num_list[int(j / num_class)]
        M[i, var_num] = cls_num_list[int(i % num_class)]

    A_eq = M[:, :var_num]
    b_eq = M[:, var_num]

    c = np.ones(var_num)

    ans = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    assert ans['success'] == True
    p = ans['x'].reshape(num_class, num_class)
    print(p)
    assert_array_almost_equal(p.sum(axis=1), np.ones(num_class))
    return p


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """
        mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # flip class
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P

def noisify_sparse(y_train, noise, random_state=None, nb_classes=10, select_class=10):
    # init the matrix
    P = np.zeros((nb_classes, nb_classes))
    n = noise

    if imb_type == 'none':
        for i in range(nb_classes):
            P[i, i] = 1 - n
            m = n / (select_class - 1)
            for j in range(i + 1, i + select_class):
                j = j % (nb_classes)
                P[j, i] = m
    elif imb_type == 'step':
        num_c = nb_classes / 2
        for i in range(num_c):
            P[i, i] = 1 - n 
            j = (i + 1) % num_c 
            P[j, i] = n
        for i in range(num_c, nb_classes):
            P[i, i] = 1 - n 
            j = num if i + 1 == nb_classes else i + 1
            P[j, i] = n

    elif imb_type == 'exp':
        for i in range(5):
            p[2 * i, 2 * i] = 1 - n
            p[2 * i, 2 * i + 1] =  n
            p[2 * i + 1, 2 * i] = n
            p[2 * i + 1, 2 * i + 1] = 1 - n

    y_train_noisy = multiclass_noisify(y_train, P=P,
                    random_state=random_state)
    
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P


# imb_noisify
def noisify_imb(y_train, noise, random_state=None, nb_classes=10, imb_type='step', imb_rate=0.1):
    # init the matrix
    n = noise
    P = solver(n, imb_type, imb_rate)
    y_train_noisy = multiclass_noisify(y_train, P=P,
                    random_state=random_state)
    
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0, select_class=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate, P = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate, P = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'sparse':
        train_noisy_labels, actual_noise_rate, P = noisify_sparse(train_labels, noise_rate, random_state=0, nb_classes=nb_classes, select_class=select_class)
    if 'imb' in noise_type:
        imb_type = noise_type.split('_')[1]
        imb_rate = float(noise_type.split('_')[2])
        train_noisy_labels, actual_noise_rate, P = noisify_imb(train_labels, noise_rate, random_state=0, nb_classes=nb_classes, \
            imb_type=imb_type, imb_rate=imb_rate)

    return train_noisy_labels, actual_noise_rate, P



