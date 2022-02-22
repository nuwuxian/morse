import numpy as np
import os
from scipy import stats
import pandas as pd

path = './best/real/reweight-start-50_dist-alignment-False_use-true-distribution-False_use-proto-False'
num_class = 12

#  Baseline Accuracy for malware-real 
base_acc = []
# add 0-3
class_0 = [0.81,0.87,0.83,0.86,0.91,0.84]
class_0 = np.array(class_0)
base_acc.append(class_0)
class_1 = [1.0,1.0,1.0,1.0,1.0,1.0]
class_1 = np.array(class_1)
base_acc.append(class_1)
class_2 = [0.99,0.98,0.99,0.99,0.99,0.98]
class_2 = np.array(class_2)
base_acc.append(class_2)
class_3 = [0.98,0.97,0.98,0.98,0.98,0.98]
class_3 = np.array(class_3)
base_acc.append(class_3)
# add 4-7
class_4 = [1.0,0.97,0.96,1.0,0.99,1.0]
class_4 = np.array(class_4)
base_acc.append(class_4)
class_5 = [1.0,1.0,1.0,1.0,1.0,1.0]
class_5 = np.array(class_5)
base_acc.append(class_5)
class_6 = [0.97,0.97,0.96,0.96,0.96,0.97]
class_6 = np.array(class_6)
base_acc.append(class_6)
class_7 = [1.0,1.0,0.99,1.0,1.0,1.0]
class_7 = np.array(class_7)
base_acc.append(class_7)
# add 8-11
class_8 = [1.0,0.98,0.96,0.98,0.95,0.98]
class_8 = np.array(class_8)
base_acc.append(class_8)
class_9 = [1.0,1.0,1.0,1.0,1.0,1.0]
class_9 = np.array(class_9)
base_acc.append(class_9)
class_10 = [0.99,0.98,0.98,0.99,0.98,0.98]
class_10 = np.array(class_10)
base_acc.append(class_10)
class_11 = [0.46,0.41,0.51,0.4,0.46,0.42]
class_11 = np.array(class_11)
base_acc.append(class_11)

def diff_p_value(diff):
    ref = np.zeros_like(diff)
    t, p = stats.ttest_rel(diff, ref, alternative='greater')
    return p

avg_noise = []
avg_noise_acc = []

for fold in os.listdir(path):
    test_acc = np.load(path + '/' + fold + '/best_results.npz')['test_acc']
    class_acc = np.load(path + '/' +fold + '/best_results.npz')['test_class_acc']
    avg_noise.append(test_acc)
    avg_noise_acc.append(class_acc)
avg_acc = np.array(avg_noise)
# Format print
print('Mean Accuracy is %.2f Std is %.2f' %(np.mean(avg_acc), np.std(avg_acc)))
class_acc = np.vstack(avg_noise_acc)
print(class_acc)

for i in range(num_class):
    p_value = 0.0
    #diff_class_i = class_acc[:, i] - base_acc[i]
    #p_value = diff_p_value(diff_class_i)
    print('Class' + str(i) + ' Mean Accuracy is %.2f, Std is %.2f, P is %.2f' %(np.mean(class_acc[:, i]), np.std(class_acc[:, i]), p_value))
