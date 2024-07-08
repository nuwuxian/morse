import numpy as np
import os
from scipy import stats


def diff_p_value(diff):
    ref = np.zeros_like(diff)
    t, p = stats.ttest_rel(diff, ref, alternative='greater')
    return p

path = '/home/xkw5132/ccs_noise/tmp/syn/imb-method_reweight_noise-type_imb_step_0.05_noise-rate_0.6_imb-type_step_imb-rate_0.05_reweight-start-20_dist-alignment-False_use-hard-labels_False_ratio_0.95'

avg_noise, avg_noise_acc = [], []
large_acc, small_acc = [], []

for fold in os.listdir(path):
    test_acc = np.load(path + '/' + fold + '/best_results.npz')['test_acc']
    class_acc = np.load(path + '/' +fold + '/best_results.npz')['test_class_acc']
    avg_noise.append(test_acc)
    avg_noise_acc.append(class_acc)
    large_acc.append(np.mean(class_acc[:5]))
    small_acc.append(np.mean(class_acc[5:]))

avg_acc = np.array(avg_noise)
large_acc = np.array(large_acc)
small_acc = np.array(small_acc)
# Format print
print('Mean Accuracy is %.4f Std is %.4f' %(np.mean(avg_acc), np.std(avg_acc)))
print('Large Accuracy is %.4f Std is %.4f' %(np.mean(large_acc), np.std(large_acc)))
print('Small Accuracy is %.4f Std is %.4f' %(np.mean(small_acc), np.std(small_acc)))
class_acc = np.vstack(avg_noise_acc)
print(class_acc)
'''
# Calculate the p-value compared to the baseline results
base_path=None
base, base_large, base_small = [], [], []
for fold in os.listdir(base_path):
    test_acc = np.load(base_path + '/' + fold + '/best_results.npz')['test_acc']
    class_acc = np.load(base_path + '/' +fold + '/best_results.npz')['test_class_acc']
    base.append(test_acc)
    base_large.append(np.mean(class_acc[:5]))
    base_small.append(np.mean(class_acc[5:]))
    
base = np.array(base)
base_large = np.array(base_large)
base_small = np.array(base_small)

diff_avg = avg_acc - base
diff_large = large_acc - base_large
diff_small = small_acc - base_small
print('P is %.3f P_large is %.3f P_small is %.3f' %(diff_p_value(diff_avg), diff_p_value(diff_large), diff_p_value(diff_small)))
'''