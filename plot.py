import numpy as np
import os
from scipy import stats

path = '/home/xkw5132/ccs_noise/tmp/syn/imb-method_logits_noise-type_imb_step_0.1_noise-rate_0.5_imb-type_step_imb-rate_0.1_reweight-start-50_dist-alignment-False_use-hard-labels_False_ratio_0.7'
base_path=path
#  Baseline Accuracy for malware-real 

# recording the baseline acc for each class
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

def diff_p_value(diff):
    ref = np.zeros_like(diff)
    t, p = stats.ttest_rel(diff, ref, alternative='greater')
    return p

avg_noise = []
avg_noise_acc = []
large_acc = []
small_acc = []

coteaching = False

for fold in os.listdir(path):
    if not coteaching:
        test_acc = np.load(path + '/' + fold + '/best_results.npz')['test_acc']
        class_acc = np.load(path + '/' +fold + '/best_results.npz')['test_class_acc']
        avg_noise.append(test_acc)
        avg_noise_acc.append(class_acc)
        large_acc.append(np.mean(class_acc[:5]))
        small_acc.append(np.mean(class_acc[5:]))
    else:
        test_acc1 = np.load(path + '/' + fold + '/best_results.npz')['test_acc1']
        class_acc1 = np.load(path + '/' +fold + '/best_results.npz')['test_class_acc1']
        test_acc2 = np.load(path + '/' + fold + '/best_results.npz')['test_acc2']
        class_acc2 = np.load(path + '/' +fold + '/best_results.npz')['test_class_acc2']
        if test_acc1 > test_acc2:
            avg_noise.append(test_acc1)
            avg_noise_acc.append(class_acc1)
            large_acc.append(np.mean(class_acc1[:5]))
            small_acc.append(np.mean(class_acc1[5:]))
        else:
            avg_noise.append(test_acc2)
            avg_noise_acc.append(class_acc2)
            large_acc.append(np.mean(class_acc2[:5]))
            small_acc.append(np.mean(class_acc2[5:]))

avg_acc = np.array(avg_noise)
large_acc = np.array(large_acc)
small_acc = np.array(small_acc)
# Format print
print('Mean Accuracy is %.4f Std is %.4f' %(np.mean(avg_acc), np.std(avg_acc)))
print('Large Accuracy is %.4f Std is %.4f' %(np.mean(large_acc), np.std(large_acc)))
print('Small Accuracy is %.4f Std is %.4f' %(np.mean(small_acc), np.std(small_acc)))
class_acc = np.vstack(avg_noise_acc)
print(class_acc)

diff_avg = avg_acc - base
diff_large = large_acc - base_large
diff_small = small_acc - base_small
print('P is %.3f P_large is %.3f P_small is %.3f' %(diff_p_value(diff_avg), diff_p_value(diff_large), diff_p_value(diff_small)))