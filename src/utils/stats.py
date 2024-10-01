
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

def bootstrap(yprobs, ytrue, runs=10000):
    """Runs bootstrapping to plot the 95% confidence interval

    Args:
        sample_a (list): samples belonging to group A
        sample_b (list): samples belonging to group B
        runs (int, optional): Number of bootstrapping. Defaults to 10000.
    """
    
    sample_indices = [i for i in range(len(yprobs))]
    scores = []
    for _ in (range(runs)):
        boot_sample = np.random.choice(sample_indices, size=len(sample_indices), replace=True)
        boot_true = [ytrue[bs] for bs in boot_sample]
        boot_prob = [yprobs[bs] for bs in boot_sample]
        scores.append(roc_auc_score(boot_true, np.array(boot_prob)[:, 1]))
    scores.sort()
    lower = int(runs * 0.025)
    upper = int(runs * 0.975)
    return np.mean(scores), scores[lower], scores[upper] 

def fischer_mean(sample_a, sample_b, runs=10000):
    """Runs the 'mean'-fisherian random invariance test.

    Computes the difference of means between group A and group B.
    Then randomly shuffles who belongs to A and who belongs to B according to 
    these criteria during the bootstrap run:
    - the number of samples attributed to A/B need to be equal to the number of
    samples in A/B
    - all samples need to be attributed to either A or B, but not both of them, 
    and not neither of them

    Args:
        sample_a (list): samples belonging to group A
        sample_b (list): samples belonging to group B
        runs (int, optional): Number of bootstrapping. Defaults to 10000.
    """
    na = len(sample_a)
    nb = len(sample_b)
    n = na + nb
    all_samples = [*sample_a, *sample_b]
    original_mean_a = np.mean(sample_a)
    original_mean_b = np.mean(sample_b)
    original_mean_diff = np.abs(original_mean_a - original_mean_b)

    sup_means = 0
    for _ in (range(runs)):
        np.random.shuffle(all_samples)
        bootstrap_a = all_samples[:na]
        bootstrap_b = all_samples[na:]
        assert len(bootstrap_a) == na and len(bootstrap_b) == nb
        bootrstrap_mean_diff = np.abs(np.mean(bootstrap_a) - np.mean(bootstrap_b))
        if bootrstrap_mean_diff >= original_mean_diff:
            sup_means += 1
    fischer = sup_means / runs
    return fischer

def fischer_var(sample_a, sample_b, runs=10000):
    """Runs the 'variance'-fisherian random invariance test.

    Computes the difference of variance between group A and group B.
    Then randomly shuffles who belongs to A and who belongs to B according to 
    these criteria during the bootstrap run:
    - the number of samples attributed to A/B need to be equal to the number of
    samples in A/B
    - all samples need to be attributed to either A or B, but not both of them, 
    and not neither of them

    Args:
        sample_a (list): samples belonging to group A
        sample_b (list): samples belonging to group B
        runs (int, optional): Number of bootstrapping. Defaults to 10000.
    """
    na = len(sample_a)
    nb = len(sample_b)
    n = na + nb
    all_samples = [*sample_a, *sample_b]
    original_mean_a = np.std(sample_a)
    original_mean_b = np.std(sample_b)
    original_mean_diff = np.abs(original_mean_a - original_mean_b)

    sup_means = 0
    for _ in (range(runs)):
        np.random.shuffle(all_samples)
        bootstrap_a = all_samples[:na]
        bootstrap_b = all_samples[na:]
        assert len(bootstrap_a) == na and len(bootstrap_b) == nb
        bootrstrap_mean_diff = np.abs(np.std(bootstrap_a) - np.std(bootstrap_b))
        if bootrstrap_mean_diff >= original_mean_diff:
            sup_means += 1
    fischer = sup_means / runs
    return fischer

