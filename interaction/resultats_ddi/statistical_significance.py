import numpy as np
from scipy.stats import ttest_ind

# Results from the table
results = {
    'Unseen DDI': {
        'ACC': {'SoTA': (97.3, 0.1), 'MFPS': (93.2, 0.0), 'GCN_T': (89.6, 0.0), 'MoLF': (92.8, 0.1)},
        'AUROC': {'SoTA': (99.2, 0.0), 'MFPS': (99.4, 0.0), 'GCN_T': (98.5, 0.0), 'MoLF': (99.3, 0.0)},
        'AUPR': {'SoTA': (97.9, 0.1), 'MFPS': (98.4, 0.0), 'GCN_T': (96.5, 0.0), 'MoLF': (98.2, 0.0)}
    },
    'Unseen 1 Drug': {
        'ACC': {'SoTA': (68.0, 0.7), 'MFPS': (68.6, 0.1), 'GCN_T': (68.6, 0.2), 'MoLF': (67.2, 0.2)},
        'AUROC': {'SoTA': (88.6, 0.7), 'MFPS': (86.3, 0.3), 'GCN_T': (86.0, 0.2), 'MoLF': (85.7, 0.1)},
        'AUPR': {'SoTA': (73.7, 1.1), 'MFPS': (73.1, 0.3), 'GCN_T': (72.1, 0.2), 'MoLF': (71.6, 0.3)}
    },
    'Unseen 2 Drugs': {
        'ACC': {'SoTA': (53.6, 2.2), 'MFPS': (52.9, 0.6), 'GCN_T': (52.9, 0.5), 'MoLF': (50.1, 0.7)},
        'AUROC': {'SoTA': (78.9, 1.5), 'MFPS': (70.9, 0.7), 'GCN_T': (72.4, 0.4), 'MoLF': (70.1, 0.6)},
        'AUPR': {'SoTA': (53.7, 1.9), 'MFPS': (51.5, 0.5), 'GCN_T': (51.8, 0.4), 'MoLF': (49.4, 0.6)}
    }
}

def simulate_values(mean, std, n=5):
    """Simulate data based on mean and standard deviation"""
    return np.random.normal(loc=mean, scale=std, size=n)

output_lines = []

for split, metrics in results.items():
    output_lines.append(f"\n=== {split} ===")
    for metric, vals in metrics.items():
        sota = simulate_values(*vals['SoTA'])
        for model in ['MFPS', 'GCN_T', 'MoLF']:
            comp = simulate_values(*vals[model])
            t_stat, p_val = ttest_ind(sota, comp)
            significance = '*' if p_val < 0.05 else ''
            output_lines.append(f"{metric} - SoTA vs {model}: t={t_stat:.3f}, p={p_val:.4e} {significance}")

output_path = "statistical_significance.txt"
with open(output_path, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"Statistical significance results saved to {output_path}")
