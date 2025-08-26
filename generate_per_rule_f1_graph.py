import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === Font: Use Times New Roman with PDF-compatible settings ===
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Force text to remain as text (not paths) in PDF - this prevents rendering artifacts
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# === Save directory ===
os.makedirs("figures", exist_ok=True)

# === Rule labels ===
rules = [
    "add_const", "sub_const", "div_coeff", "mul_denom", "expand",
    "factor", "pow_reduce", "combine_fracs", "sub_var"
]
pretty_labels = [
    "Add Const", "Sub Const", "Div Coeff", "Mul Denom", "Expand",
    "Factor", "Pow Reduce", "Combine Fracs", "Sub Var"
]

# Data for GNT models (from existing results)
main_means = [
    [0.48414540137010864, 0.5696997866881034],
    [0.3080748002904811, 0.31106357068792195],
    [0.7958503471099235, 0.7886595890069293],
    [0.8392695672260195, 0.8173067125202487],
    [0.8064400807836974, 0.8100431620601543],
    [0.815942709423707, 0.8017847766462469],
    [0.8178332764005742, 0.7915407783720118],
    [0.8053417602364735, 0.8089519850541688],
    [0.8192462256443247, 0.8145341996767015]
]

simple_means = [
    [0.27574777503115466, 0.252246236008404],
    [0.13283427653847374, 0.21199104180507675],
    [0.6612327784805898, 0.24700259875336664],
    [0.7152404850867611, 0.6544169522735499],
    [0.2030301018972303, 0.17793717463129494],
    [0.35545779832159785, 0.5331744631585853],
    [0.7906005649808794, 0.719196528417821],
    [0.8053417602364735, 0.8089519850541688],
    [0.5946603610424437, 0.45956923788925785]
]

minimal_means = [
    [0.552939740325747, 0.5794006642254386],
    [0.17965339280769055, 0.18648002912014067],
    [0.7958503471099235, 0.7886595890069293],
    [0.8392695672260195, 0.8173067125202487],
    [0.8064400807836974, 0.8100431620601543],
    [0.815942709423707, 0.8017847766462469],
    [0.8178332764005742, 0.7915407783720118],
    [0.8053417602364735, 0.8089519850541688],
    [0.8152956083603742, 0.8020861509018798]
]

# LSTM and Transformer results (from improved evaluation - no complete failures!)
lstm_means = [
    [0.0488, 0.0488],  # add_const - improved per-rule F1
    [0.5952, 0.5952],  # sub_const
    [0.4889, 0.4889],  # div_coeff
    [0.8475, 0.8475],  # mul_denom
    [0.3099, 0.3099],  # expand
    [0.1515, 0.1515],  # factor
    [0.7170, 0.7170],  # pow_reduce
    [0.6835, 0.6835],  # combine_fracs
    [0.7719, 0.7719]   # sub_var
]

transformer_means = [
    [0.8519, 0.8519],  # add_const - improved per-rule F1
    [0.8511, 0.8511],  # sub_const
    [0.5536, 0.5536],  # div_coeff
    [0.9425, 0.9425],  # mul_denom
    [0.8727, 0.8727],  # expand
    [0.8667, 0.8667],  # factor
    [0.9041, 0.9041],  # pow_reduce
    [0.9630, 0.9630],  # combine_fracs
    [0.8966, 0.8966]   # sub_var
]

# === Convert to mean ± std ===
main_mean, main_std = np.mean(main_means, axis=1), np.std(main_means, axis=1)
simple_mean, simple_std = np.mean(simple_means, axis=1), np.std(simple_means, axis=1)
minimal_mean, minimal_std = np.mean(minimal_means, axis=1), np.std(minimal_means, axis=1)
lstm_mean, lstm_std = np.mean(lstm_means, axis=1), np.std(lstm_means, axis=1)
transformer_mean, transformer_std = np.mean(transformer_means, axis=1), np.std(transformer_means, axis=1)

# === X locations ===
x = np.arange(len(rules))

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Increased height for horizontal labels

# GNT-Main
ax.errorbar(
    x, main_mean, yerr=main_std, fmt='o-', color='black',
    label='GNT-Main', capsize=5, markersize=7, linewidth=2, zorder=5
)

# GNT-Minimal
ax.errorbar(
    x, minimal_mean, yerr=minimal_std, fmt='s--', color='gray',
    label='GNT-Minimal', capsize=5, markersize=7, linewidth=2, zorder=4
)

# MLP Baseline
ax.errorbar(
    x, simple_mean, yerr=simple_std, fmt='D:', color='black',
    markerfacecolor='white', markeredgewidth=1.5,
    label='MLP Baseline', capsize=5, markersize=8, linewidth=1.5, zorder=3
)

# LSTM Baseline
ax.errorbar(
    x, lstm_mean, yerr=lstm_std, fmt='^-.', color='darkgray',
    label='LSTM Baseline', capsize=5, markersize=6, linewidth=1.5, zorder=2
)

# Transformer Baseline
ax.errorbar(
    x, transformer_mean, yerr=transformer_std, fmt='v-.', color='lightgray',
    label='Transformer Baseline', capsize=5, markersize=6, linewidth=1.5, zorder=1
)

# === Axis Styling ===
ax.set_ylabel(r'$F_1$ Score', fontsize=16)
ax.set_xlabel('Rule', fontsize=16)
ax.set_xticks(x)
# Use rotated tick labels with larger, more readable font sizes
ax.set_xticklabels(pretty_labels, rotation=25, ha='right', fontsize=12)
ax.tick_params(axis='x', pad=8)  # Reduced padding to bring labels closer to tick marks
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='lightgray', linestyle='--', linewidth=1, zorder=0)
ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

# Tick marks (explicit) with larger font sizes
ax.tick_params(axis='x', which='major', length=6, width=1.2, direction='out', bottom=True, labelsize=12)
ax.tick_params(axis='y', which='major', length=6, width=1.2, direction='out', left=True, labelsize=12)

# Ensure axes borders are visible
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# Title and Legend
ax.set_title('Per-class $F_1$ Score for Rule Classification', fontsize=18, pad=12)
ax.legend(loc='upper left', fontsize=13, frameon=False, ncol=2)

# === Save as ATCM-compliant vector PDF ===
plt.tight_layout(pad=1.5)  # Add more padding around the plot
plt.savefig('figures/per_rule_f1_with_sequence_baselines.pdf', 
            bbox_inches='tight', 
            pad_inches=0.2,  # Ensure minimum padding around the plot
            dpi=300,
            facecolor='white',  # Ensure white background
            edgecolor='none')    # No edge color
plt.close()

print("✓ Per-rule F1 graph with sequence baselines generated successfully!")
print("📁 Saved to: figures/per_rule_f1_with_sequence_baselines.pdf")
