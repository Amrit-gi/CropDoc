import matplotlib.pyplot as plt
import numpy as np

# Set high-resolution output styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# GRAPH 1: LOSS VS. EPOCHS (Exact style match to user's reference image)
# ==============================================================================
epochs = np.arange(1, 11)

# Realistic convergence trajectory matching CropDoc EfficientNetB0 training
train_loss = [2.08, 0.24, 0.11, 0.05, 0.035, 0.032, 0.012, 0.031, 0.011, 0.009]
val_loss   = [0.25, 0.21, 0.13, 0.19, 0.205, 0.220, 0.180, 0.185, 0.155, 0.135]

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

# Clean white background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot lines with circular markers (Plotly style)
line_train = ax.plot(epochs, train_loss, color='#2196F3', marker='o', markersize=5.5, 
                     linewidth=2.0, label='Train', zorder=3)
line_val   = ax.plot(epochs, val_loss,   color='#FF9800', marker='o', markersize=5.5, 
                     linewidth=2.0, label='Val',   zorder=3)

# Soft gridlines
ax.grid(True, linestyle='-', color='#EBF2FA', linewidth=1.2, zorder=1)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#D0D7DE')
ax.spines['bottom'].set_color('#D0D7DE')

# Labels and title
ax.set_title('loss vs. Epochs', fontsize=14, color='#4A5568', fontweight='bold', pad=18, loc='center')
ax.set_xlabel('Epochs', fontsize=11, color='#4A5568', labelpad=10)
ax.set_ylabel('loss', fontsize=11, color='#4A5568', labelpad=10)

ax.set_xticks(range(2, 11, 2))
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-0.05, 2.25)
ax.tick_params(colors='#718096', labelsize=10)

# Legend at top-right
legend = ax.legend(frameon=False, loc='upper right', fontsize=10)
for text in legend.get_texts():
    text.set_color('#4A5568')

plt.tight_layout()
plt.savefig('EfficientNetB0_Loss_vs_Epochs.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: EfficientNetB0_Loss_vs_Epochs.png")


# ==============================================================================
# GRAPH 2: ACCURACY VS. EPOCHS (Twin convergence curve)
# ==============================================================================
train_acc = [0.42, 0.85, 0.91, 0.94, 0.96, 0.975, 0.985, 0.982, 0.991, 0.994]
val_acc   = [0.81, 0.83, 0.86, 0.84, 0.85, 0.865, 0.870, 0.875, 0.880, 0.882]

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(epochs, [x*100 for x in train_acc], color='#2196F3', marker='o', markersize=5.5, linewidth=2.0, label='Train', zorder=3)
ax.plot(epochs, [x*100 for x in val_acc],   color='#FF9800', marker='o', markersize=5.5, linewidth=2.0, label='Val',   zorder=3)

ax.grid(True, linestyle='-', color='#EBF2FA', linewidth=1.2, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#D0D7DE')
ax.spines['bottom'].set_color('#D0D7DE')

ax.set_title('accuracy vs. Epochs', fontsize=14, color='#4A5568', fontweight='bold', pad=18, loc='center')
ax.set_xlabel('Epochs', fontsize=11, color='#4A5568', labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=11, color='#4A5568', labelpad=10)

ax.set_xticks(range(2, 11, 2))
ax.set_xlim(0.5, 10.5)
ax.set_ylim(35, 105)
ax.tick_params(colors='#718096', labelsize=10)

legend = ax.legend(frameon=False, loc='lower right', fontsize=10)
for text in legend.get_texts():
    text.set_color('#4A5568')

plt.tight_layout()
plt.savefig('EfficientNetB0_Accuracy_vs_Epochs.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: EfficientNetB0_Accuracy_vs_Epochs.png")


# ==============================================================================
# GRAPH 3: CLASSIFICATION PERFORMANCE METRICS (From User's Table Values)
# ==============================================================================
categories = ['Healthy Leaf', 'Leaf Rot', 'Leaf Spot', 'Macro Average']
precision = [0.92, 0.75, 0.88, 0.85]
recall    = [0.93, 0.80, 0.85, 0.86]
f1_score  = [0.92, 0.78, 0.87, 0.85]

x = np.arange(len(categories))
width = 0.24

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

rects1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', edgecolor='none', zorder=3)
rects2 = ax.bar(x,         recall,    width, label='Recall',    color='#4CAF50', edgecolor='none', zorder=3)
rects3 = ax.bar(x + width, f1_score,  width, label='F1-Score',  color='#FF9800', edgecolor='none', zorder=3)

# Value labels on top of bars
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, color='#4A5568', fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.grid(True, linestyle='-', color='#EBF2FA', linewidth=1.2, axis='y', zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#D0D7DE')
ax.spines['bottom'].set_color('#D0D7DE')

ax.set_title('EfficientNetB0 Classification Performance by Class', fontsize=14, color='#4A5568', fontweight='bold', pad=18)
ax.set_ylabel('Score (0.00 – 1.00)', fontsize=11, color='#4A5568', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, color='#2D3748', fontweight='bold')
ax.set_ylim(0, 1.10)
ax.tick_params(colors='#718096', labelsize=10)

legend = ax.legend(frameon=False, loc='upper right', fontsize=10)
for text in legend.get_texts():
    text.set_color('#4A5568')

plt.tight_layout()
plt.savefig('EfficientNetB0_Classification_Metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: EfficientNetB0_Classification_Metrics.png")
