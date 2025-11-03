import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")  # Hide axis

# --- Components (Boxes) ---

# Drug A Processing
ax.text(2, 9, "Drug A Embedding (emb. dim.)", fontsize=12, ha="center",
        bbox=dict(facecolor="lightblue", edgecolor="black"))
ax.text(2, 7.5, "Encoder", fontsize=12, ha="center",
        bbox=dict(facecolor="lightgray", edgecolor="black"))
ax.text(2, 6, "Encoded A (256)", fontsize=12, ha="center",
        bbox=dict(facecolor="lightgreen", edgecolor="black"))

# Drug B Processing
ax.text(8, 9, "Drug B Embedding (emb. dim.)", fontsize=12, ha="center",
        bbox=dict(facecolor="lightblue", edgecolor="black"))
ax.text(8, 7.5, "Encoder", fontsize=12, ha="center",
        bbox=dict(facecolor="lightgray", edgecolor="black"))
ax.text(8, 6, "Encoded B (256)", fontsize=12, ha="center",
        bbox=dict(facecolor="lightgreen", edgecolor="black"))

# Concatenation
ax.text(5, 5, "Concatenation (512)", fontsize=12, ha="center",
        bbox=dict(facecolor="orange", edgecolor="black"))

# Classifier Block
ax.text(5, 3.5, "Classifier", fontsize=12, ha="center",
        bbox=dict(facecolor="lightgray", edgecolor="black"))
ax.text(5, 2.5, "Linear (512 → 256)", fontsize=10, ha="center",
        bbox=dict(facecolor="lightyellow", edgecolor="black"))
ax.text(5, 1.8, "BatchNorm + ReLU", fontsize=10, ha="center",
        bbox=dict(facecolor="lightyellow", edgecolor="black"))
ax.text(5, 1.2, "Dropout", fontsize=10, ha="center",
        bbox=dict(facecolor="lightyellow", edgecolor="black"))
ax.text(5, 0.5, "Final Linear (256 → num classes)", fontsize=10, ha="center",
        bbox=dict(facecolor="lightyellow", edgecolor="black"))

# Prediction
ax.text(5, -0.5, "Prediction (num classes)", fontsize=12, ha="center",
        bbox=dict(facecolor="lightcoral", edgecolor="black"))

# --- Arrows (Connections) ---
def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Arrows for Drug A
draw_arrow(2, 8.7, 2, 8)  # Embedding → Encoder
draw_arrow(2, 7.2, 2, 6.5)  # Encoder → Encoded A

# Arrows for Drug B
draw_arrow(8, 8.7, 8, 8)  # Embedding → Encoder
draw_arrow(8, 7.2, 8, 6.5)  # Encoder → Encoded B

# Arrows to Concatenation
draw_arrow(2, 5.7, 5, 5.3)  # Encoded A → Concatenation
draw_arrow(8, 5.7, 5, 5.3)  # Encoded B → Concatenation

# Arrows for Classifier
draw_arrow(5, 4.8, 5, 4.2)  # Concatenation → Classifier
draw_arrow(5, 3.3, 5, 2.9)  # Classifier → Linear 512 → 256
draw_arrow(5, 2.3, 5, 2.1)  # Linear → BatchNorm + ReLU
draw_arrow(5, 1.6, 5, 1.4)  # BatchNorm + ReLU → Dropout
draw_arrow(5, 0.9, 5, 0.7)  # Dropout → Final Linear
draw_arrow(5, 0.3, 5, -0.2)  # Final Linear → Prediction

# Save figure
output_path = "model_schematic.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"✅ Model schematic saved as: {output_path}")
