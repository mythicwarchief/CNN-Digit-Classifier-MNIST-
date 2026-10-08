import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("mnist_cnn.keras")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_norm = x_test.reshape(-1, 28, 28, 1) / 255.0

probs = model.predict(x_test_norm, verbose=0)
preds = np.argmax(probs, axis=1)
conf = np.max(probs, axis=1) * 100

# Show 12 correct and 4 wrong predictions, so the misses are visible
wrong = np.where(preds != y_test)[0][:4]
right = np.where(preds == y_test)[0][:12]
idx = np.concatenate([right, wrong])

fig, axes = plt.subplots(4, 4, figsize=(8, 9))
for ax, i in zip(axes.flat, idx):
    ax.imshow(x_test[i], cmap="gray")
    correct = preds[i] == y_test[i]
    ax.set_title(
        f"Actual {y_test[i]}, Predicted {preds[i]}\n{conf[i]:.1f}%",
        color="green" if correct else "red",
        fontsize=10,
    )
    ax.axis("off")

plt.suptitle("CNN predictions on MNIST test set (red = wrong)", fontsize=12)
plt.tight_layout()
plt.savefig("predictions_grid.png", dpi=150)
plt.show()
print(f"Overall test accuracy: {(preds == y_test).mean() * 100:.2f}%")