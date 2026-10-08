import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mnist_cnn.keras")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test / 255.0

print("\nCNN Digit Classifier")
print("--------------------")
print("MNIST test images: 0 - 9999")

while True:
    choice = input("\nEnter image number or 'q' to quit: ")

    if choice.lower() == "q":
        break

    if not choice.isdigit() or not 0 <= int(choice) <= 9999:
        print("Enter a number between 0 and 9999.")
        continue

    index = int(choice)

    image = x_test[index]
    actual = y_test[index]

    prediction = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )

    predicted = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"\nActual digit:    {actual}")
    print(f"Predicted digit: {predicted}")
    print(f"Confidence:      {confidence:.2f}%")