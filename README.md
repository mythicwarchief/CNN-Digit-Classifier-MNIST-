# CNN Digit Classifier (MNIST)

A convolutional neural network that recognises handwritten digits from the MNIST dataset, built with TensorFlow and Keras. The project includes a command-line app for checking individual predictions and scripts for visualising results.

## Results

- **Test accuracy:** 98.55% on the 10,000 MNIST test images
- **Architecture:** one convolutional layer (32 filters, 3×3), max pooling (2×2), a dense layer with 128 units, and a 10-unit softmax output
- **Training:** 5 epochs, Adam optimiser, 10% of the training data held out for validation

## Project files

| File | Purpose |
|---|---|
| `train.py` | Trains the model, prints test accuracy, and saves it to `mnist_cnn.keras` |
| `app.py` | Loads the saved model and predicts a chosen test image (0-9999) |
| `plot_predictions.py` | Shows a grid of correct and incorrect predictions and saves `predictions_grid.png` |
| `mnist_cnn.keras` | Trained model (created by `train.py`) |
| `training_curve.png` | Training and validation accuracy chart (created by `train.py`) |
| `requirements.txt` | Python packages needed to run the project |

## Setup

Requires Python 3.10 or later.

```
pip install -r requirements.txt
```

## Usage

**1. Train the model** (optional, since `mnist_cnn.keras` is included):

```
python train.py
```

This downloads MNIST the first time it runs, trains for 5 epochs, and saves the model.

**2. Predict individual digits:**

```
python app.py
```

Enter a number from 0 to 9999 to see the actual digit, the prediction, and the confidence. Enter `q` to quit.

**3. View a grid of predictions:**

```
python plot_predictions.py
```

This shows 12 correct and 4 incorrect predictions and saves them as `predictions_grid.png`.

## Notes

- The model runs on the CPU. A GPU isn't required.
- Training takes roughly 9 seconds per epoch on a typical laptop CPU.
- Running `train.py` again overwrites `mnist_cnn.keras`, so the accuracy may differ slightly from the figure above.

## Author

**Vaishnav Sunil Nair**
Second Year B.Tech, Computer Science and Engineering (Artificial Intelligence)

## Licence

Released for learning purposes. MNIST is provided by Yann LeCun, Corinna Cortes and Christopher J.C. Burges.