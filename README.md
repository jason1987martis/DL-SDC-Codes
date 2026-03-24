# Deep learning mini-project codes

This pack contains 10 separate Python files for the topics in your image.

## Recommended format: CLI or notebook?

For your use case, **Python CLI scripts are better as the main submission format**.

Why:
- easier to run again with different hyperparameters
- cleaner for faculty review
- simpler to keep each topic in a separate file
- better for version control and final project organization

Use **notebooks** when:
- you want to explain step by step in class
- you want plots, intermediate outputs, and experiments in one place
- you are still learning and debugging

Best practical choice:
- keep these `.py` files as the main working codes
- optionally convert 1 or 2 into notebooks later for demo/explanation

## File list

1. `01_mnist_mlp.py` - MLP for MNIST with activation/loss options  
2. `02_creditcard_autoencoder.py` - autoencoder for anomaly detection  
3. `03_fashionmnist_rbm.py` - RBM for unsupervised feature learning  
4. `04_cifar10_alexnet.py` - AlexNet-style CNN for CIFAR-10  
5. `05_stock_lstm_gru.py` - LSTM/GRU for stock prediction  
6. `06_celeba_basic_gan.py` - basic GAN for a CelebA subset  
7. `07_cartpole_q_learning.py` - Q-learning on CartPole  
8. `08_cnn_dropout_l2.py` - dropout + L2 regularization  
9. `09_adam_vs_sgd.py` - optimizer comparison  
10. `10_data_aug_early_stopping.py` - augmentation + early stopping  

## Notes

### Scripts that auto-download standard datasets
- MNIST
- FashionMNIST
- CIFAR-10
- CartPole environment

### Scripts that need your own dataset path
- `02_creditcard_autoencoder.py` -> `creditcard.csv`
- `05_stock_lstm_gru.py` -> CSV with `Close` column
- `06_celeba_basic_gan.py` -> image folder

## How to run

Create environment and install requirements:

```bash
pip install -r requirements.txt
```

Example runs:

```bash
python 01_mnist_mlp.py --epochs 5 --activation relu --loss cross_entropy
python 02_creditcard_autoencoder.py --csv_path creditcard.csv --epochs 20
python 03_fashionmnist_rbm.py --epochs 10
python 04_cifar10_alexnet.py --epochs 10
python 05_stock_lstm_gru.py --csv_path stock.csv --model lstm --epochs 20
python 06_celeba_basic_gan.py --data_dir celeba_subset --epochs 20
python 07_cartpole_q_learning.py --episodes 2000
python 08_cnn_dropout_l2.py --epochs 10 --dropout 0.5 --weight_decay 0.0001
python 09_adam_vs_sgd.py --epochs 5
python 10_data_aug_early_stopping.py --epochs 30 --patience 5
```

## Very short working explanation

### 1) MLP for MNIST
Flattens the image, passes it through dense layers, and predicts one of 10 digits.

### 2) Autoencoder for anomaly detection
Learns normal transactions. High reconstruction error suggests anomaly/fraud.

### 3) RBM on Fashion-MNIST
Learns hidden features without labels using contrastive divergence.

### 4) AlexNet-style CNN
Uses convolution layers to learn spatial image features and classify CIFAR-10 images.

### 5) LSTM/GRU for stock prediction
Uses previous closing prices to predict the next one.

### 6) Basic GAN
Generator creates fake images, discriminator tries to detect them, both improve through adversarial training.

### 7) Q-learning for CartPole
Learns a state-action value table and improves balancing over many episodes.

### 8) Dropout + L2
Applies two regularization techniques to reduce overfitting.

### 9) Adam vs SGD
Trains the same model with two optimizers and compares convergence.

### 10) Data augmentation + early stopping
Augments training images and stops training when validation stops improving.

## Suggestion

For viva/demo:
- run 1, 4, 7, 9 first
- explain 2, 5, 6 as dataset-dependent extensions
