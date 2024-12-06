# MNIST Classification with PyTorch
![Build pass](https://github.com/divyanipatil/UltraCompactMNIST/actions/workflows/model_validation.yml/badge.svg)

A deep learning project implementing CNN for MNIST digit classification, optimized for parameter efficiency while maintaining high test accuracy.

## 🎯 Project Goals

- Implement a CNN model with less than 20,000 parameters
- Achieve classification test accuracy of ≥99.4% on MNIST dataset within 20 epochs

## 🛠️ Technologies Used

- Python 3.8
- PyTorch
- torchvision
- pytest for testing

## 📦 Installation

1. Clone the repository:
```bash
    git clone https://github.com/divyanipatil/UltraCompactMNIST.git
```

2. Install dependencies:
```bash
    pip install -r requirements.txt
```
## 🏗️ Project Structure

- `train.py` - CNN implementation and training logic
- `test_model.py` - Model testing and validation
- GitHub Actions workflow for automated testing

## 🚀 Models

### CompactCNN
- Convolutional Neural Network optimized for parameter efficiency (<20k)
- Features batch normalization, ReLU activation, DropOut and Fully connected layers.
- Uses SGD optimizer with OneCycleLR scheduler

## 💻 Usage

To train the CNN model:
```bash
    python train.py
```

To run tests:
```bash
    pytest test_model.py -v
```

## 🔍 Model Performance

The model is designed to meet the following criteria:
- Parameter count: < 20,000
- Uses DropOut, Batch normalization, Fully connected layers.
- Test Accuracy threshold: ≥99.4% within 20 epochs

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚙️ CI/CD

The project includes GitHub Actions workflow for automated testing:
- Runs on Ubuntu latest
- Uses Python 3.8
- Automatically runs tests on push to main and pull requests
- Validates few critical aspects:
     - Model parameter count (must be < 20,000)
     - Usage of BN, Drop-out, FC layers in the Model architecture.

## 📊 App Logs

Logs related to the test accuracy attainment of the model.

### Test Accuracy Attainment

- Parameters count: 13306
- Epoch 17 test accuracy: 99.40%
- Final Epoch 20 test accuracy: 99.44%

```shell
Parameter count breakdown:
features.0.weight: 144
features.0.bias: 16
features.1.weight: 16
features.1.bias: 16
features.5.weight: 4608
features.5.bias: 32
features.6.weight: 32
features.6.bias: 32
features.10.weight: 512
features.10.bias: 16
features.11.weight: 16
features.11.bias: 16
classifier.0.weight: 7840
classifier.0.bias: 10

Total trainable parameters: 13306

Batch: 0/469, Loss: 2.3385, Accuracy: 10.94%, LR: 0.028002
Batch: 100/469, Loss: 0.5534, Accuracy: 82.84%, LR: 0.047568
Batch: 200/469, Loss: 0.3724, Accuracy: 88.48%, LR: 0.099557
Batch: 300/469, Loss: 0.2986, Accuracy: 90.77%, LR: 0.168133
Batch: 400/469, Loss: 0.2599, Accuracy: 92.03%, LR: 0.232402
Epoch: 1/20
Test Accuracy: 98.49%

Batch: 0/469, Loss: 0.2387, Accuracy: 92.68%, LR: 0.263759
Batch: 100/469, Loss: 0.2160, Accuracy: 93.37%, LR: 0.279999
Batch: 200/469, Loss: 0.1990, Accuracy: 93.88%, LR: 0.279896
Batch: 300/469, Loss: 0.1860, Accuracy: 94.27%, LR: 0.279615
Batch: 400/469, Loss: 0.1756, Accuracy: 94.58%, LR: 0.279158
Epoch: 2/20
Test Accuracy: 98.40%

Batch: 0/469, Loss: 0.1698, Accuracy: 94.76%, LR: 0.278739
Batch: 100/469, Loss: 0.1620, Accuracy: 95.01%, LR: 0.277983
Batch: 200/469, Loss: 0.1546, Accuracy: 95.24%, LR: 0.277053
Batch: 300/469, Loss: 0.1492, Accuracy: 95.42%, LR: 0.275948
Batch: 400/469, Loss: 0.1440, Accuracy: 95.58%, LR: 0.274672
Epoch: 3/20
Test Accuracy: 98.79%

Batch: 0/469, Loss: 0.1406, Accuracy: 95.69%, LR: 0.273691
Batch: 100/469, Loss: 0.1362, Accuracy: 95.83%, LR: 0.272127
Batch: 200/469, Loss: 0.1319, Accuracy: 95.95%, LR: 0.270395
Batch: 300/469, Loss: 0.1287, Accuracy: 96.05%, LR: 0.268498
Batch: 400/469, Loss: 0.1255, Accuracy: 96.13%, LR: 0.266438
Epoch: 4/20
Test Accuracy: 98.66%

Batch: 0/469, Loss: 0.1237, Accuracy: 96.19%, LR: 0.264923
Batch: 100/469, Loss: 0.1206, Accuracy: 96.29%, LR: 0.262593
Batch: 200/469, Loss: 0.1182, Accuracy: 96.37%, LR: 0.260108
Batch: 300/469, Loss: 0.1159, Accuracy: 96.43%, LR: 0.257471
Batch: 400/469, Loss: 0.1140, Accuracy: 96.49%, LR: 0.254685
Epoch: 5/20
Test Accuracy: 99.05%

Batch: 0/469, Loss: 0.1124, Accuracy: 96.54%, LR: 0.252678
Batch: 100/469, Loss: 0.1103, Accuracy: 96.60%, LR: 0.249648
Batch: 200/469, Loss: 0.1089, Accuracy: 96.64%, LR: 0.246480
Batch: 300/469, Loss: 0.1072, Accuracy: 96.70%, LR: 0.243176
Batch: 400/469, Loss: 0.1054, Accuracy: 96.76%, LR: 0.239742
Epoch: 6/20
Test Accuracy: 98.85%

Batch: 0/469, Loss: 0.1044, Accuracy: 96.79%, LR: 0.237298
Batch: 100/469, Loss: 0.1028, Accuracy: 96.84%, LR: 0.233652
Batch: 200/469, Loss: 0.1013, Accuracy: 96.89%, LR: 0.229888
Batch: 300/469, Loss: 0.0999, Accuracy: 96.93%, LR: 0.226010
Batch: 400/469, Loss: 0.0987, Accuracy: 96.96%, LR: 0.222023
Epoch: 7/20
Test Accuracy: 98.82%

Batch: 0/469, Loss: 0.0977, Accuracy: 96.99%, LR: 0.219211
Batch: 100/469, Loss: 0.0968, Accuracy: 97.02%, LR: 0.215051
Batch: 200/469, Loss: 0.0957, Accuracy: 97.06%, LR: 0.210797
Batch: 300/469, Loss: 0.0945, Accuracy: 97.09%, LR: 0.206452
Batch: 400/469, Loss: 0.0934, Accuracy: 97.12%, LR: 0.202023
Epoch: 8/20
Test Accuracy: 98.87%

Batch: 0/469, Loss: 0.0927, Accuracy: 97.14%, LR: 0.198922
Batch: 100/469, Loss: 0.0919, Accuracy: 97.17%, LR: 0.194363
Batch: 200/469, Loss: 0.0909, Accuracy: 97.20%, LR: 0.189736
Batch: 300/469, Loss: 0.0900, Accuracy: 97.23%, LR: 0.185047
Batch: 400/469, Loss: 0.0892, Accuracy: 97.25%, LR: 0.180300
Epoch: 9/20
Test Accuracy: 99.10%

Batch: 0/469, Loss: 0.0885, Accuracy: 97.27%, LR: 0.176994
Batch: 100/469, Loss: 0.0877, Accuracy: 97.29%, LR: 0.172165
Batch: 200/469, Loss: 0.0869, Accuracy: 97.32%, LR: 0.167295
Batch: 300/469, Loss: 0.0860, Accuracy: 97.34%, LR: 0.162390
Batch: 400/469, Loss: 0.0853, Accuracy: 97.37%, LR: 0.157457
Epoch: 10/20
Test Accuracy: 99.16%

Batch: 0/469, Loss: 0.0847, Accuracy: 97.39%, LR: 0.154040
Batch: 100/469, Loss: 0.0839, Accuracy: 97.41%, LR: 0.149074
Batch: 200/469, Loss: 0.0832, Accuracy: 97.43%, LR: 0.144096
Batch: 300/469, Loss: 0.0825, Accuracy: 97.45%, LR: 0.139113
Batch: 400/469, Loss: 0.0819, Accuracy: 97.47%, LR: 0.134132
Epoch: 11/20
Test Accuracy: 99.23%

Batch: 0/469, Loss: 0.0813, Accuracy: 97.49%, LR: 0.130699
Batch: 100/469, Loss: 0.0807, Accuracy: 97.51%, LR: 0.125734
Batch: 200/469, Loss: 0.0800, Accuracy: 97.53%, LR: 0.120788
Batch: 300/469, Loss: 0.0794, Accuracy: 97.55%, LR: 0.115866
Batch: 400/469, Loss: 0.0788, Accuracy: 97.56%, LR: 0.110974
Epoch: 12/20
Test Accuracy: 99.17%

Batch: 0/469, Loss: 0.0784, Accuracy: 97.57%, LR: 0.107621
Batch: 100/469, Loss: 0.0777, Accuracy: 97.60%, LR: 0.102796
Batch: 200/469, Loss: 0.0770, Accuracy: 97.61%, LR: 0.098018
Batch: 300/469, Loss: 0.0764, Accuracy: 97.64%, LR: 0.093294
Batch: 400/469, Loss: 0.0759, Accuracy: 97.65%, LR: 0.088629
Epoch: 13/20
Test Accuracy: 99.24%

Batch: 0/469, Loss: 0.0755, Accuracy: 97.66%, LR: 0.085448
Batch: 100/469, Loss: 0.0749, Accuracy: 97.68%, LR: 0.080898
Batch: 200/469, Loss: 0.0743, Accuracy: 97.70%, LR: 0.076422
Batch: 300/469, Loss: 0.0737, Accuracy: 97.71%, LR: 0.072027
Batch: 400/469, Loss: 0.0733, Accuracy: 97.72%, LR: 0.067719
Epoch: 14/20
Test Accuracy: 99.31%

Batch: 0/469, Loss: 0.0730, Accuracy: 97.74%, LR: 0.064800
Batch: 100/469, Loss: 0.0725, Accuracy: 97.75%, LR: 0.060650
Batch: 200/469, Loss: 0.0720, Accuracy: 97.77%, LR: 0.056601
Batch: 300/469, Loss: 0.0715, Accuracy: 97.79%, LR: 0.052658
Batch: 400/469, Loss: 0.0710, Accuracy: 97.80%, LR: 0.048827
Epoch: 15/20
Test Accuracy: 99.34%

Batch: 0/469, Loss: 0.0706, Accuracy: 97.81%, LR: 0.046250
Batch: 100/469, Loss: 0.0702, Accuracy: 97.82%, LR: 0.042617
Batch: 200/469, Loss: 0.0696, Accuracy: 97.84%, LR: 0.039108
Batch: 300/469, Loss: 0.0691, Accuracy: 97.86%, LR: 0.035727
Batch: 400/469, Loss: 0.0686, Accuracy: 97.87%, LR: 0.032478
Epoch: 16/20
Test Accuracy: 99.39%

Batch: 0/469, Loss: 0.0683, Accuracy: 97.88%, LR: 0.030316
Batch: 100/469, Loss: 0.0678, Accuracy: 97.90%, LR: 0.027301
Batch: 200/469, Loss: 0.0674, Accuracy: 97.91%, LR: 0.024429
Batch: 300/469, Loss: 0.0669, Accuracy: 97.92%, LR: 0.021704
Batch: 400/469, Loss: 0.0665, Accuracy: 97.94%, LR: 0.019129
Epoch: 17/20
Test Accuracy: 99.40%

Batch: 0/469, Loss: 0.0662, Accuracy: 97.95%, LR: 0.017442
Batch: 100/469, Loss: 0.0657, Accuracy: 97.96%, LR: 0.015129
Batch: 200/469, Loss: 0.0653, Accuracy: 97.97%, LR: 0.012974
Batch: 300/469, Loss: 0.0648, Accuracy: 97.99%, LR: 0.010981
Batch: 400/469, Loss: 0.0643, Accuracy: 98.00%, LR: 0.009151
Epoch: 18/20
Test Accuracy: 99.41%

Batch: 0/469, Loss: 0.0641, Accuracy: 98.01%, LR: 0.007986
Batch: 100/469, Loss: 0.0637, Accuracy: 98.02%, LR: 0.006439
Batch: 200/469, Loss: 0.0632, Accuracy: 98.04%, LR: 0.005062
Batch: 300/469, Loss: 0.0628, Accuracy: 98.05%, LR: 0.003856
Batch: 400/469, Loss: 0.0624, Accuracy: 98.06%, LR: 0.002823
Epoch: 19/20
Test Accuracy: 99.44%

Batch: 0/469, Loss: 0.0622, Accuracy: 98.07%, LR: 0.002212
Batch: 100/469, Loss: 0.0618, Accuracy: 98.08%, LR: 0.001474
Batch: 200/469, Loss: 0.0615, Accuracy: 98.09%, LR: 0.000912
Batch: 300/469, Loss: 0.0611, Accuracy: 98.10%, LR: 0.000528
Batch: 400/469, Loss: 0.0607, Accuracy: 98.11%, LR: 0.000320
Epoch: 20/20
Test Accuracy: 99.44%

Model saved to models/mnist_model.pth
```