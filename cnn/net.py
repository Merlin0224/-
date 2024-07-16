import pandas as pd
import matplotlib.pyplot as plt
from class_ import Conv_nxn, MaxPool2, FlattenLayer, DropoutLayer, read_matrices_from_csv, Softmax, \
    Conv_nxn_, FullyConnectedLayer, _ReLU_, BatchNormalizationLayer
import numpy as np


# 定义AlexNet网络结构
def relu(x):
    """
    ReLU（修正线性单元）激活函数

    参数：
    x：输入张量

    返回值：
    relu(x)：ReLU 激活后的张量
    """
    return np.maximum(0, x)


data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_test_ = read_matrices_from_csv(data_test)
X_train = data.drop('label', axis=1)[:10000]
y_train = data['label'][:10000]
X_test = data.drop('label', axis=1)[11000:12000]
y_test = data['label'][11000:12000]
X_train = read_matrices_from_csv(X_train)
X_test = read_matrices_from_csv(X_test)
num_filters = 8
depth = 8
conv = Conv_nxn(num_filters, 3, 1)
conv_ = Conv_nxn(num_filters, 3, 1)
Relu1 = _ReLU_()
Relu1_ = _ReLU_()
Relu2 = _ReLU_()
Relu2_ = _ReLU_()
Relu3 = _ReLU_()
Relu4 = _ReLU_()
Relu5 = _ReLU_()
Relu6 = _ReLU_()
BN = BatchNormalizationLayer(128)
pool1 = MaxPool2()
pool1_ = MaxPool2()
pool2 = MaxPool2()
pool2_ = MaxPool2()
conv2 = Conv_nxn_(num_filters, 3, 1)
conv2_ = Conv_nxn_(num_filters, 3, 1)
conv3 = Conv_nxn_(num_filters, 3, 1)
conv4 = Conv_nxn_(num_filters, 3, 1)
conv5 = Conv_nxn_(num_filters, 3, 1)
flatten = FlattenLayer()
dropout = DropoutLayer(0.15)
fc = FullyConnectedLayer(7 * 7 * num_filters, 128, BN, Relu6.forward)
softmax = Softmax(128, 10)
softmax_ = Softmax(128, 10)


def forward(image, label):
    """
    完成CNN的前向传播，并计算精度与交叉熵损失。
    - image是一个2维numpy数组
    - label是一个数字
    """
    # image: [0, 255] -> [-0.5, 0.5]
    out = conv.forward((image / 255) - 0.5, 1)
    out = Relu1.forward(out)
    out = pool1.forward(out)
    out = conv2.forward(out)
    out = Relu2.forward(out)
    out = pool2.forward(out)
    out = flatten.forward(out)
    out = fc.forward(out)
    out = BN.forward(out)
    out = Relu6.forward(out)
    out = softmax.forward(out)
    # 计算交叉熵损失函数和准确率.
    # 损失函数的计算只与 label 的数有关，相当于索引
    loss = -np.log(out[label])
    # 如果 softmax 输出的最大值就是 label 的值，表示正确，否则错误
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def backward(out, learning_rate):
    # 计算初始梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    gradient = softmax.backward(gradient, learning_rate)
    gradient = Relu6.backward(gradient)
    gradient = BN.backward(gradient, learning_rate)
    gradient = fc.backward(gradient, learning_rate)
    gradient = flatten.backward(gradient)
    gradient = pool2.backward(gradient)
    gradient = Relu2.backward(gradient)
    gradient = conv2.backward(gradient, learning_rate)
    gradient = pool1.backward(gradient)
    gradient = Relu1.backward(gradient)
    gradient = conv.backward(gradient, learning_rate)

    return None


def train(im, label, learning_rate, flag=True):
    # Forward pass
    out, loss, acc = forward(im, label)

    # Backward pass
    backward(out, learning_rate)
    return loss, acc


print('digit CNN initialized!')

loss = 0
num_correct = 0

# 记录每个样本的损失值和准确率
losses = []
accuracies = []
losses_1 = []
accuracies_1 = []

# 训练CNN: epoch = 2
for epoch in range(2):
    print('--- Epoch %d ---' % (epoch + 1))

    # Train!
    loss = 0
    num_correct = 0
    loss_1 = 0
    num_correct_1 = 0
    # i: index
    # im: image
    # label: label
    for i, (im, label) in enumerate(zip(X_train, y_train)):
        l, acc = train(im, label, 0.05)
        loss += l
        num_correct += acc

        if i > 0 and i % 100 == 99:
            avg_loss = loss / 100
            accuracy = num_correct / 100

            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )

            losses.append(avg_loss)
            accuracies.append(accuracy)
            loss = 0
            num_correct = 0

    # Test the CNN
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(X_test, y_test):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(X_test)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)

    # Save the model
# 预测并保存到 sample_submission.csv 文件中
predictions = []
for im in data_test_:
    y_pred = forward(im, None)
    predictions.append(y_pred)

# 创建 DataFrame 并保存
submission = pd.DataFrame({'ImageId': np.arange(1, len(predictions) + 1), 'Label': predictions})

# 尝试不同的文件名来避免权限问题
submission_file_path = 'sample_submission.csv'
try:
    submission.to_csv(submission_file_path, index=False)
    print(f"Predictions saved to {submission_file_path}")
except PermissionError as e:
    print(f"PermissionError: {e}. Trying a different file name.")
    submission_file_path = 'sample_submission_new.csv'
    submission.to_csv(submission_file_path, index=False)
    print(f"Predictions saved to {submission_file_path}")

# 绘制损失值和准确率变化图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(losses, label='Loss with BN', linestyle='-')
ax1.set_title('Loss over time')
ax1.set_xlabel('Training step')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(accuracies, label='Accuracy with BN', linestyle='-')
ax2.set_title('Accuracy over time')
ax2.set_xlabel('Training step')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.savefig('Ac_2.png')
plt.tight_layout()
plt.show()
