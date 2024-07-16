import numpy as np


def read_matrices_from_csv(x):
    matrices = []  # 创建一个空列表来存储矩阵
    for index, row in x.iterrows():
        # 将每一行转换为NumPy数组，并重塑为28x28的矩阵
        matrix = np.array(row).reshape(28, 28)
        matrices.append(matrix)
    return matrices



class Conv_nxn:  # 卷积层 基于mxm的卷积核
    def __init__(self, num_filters, m=3, padding=0):
        self.last_input = None
        self.num_filters = num_filters
        self.filters = np.random.rand(num_filters, m, m) / 9  # 初始化卷积层 使用 mxm 的filter
        self.padding = padding
        self.m = m

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.m + 1):
            for j in range(w - self.m + 1):
                im_region = image[i:(i + self.m), j:(j + self.m)]
                yield im_region, i, j
        # 将 im_region, i, j 以 tuple 形式存储到迭代器中
        # 以便后面遍历使用

    def forward(self, input, padding=0):  # 卷积层前向传播，输入规模较小，因此stride为1
        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0
        # input: 28x28
        # output: 28 x 28 x num_filters
        h, w = input.shape
        self.last_input = input
        self.last_input.shape = input.shape
        padded_input = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant',
                              constant_values=0)
        output_h, output_w = padded_input.shape
        output = np.zeros((output_h - self.m + 1, output_w - self.m + 1, self.num_filters))

        for im_region, i, j in self.iterate_regions(padded_input):
            # 卷积运算，点乘再相加，output[i, j] 为向量，num_filters 层
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        # 最后将输出数据返回，便于下一层的输入使用
        return output

    def backward(self, d_L_d_out, learning_rate):
        # 执行卷积层的反向传播
        # d_L_d_out: 梯度，28x28x8
        # 计算梯度
        d_L_d_filters = np.zeros(self.filters.shape)
        padding_input = np.pad(self.last_input, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant',
                              constant_values=0)
        # 每个im_region为 3x3 小矩阵
        for im_region, i, j in self.iterate_regions(padding_input):
            for f in range(self.num_filters):
                # 按 f 分层计算，一次算一层，然后累加起来
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        # Update filters
        self.filters -= learning_rate * d_L_d_filters


class Conv_nxn_:
    def __init__(self, num_filters, m=3, padding=0):
        self.last_input = None
        self.num_filters = num_filters
        self.m = m
        self.filters = np.random.rand(num_filters, m, m) / 9  # 初始化卷积层 使用 m x m 的 filter
        self.padding = padding  # 添加 padding 参数

    def iterate_regions(self, image, f):
        h, w, _ = image.shape
        for i in range(h - self.m + 1):
            for j in range(w - self.m + 1):
                # 裁剪填充后的区域以保持大小为 m x m
                im_region = image[i:(i + self.m), j:(j + self.m), f]
                yield im_region, i, j

    def forward(self, input):  # 卷积层前向传播，输入规模较小，因此 stride 为 1
        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0
        # input: 三维张量
        # output: x,y与input相同, z为num_filters的三维张量
        h, w, _ = input.shape
        self.last_input = input
        self.last_input.shape = input.shape
        padded_input = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                              mode='constant',
                              constant_values=0)
        output_h, output_w, output_ = padded_input.shape
        output = np.zeros((output_h - self.m + 1, output_w - self.m + 1, self.num_filters))
        for f in range(_):
            for im_region, i, j in self.iterate_regions(padded_input, f):
                # 卷积运算，点乘再相加，output[i, j] 为向量，num_filters 层
                # 这里使用 im_region * self.filters[k] 是因为 im_region 是一个三维数组，
                output[i, j, f] = np.sum(im_region * self.filters[f])

        # 最后将输出数据返回，便于下一层的输入使用
        return output

    def backward(self, d_L_d_out, learning_rate):
        # 执行卷积层的反向传播
        # d_L_d_out: 梯度，三维张量
        # 计算梯度
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_input = np.zeros(self.last_input.shape)
        h, w, _ = self.last_input.shape
        # 这里需要对 d_L_d_out 进行 padding，以便与 im_region 进行卷积操作
        d_L_d_out_reshaped = np.pad(d_L_d_out, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                                    mode='constant',
                                    constant_values=0)

        input_reshaped = np.pad(self.last_input, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                                    mode='constant',
                                    constant_values=0)

        # 每个im_region为 m x m x depth 三维张量
        for f in range(self.num_filters):
            for im_region, i, j in self.iterate_regions(input_reshaped, f):
                # 按 f 分层计算，一次算一层，然后累加起来
                # d_L_d_filters[f]: m x m x depth 三维张量
                # d_L_d_out[i, j, f]: num
                # im_region: m x m x depth 三维张量
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        filter_rot_180 = np.rot90(self.filters, 2, (1, 2))
        for g in range(_):
            for im_region, i, j in self.iterate_regions(d_L_d_out_reshaped, g):
                # 按 f 分层计算，一次算一层
                # d_L_d_filters: m x m x depth 三维张量
                # d_L_d_out[i, j, f]: num
                # im_region: m x m x 1 三维张量  in d_L_d_out
                d_L_d_input[i, j, g] = np.sum(filter_rot_180[g] * im_region)
        # Update filters
        self.filters -= learning_rate * d_L_d_filters
        return d_L_d_input


class MaxPool2:  # 池化层 2x2 的 Max Pooling

    def __init__(self):
        self.last_input = None

    def iterate_regions(self, image):
        # image: 三维张量
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        # input: 卷积层的输出，池化层的输入
        h, w, num_filters = input.shape
        self.last_input = input
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backward(self, d_L_d_out):
        # 池化层没有参数
        d_L_d_input = np.zeros(self.last_input.shape)
        # 修改 max 的部分，首先查找 max
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, _ = im_region.shape
            # 获取 im_region 里面最大值的索引向量，一叠的感觉
            amax = np.amax(im_region, axis=(0, 1))

            # 遍历整个 im_region，对于传递下去的像素点，修改 gradient 为 loss 对 output 的gradient
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(_):
                        # 如果pixel是最大值，直接复制梯度到d_L_d_input
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
                        # 如果pixel不是最大值，梯度为 0
        return d_L_d_input


class Softmax:
    # 实现了一个标准的全连接层（fully-connected layer）并使用了softmax激活函数。

    def __init__(self, input_len, nodes=10):
        # input_len: 输入层的节点个数
        # nodes: 输出层的节点个数，本例中为 10
        # 构建权重矩阵，初始化随机数
        self.last_totals = None
        self.last_input = None
        self.last_input_shape = None
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        """
        执行softmax层的前向传播。
        返回一个1维numpy数组，其中包含给定输入的相应概率值。
        - input可以是任何具有任何维度的数组。
        """
        input_len, nodes = self.weights.shape

        # input: 128
        self.last_input = input
        self.last_input.shape = input.shape
        # self.weights: (input_len, nodes)
        # 以上叉乘之后为 向量，input_len个节点与对应的权重相乘再加上bias得到输出的节点
        # totals: 向量, nodes
        totals = np.dot(input, self.weights) + self.biases
        # exp: 向量, nodes
        exp = np.exp(totals)
        self.last_totals = totals
        return exp / np.sum(exp, axis=0)

    def backward(self, d_L_d_outputs, learning_rate):
        """
        执行softmax层的反向传播。
        返回一个1维numpy数组，其中包含给定输入的相应误差梯度。
        - d_L_d_outputs是上一层传回的误差梯度，形状为 (batch_size, output_size)
        """
        # 计算误差梯度
        # d_L_d_outputs: nodes
        # 仅有一个值非0
        global d_L_d_inputs
        for i, gradient in enumerate(d_L_d_outputs):
            # 找到 label 的值，就是 gradient 不为 0 的
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # e^totals的和
            S = np.sum(t_exp)

            # out[i] 对 totals的梯度
            # 初始化都设置为 非 c 的值，再单独修改 c 的值
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            # totals 对 weights/biases/input  的梯度
            # d_t_d_w 的结果是 softmax 层的输入数据，input_len个元素的向量
            d_t_d_w = self.last_input
            d_t_d_b = 1
            # d_t_d_input 的结果是 weights 值
            d_t_d_inputs = self.weights

            # loss 对 totals 的梯度
            # 向量，nodes
            d_L_d_t = gradient * d_out_d_t

            # loss 对 weights/biases/input 的梯度
            d_L_d_weights = np.dot(self.last_input[np.newaxis].T, d_L_d_t[np.newaxis])
            d_L_d_biases = d_L_d_t * d_t_d_b
            d_L_d_inputs = np.dot(d_L_d_t, self.weights.T)

            # 更新权重和偏差
            self.weights -= learning_rate * d_L_d_weights
            self.biases -= learning_rate * d_L_d_biases
        return d_L_d_inputs


class FullyConnectedLayer:
    def __init__(self, input_size, output_size, batchnormal=None, activation_function=None):
        """
        初始化全连接层的权重和偏差，并指定激活函数

        参数：
        input_size：输入大小
        output_size：输出大小
        activation_function：激活函数，如 np.maximum

        """
        self.last_inputs = None
        self.outputs = None
        self.z = None
        self.inputs = None
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.batchnormal = batchnormal

        # 构建权重矩阵，初始化随机数
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)

    def forward(self, inputs):
        """
        前向传播：计算全连接层的输出

        参数：
        inputs：输入数据，形状为 (input_size)

        返回值：
        outputs：输出数据，形状为 (output_size)
        """
        self.inputs = inputs
        self.last_inputs = inputs
        input_size, output_size = self.weights.shape
        self.z = np.dot(inputs, self.weights) + self.biases
        self.outputs = self.z
        # if self.batchnormal is not None:
        #     self.z = self.batchnormal.forward(self.z)
        # if self.activation_function is not None:
        #     self.outputs = self.activation_function(self.z)
        # else:
        #     self.outputs = self.z
        return self.outputs

    def backward(self, d_L_d_outputs, learning_rate):
        """
        反向传播：计算梯度并更新权重和偏差

        参数：
        d_L_d_outputs：上一层传回的误差梯度，形状为 (output_size)
        learning_rate：学习率
        """
        # 计算权重的梯度
        # if self.batchnormal is not None:
        #     d_L_d_outputs = self.batchnormal.backward(d_L_d_outputs, learning_rate)
        d_L_d_weights = np.dot(self.last_inputs[np.newaxis].T, d_L_d_outputs[np.newaxis])
        # 计算偏差的梯度
        d_L_d_biases = d_L_d_outputs
        # 计算输入的梯度
        d_L_d_inputs = np.dot(d_L_d_outputs, self.weights.T)

        # 更新权重和偏差
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_inputs


class FlattenLayer:
    def __init__(self):
        self.last_input_shape = None

    def forward(self, input):
        """
        将输入张量展平为一维数组

        参数：
        input：输入张量，形状为 (height, width, channels)

        返回值：
        flattened_tensor：展平后的一维数组，形状为 (height * width * channels)
        """
        # 展平操作
        self.last_input_shape = input.shape
        flattened_tensor = input.flatten()
        return flattened_tensor

    def backward(self, d_L_d_output):
        """
        计算反向传播的梯度

        参数：
        d_L_d_output：上一层传回的误差梯度，形状为 (batch_size, output_size)

        返回值：
        d_L_d_input：输入的误差梯度，
        """
        d_L_d_input = d_L_d_output.reshape(self.last_input_shape)
        return d_L_d_input


class DropoutLayer:
    def __init__(self, dropout_prob):
        """
        初始化 Dropout 层

        参数：
        dropout_prob：丢弃概率，介于 0 和 1 之间
        """
        self.dropout_prob = dropout_prob
        self.mask = None  # 用于保存丢弃的神经元的掩码

    def forward(self, input, is_training=True):
        """
        Dropout 前向传播

        参数：
        input：输入张量
        is_training：是否处于训练模式，默认为 True

        返回值：
        output：输出张量
        """
        if is_training:
            self.mask = np.random.rand(*input.shape) > self.dropout_prob
            output = input * self.mask / 1 - self.dropout_prob
        else:
            output = input
        return output

    def backward(self, d_L_d_output):
        """
        Dropout 反向传播

        参数：
        d_L_d_output：上一层传回的误差梯度

        返回值：
        d_L_d_input：输入的误差梯度
        """
        d_L_d_input = d_L_d_output * self.mask / 1 - self.dropout_prob
        return d_L_d_input


class _ReLU_:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, d_L_d_output):
        return d_L_d_output * (self.input > 0)



import numpy as np

class BatchNormalizationLayer:
    def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
        """
        初始化批量归一化层。

        参数：
        input_size：输入大小。
        epsilon：避免除零错误的小常数。
        momentum：用于计算移动平均的动量参数。
        """
        self.inputs = None
        self.normalized_inputs = None
        self.batch_var = None
        self.input_size = input_size
        self.epsilon = epsilon
        self.momentum = momentum
        self.batch_mean = np.zeros(input_size)
        # 初始化参数
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.zeros(input_size)

    def forward(self, inputs, training=True):
        """
        前向传播：对输入数据进行批量归一化。

        参数：
        inputs：输入数据，形状为 (batch_size, input_size)。
        training：是否处于训练模式。

        返回值：
        outputs：批量归一化后的输出数据，形状与输入相同。
        """
        if training:
            # 计算批量的均值和方差
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            self.inputs = inputs
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            # 更新移动平均值
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # 标准化输入数据
            normalized_inputs = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.normalized_inputs = normalized_inputs
        else:
            # 使用移动平均值进行标准化
            normalized_inputs = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # 应用缩放和偏移
        outputs = self.gamma * normalized_inputs + self.beta

        return outputs

    def backward(self, d_L_d_outputs, learning_rate):
        """
        反向传播：计算梯度。

        参数：
        d_L_d_outputs：上一层传回的误差梯度，形状为 (batch_size, input_size)。
        learning_rate：学习率。

        返回值：
        d_L_d_inputs：上一层的误差梯度，形状为 (batch_size, input_size)。
        """
        batch_size = d_L_d_outputs.shape[0]

        # 计算对参数的梯度
        d_gamma = np.sum(d_L_d_outputs * self.normalized_inputs, axis=0)
        d_beta = np.sum(d_L_d_outputs, axis=0)

        # 计算对输入的梯度
        d_L_d_normalized_inputs = d_L_d_outputs * self.gamma

        # 计算对均值和方差的梯度
        d_batch_var = np.sum(d_L_d_normalized_inputs * (self.inputs - self.batch_mean) * (-0.5) * (self.batch_var + self.epsilon) ** (-1.5), axis=0)
        d_batch_mean = np.sum(d_L_d_normalized_inputs * (-1) / np.sqrt(self.batch_var + self.epsilon), axis=0)

        # 计算对输入的梯度
        d_L_d_inputs = d_L_d_normalized_inputs / np.sqrt(self.batch_var + self.epsilon) + d_batch_var * 2 * (self.inputs - self.batch_mean) / batch_size + d_batch_mean / batch_size

        # 更新参数
        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta

        return d_L_d_inputs
