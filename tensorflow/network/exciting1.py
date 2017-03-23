import tensorflow as tf
import numpy as np
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
# create tensorflow structure start
#权重是一维，-1.0到1.0的数
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
#预测y值
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
#学习效率<1,这里选择0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化
init = tf.initialize_all_variables()
# create tensorflow structure end
sess = tf.Session()
# 激活
sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))