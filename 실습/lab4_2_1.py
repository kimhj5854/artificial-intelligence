#To create each Variables and Placeholders is unuseful
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter = ',', dtype = np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]] #그냥 -1하면 안되는 걸까?

#Make sure the shape and data are ok.
print(x_data.shape, x_data, len(x_data))
print(y_data.shpae, y_data)

#placeholders for a tensor that will be always fed?
X = tf.placeholder(tf.float32, shape=[None,3]) #when data instance is not setted, first elements is None
Y = tf.placeholder(tf.float32, shape=[None,1]) #N data

W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#hypothesis using matrix multiplication
hypothesis = tf.matmul(X,W) + b 


#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize. Need a very small Learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

#launch the graph in a session
sess = tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001) :
     cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={X: x_data, Y: y_data})
     if step%10 == 0 :
         print(step, "cost:" , cost_val, "\nPrediction\n", hy_val)



#Ask my score

print("Your score will be ", sess.run(hypothesis,
                                      feed_dict = {X : [[100,70,101]]}))

print("Your score will be ", sess.run(hypothesis,
                                      feed_dict = {X : [[60,70,110],[90,100,80]]}))

