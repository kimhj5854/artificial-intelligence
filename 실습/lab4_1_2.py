#To create each Variables and Placeholders is unuseful
import tensorflow as tf

x_data = [[73.,80.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
#Y matrix will be [#ofinstance, 1] matrix. Then we should use 2D matrix
y_data = [[152.], [185.], [180.], [196.], [142.]]



#placeholders for a tensor that will be always fed?
X = tf.placeholder(tf.float32, shape=[None,3]) #when data instance is not setted, first elements is None
Y = tf.placeholder(tf.float32, shape=[None,1]) #N data

#Second matrix is weight matrix, which is will [3,1] matrix in this case
W = tf.Variable(tf.random_normal([3,1]), name = 'weight')

#bias will just a float number
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


         
     
