import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant(2)
    b = tf.constant(3)
    print(sess.run(a+b))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))