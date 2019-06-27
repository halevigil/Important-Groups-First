import numpy as np
import time
import tensorflow as tf
import tensorflow_hub as hub
import gc
import sys
print(sys.argv)
"""/Users/Gil/Documents/Code/getReps/"""
start = time.time()
pics = np.load("dsprites_full_pics.npy")
i=sys.argv[1]
with tf.Session() as sess:
    model = hub.Module("models/" + str(i) + "/" + str(i) + "/postprocessed/mean/tfhub")
    sess.run(tf.global_variables_initializer())
    np.savetxt("reps/reps" + str(i),
              model(dict(images=pics), signature="representation", as_dict=True)['default'].eval())
gc.collect()
end = time.time()
print(end - start)