import tensorflow as tf

table = tf.contrib.lookup.HashTable(
  tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
)
out = table.lookup(input_tensor)
table.init.run()
print out.eval()