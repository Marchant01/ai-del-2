import tensorflow as tf

# Check if GPU is detected
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# Check if operations are on GPU
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("GPU available: ", tf.test.is_gpu_available())  # deprecated but useful
