import tensorflow as tf
print(f"TensorFlow {tf.__version__} working!")
print(f"Keras available: {hasattr(tf, 'keras')}")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(5,)),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='mse')
print("Model created successfully!")