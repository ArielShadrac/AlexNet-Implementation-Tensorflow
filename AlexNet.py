from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the AlexNet model
model = Sequential([
    # First convolutional layer: 96 filters, 11x11 kernel, stride 4, ReLU activation
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)),
    # First max pooling layer: 3x3 pool size, stride 2
    MaxPooling2D(pool_size=(3, 3), strides=2),
    # Second convolutional layer: 384 filters, 3x3 kernel, same padding, ReLU activation
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    # Third convolutional layer: 384 filters, 3x3 kernel, same padding, ReLU activation
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    # Fourth convolutional layer: 256 filters, 3x3 kernel, same padding, ReLU activation
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    # Second max pooling layer: 3x3 pool size, stride 2
    MaxPooling2D(pool_size=(3, 3), strides=2),
    # Flatten the output for dense layers
    Flatten(),
    # First fully connected layer: 4096 units, ReLU activation
    Dense(4096, activation='relu'),
    # Dropout for regularization (50% dropout rate)
    Dropout(0.5),
    # Second fully connected layer: 4096 units, ReLU activation
    Dense(4096, activation='relu'),
    # Dropout for regularization (50% dropout rate)
    Dropout(0.5),
    # Output layer: 1000 units for ImageNet classification, softmax activation
    Dense(1000, activation='softmax')
])

# Compile the model (example configuration for classification)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the model to a file
model.save('alexnet_model.h5')

# Print model summary
model.summary()