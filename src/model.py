import tensorflow as tf

def vgg16_1d(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
    # Block 1
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Block 2
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Block 3
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Block 4
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Block 5
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
    
    # Classification block
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    
    return model

def simple_model_lung():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=(193, 1)))

    model.add(tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2)) 
    
    model.add(tf.keras.layers.Conv1D(256, kernel_size=5, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2)) 

    model.add(tf.keras.layers.Conv1D(512, kernel_size=5, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2)) 

    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, activation='relu'))   
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(512, activation='relu'))  
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(256, activation='relu'))  
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    
    return model


