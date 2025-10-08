import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold 
import matplotlib.pyplot as plt
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from glob import glob
import cv2
import numpy as np
import os

class CustomDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, img_size, num_classes, shuffle=True, class_indices=None):
        self.image_paths = image_paths
        self.labels = labels
        self.classes = np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.class_indices = class_indices or self._build_class_indices(labels)
        self.on_epoch_end()
        self.current_index=0
        
    def __iter__(self):
        """Initialize the iterator."""
        self.current_index = 0
        return self

    def __next__(self):
        """Get the next batch."""
        if self.current_index >= len(self):
            # End of epoch
            self.on_epoch_end()
            self.current_index = 0
            raise StopIteration
        
        batch = self.__getitem__(self.current_index)
        self.current_index += 1
        return batch
    
    def reset(self):
        """Reset the generator for the next epoch."""
        self.current_index = 0
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            np.random.shuffle(combined)
            self.image_paths, self.labels = map(list, zip(*combined))  # Convert back to lists
            self.classes = np.array(self.labels)  # Update classes array as well
    
    def _build_class_indices(self, labels):
        unique_labels = sorted(set(labels))
        return {str(lbl): lbl for lbl in unique_labels}

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            stacked =np.stack([img], axis=-1)
            batch_images.append(stacked)

        batch_images = np.array(batch_images)
        batch_labels = tf.keras.utils.to_categorical(batch_labels, self.num_classes)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            np.random.shuffle(combined)
            self.image_paths, self.labels = map(list, zip(*combined))  # Convert back to lists
            self.classes = np.array(self.labels)  # Update classes array as well


train_dir = "/mnt/c/newTrainHV/Train"

test_dir = "/mnt/c/newTrainHV/Test"

from sklearn.preprocessing import LabelEncoder

def get_filepaths_and_labels(directory):
    class_names = sorted(os.listdir(directory))
    filepaths = []
    labels = []

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in glob(os.path.join(class_dir, "*")):
            filepaths.append(fname)
            labels.append(class_name)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    class_indices = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return filepaths, encoded_labels, len(label_encoder.classes_), label_encoder,class_indices
# Image parameters
IMG_SIZE = (256,256 )  # Image size
BATCH_SIZE = 40       # Number of images in each batch
# Train
train_filepaths, train_labels, NUM_CLASSES, label_encoder,class_indices = get_filepaths_and_labels(train_dir)
test_filepaths, test_labels, _, _,_ = get_filepaths_and_labels(test_dir)
# Combine for full dataset cross-validation
all_filepaths = np.array(train_filepaths + test_filepaths)
all_labels = np.array(train_labels.tolist() + test_labels.tolist())
train_generator = CustomDataGenerator(train_filepaths, train_labels, BATCH_SIZE, IMG_SIZE, NUM_CLASSES,class_indices=class_indices)
test_generator = CustomDataGenerator(test_filepaths, test_labels, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, shuffle=False,class_indices=class_indices)

train_set= train_generator
test_set= test_generator
print("Class Indices:", train_set.class_indices)
print("Number of Classes:", train_set.num_classes)
print("Number of Classes:", test_set.num_classes)
from collections import Counter
print(Counter(train_set.classes))
print(Counter(test_set.classes))
#***MODEL CREATION***

INPUT_SHAPE = (256, 256, 1)  # Input size of images
NUM_CLASSES =  train_generator.num_classes             # many class
LEARNING_RATE = 0.001
EPOCHS = 100                 # Increased epochs
BATCH_SIZE = 40

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout


import tensorflow as tf

# Define quantization functions
def fake_quantize(x, bits=8, min_value=None, max_value=None):
    if min_value is None:
        min_value = tf.reduce_min(x)
    if max_value is None:
        max_value = tf.reduce_max(x)
    
    # Ensure min doesn't equal max to prevent division by zero
    max_value = tf.maximum(max_value, min_value + 1e-6)
    # Calculate the step size (the value of 1 bit)
    step = (max_value - min_value) / (2**bits - 1)
    # Quantize the values
    x_int = tf.round((x - min_value) / step)
    # Clip values to the quantization range
    x_int = tf.clip_by_value(x_int, 0, 2**bits - 1)
    # Convert back to original range
    x_q = x_int * step + min_value
    # During training, pass through the gradients using STE (Straight Through Estimator)
    return x + tf.stop_gradient(x_q - x)

@tf.keras.utils.register_keras_serializable()
class QuantizedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', strides=1, activation=None, weight_bits=8, activation_bits=8, **kwargs):
        super(QuantizedConv2D, self).__init__(**kwargs)
        self.filters = int(filters)
        # Ensure kernel_size is stored as a tuple of integers
        if isinstance(kernel_size, int):
            self.kernel_size = (int(kernel_size), int(kernel_size))
        else:
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.padding = padding
        if isinstance(strides, int):
            self.strides = (int(strides), int(strides))
        else:
            self.strides = (int(strides[0]), int(strides[1]))
        self.activation_fn = tf.keras.activations.get(activation)
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        
    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        kernel_size = tuple(self.kernel_size)
        kernel_shape = kernel_shape = (
            int(self.kernel_size[0]),
            int(self.kernel_size[1]),
            int(input_channels),
            int(self.filters)
            )
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        
        # Track min and max values for weights (needed for quantization)
        self.w_min = self.add_weight(
            name='w_min',
            shape=(1,),
            initializer=tf.constant_initializer(-1.0),
            trainable=False
        )
        
        self.w_max = self.add_weight(
            name='w_max',
            shape=(1,),
            initializer=tf.constant_initializer(1.0),
            trainable=False
        )
        
        # Track min and max values for activations
        self.a_min = self.add_weight(
            name='a_min',
            shape=(1,),
            initializer=tf.constant_initializer(0.0),
            trainable=False
        )
        
        self.a_max = self.add_weight(
            name='a_max',
            shape=(1,),
            initializer=tf.constant_initializer(6.0),  # ReLU typically has max around 6
            trainable=False
        )
        
        self.built = True
    
    def call(self, inputs, training=None):
        if training:
            curr_w_min = tf.reduce_min(self.kernel)
            curr_w_max = tf.reduce_max(self.kernel)
            
            # Use EMA (Exponential Moving Average) to update min/max values
            momentum = 0.9
            self.w_min.assign(momentum * self.w_min + (1 - momentum) * curr_w_min)
            self.w_max.assign(momentum * self.w_max + (1 - momentum) * curr_w_max)
        
        # Quantize weights
        quantized_kernel = fake_quantize(
            self.kernel, 
            bits=self.weight_bits,
            min_value=self.w_min,
            max_value=self.w_max
        )
        
        # Standard convolution with quantized weights
        outputs = tf.nn.conv2d(
            inputs,
            quantized_kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper()
        )
        
        outputs = tf.nn.bias_add(outputs, self.bias)
        
        # Apply activation if specified
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
            
            # Update activation min/max during training
            if training:
                curr_a_min = tf.reduce_min(outputs)
                curr_a_max = tf.reduce_max(outputs)
                
                # Use EMA to update min/max values
                momentum = 0.9
                self.a_min.assign(momentum * self.a_min + (1 - momentum) * curr_a_min)
                self.a_max.assign(momentum * self.a_max + (1 - momentum) * curr_a_max)
            
            # Quantize activations
            outputs = fake_quantize(
                outputs,
                bits=self.activation_bits,
                min_value=self.a_min,
                max_value=self.a_max
            )
        
        return outputs

    def get_config(self):
        config = super(QuantizedConv2D, self).get_config()
        config.update({
            'filters': int(self.filters),  # Ensure filters is serialized as int
            'kernel_size': (int(self.kernel_size[0]), int(self.kernel_size[1])),  # Ensure tuple of ints
            'padding': self.padding,
            'strides': (int(self.strides[0]), int(self.strides[1])),  # Ensure tuple of ints
            'activation': tf.keras.activations.serialize(self.activation_fn),
            'weight_bits': int(self.weight_bits),
            'activation_bits': int(self.activation_bits)
        })
        return config

@tf.keras.utils.register_keras_serializable()
class QuantizedDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, weight_bits=8, activation_bits=8, **kwargs):
        super(QuantizedDense, self).__init__(**kwargs)
        self.units = units
        self.activation_fn = tf.keras.activations.get(activation)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        # Track min and max values for weights
        self.w_min = self.add_weight(
            name='w_min',
            shape=(1,),
            initializer=tf.constant_initializer(-1.0),
            trainable=False
        )
        
        self.w_max = self.add_weight(
            name='w_max',
            shape=(1,),
            initializer=tf.constant_initializer(1.0),
            trainable=False
        )
        
        # Track min and max values for activations
        self.a_min = self.add_weight(
            name='a_min',
            shape=(1,),
            initializer=tf.constant_initializer(0.0),
            trainable=False
        )
        
        self.a_max = self.add_weight(
            name='a_max',
            shape=(1,),
            initializer=tf.constant_initializer(6.0),
            trainable=False
        )
        
        self.built = True
    
    def call(self, inputs, training=None):
        if training:
            curr_w_min = tf.reduce_min(self.kernel)
            curr_w_max = tf.reduce_max(self.kernel)
            
            # Use EMA to update min/max values
            momentum = 0.9
            self.w_min.assign(momentum * self.w_min + (1 - momentum) * curr_w_min)
            self.w_max.assign(momentum * self.w_max + (1 - momentum) * curr_w_max)
        
        # Quantize weights
        quantized_kernel = fake_quantize(
            self.kernel,
            bits=self.weight_bits,
            min_value=self.w_min,
            max_value=self.w_max
        )
    
        # Standard dense operation with quantized weights
        outputs = tf.matmul(inputs, quantized_kernel)
        outputs = tf.nn.bias_add(outputs, self.bias)
        
        # Apply activation if specified
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
            
            # Update activation min/max during training
            if training:
                curr_a_min = tf.reduce_min(outputs)
                curr_a_max = tf.reduce_max(outputs)
                
                # Use EMA to update min/max values
                momentum = 0.9
                self.a_min.assign(momentum * self.a_min + (1 - momentum) * curr_a_min)
                self.a_max.assign(momentum * self.a_max + (1 - momentum) * curr_a_max)
            
            # Quantize activations
            outputs = fake_quantize(
                outputs,
                bits=self.activation_bits,
                min_value=self.a_min,
                max_value=self.a_max
            )
        
        return outputs
    
    def get_config(self):
        config = super(QuantizedDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation_fn),
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits
        })
        return config
    
def create_quantized_cnn_model(input_shape=(128, 128, 1), num_classes=10, weight_bits=8, activation_bits=8):
    """
    Creates a quantization-aware version of the model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # First Convolutional Block
    x = QuantizedConv2D(16, (3, 3), padding='same', activation='relu', 
                        weight_bits=weight_bits, activation_bits=activation_bits)(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(6, 6), strides=6)(x)
    
    # Second Convolutional Block
    x = QuantizedConv2D(32, (3, 3), padding='same', activation='relu',
                        weight_bits=weight_bits, activation_bits=activation_bits)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=3)(x)
    
    # Third Convolutional Block
    x = QuantizedConv2D(64, (3, 3), padding='same', activation='relu',
                        weight_bits=weight_bits, activation_bits=activation_bits)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    
    # Fourth Convolutional Block
    x = QuantizedConv2D(96, (3, 3), padding='same', activation='relu',
                        weight_bits=weight_bits, activation_bits=activation_bits)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    
    # Flatten layer
    x = tf.keras.layers.Flatten()(x)
    
    # First Dense Layer with Dropout
    x = QuantizedDense(512, activation='relu',
                      weight_bits=weight_bits, activation_bits=activation_bits)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Second Dense Layer
    x = QuantizedDense(256, activation='relu',
                      weight_bits=weight_bits, activation_bits=activation_bits)(x)
    
    # Output Layer 
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# load the model with custom_objects
custom_objects = {
    'QuantizedConv2D': QuantizedConv2D,
    'QuantizedDense': QuantizedDense
}


# %% [markdown]
# **OPTIMIZERS AND CALLBACK DEFINITIONS**

# Create the model
model = create_quantized_cnn_model(INPUT_SHAPE, NUM_CLASSES)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',  # Multi-class classification
    metrics=['accuracy']
)

# Display model summary
model.summary()

from tensorflow.keras.callbacks import Callback, EarlyStopping

class StopAtAccuracy(Callback):
    def __init__(self, target_acc=0.85):
        super(StopAtAccuracy, self).__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if logs.get("val_accuracy") >= self.target_acc:  # Stop when val_accuracy reaches target
            print(f"\nStopping training: Reached {self.target_acc * 100:.1f}% validation accuracy")
            self.model.stop_training = True

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
)
stop_at_100 = StopAtAccuracy(target_acc=1)
# Learning rate scheduler to reduce LR when validation loss plateaus
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1
)


# %%
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
history_list = []
i = 0
for fold, (train_idx, val_idx) in enumerate(skf.split(all_filepaths, all_labels)):
    i += 1
    if i == 2:
        break
    print(f"\n[INFO] Fold {fold + 10}")

    X_train, y_train = all_filepaths[train_idx], all_labels[train_idx]
    X_val, y_val = all_filepaths[val_idx], all_labels[val_idx]

    # Create data generators for this fold
    train_generator = CustomDataGenerator(X_train, y_train, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, shuffle=True, class_indices=class_indices)
    val_generator = CustomDataGenerator(X_val, y_val, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, shuffle=False, class_indices=class_indices)

    # Initialize model (make sure create_model is defined)
    model = create_quantized_cnn_model(INPUT_SHAPE, NUM_CLASSES, weight_bits=8, activation_bits=8)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    with tf.device('/GPU:0'):
        history = model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=EPOCHS,
            callbacks=[early_stopping, lr_scheduler, ModelCheckpoint(f'/mnt/c/modelFiles/10splitsFlatten/GreyColor_learn_{fold+9}.keras', monitor='val_loss', save_best_only=True, verbose=1), stop_at_100],
            verbose=1
        )
    model.save(f"/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_{fold+9}.keras")
    if i == 1:
        model.summary()
    history_list.append(history)
    model.save(f"/mnt/c/modelFiles/10splitsFlatten/qat_kfold_model_{fold+9}.h5")  # Save in HDF5 format
    print("model saved")
    # Reload it 
    modelh5 = tf.keras.models.load_model(f"/mnt/c/modelFiles/10splitsFlatten/qat_kfold_model_{fold+9}.h5", custom_objects=custom_objects)
    print("model loaded")
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(modelh5)
    print("converter set")
    # Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    print("optimizations initialized")
    # Define a representative dataset function for calibration
    def representative_dataset_gen():
        for i in range(100):  # Just use first 100 images for calibration
            img, _ = train_generator[i]
            for x in img:
                yield [np.expand_dims(x, axis=0).astype(np.float32)]


    converter.representative_dataset = representative_dataset_gen
    print("representative dataset initialized")
    # Ensure all tensors are int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    print("type bit converstions done")
    # Perform conversion
    quantized_tflite_model = converter.convert()
    print("quantized model obtained")
    # Save the quantized model to disk
    with open(f"/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_{fold+9}.tflite", "wb") as f:
        f.write(quantized_tflite_model)
    print("tfmodel saved") #tflite 


# %%
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_individual_graphs(history_list):
    x = random.randint(1,10)
    # Create figure for accuracy plots (top row)
    plt.figure(figsize=(20, 10))
    
    for i, history in enumerate(history_list):
        # Get the history dictionary
        history_dict = history.history
        
        # Check for accuracy metric name
        acc_key = 'accuracy' if 'accuracy' in history_dict else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'
        
        # Get the number of epochs
        epochs = range(1, len(history_dict[acc_key]) + 1)
        
        # Create subplot for this run's accuracy
        plt.subplot(1, 5, i+1)
        plt.plot(epochs, history_dict[acc_key], 'b-', label='Training Accuracy')
        plt.plot(epochs, history_dict[val_acc_key], 'r-', label='Validation Accuracy')
        plt.title(f'Run {i+1} Accuracy'  if i%2==0 else f'Run {i+x} Accuracy' )
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"/mnt/c/modelFiles/10splitsFlatten/accuracy_plots.png")
    plt.show()
    print("Accuracy plots saved as 'accuracy_plots.png'")
    
    # Create figure for loss plots 
    plt.figure(figsize=(20, 10))
    
    for i, history in enumerate(history_list):
        # Get the history dictionary
        history_dict = history.history
        
        # Get the number of epochs
        epochs = range(1, len(history_dict['loss']) + 1)
        
        # Create subplot for this run's loss
        plt.subplot(1, 5, i+1)
        plt.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
        plt.title(f'Run {i+1} Loss' if i%2==0 else f'Run {i+x} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"/mnt/c/modelFiles/10splitsFlatten/loss_plots.png")
    plt.show()
    print("Loss plots saved as 'loss_plots.png'")

    # Optional: Print summary statistics for each run
    print("\nSummary Statistics:")
    for i, history in enumerate(history_list):
        history_dict = history.history
        acc_key = 'accuracy' if 'accuracy' in history_dict else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'
        
        final_train_acc = history_dict[acc_key][-1]
        final_val_acc = history_dict[val_acc_key][-1]
        final_train_loss = history_dict['loss'][-1]
        final_val_loss = history_dict['val_loss'][-1]
        
        print(f"Run {i+1}:")
        print(f"  Final training accuracy: {final_train_acc:.4f}")
        print(f"  Final validation accuracy: {final_val_acc:.4f}")
        print(f"  Final training loss: {final_train_loss:.4f}")
        print(f"  Final validation loss: {final_val_loss:.4f}")

# Call the function with your list of history objects
plot_individual_graphs(history_list)

# %% [markdown]
# CHOOSING BEST MODEL - KERAS

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.models import load_model

# Load all 10 Keras models
model_paths = [f"/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_{fold}.keras" for fold in range(10)]
models = [load_model(path) for path in model_paths]

def evaluate_individual_models(models, test_generator):
    class_names = list(test_generator.class_indices.keys())
    num_samples = len(test_generator.image_paths)
    steps = int(np.ceil(num_samples / test_generator.batch_size))
    f1_scores = []

    for model_idx, model in enumerate(models):
        y_true_all, y_pred_all = [], []
        test_generator.reset()

        for _ in range(steps):
            x_batch, y_batch = next(test_generator)
            preds = model.predict(x_batch, verbose=0)
            y_pred_all.extend(np.argmax(preds, axis=1))
            y_true_all.extend(np.argmax(y_batch, axis=1))

        y_true_all = np.array(y_true_all[:num_samples])
        y_pred_all = np.array(y_pred_all[:num_samples])

        print(f"\n=== Model {model_idx + 1} ===")
        print("Classification Report:")
        print(classification_report(y_true_all, y_pred_all, target_names=class_names))

        # F1-score (weighted)
        f1 = f1_score(y_true_all, y_pred_all, average='weighted')
        f1_scores.append(f1)

        accuracy = np.mean(y_true_all == y_pred_all)
        print(f"Model {model_idx + 1} Accuracy: {accuracy:.4f}")
        print(f"Model {model_idx + 1} Weighted F1-Score: {f1:.4f}")

        print("Per-class Accuracy:")
        for i, class_name in enumerate(class_names):
            mask = y_true_all == i
            correct = np.sum((y_true_all == i) & (y_pred_all == i))
            acc = correct / np.sum(mask) if np.sum(mask) > 0 else 0.0
            print(f"{class_name}: {acc:.4f}")

    # --- Plot F1-Scores ---
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(models)), f1_scores, color='skyblue')
    plt.title('F1-Scores of 10 Keras Models')
    plt.xlabel('Model Index')
    plt.ylabel('F1-Score (Weighted)')
    plt.ylim(0, 1.05)
    plt.xticks(range(10), [f'Model {i}' for i in range(10)])

    # Annotate each bar with F1-score
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("keras_f1_scores.png")
    plt.show()

# Run the evaluation and plot
evaluate_individual_models(models, test_generator)


# %% [markdown]
# CHOOSING BEST MODEL - TFLite

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score

# Paths to the 10 TFLite models
tflite_model_paths = [f"/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_{i}.tflite" for i in range(10)]

# Load all interpreters
interpreters = []
input_details_list = []
output_details_list = []
quant_params_list = []

for path in tflite_model_paths:
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scale, zero_point = input_details[0]['quantization']

    interpreters.append(interpreter)
    input_details_list.append(input_details)
    output_details_list.append(output_details)
    quant_params_list.append((scale, zero_point))

# Evaluate each model and store F1-scores
f1_scores = []

print(f"\nRunning evaluation for {len(interpreters)} individual TFLite models...\n")

for model_idx in range(len(interpreters)):
    y_true = []
    y_pred = []

    print(f"\n--- Evaluating TFLite Model {model_idx + 1} ---")

    for batch_images, batch_labels in test_generator:
        for img, label in zip(batch_images, batch_labels):
            img = np.expand_dims(img, axis=0).astype(np.float32)

            scale, zero_point = quant_params_list[model_idx]
            img_int8 = (img / scale + zero_point).astype(np.int8)

            interpreters[model_idx].set_tensor(input_details_list[model_idx][0]['index'], img_int8)
            interpreters[model_idx].invoke()
            output_data = interpreters[model_idx].get_tensor(output_details_list[model_idx][0]['index'])

            pred_class = np.argmax(output_data[0])
            true_class = np.argmax(label)

            y_pred.append(pred_class)
            y_true.append(true_class)

    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_scores.append(f1)

    print(f"Model {model_idx + 1} Weighted F1-Score: {f1:.4f}")
    print("Classification Report:")
    target_names = list(label_encoder.classes_)  # assumes a fitted label_encoder
    print(classification_report(y_true, y_pred, target_names=target_names))

# --- Plot F1-Scores ---
plt.figure(figsize=(10, 5))
bars = plt.bar(range(10), f1_scores, color='orange')
plt.title('F1-Scores of 10 TFLite Models')
plt.xlabel('Model Index')
plt.ylabel('F1-Score (Weighted)')
plt.ylim(0, 1.05)
plt.xticks(range(10), [f'Model {i}' for i in range(10)])

# Annotate each bar with F1-score
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("tflite_f1_scores.png")
plt.show()


# %% [markdown]
# **QUANTIZED MODEL EVALUATION**

# %% [markdown]
# .tflite Model

# %%

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the TFLite model
model_path = "/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_0.tflite" 

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = input_details[0]['quantization']

# Inference
y_true = []
y_pred = []

print("\nRunning prediction using TFLite model...\n")

for batch_images, batch_labels in test_generator:
    for img, label in zip(batch_images, batch_labels):
        img = np.expand_dims(img, axis=0).astype(np.float32)  # [1, H, W, C]

        img_int8 = (img / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], img_int8)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_class = np.argmax(output_data[0])
        true_class = np.argmax(label)

        y_pred.append(pred_class)
        y_true.append(true_class)

# Evaluation
print("TFLite Model Accuracy:", np.mean(np.array(y_true) == np.array(y_pred)))

# Classification report
target_names = list(label_encoder.classes_)  # assumes you have a fitted label_encoder
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# %% [markdown]
# ***KERAS MODEL EVALUATION***

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Load one model 
model_path = "/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras"
model = load_model(model_path)

# %%
def evaluate_single_model(model, test_generator):
    num_samples = len(test_generator.image_paths)
    steps = int(np.ceil(num_samples / test_generator.batch_size))

    y_true_all, y_pred_all = [], []

    for _ in range(steps):
        x_batch, y_batch = next(test_generator)
        preds = model.predict(x_batch, verbose=0)
        y_pred_all.extend(np.argmax(preds, axis=1))
        y_true_all.extend(np.argmax(y_batch, axis=1))

    y_true_all = np.array(y_true_all[:num_samples])
    y_pred_all = np.array(y_pred_all[:num_samples])

    class_names = list(test_generator.class_indices.keys())

    # Confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Single Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=class_names))

    accuracy = np.sum(y_true_all == y_pred_all) / len(y_true_all)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        mask = y_true_all == i
        correct = np.sum((y_true_all == i) & (y_pred_all == i))
        acc = correct / np.sum(mask) if np.sum(mask) > 0 else 0.0
        print(f"{class_name}: {acc:.4f}")

# Usage
evaluate_single_model(model, test_generator)


# %% [markdown]
# (SIMULATION) 
# **COMPARISON BETWEEN A GENERIC MODEL AND QUANTIZATION AWARE TRAINING MODEL**

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load both models
original_model = tf.keras.models.load_model('/mnt/c/modelFiles/10splitsFlatten/GreyColorisback.keras')
quantized_model = tf.keras.models.load_model("/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras", custom_objects=custom_objects)

# Build models if they aren't already
sample_input_shape = (None, 256, 256, 1)
original_model.build(input_shape=sample_input_shape)
quantized_model.build(input_shape=sample_input_shape)

# Extract weights for all convolutional layers
def get_all_conv_weights(model):
    weights = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, QuantizedConv2D)):
            w = layer.get_weights()
            if w:
                weights.append(w[0])  # Only use kernel weights
    return weights

orig_weights_list = get_all_conv_weights(original_model)
quant_weights_list = get_all_conv_weights(quantized_model)

# Sanity check
num_layers = min(len(orig_weights_list), len(quant_weights_list))
print(f"Comparing {num_layers} convolutional layers")

# Plot distributions for each layer
for i in range(num_layers):
    orig_weights = orig_weights_list[i].flatten()
    quant_weights = quant_weights_list[i].flatten()
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(orig_weights, bins=50, alpha=0.6, label='Original')
    plt.title(f'Layer {i+1} Original Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(quant_weights, bins=50, alpha=0.6, label='Quantized', color='orange')
    plt.title(f'Layer {i+1} Quantized Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.hist(orig_weights, bins=50, alpha=0.5, label='Original')
    plt.hist(quant_weights, bins=50, alpha=0.5, label='Quantized')
    plt.title(f'Layer {i+1} Overlaid Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'layer_{i+1}_weight_distribution_comparison.png')
    plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load both models
original_model = tf.keras.models.load_model('/mnt/c/modelFiles/10splitsFlatten/GreyColorisback.keras')
quantized_model = tf.keras.models.load_model("/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras", custom_objects=custom_objects)

# Build models if they aren't already
sample_input_shape = (None, 256, 256, 1)
original_model.build(input_shape=sample_input_shape)
quantized_model.build(input_shape=sample_input_shape)

# Extract weights for all convolutional layers
def get_all_conv_weights(model):
    weights = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, QuantizedDense)):
            w = layer.get_weights()
            if w:
                weights.append(w[0])  # Only use kernel weights
    return weights

orig_weights_list = get_all_conv_weights(original_model)
quant_weights_list = get_all_conv_weights(quantized_model)

# Sanity check
num_layers = min(len(orig_weights_list), len(quant_weights_list))
print(f"Comparing {num_layers} Dense layers")

# Plot distributions for each layer
for i in range(num_layers):
    orig_weights = orig_weights_list[i].flatten()
    quant_weights = quant_weights_list[i].flatten()
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(orig_weights, bins=50, alpha=0.6, label='Original')
    plt.title(f'Layer {i+1} Original Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(quant_weights, bins=50, alpha=0.6, label='Quantized', color='orange')
    plt.title(f'Layer {i+1} Quantized Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.hist(orig_weights, bins=50, alpha=0.5, label='Original')
    plt.hist(quant_weights, bins=50, alpha=0.5, label='Quantized')
    plt.title(f'Layer {i+1} Overlaid Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'layer_{i+1}_weight_distribution_comparison.png')
    plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Supported layer types for activation visualization
ACTIVATION_LAYERS = (tf.keras.layers.Conv2D, tf.keras.layers.Dense, QuantizedConv2D, QuantizedDense)

def create_activation_models(model):
    input_shape = (256, 256, 1)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    outputs = []
    layer_names = []
    x = input_layer

    for layer in model.layers:
        try:
            x = layer(x)
            if isinstance(layer, ACTIVATION_LAYERS):
                outputs.append(x)
                layer_names.append(layer.name)
        except:
            continue
    
    activation_model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    return activation_model, layer_names

# Create activation models
orig_activation_model, orig_layer_names = create_activation_models(original_model)
quant_activation_model, quant_layer_names = create_activation_models(quantized_model)

# Ensure layer alignment
num_layers = min(len(orig_layer_names), len(quant_layer_names))

# Get a batch of input
x_batch, _ = next(iter(test_generator))

# Predict activations
orig_activations = orig_activation_model.predict(x_batch)
quant_activations = quant_activation_model.predict(x_batch)

# Plot activation distributions for each layer
for i in range(num_layers):
    orig_act = orig_activations[i].flatten()
    quant_act = quant_activations[i].flatten()

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(orig_act, bins=50, alpha=0.6, label='Original')
    plt.title(f'Layer {orig_layer_names[i]} Original')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(quant_act, bins=50, alpha=0.6, label='Quantized', color='orange')
    plt.title(f'Layer {quant_layer_names[i]} Quantized')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(orig_act, bins=50, alpha=0.5, label='Original')
    plt.hist(quant_act, bins=50, alpha=0.5, label='Quantized')
    plt.title(f'Layer {orig_layer_names[i]} Overlaid')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'activation_distribution_layer_{i+1}_{orig_layer_names[i]}.png')
    plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict

original_model = tf.keras.models.load_model('/mnt/c/modelFiles/10splitsFlatten/GreyColorisback.keras')
quantized_model = tf.keras.models.load_model("/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras", custom_objects=custom_objects)

# 1. Collect one image per class (assumes one-hot encoded labels)
def collect_one_sample_per_class(test_generator, num_classes=5):
    class_samples = {}
    for x_batch, y_batch in test_generator:
        for img, label in zip(x_batch, y_batch):
            class_idx = np.argmax(label)
            if class_idx not in class_samples:
                class_samples[class_idx] = img[np.newaxis, ...]  # Add batch dimension
            if len(class_samples) == num_classes:
                return class_samples
    return class_samples

# 2. Function to get intermediate layer outputs
def get_layer_outputs(model, input_data, num_layers=4):
    model.build(input_shape=(None, 256, 256, 1))
    outputs = []
    count = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, QuantizedConv2D)):
            temp_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            out = temp_model.predict(input_data)
            outputs.append(out)
            count += 1
            if count >= num_layers:
                break
    return outputs

# 3. Main logic
samples_per_class = collect_one_sample_per_class(test_generator, num_classes=5)

for class_idx, sample_img in samples_per_class.items():
    orig_features = get_layer_outputs(original_model, sample_img)
    quant_features = get_layer_outputs(quantized_model, sample_img)

    for layer_idx in range(min(len(orig_features), len(quant_features))):
        orig_feature = orig_features[layer_idx][0]   # Shape: (H, W, C)
        quant_feature = quant_features[layer_idx][0]

        # Compute average activation per channel
        orig_channel_strength = np.mean(orig_feature, axis=(0, 1))  # Shape: (C,)
        quant_channel_strength = np.mean(quant_feature, axis=(0, 1))  # Shape: (C,)

        # Get indices of top 8 channels
        top8_orig_indices = np.argsort(orig_channel_strength)[-8:][::-1]
        top8_quant_indices = np.argsort(quant_channel_strength)[-8:][::-1]

        plt.figure(figsize=(16, 8))
        plt.suptitle(f'Layer {layer_idx+1} Top-Activated Feature Maps', fontsize=16)

        for idx, (orig_ch, quant_ch) in enumerate(zip(top8_orig_indices, top8_quant_indices)):
            # Original model top channel
            plt.subplot(2, 8, idx + 1)
            plt.imshow(orig_feature[:, :, orig_ch], cmap='viridis')
            plt.title(f'Orig Ch {orig_ch}')
            plt.axis('off')

            # Quantized model top channel
            plt.subplot(2, 8, idx + 9)
            plt.imshow(quant_feature[:, :, quant_ch], cmap='viridis')
            plt.title(f'Quant Ch {quant_ch}')
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'top_feature_map_layer_{layer_idx+1}_{class_idx+1}.png')
        plt.show()



# %%
def visualize_quantization_effects(original_model_path, quantized_model_path, test_generator):
    """
    Visualize the effects of quantization on model weights and activations.
    
    Args:
        original_model_path: Path to the original (non-quantized) model
        quantized_model_path: Path to the quantized-aware trained model
        test_generator: A data generator providing test data
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    
    # Define custom objects for loading the quantized model
    custom_objects = {'QuantizedConv2D': QuantizedConv2D, 'QuantizedDense': QuantizedDense}
    
    # Load both models
    original_model = tf.keras.models.load_model(original_model_path)
    quantized_model = tf.keras.models.load_model(quantized_model_path, custom_objects=custom_objects)
    # Build models if they aren't already
    sample_input_shape = (None, 256, 256, 1)
    original_model.build(input_shape=sample_input_shape)
    quantized_model.build(input_shape=sample_input_shape)
    x_batch, _ = next(iter(test_generator))
    # Calculate size reduction
    def get_model_size(model):
        """Get approximate model size in MB"""
        weights = [w.numpy() for w in model.weights]
        total_params = sum(w.size for w in weights)
        
        # Calculate size in MB (32-bit float = 4 bytes)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    orig_size = get_model_size(original_model)
    
    # Simulate quantized model size (8-bit = 1 byte)
    quant_size = get_model_size(quantized_model) / 4  # Approximation: 8-bit is 1/4 of 32-bit
    
    print(f"Original model size: {orig_size:.2f} MB")
    print(f"Quantized model size: {quant_size:.2f} MB")
    print(f"Size reduction: {(1 - quant_size/orig_size) * 100:.1f}%")
    
    # Measure and compare inference speed
    import time
    
    # Warm up
    _ = original_model.predict(x_batch[:10])
    _ = quantized_model.predict(x_batch[:10])
    
    # Measure original model speed
    start_time = time.time()
    _ = original_model.predict(x_batch)
    orig_time = time.time() - start_time
    
    # Measure quantized model speed
    start_time = time.time()
    _ = quantized_model.predict(x_batch)
    quant_time = time.time() - start_time
    
    print(f"Original model inference time: {orig_time:.4f} seconds")
    print(f"Quantized model inference time: {quant_time:.4f} seconds")
    print(f"Speed improvement: {(1 - quant_time/orig_time) * 100:.1f}%")
    
    # Return summary as dictionary for further analysis
    return {
        "original_size_mb": orig_size,
        "quantized_size_mb": quant_size,
        "size_reduction_percent": (1 - quant_size/orig_size) * 100,
        "original_inference_time": orig_time,
        "quantized_inference_time": quant_time,
        "speed_improvement_percent": (1 - quant_time/orig_time) * 100
    }

visualize_quantization_effects('/mnt/c/modelFiles/10splitsFlatten/GreyColorisback.keras', "/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras", test_generator)

# %% [markdown]
# **CLASS BASED ADAPTIVE ACCURACY THRESHOLDING**

# %%
#PER CLASS THRESHOLDING
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, auc
import tensorflow as tf

def calculate_metrics_multiclass(y_true, y_pred_probs, thresholds):
    """
    Apply per-class thresholds and compute evaluation metrics.
    
    Args:
        y_true: True class labels
        y_pred_probs: Predicted probabilities for each class
        thresholds: List of per-class thresholds
    
    Returns:
        y_pred: Final predicted labels based on adaptive thresholds
        cm: Confusion matrix
        f1_scores: Per-class F1 scores
    """
    num_samples, num_classes = y_pred_probs.shape
    y_pred = np.full(y_true.shape, -1)

    for idx in range(num_samples):
        valid_classes = []
        for i in range(num_classes):
            if y_pred_probs[idx, i] >= thresholds[i]:
                valid_classes.append(i)

        if valid_classes:
            # Among the valid classes, choose the one with the highest probability
            best_class = max(valid_classes, key=lambda i: y_pred_probs[idx, i])
            y_pred[idx] = best_class
        else:
            # Fallback to argmax if no class exceeds its threshold
            y_pred[idx] = np.argmax(y_pred_probs[idx])

    cm = confusion_matrix(y_true, y_pred)
    f1_scores = f1_score(y_true, y_pred, average=None)
    overall_f1 = f1_score(y_true, y_pred, average="macro")

    return y_pred, cm, f1_scores, overall_f1


def adaptive_threshold_determination_multiclass2(model, test_generator, threshold_range=None):
    """
    Determine adaptive per-class thresholds and evaluate the model.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        threshold_range: Range of thresholds to test (default: 0 to 1 in 0.01 steps)
    
    Returns:
        optimal_thresholds: List of per-class optimal thresholds
    """
    # Reset generator and get predictions
    test_generator.reset()
    steps = len(test_generator)
    
    y_true = []
    y_pred_probs = []
    
    for _ in range(steps):
        x_batch, y_batch = next(test_generator)
        batch_pred = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot labels to indices
        y_pred_probs.extend(batch_pred)  # Store full probability distribution
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    num_classes = y_pred_probs.shape[1]
    class_names = list(test_generator.class_indices.keys())

    # Set threshold search range
    if threshold_range is None:
        threshold_range = np.arange(0, 1.01, 0.01)

    # Find best threshold for each class based on F1-score
    best_thresholds = []
    best_f1_scores = []

    for i in range(num_classes):
        best_threshold = 0.5  # Default
        best_f1 = 0.0

        for threshold in threshold_range:
            temp_thresholds = [0.5] * num_classes  # Start with default 0.5 for all
            temp_thresholds[i] = threshold  # Vary only the current class
            
            _, _, f1_scores, _ = calculate_metrics_multiclass(y_true, y_pred_probs, temp_thresholds)
            if f1_scores[i] > best_f1:
                best_f1 = f1_scores[i]
                best_threshold = threshold

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)

    # Apply optimal thresholds
    y_pred, cm, f1_scores, overall_f1 = calculate_metrics_multiclass(y_true, y_pred_probs, best_thresholds)

    # Compute ROC Curve & AUC per class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = y_pred_probs[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc[i] = auc(fpr[i], tpr[i])

    macro_auc = np.mean(list(roc_auc.values()))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve (Macro AUC = {macro_auc:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Print optimal thresholds
    print("\nOptimal Per-Class Thresholds:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {best_thresholds[i]:.2f} (F1 = {best_f1_scores[i]:.4f})")

    print(f"\nOverall Macro F1 Score: {overall_f1:.4f}")
    print(f"Macro-Averaged ROC AUC: {macro_auc:.4f}")

    return best_thresholds


# Usage:
def evaluate_multiclass_with_adaptive_threshold(model, test_generator):
    """
    Evaluate model on a multi-class dataset using adaptive thresholding.
    """
    optimal_thresholds = adaptive_threshold_determination_multiclass2(model, test_generator)
    return optimal_thresholds

cnn = tf.keras.models.load_model("/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras")
try:
    evaluate_multiclass_with_adaptive_threshold(cnn,test_generator)
except StopIteration as e:
    print("iteration stopped. evaluating again.")
    evaluate_multiclass_with_adaptive_threshold(cnn,test_generator)


# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, auc
import tensorflow as tf

def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantization parameters for input
    input_index = input_details[0]['index']
    input_dtype = input_details[0]['dtype']
    input_scale, input_zero_point = input_details[0]['quantization']

    # Quantize input if needed
    if input_dtype == np.int8:
        input_data = input_data / input_scale + input_zero_point
        input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)
    elif input_dtype == np.uint8:
        input_data = input_data / input_scale + input_zero_point
        input_data = np.clip(np.round(input_data), 0, 255).astype(np.uint8)
    else:
        input_data = input_data.astype(input_dtype)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Dequantize output 
    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)

    output_dtype = output_details[0]['dtype']
    output_scale, output_zero_point = output_details[0]['quantization']

    if output_dtype == np.int8:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    elif output_dtype == np.uint8:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    return output_data

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def calculate_metrics_multiclass(y_true, y_pred_probs, thresholds):
    num_samples, num_classes = y_pred_probs.shape
    y_pred = np.full(y_true.shape, -1)

    for idx in range(num_samples):
        valid_classes = []
        for i in range(num_classes):
            if y_pred_probs[idx, i] >= thresholds[i]:
                valid_classes.append(i)

        if valid_classes:
            # Among the valid classes, choose the one with the highest probability
            best_class = max(valid_classes, key=lambda i: y_pred_probs[idx, i])
            y_pred[idx] = best_class
        else:
            # Fallback to argmax if no class exceeds its threshold
            y_pred[idx] = np.argmax(y_pred_probs[idx])

    cm = confusion_matrix(y_true, y_pred)
    f1_scores = f1_score(y_true, y_pred, average=None)
    overall_f1 = f1_score(y_true, y_pred, average="macro")

    return y_pred, cm, f1_scores, overall_f1

def adaptive_threshold_determination_multiclass2_tflite(tflite_model_path, test_generator, threshold_range=None):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    steps = len(test_generator)
    y_true = []
    y_pred_probs = []

    for _ in range(steps):
        x_batch, y_batch = next(test_generator)
        batch_pred = []

        for i in range(x_batch.shape[0]):
            input_tensor = np.expand_dims(x_batch[i], axis=0).astype(np.float32)
            output = run_tflite_inference(interpreter, input_tensor)
            batch_pred.append(output[0])  

        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_probs.extend(batch_pred)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    num_classes = y_pred_probs.shape[1]
    class_names = list(test_generator.class_indices.keys())

    if threshold_range is None:
        threshold_range = np.arange(0, 1.01, 0.01)

    best_thresholds = []
    best_f1_scores = []

    for i in range(num_classes):
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in threshold_range:
            temp_thresholds = [0.5] * num_classes
            temp_thresholds[i] = threshold

            _, _, f1_scores, _ = calculate_metrics_multiclass(y_true, y_pred_probs, temp_thresholds)
            if f1_scores[i] > best_f1:
                best_f1 = f1_scores[i]
                best_threshold = threshold

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)

    y_pred, cm, f1_scores, overall_f1 = calculate_metrics_multiclass(y_true, y_pred_probs, best_thresholds)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = y_pred_probs[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc[i] = auc(fpr[i], tpr[i])

    macro_auc = np.mean(list(roc_auc.values()))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve (Macro AUC = {macro_auc:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nOptimal Per-Class Thresholds:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {best_thresholds[i]:.2f} (F1 = {best_f1_scores[i]:.4f})")

    print(f"\nOverall Macro F1 Score: {overall_f1:.4f}")
    print(f"Macro-Averaged ROC AUC: {macro_auc:.4f}")

    return best_thresholds

def evaluate_multiclass_with_adaptive_threshold_tflite(tflite_model_path, test_generator):
    optimal_thresholds = adaptive_threshold_determination_multiclass2_tflite(tflite_model_path, test_generator)
    return optimal_thresholds


# %%
try:
    optimal_thresholds = evaluate_multiclass_with_adaptive_threshold_tflite("/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_0.tflite" , test_generator)
except StopIteration as e:
    print("iteration stopped. evaluating again.")
    optimal_thresholds = evaluate_multiclass_with_adaptive_threshold_tflite("/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_0.tflite" , test_generator)

# %% [markdown]
# ***QAT GENERIC KERAS MODEL RESOURCE UTILISATIONS AND INFERENCE***

# %%

import time
import psutil
import GPUtil
import pynvml
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.compat.v1.profiler import profile
from tensorflow.compat.v1.profiler import ProfileOptionBuilder
import keras

#best model
model=keras.models.load_model("/mnt/c/modelFiles/10splitsFlatten/greyColor_quantized_kfold_model_0.keras")

# Load a single test image
img_path = "/mnt/c/newTrain/Train/line/17.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0

# Stack into 1 channel
stacked = np.stack([img], axis=-1)  # shape: (256, 256, 1)
img_array = np.expand_dims(stacked, axis=0)  # shape: (1, 256, 256, 1)

# Cold start timing
t1 = time.time()
ensemble_probs = model.predict(img_array, verbose=0)
t2 = time.time()
cold_start_time = t2 - t1

# Inference timing
num_trials = 10
times = []
for _ in range(num_trials):
    t1 = time.time()
    _ = model.predict(img_array, verbose=0)
    t2 = time.time()
    times.append(t2 - t1)
inference_time = np.mean(times)

def get_model_size(model):
        """Get approximate model size in MB"""
        weights = [w.numpy() for w in model.weights]
        total_params = sum(w.size for w in weights)
        
        # Calculate size in MB (32-bit float = 4 bytes)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
# model size
model_size = get_model_size(model)

# Memory usage
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)
_ = np.mean([model.predict(img_array, verbose=0)], axis=0)
mem_after = process.memory_info().rss / (1024 * 1024)
memory_usage = mem_after - mem_before

# GPU and CPU stats
gpus = GPUtil.getGPUs()
gpu_usage = gpus[0].load * 100 if gpus else None
cpu_util = psutil.cpu_percent(interval=1)

# Power usage (CPU only)
try:
    power_usage = psutil.sensors_battery().power_plugged
except AttributeError:
    power_usage = None

# GPU power usage
def get_gpu_power():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    pynvml.nvmlShutdown()
    return power
power_watts = get_gpu_power()

# FLOPs
import io
from contextlib import redirect_stdout

def get_flops(model, input_shape):
    concrete = tf.function(lambda inputs: model(inputs)).get_concrete_function(
        tf.TensorSpec([1] + list(input_shape), model.inputs[0].dtype)
    )
    frozen_func = convert_variables_to_constants_v2(concrete)
    graph = frozen_func.graph

    run_meta = tf.compat.v1.RunMetadata()
    opts = ProfileOptionBuilder.float_operation()

    # Suppress stdout
    with io.StringIO() as buf, redirect_stdout(buf):
        flops = profile(graph, run_meta=run_meta, options=opts)
    
    return flops.total_float_ops

input_shape = (256, 256, 1)
flops = get_flops(models[0], input_shape) * len(models)  # total ensemble FLOPs

# Parameter count
num_params = model.count_params()

# Data load timing
t1 = time.time()
_ = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
t2 = time.time()
data_loading_time = t2 - t1

# Sparsity
def get_sparsity_ratio(model):
    total_params = np.sum([np.prod(w.shape) for w in model.weights])
    zero_params = np.sum([np.sum(w.numpy() == 0) for w in model.weights])
    return zero_params / total_params

avg_sparsity= get_sparsity_ratio(model)


# Energy efficiency
def get_energy_efficiency(flops, power_watts=None):
    if power_watts is None:
        return "Power consumption data missing"
    return flops / power_watts

energy_efficiency = get_energy_efficiency(flops, power_watts)

# results
print(f"Cold Start Time: {cold_start_time:.4f} seconds")
print(f"Average Inference Time: {inference_time:.4f} seconds")
print(f"Model Size: {model_size:.2f} MB")
print(f"Memory Usage: {memory_usage:.2f} MB")
print(f"Total FLOPs (Ensemble): {flops}")
print(f"Total Parameters (Ensemble): {num_params}")
print(f"GPU Utilization: {gpu_usage:.2f}%" if gpu_usage is not None else "GPU Utilization: Not Available")
print(f"CPU Utilization: {cpu_util:.2f}%")
print(f"Power Consumption (CPU): {power_usage}")
print(f"Power Watts (GPU): {power_watts}")
print(f"Data Loading Time: {data_loading_time:.4f} seconds")
print(f"Sparsity Ratio: {avg_sparsity:.4f}")
print(f"Energy Efficiency (FLOPs/Watt): {energy_efficiency}")


# %% [markdown]
# ***QUANTIZED TFLITE MODEL RESOURCE UTILISATIONS AND INFERENCE***

# %%

import time
import psutil
import GPUtil
import pynvml



model_paths = [
   f"/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_{i}.tflite" for i in range(10) 
]

# Load interpreters
interpreters = [tf.lite.Interpreter(model_path=path) for path in model_paths]
for interpreter in interpreters:
    interpreter.allocate_tensors()
    
#using first model
input_details = interpreters[0].get_input_details() 
output_details = interpreters[0].get_output_details()

# Preprocess image
img_path = "/mnt/c/newTrain/Train/line/17.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0

# Stack channels
stacked = np.stack([img], axis=-1)

# Quantize input
input_scale, input_zero_point = input_details[0]['quantization']
img_input = stacked / input_scale + input_zero_point
img_input = np.clip(img_input, -128, 127).astype(np.int8)
img_input = np.expand_dims(img_input, axis=0)

# Cold start time (for one model)
t1 = time.time()
interpreters[0].set_tensor(input_details[0]['index'], img_input)
interpreters[0].invoke()
_ = interpreters[0].get_tensor(output_details[0]['index'])
t2 = time.time()
cold_start_time = t2 - t1

# Inference time over 10 trials (ensemble)
times = []
for _ in range(10):
    t1 = time.time()
    predictions = []
    for interpreter in interpreters:
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
    _ = np.mean(predictions, axis=0)
    t2 = time.time()
    times.append(t2 - t1)
inference_time = np.mean(times)

# Model size of tflite model 
model_size = os.path.getsize("/mnt/c/modelFiles/10splitsFlatten/quantized_kfold_model_0.tflite") / (1024 * 1024)

# Memory usage
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)
for interpreter in interpreters:
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
mem_after = process.memory_info().rss / (1024 * 1024)
memory_usage = mem_after - mem_before

# GPU usage
gpus = GPUtil.getGPUs()
gpu_usage = gpus[0].load * 100 if gpus else None

# CPU usage
cpu_util = psutil.cpu_percent(interval=1)

# Power (battery)
try:
    power_usage = psutil.sensors_battery().power_plugged
except:
    power_usage = None

# GPU power
def get_gpu_power():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    pynvml.nvmlShutdown()
    return power

power_watts = get_gpu_power()

# Data loading time
t1 = time.time()
_ = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
t2 = time.time()
data_loading_time = t2 - t1

# Print results
print(f"Cold Start Time: {cold_start_time:.4f} s")
print(f"Average Inference Time: {inference_time:.4f} s")
print(f"average Model Size: {model_size:.2f} MB")
print(f"Memory Usage: {memory_usage:.2f} MB")
print(f"GPU Utilization: {gpu_usage:.2f}%" if gpu_usage else "GPU Utilization: Not Available")
print(f"CPU Utilization: {cpu_util:.2f}%")
print(f"Power Plugged In: {power_usage}")
print(f"GPU Power (W): {power_watts}")
print(f"Data Loading Time: {data_loading_time:.4f} s")


# %% [markdown]
# **ENDING CODES**

# %%
!pip freeze > req.txt


