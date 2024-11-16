##################################################################### 1)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

data_path = '/kaggle/input/phising-detection-dataset/Phising_Detection_Dataset.csv'
data = pd.read_csv(data_path)

if data.isnull().sum().sum() > 0:
    data = data.dropna()

X = data.drop('Phising', axis=1)
y = data['Phising']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
##################################################################### 2)
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import random

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

train_dir = '/kaggle/input/animal-dataset/animal_dataset_intermediate/train'
test_dir = '/kaggle/input/animal-dataset/animal_dataset_intermediate/test'

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
)
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
)

model_from_scratch = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax'),
])

model_from_scratch.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history_scratch = model_from_scratch.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

model_transfer = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax'),
])

model_transfer.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history_transfer = model_transfer.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
)

def predict_and_analyze(model, img_path, class_names):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    print("Prediction results:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i]*100:.2f}%")
    print(f"\nThe image belongs to '{predicted_class_name}' with a confidence of {confidence*100:.2f}%")
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_name} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

all_images = [img for img in os.listdir(test_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
random_images = random.sample(all_images, 10)

class_names = list(train_data.class_indices.keys())

for img_name in random_images:
    img_path = os.path.join(test_dir, img_name)
    print(f"\nProcessing image: {img_name}")
    print("\nResults from the model trained from scratch:")
    predict_and_analyze(model_from_scratch, img_path, class_names)
    print("\nResults from the transfer learning model:")
    predict_and_analyze(model_transfer, img_path, class_names)

def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history_scratch, 'Training from Scratch')
plot_history(history_transfer, 'Transfer Learning')
##################################################################### 3)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/spam-emails/spam.csv')
data['label'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
texts = data['Message'].values
labels = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vocab_size = 10000
max_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

embedding_dim = 200
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
#class_weights[0] *= 0.5
#class_weights[1] *= 1.5

model_scratch = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model_scratch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_scratch = model_scratch.fit(X_train_padded, y_train, epochs=5, validation_data=(X_test_padded, y_test), batch_size=64, class_weight=class_weights)
y_pred_scratch = (model_scratch.predict(X_test_padded) > 0.5).astype("int32")

embeddings_index = {}
with open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model_pretrained = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model_pretrained.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_pretrained = model_pretrained.fit(X_train_padded, y_train, epochs=5, validation_data=(X_test_padded, y_test), batch_size=64, class_weight=class_weights)
y_pred_pretrained = (model_pretrained.predict(X_test_padded) > 0.5).astype("int32")

print("Comparative results:")
print("From scratch model:")
print(classification_report(y_test, y_pred_scratch))
print("\nPretrained embeddings model:")
print(classification_report(y_test, y_pred_pretrained))

plt.plot(history_scratch.history['accuracy'], label='Scratch Training Accuracy')
plt.plot(history_pretrained.history['accuracy'], label='Pretrained Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def predict_spam(model, text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    return "Spam" if prediction > 0.5 else "Not Spam"

sample_text = input("Enter text for spam prediction: ")

print("\nPrediction using model with embeddings trained from scratch:")
print(predict_spam(model_scratch, sample_text))

print("\nPrediction using model with pretrained embeddings:")
print(predict_spam(model_pretrained, sample_text))