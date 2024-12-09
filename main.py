import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import TextVectorization
import tensorflow_hub as hub
from tensorflow.keras import layers

# Load dataset
df = pd.read_csv(r'D:\SMS_Spam_Detection_using_TensorFlow_in_Python\spam.csv')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'Text'})
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

# Visualize data
sns.countplot(x=df['label'])
plt.show()

# Analyze text data
avg_words_len = round(sum([len(i.split()) for i in df['Text']]) / len(df['Text']))
print(f"Average words per message: {avg_words_len}")

s = set()
for sent in df['Text']:
    for word in sent.split():
        s.add(word)
total_words_length = len(s)
print(f"Total unique words: {total_words_length}")

# Split data
X, y = np.asarray(df['Text']), np.asarray(df['label_enc'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer for baseline model
tfidf_vec = TfidfVectorizer().fit(X_train)
X_train_vec, X_test_vec = tfidf_vec.transform(X_train), tfidf_vec.transform(X_test)

# Baseline model using MultinomialNB
baseline_model = MultinomialNB()
baseline_model.fit(X_train_vec, y_train)

# TextVectorization setup
MAXTOKENS = total_words_length
OUTPUTLEN = avg_words_len

text_vec = TextVectorization(
    max_tokens=MAXTOKENS,
    standardize=tf.strings.lower,  # Lowercase and remove punctuation
    output_mode='int',
    output_sequence_length=OUTPUTLEN
)
text_vec.adapt(X_train)

# Custom-Embedding Model
embedding_layer = layers.Embedding(
    input_dim=MAXTOKENS,
    output_dim=128,
    embeddings_initializer='uniform',
    input_length=OUTPUTLEN
)

input_layer = layers.Input(shape=(1,), dtype=tf.string)
vec_layer = text_vec(input_layer)
embedding_layer_model = embedding_layer(vec_layer)
x = layers.GlobalAveragePooling1D()(embedding_layer_model)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)
model_1 = keras.Model(input_layer, output_layer)

model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Bidirectional-LSTM Model
model_2 = keras.Sequential([
    layers.Input(shape=(1,), dtype=tf.string),
    text_vec,
    layers.Embedding(input_dim=MAXTOKENS, output_dim=128, input_length=OUTPUTLEN),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Universal Sentence Encoder (USE) Model
tf.config.optimizer.set_jit(False)  # Ensure no jit optimizations conflict
use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, dtype=tf.string)

input_layer = keras.Input(shape=[], dtype=tf.string)
x = layers.Lambda(lambda x: use_layer(x), output_shape=(512,))(input_layer)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)
model_3 = keras.Model(input_layer, output_layer)
model_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training Function
def fit_model(model, epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return history

# Evaluation Function
def evaluate_model(model, X, y):
    y_preds = np.round(model.predict(X).flatten())
    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1-score': f1}

# Train models
baseline_model_results = evaluate_model(baseline_model, X_test_vec, y_test)
fit_model(model_1, epochs=5)
model_1_results = evaluate_model(model_1, X_test, y_test)
fit_model(model_2, epochs=5)
model_2_results = evaluate_model(model_2, X_test, y_test)
fit_model(model_3, epochs=5)
model_3_results = evaluate_model(model_3, X_test, y_test)

# Combine results
total_results = pd.DataFrame({
    'MultinomialNB Model': baseline_model_results,
    'Custom-Vec-Embedding Model': model_1_results,
    'Bidirectional-LSTM Model': model_2_results,
    'USE-Transfer learning Model': model_3_results
}).transpose()

print(total_results)
