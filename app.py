import tkinter as tk
from tkinter import scrolledtext, messagebox
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import nltk
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)

# Function to perform text classification
def classify_text(text):
    try:
        # Load your dataset or use a predefined corpus
        corpus = ["Text document 1", "Text document 2", "Text document 3"]
        labels = [0, 1, 0]  # Example labels

        # Preprocess the corpus
        preprocessed_corpus = [preprocess_text(doc) for doc in corpus]

        # Vectorize using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(preprocessed_corpus)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        # Train SVM classifier
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # Prepare input text for prediction
        input_text = preprocess_text(text)
        input_vector = vectorizer.transform([input_text])

        # Perform prediction
        prediction = clf.predict(input_vector)

        return prediction[0]  # Return the predicted class (0 or 1)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to handle button click event
def classify_text_click():
    input_text = input_text_area.get("1.0", tk.END).strip()
    if input_text:
        prediction = classify_text(input_text)
        if prediction is not None:
            result_text_area.config(state=tk.NORMAL)
            result_text_area.delete("1.0", tk.END)
            result_text_area.insert(tk.END, f"Predicted Class: {prediction}\n")
            result_text_area.config(state=tk.DISABLED)
        else:
            messagebox.showerror("Error", "An error occurred during classification.")
    else:
        messagebox.showwarning("Warning", "Please enter text to classify.")

# Create main application window
app = tk.Tk()
app.title("Text Classification with SVM")

# Input Text Area
input_label = tk.Label(app, text="Enter Text to Classify:")
input_label.pack(pady=10)

input_text_area = scrolledtext.ScrolledText(app, width=60, height=10)
input_text_area.pack(padx=20, pady=10)

# Classify Button
classify_button = tk.Button(app, text="Classify Text", command=classify_text_click)
classify_button.pack(pady=10)

# Result Text Area dwdwdw,awlcm
result_label = tk.Label(app, text="Classification Resultxcdvseg:")
result_label.pack(pady=10)

result_text_area = scrolledtext.ScrolledText(app, width=60, height=5, state=tk.DISABLED)
result_text_area.pack(padx=20, pady=10)

# Run the application
app.mainloop()
