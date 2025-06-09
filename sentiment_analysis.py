import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK data
nltk.download('movie_reviews')

def load_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    texts = [" ".join(doc) for doc, _ in documents]
    labels = [label for _, label in documents]
    return texts, labels

def preprocess_and_vectorize(texts):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_model(X_train, y_train):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def predict_sentiment(text, clf, vectorizer):
    X = vectorizer.transform([text])
    prediction = clf.predict(X)[0]
    return prediction

def main():
    print("Loading data...")
    texts, labels = load_data()
    print(f"Loaded {len(texts)} documents.")

    print("Preprocessing and vectorizing...")
    X, vectorizer = preprocess_and_vectorize(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    print("Training model...")
    clf = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(clf, X_test, y_test)

    # Example prediction
    example_text = "This movie was fantastic! I really enjoyed it."
    prediction = predict_sentiment(example_text, clf, vectorizer)
    print(f"Example text: {example_text}")
    print(f"Predicted sentiment: {prediction}")

    # Interactive user input for sentiment prediction
    user_text = input("Enter text to analyze sentiment: ")
    user_prediction = predict_sentiment(user_text, clf, vectorizer)
    print(f"Predicted sentiment: {user_prediction}")

if __name__ == "__main__":
    main()
