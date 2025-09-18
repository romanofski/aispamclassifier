import notmuch
import pandas
import nltk
import re
from transformers import AutoTokenizer
from datasets import Dataset

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

DATABASE = '/home/rjoost/Maildir'


def load_mail_data() -> list[tuple[str, str]]:
    db = notmuch.Database(DATABASE)

    # Example query: All emails
    query = db.create_query(
        'tag:spam or tag:archive')  # Or use 'tag:spam' for spam emails

    # Iterate over the emails in the query
    emails = []

    count = 0
    for msg in query.search_messages():
        #subject = msg.get_header('subject')
        body = [p.as_string() for p in msg.get_message_parts()]
        labels = list(msg.get_tags())
        classifier = 'spam' if 'spam' in labels else 'normal'
        count += 1
        emails.append((' '.join(body), classifier))

        if count > 500:
            break

    db.close()
    return emails


def preprocess_mails(emails: list[tuple[str, str]]) -> pandas.DataFrame:
    # Convert to DataFrame for convenience
    df = pandas.DataFrame(emails, columns=['body', 'classifier'])

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Preprocessing function to clean email text
    def preprocess_email(text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove non-alphanumeric characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords
        text = ' '.join(
            [word for word in text.split() if word not in stop_words])
        return text

    df['cleaned_body'] = df['body'].apply(preprocess_email)
    return df


def run_tfid_model(df: pandas.DataFrame):
    vectorizer = TfidfVectorizer(
        max_features=5000)  # Adjust features as needed
    X = vectorizer.fit_transform(df['cleaned_body'])

    # Labels
    y = df['classifier'].map({'normal': 0, 'spam': 1})  # Binary classification

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == '__main__':
    emails = load_mail_data()
    df = preprocess_mails(emails)
    run_tfid_model(df)
