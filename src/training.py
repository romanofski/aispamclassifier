import notmuch
import pandas
import nltk
import re
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

DATABASE = '/home/rjoost/Maildir'

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           num_labels=2)
# Use BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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


def tokenize_dataset(df: pandas.DataFrame):
    df['classifier'] = df['classifier'].map({
        'normal': 0,
        'spam': 1
    })  # Binary classification
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenize text
    def tokenize(example):
        return tokenizer(example["body"],
                         truncation=True,
                         padding="max_length",
                         max_length=128)

    dataset = dataset.map(tokenize)
    dataset = dataset.rename_column("classifier", "labels")
    dataset.set_format(type="torch",
                       columns=["input_ids", "attention_mask", "labels"])

    # Train/test split
    split = dataset.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    return (train_dataset, eval_dataset)


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


def create_deep_model_trainer(train_dataset, eval_dataset) -> Trainer:

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    training_args = TrainingArguments(
        output_dir="./bert-spam-classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
    )

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)
    return trainer


def train_tfid_model(df: pandas.DataFrame):
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
    train_dataset, eval_dataset = tokenize_dataset(df)
    trainer = create_deep_model_trainer(train_dataset, eval_dataset)
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
