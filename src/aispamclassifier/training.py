import argparse

import torch
from torch.nn import CrossEntropyLoss
import pandas
import notmuch
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

from corpus import clean_mailcorpus


BASEMODEL = 'bert-base-multilingual-cased'

model = AutoModelForSequenceClassification.from_pretrained(BASEMODEL,
                                                           num_labels=2)
# Use BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
MAX_TRAINING = 1000

class TrainingData:
    emails: pandas.DataFrame
    spamcount: int
    hamcount: int

    def __init__(self, emails: list[tuple[str, str]], spamcount: int, hamcount: int):
        self.emails = self._preprocess_mails(emails)
        self.spamcount = spamcount
        self.hamcount = hamcount

    def _preprocess_mails(self, emails: list[tuple[str, str]]) -> pandas.DataFrame:
        # Convert to DataFrame for convenience
        df = pandas.DataFrame(emails, columns=['body', 'classifier'])

        df['cleaned_body'] = df['body'].apply(clean_mailcorpus)
        return df

    def get_tokenized_dataset(self):
        return tokenize_dataset(self.emails)

    def get_loss_function(self) -> CrossEntropyLoss:
        total = self.spamcount + self.hamcount
        weight_spam = total / (2 * self.spamcount)
        weight_ham = total / (2 * self.hamcount)

        class_weights = torch.tensor([weight_ham, weight_spam])
        device = torch.device("cpu")
        assigned_weights = class_weights.to(device)

        return CrossEntropyLoss(weight=assigned_weights)


def load_mail_data(database_path: str) -> TrainingData:
    with notmuch.Database(database_path) as db:

        # Example query: All emails
        query = db.create_query('*')  # Or use 'tag:spam' for spam emails

        # Iterate over the emails in the query
        emails = []

        spamcount = 0
        normalcount = 0
        for msg in query.search_messages():
            bytes = msg.get_message_parts()[0].get_payload(decode=True)
            body = bytes.decode('utf-8', errors='ignore')
            labels = list(msg.get_tags())
            classifier = 'spam' if 'spam' in labels else 'normal'

            if classifier == 'spam':
                spamcount += 1
            if classifier == 'normal':
                normalcount += 1

            emails.append((body, classifier))
            if spamcount + normalcount > MAX_TRAINING:
                print(
                    f'Gathered {normalcount} normal and {spamcount} spam mails'
                )
                break

        return TrainingData(emails, spamcount, normalcount)


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


class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_fn=CrossEntropyLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def create_deep_model_trainer(train_dataset, eval_dataset, loss_fn: CrossEntropyLoss) -> Trainer:

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

    trainer = WeightedTrainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics, loss_fn=loss_fn)
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
    parser = argparse.ArgumentParser(description='classify mail/spam non spam')
    parser.add_argument('database',
                        type=str,)
    args = parser.parse_args()

    data = load_mail_data(args.database)
    train_dataset, eval_dataset = data.get_tokenized_dataset()
    loss_fn = data.get_loss_function()

    trainer = create_deep_model_trainer(train_dataset, eval_dataset, loss_fn)
    trainer.train()
    trainer.save_model("bert-spam-classifier-final")

    metrics = trainer.evaluate()
    print(metrics)
