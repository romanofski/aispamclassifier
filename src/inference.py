import argparse
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def main(emailfile, modelpath: str):
    # Load tokenizer and model from saved directory
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForSequenceClassification.from_pretrained(modelpath)

    # Set model to evaluation mode
    model.eval()

    email = emailfile.read()
    # Tokenize
    inputs = tokenizer(email,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    print("Spam" if predicted_class == 1 else "Normal")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classify mail/spam non spam')
    parser.add_argument('emailfile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('--modelpath', type=str, required=True)

    args = parser.parse_args()
    main(**args.__dict__)
