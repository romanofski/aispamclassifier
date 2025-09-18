import argparse
from email.message import EmailMessage
import sys
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup


def clean_email(raw_email: str) -> str:
    text = BeautifulSoup(raw_email, "html.parser").get_text()
    text = re.sub(r'(?i)(On .* wrote:|From: .*|Sent: .*|To: .*|Subject: .*)',
                  '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main(emailfile, modelpath: str):
    # Load tokenizer and model from saved directory
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForSequenceClassification.from_pretrained(modelpath)

    # Set model to evaluation mode
    model.eval()

    msg = EmailMessage()
    msg.set_content(emailfile.read())
    rawbody = msg.get_body(preferencelist=('plain', 'html'))
    if rawbody is None:
        print("Unable to extract body part", file=sys.stderr)
        sys.exit(1)

    cleaned = clean_email(rawbody.get_content())
    inputs = tokenizer(cleaned,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128)
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
