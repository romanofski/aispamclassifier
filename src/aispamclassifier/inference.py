import argparse
import pathlib
import io
from email import policy
from email.parser import BytesParser
import sys
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup

from aispamclassifier.corpus import clean_mailcorpus

def clean_email(raw_email: str) -> str:
    text = BeautifulSoup(raw_email, "html.parser").get_text()
    text = re.sub(r'(?i)(On .* wrote:|From: .*|Sent: .*|To: .*|Subject: .*)',
                  '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_spam_or_ham(emailfile: io.BufferedReader, tokenizer:
                       AutoTokenizer, model:
                       AutoModelForSequenceClassification) -> str:
    # Set model to evaluation mode
    model.eval()

    msg = BytesParser(policy=policy.default).parse(emailfile)
    rawbody = msg.get_body(preferencelist=('plain', 'html'))
    if rawbody is None:
        print("Unable to extract body part", file=sys.stderr)
        sys.exit(1)

    cleaned = clean_mailcorpus(rawbody.get_content())
    print(f'{cleaned}')
    inputs = tokenizer(cleaned,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return "Spam" if predicted_class == 1 else "Normal"


def main():
    parser = argparse.ArgumentParser(description='classify mail/spam non spam')
    parser.add_argument('emailfile',
                        type=argparse.FileType('rb'))
    parser.add_argument('--modelpath', type=pathlib.Path, required=True)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    model = AutoModelForSequenceClassification.from_pretrained(args.modelpath)
    label = detect_spam_or_ham(args.emailfile, tokenizer=tokenizer, model=model)
    print(label)

if __name__ == '__main__':
    main()
