import re
import unicodedata
from bs4 import BeautifulSoup
import langid

KEEP_HEADERS = {"subject", "from", "to", "sent"}


def _remove_emojis_and_symbols(text: str) -> str:
    return re.sub(
        r'['
        '\U0001F600-\U0001F64F'  # emoticons
        '\U0001F300-\U0001F5FF'  # symbols & pictographs
        '\U0001F680-\U0001F6FF'  # transport & map symbols
        '\U0001F700-\U0001F77F'  # alchemical symbols
        '\U0001F780-\U0001F7FF'  # geometric symbols
        '\U0001F800-\U0001F8FF'  # supplemental arrows
        '\U0001F900-\U0001F9FF'  # supplemental symbols and pictographs
        '\U0001FA00-\U0001FA6F'  # chess symbols etc.
        '\U0001FA70-\U0001FAFF'  # extended symbols
        '\U00002702-\U000027B0'  # Dingbats
        '\U000024C2-\U0001F251'  # Enclosed characters
        '\uE000-\uF8FF'          # Private use area
        '\uD800-\uDFFF'          # Surrogates
        ']+',
        '',
        text
    )

def _detect_language(body: str) -> str:
    try:
        lang, confidence = langid.classify(body)
    except Exception:
        import pdb; pdb.set_trace() 
    return lang


def clean_mailcorpus(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    # Umlauts stay umlauts but normalise weird symbols
    text = unicodedata.normalize("NFKC", text)
    text = _remove_emojis_and_symbols(text)
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        # Ignore quoted text
        if re.match(r'^\s*>', line):
            continue
        # Ignore most headers as well
        match = re.match(r'^([A-Za-z\-]+):\s*(.*)', line)
        if match:
            header_name = match.group(1).lower()
            if header_name not in KEEP_HEADERS:
                continue
        clean_lines.append(line)

    text = ' '.join(clean_lines)
    # Normalise spacing
    text = re.sub(r'\s+', ' ', text).strip()
    language = _detect_language(text)
    return f'[{language}]{text}'
