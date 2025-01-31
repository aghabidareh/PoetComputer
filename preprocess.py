import re
from pathlib import Path
from configs import config

def clean_text(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[ـ\r\n\u200c]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess():
    raw_text = config.DATA_PATH.read_text(encoding='utf-8')
    cleaned = clean_text(raw_text)
    config.PROCESSED_DATA.write_text(cleaned, encoding='utf-8')

if __name__ == "__main__":
    preprocess()
