from SentiCR import SentiCR
from SentiCR import tokenize_and_stem, stem_tokens
import pandas as pd

sentiment_analyzer = SentiCR()

df = pd.read_csv(
    "G:\Personal\Edu\Academics\Research-Works\datasets-cleaned\Combined_Data\Combined.test.csv")
sentences = df['text'].tolist()


for sent in sentences:
    score = sentiment_analyzer.get_sentiment_polarity(
        sent, "models/model_0_9.sav", "vectorizers/vectorizer_0_9.sav")
    print(sent+"\n Score: "+str(score))
