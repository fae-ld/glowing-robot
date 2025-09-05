from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import torch, re, regex
import pandas as pd
from stqdm import stqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

class HFExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=16):
        self.batch_size = batch_size

        # Model 1: SentenceTransformer
        self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")

        # Model 2: Emotion classifier
        self.emotion_model_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            "SamLowe/roberta-base-go_emotions", 
            problem_type="multi_label_classification"
        ).to(device)

        self.emotions = [
            "admiration","amusement","anger","annoyance","approval","caring","confusion",
            "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
            "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
            "pride","realization","relief","remorse","sadness","surprise","neutral"
        ]
        
    def __clean_text__(self, text):
        text = str(text).lower()
        text = re.sub(r'@\w+', "@USER", text)
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = regex.sub(r'(\S)(\p{Emoji})', r'\1 \2', text)
        text = regex.sub(r'(\p{Emoji})(\S)', r'\1 \2', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = regex.sub(r'[^\p{L}\p{N}\s@\p{Emoji}]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['text'] = X['text'].apply(self.__clean_text__)
        texts = X["text"].tolist() # TODO: Ini harusnya translated text

        # --- Emotion features ---
        all_probs = []
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        for batch in stqdm(batches, desc="Processing emotion score in batches"):
            inputs = self.emotion_model_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = self.emotion_model(**inputs).logits
                probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
        
        final_probs = torch.cat(all_probs, dim=0).numpy()
        df_emotions = pd.DataFrame(final_probs, columns=self.emotions)

        # --- Embedding features ---
        embeddings = self.sentence_transformer.encode(texts, show_progress_bar=True)
        df_embeddings = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

        # --- Gabung ---
        df_out = pd.concat([df_emotions, df_embeddings], axis=1)
        return df_out