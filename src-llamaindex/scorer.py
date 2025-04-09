#Cross-Encoder scorer, utilizing BioBERT transformer model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as FNC

class CrossEncoderScorer:
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(self.device)
        self.model.eval()

    def score(self, query, chunks, max_length=512):
        #given a query and list of chunks, return relevance scores using the cross-encoder
        
        scores = []
        for chunk in chunks:
            inputs = self.tokenizer(query, chunk, return_tensors="pt", truncation=True,
                                    padding="max_length", max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                score = torch.sigmoid(logits).item()  # For binary classification style scoring
                scores.append(score)
        return scores


# if __name__ == "__main__":
#     scorer = CrossEncoderScorer()
#     query = "What caused the patient's kidney failure?"
#     chunks = [
#         "Patient was admitted for flu symptoms in 2017.",
#         "Chronic hypertension noted since 2015; poorly controlled.",
#         "Creatinine levels spiked significantly during hospitalization in 2023.",
#         "Physical therapy recommended for lower back pain."
#     ]

#     salience_scores = scorer.score(query, chunks)
#     for i, score in enumerate(salience_scores):
#         print(f"Chunk {i+1}: {score:.4f}")