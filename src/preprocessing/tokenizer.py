from transformers import AutoTokenizer

def get_tokenizer(model_name="nlpaueb/legal-bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)
