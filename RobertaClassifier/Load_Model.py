from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name from Hugging Face
model = 'FacebookAI/roberta-base'

# Path to save the model download from Hugging Face
cache_dir = './model'

# Download model and tokenizer to the path
AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model,cache_dir=cache_dir)