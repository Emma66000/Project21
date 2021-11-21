from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")

model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")

help(model)

