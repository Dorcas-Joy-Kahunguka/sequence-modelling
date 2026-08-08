from data_utils import load_json_dataset

ids, clean_sentences, scrambled_sentences, target_words = load_json_dataset("data/processed/clean.json", "data/processed/scramble.json","data/processed/target.json")
print ("ids", ids)
print ("clean_sentences", clean_sentences)
print ("scrambled_sentences", scrambled_sentences)
print ("target_words", target_words)