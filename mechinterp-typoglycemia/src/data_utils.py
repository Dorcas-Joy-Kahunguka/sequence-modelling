
import json
from src import scramble_utils 
from scramble_utils import scramble_sentence   

def load_raw_sentences(path):
 with open(path, "r", encoding="utf-8") as file:
   sentences = [line.strip() for line in file if line.strip()]
   return sentences

def load_prose_text(path):
   with open(path, "r", encoding="utf-8") as file:
    prose_text = file.read()
    return prose_text
   
def split_context_target(sentences):
    
    if isinstance (sentences, str):
       sentences = [sentences]

    pairs = []

    #Separate last word from each sentence
    for s in sentences:
        words = s.rstrip(".!?").split()
        target = words[-1]
        context = " ".join(words[:-1])
        
        pairs.append((context, target))
        
    return pairs

def build_cst_triplets(sentences):
    #For each sentence, produce a clean/scrambled pair and the target last word
    cst_triplets = []
    for context, target in split_context_target(sentences):
        scrambled_context = scramble_sentence(context)
        cst_triplets.append({
            "clean_context": context,
            "scrambled_context": scrambled_context,
            "target": target
        })
        
    return cst_triplets

def split_by_category(cst_triplets):
   # Unpack each field into its own list; zip(*cst_triplets) transposes rows into columns returns tuples
       clean_tuple, scrambled_tuple, target_tuple = zip(*(dict_entry.values() for dict_entry in cst_triplets))
       
   
       clean_dict = {str(i): v for i, v in enumerate(clean_tuple)}
       scrambled_dict = {str(i): v for i, v in enumerate(scrambled_tuple)}
       target_dict = {str(i): v for i, v in enumerate(target_tuple)}

       return clean_dict, scrambled_dict, target_dict

def generate_cst_json(cst_triplets):
    
    clean_dict, scrambled_dict,target_dict = split_by_category(cst_triplets)

    # Write each to its own JSON file
    with open("data/processed/clean.json", "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2, ensure_ascii=False)

    with open("data/processed/scramble.json", "w", encoding="utf-8") as f:
        json.dump(scrambled_dict, f, indent=2, ensure_ascii=False)
        
    
    with open("data/processed/target.json", "w", encoding="utf-8") as f:
        json.dump(target_dict, f, indent=2, ensure_ascii=False)

def build_and_generate_cst_json(sentences):
   cst_triplets= build_cst_triplets(sentences)
   generate_cst_json(cst_triplets)

def align_by_id(clean_dict, scrambled_dict, target_dict):
   ids = sorted(set(clean_dict) & set(scrambled_dict) & set(target_dict))
   missing = (set(clean_dict) | set(scrambled_dict) | set(target_dict)) - set(ids)
   if missing:
    print(f"Warning: {len(missing)} ids not present in all three files, skipping them.")

   clean_sentences = [clean_dict[i] for i in ids]
   scrambled_sentences = [scrambled_dict[i] for i in ids]
   target_words = [target_dict[i] for i in ids]

   return ids, clean_sentences, scrambled_sentences, target_words 

def build_dataset(cst_triplets):
   
   clean_dict, scrambled_dict,target_dict = split_by_category(cst_triplets)
   ids, clean_sentences, scrambled_sentences, target_words = align_by_id(clean_dict, scrambled_dict,target_dict)

   return ids, clean_sentences, scrambled_sentences, target_words
    
def load_json_dataset(clean_path, scrambled_path, target_path):
   with open(clean_path) as f:
        clean_processed = json.load(f)
   with open(scrambled_path) as f:
        scrambled_processed = json.load(f)
   with open(target_path) as f:
        target_processed = json.load(f)

   # Compress list of dictionaries into a dictionary. 
   clean_by_id = {r["id"]: r["text"] for r in clean_processed}
   scrambled_by_id = {r["id"]: r["text"] for r in scrambled_processed}
   target_by_id = {r["id"]: r["word"] for r in target_processed}

   ids, clean_sentences, scrambled_sentences, target_words = align_by_id(clean_by_id, scrambled_by_id,target_by_id)

   return ids, clean_sentences, scrambled_sentences, target_words

