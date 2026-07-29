
import json
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
    #Separate last word from each sentence
    if isinstance (sentences, str):
       sentences = [sentences]

    pairs = []
    for s in sentences:
        words = s.rstrip(".!?").split()
        #print(words)
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

def generate_cst_json(cst_triplets):
    # Unpack each field into its own list; zip(*cst_triplets) transposes rows into columns returns tuples
    clean_tuple, scramble_tuple, target_tuple = zip(*cst_triplets)

    #For downstream conversion into JSON files with explicit index labels. Keyed by matching string index so clean_dict[i], scramble_dict[i], and target_dict[i] always refer to the same example when the three JSON files are loaded separately
    clean_dict = {str(i): v for i, v in enumerate(clean_tuple)}
    scramble_dict = {str(i): v for i, v in enumerate(scramble_tuple)}
    target_dict = {str(i): v for i, v in enumerate(target_tuple)}

    

    # Write each to its own JSON file
    with open("clean.json", "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2, ensure_ascii=False)

    with open("scramble.json", "w", encoding="utf-8") as f:
        json.dump(scramble_dict, f, indent=2, ensure_ascii=False)
        
    
    with open("target.json", "w", encoding="utf-8") as f:
        json.dump(target_dict, f, indent=2, ensure_ascii=False)


def build_and_generate_cst_json(sentences):
   cst_triplets= build_cst_triplets(sentences)
   generate_cst_json(cst_triplets)