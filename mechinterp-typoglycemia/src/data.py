# data.py

from scramble import scramble_sentence   

#def load_raw_sentences(path):
 #   """Read raw sentences from csv/txt"""
    

def split_context_target(sentences):
    """Separate last word from each sentence"""
    pairs = []
    for s in sentences:
        words = s.rstrip(".").split()
        print(words)
        target = words[-1]
        context = " ".join(words[:-1])
        
        pairs.append((context, target))
    return pairs

def build_scrambled_pairs(sentences):
    """For each sentence, produce a clean/scrambled pair"""
    pairs = []
    for context, target in split_context_target(sentences):
        scrambled_context = scramble_sentence(context)
        pairs.append({
            "clean_context": context,
            "scrambled_context": scrambled_context,
            "target": target
        })
        
    return pairs

"""def load_pairs(path):
    #Main entry point — call this from other scripts
    sentences = load_raw_sentences(path)
    return build_scrambled_pairs(sentences)"""


