import re
import random
def scramble_word(word):
        match = re.match(r"^([a-zA-Z]+)(['\-a-zA-Z]*)([^a-zA-Z]*)$", word)
        if not match:
            return word,
        core, suffix = match.group(1)+ match.group(2), match.group(3)
        if len(core) <=3:
            return word
        else:
         first, last  = core[0], core[-1]
         middle       = list(core[1:-1])
         original_mid = middle[:]
        for i in range(10):
           random.shuffle(middle)
           if middle != original_mid:
                break
        return first + ''.join(middle) + last + suffix
    
    
def scramble_sentence(sentence):
        return ' '.join(scramble_word(w) for w in sentence.split())

