import data_utils

sentences=data_utils.load_raw_sentences("data/raw/sentence-test-data.txt")
print(sentences)

cst_list = data_utils.build_cst_triplets(sentences)
print(cst_list)

data_utils.generate_cst_json(cst_list)

#data_utils.build_and_generate_cst_json(sentences)