from stud.implementation import build_model_34

input = {'words': ['According', 'to', 'information', 'provided', 'by', 'the', 'Romanian', 'Government', ',', 'the', 'military', 'prosecutor', 'decided', 'not', 'to', 'prosecute', 'the', 'police', 'officers', '.'], 'lemmas': ['accord', 'to', 'information', 'provide', 'by', 'the', 'Romanian', 'Government', ',', 'the', 'military', 'prosecutor', 'decide', 'not', 'to', 'prosecute', 'the', 'police', 'officer', '.'], 'pos_tags': ['VERB', 'ADP', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT', 'DET', 'ADJ', 'NOUN', 'VERB', 'PART', 'PART', 'VERB', 'DET', 'NOUN', 'NOUN', 'PUNCT'], 'dependency_heads': [3, 1, 13, 3, 8, 8, 8, 4, 3, 12, 12, 13, 0, 16, 16, 13, 19, 19, 16, 13], 'dependency_relations': ['case', 'fixed', 'obl', 'acl', 'case', 'det', 'amod', 'obl:agent', 'punct', 'det', 'amod', 'nsubj', 'root', 'advmod', 'mark', 'xcomp', 'det', 'compound', 'obj', 'punct'], 'predicates': ['HARMONIZE', '_', '_', 'LOAD_PROVIDE_CHARGE_FURNISH', '_', '_', '_', '_', '_', '_', '_', '_', 'DECIDE_DETERMINE', '_', '_', 'ACCUSE', '_', '_', '_', '_']}

model = build_model_34("EN","")

predictions = model.predict(input)
print(predictions)