
from utils.read_data import text_tokenizer
import readability
import pickle
import numpy as np

features_data_file = 'data/allreadability.pickle'
features_object = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: []
}
unwanted_features = [
    'paragraphs',
    'words',
    'characters',
    'sentences_per_paragraph',
    'words_per_sentence',
]
final_array = None
data_file_path = 'data/training_set_rel3.tsv'
data = open(data_file_path, encoding="ISO-8859-1")
lines = data.readlines()
data.close()
for index, line in enumerate(lines[1:]):
    if index % 50 == 0:
        print(f"processed {index} essays")
    tokens = line.strip().split('\t')
    essay_id = int(tokens[0])
    essay_set = int(tokens[1])
    content = tokens[2].strip()
    score = tokens[6]
    sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
    sentences = [' '.join(sent) + '\n' for sent in sent_tokens]
    sentences = ''.join(sentences)
    
    # Readability scores are calculated for the combined sentences using the readability.getmeasures function.
    # The readability.getmeasures function is typically used to compute various readability scores and metrics 
    # for a given text. While the exact scores returned can depend on the version of the readability library 
    # and the specific configuration, the function generally provides a comprehensive set of readability measures. 
    readability_scores = readability.getmeasures(sentences, lang='en')
    """
    OrderedDict([('readability grades', OrderedDict([('Kincaid', 6.442726838586442), ("ARI", 8.22458771092009), 
    ('Coleman-Liau', 7.590968461318052), ('FleschReadingEase', 83.40513451130215), ('GunningFogIndex", 10.7354982489653), 
    ("LIX', 36.58086596625279), ("SMOGIndex', 9.582805886043833), ('RIX', 3.3333333333333335), ('DaleChallIndex", 7.9914553645335875)])), 
    ('sentence info', OrderedDict([('characters_per_word', 4.23782234957201), ('syll_per_word', 1.2263610315186246), 
    ('words_per_sentence', 19.3888888888889), ('sentences_per_paragraph', 18.8), (“type_token_ratio', 0.4899713467048711), 
    (“characters', 1479), ('syllables', 428), ('words', 349), ('wordtypes', 171), ('sentences', 18), (“paragraphs', 1), 
    ('long_words', 60), (“complex_words', 26), ('complex_words_dc', 75)])), ('word usage', OrderedDict([('tobeverb', 10), 
    ('auxverb', 4), ('conjunction', 14), (“pronoun', 45), (“preposition', 55), ('nominalization', 3)])), 
    ('sentence beginnings', OrderedDict([("pronoun', 2), ('interrogative', 2), ('article', 0), ('subordination', 3), 
    ('conjunction', 0), ('preposition', 0)]))])
    """
    print(readability_scores)
    
    features = [essay_id]
    for cat in readability_scores.keys():
        for subcat in readability_scores[cat].keys():
            if subcat not in unwanted_features:
                ind_score = readability_scores[cat][subcat]
                features.append(ind_score)
    features_object[essay_set].append(features)
for key in features_object.keys():
    features_object[key] = np.array(features_object[key])
    min_v, max_v = features_object[key].min(axis=0), features_object[key].max(axis=0)
    features = (features_object[key] - min_v) / (max_v - min_v)
    features = np.nan_to_num(features)
    features = features_object[key]
    features_object[key][:, 1:] = features[:, 1:]
    if isinstance(final_array, type(None)):
        final_array = features_object[key]
    else:
        final_array = np.vstack((final_array, features_object[key]))
with open(features_data_file, 'wb') as fp:
    pickle.dump(final_array, fp)
