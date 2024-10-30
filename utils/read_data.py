# read_data.py
import pickle
import nltk
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils.general_utils import get_score_vector_positions


"""
url_replacer: Sets '<url>' as the replacement text for any detected URLs.
num_regex: Compiles a regex pattern to match numbers, with optional signs and decimal points.
ref_scores_dtype: Sets the reference scores' data type to int32.
MAX_SENTLEN and MAX_SENTNUM: Define maximum sentence length and number of sentences, potentially limiting input text processing.
pd.set_option('mode.chained_assignment', None): Disables chained assignment warnings in pandas. 
"""

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
MAX_SENTLEN = 50
MAX_SENTNUM = 100
pd.set_option('mode.chained_assignment', None)

# Defines the replace_url function to replace URLs in a string.
def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text

# Tokenizes string into individual words or symbols using nltk.word_tokenize.
# Loops through each token and its index in the tokens list and also accounts for NER.
def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens

# Removes any leading or trailing whitespace from sent. 
# it also splits the sentence into shorter sentences if it exceeds the max_sentlen.
def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        # Finds indices of tokens that match the split_keywords.
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            # Determines the approximate number of splits needed by dividing tokens by max_sentlen.     
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    return new_tokens


# Tokenizes text into sentences using regex to split on punctuation marks.
# Returns the list of tokenized sentences, split according to max_sentlength.
def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        return tokens

    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens

# Tokenizes text into sentences, with options to replace URLs, clean up punctuation, and optionally create a vocabulary.
def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    # Replaces repeated sequences of periods (...), question marks (??), and exclamation points (!!) with a single instance.
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text: 
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        return sent_tokens
    else:
        raise NotImplementedError

# Checks if a token represents a number using num_regex.
def is_number(token):
    return bool(num_regex.match(token))

# Builds a word vocabulary dictionary from text data in the training file.
def read_word_vocab(read_configs):
    vocab_size = read_configs['vocab_size']
    file_path = read_configs['train_path']
    word_vocab_count = {}

    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    for index, essay in enumerate(train_essays_list):
        content = essay['content_text']
        content = text_tokenizer(content, True, True, True)
        content = [w.lower() for w in content]
        for word in content:
            try:
                word_vocab_count[word] += 1
            except KeyError:
                word_vocab_count[word] = 1

    import operator
    sorted_word_freqs = sorted(word_vocab_count.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1

    word_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(word_vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        word_vocab[word] = index
        index += 1
    return word_vocab

# This function reads a list of essays and counts the occurrences of different part-of-speech tags.
# A dictionary pos_tags mapping each unique POS tag to an index, with <pad> and <unk> 
# reserved for padding and unknown tags, respectively.
def read_pos_vocab(read_configs):
    file_path = read_configs['train_path']
    pos_tags_count = {}

    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    for index, essay in enumerate(train_essays_list[:16]):
        content = essay['content_text']
        content = text_tokenizer(content, True, True, True)
        content = [w.lower() for w in content]
        tags = nltk.pos_tag(content)
        for tag in tags:
            tag = tag[1]
            try:
                pos_tags_count[tag] += 1
            except KeyError:
                pos_tags_count[tag] = 1

    pos_tags = {'<pad>': 0, '<unk>': 1}
    pos_len = len(pos_tags)
    pos_index = pos_len
    for pos in pos_tags_count.keys():
        pos_tags[pos] = pos_index
        pos_index += 1
    return pos_tags

# To read and return readability features from a binary file.
def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features

# To load linguistic features stored in a CSV file into a Pandas DataFrame.
# Handcrafted features get loaded here
def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df

# This function normalizes features in the DataFrame, excluding certain columns.
# Normalizes the relevant columns using Min-Max scaling.
def get_normalized_features(features_df):
    column_names_not_to_normalize = ['item_id', 'prompt_id', 'score']
    column_names_to_normalize = list(features_df.columns.values)
    for col in column_names_not_to_normalize:
        column_names_to_normalize.remove(col)
    final_columns = ['item_id'] + column_names_to_normalize
    normalized_features_df = None
    for prompt_ in range(1, 9):
        is_prompt_id = features_df['prompt_id'] == prompt_
        prompt_id_df = features_df[is_prompt_id]
        x = prompt_id_df[column_names_to_normalize].values
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_pd1 = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index = prompt_id_df.index)
        prompt_id_df[column_names_to_normalize] = df_temp
        final_df = prompt_id_df[final_columns]
        if normalized_features_df is not None:
            normalized_features_df = pd.concat([normalized_features_df,final_df],ignore_index=True)
        else:
            normalized_features_df = final_df
    return normalized_features_df


def read_essay_sets(essay_list, readability_features, normalized_features_df, pos_tags):
    out_data = {
        'essay_ids': [],
        'pos_x': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_tag_indices = []
        tag_indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                tags = nltk.pos_tag(sent)
                for tag in tags:
                    if tag[1] in pos_tags:
                        tag_indices.append(pos_tags[tag[1]])
                    else:
                        tag_indices.append(pos_tags['<unk>'])
                sent_tag_indices.append(tag_indices)
                tag_indices = []

        out_data['pos_x'].append(sent_tag_indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        if out_data['max_sentnum'] < len(sent_tag_indices):
            out_data['max_sentnum'] = len(sent_tag_indices)
    assert(len(out_data['pos_x']) == len(out_data['readability_x']))
    print(' pos_x size: {}'.format(len(out_data['pos_x'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essay_sets_word_flat(essay_list, readability_features, normalized_features_df, vocab):
    out_data = {
        'essay_ids': [],
        'words': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_essay_len': -1,
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                for word in sent:
                    if is_number(word):
                        indices.append(vocab['<num>'])
                    elif word in vocab:
                        indices.append(vocab[word])
                    else:
                        indices.append(vocab['<unk>'])
        out_data['words'].append(indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        if out_data['max_essay_len'] < len(indices):
            out_data['max_essay_len'] = len(indices)
    assert(len(out_data['words']) == len(out_data['readability_x']))
    print(' word_x size: {}'.format(len(out_data['words'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data

"""
For each essay, the function extracts the ID and prompt, computes the score vector, 
retrieves readability features, and normalizes additional features.
The essay content is tokenized into sentences and words, which are then processed to extract POS tags.
The function keeps track of the maximum sentence number and sentence length encountered.


Input Parameters:

essay_list: A list of essays to be processed.
readability_features: Array containing readability metrics for the essays.
normalized_features_df: DataFrame containing normalized features for the essays.
pos_tags: A dictionary mapping POS tags to their corresponding indices.

Output:
A dictionary out_data containing:

essay_ids: List of essay IDs.
pos_x: List of POS tag indices for each essay.
readability_x: List of readability features.
features_x: List of additional features.
data_y: List of scores for each essay.
prompt_ids: List of prompt IDs.
max_sentnum: Maximum number of sentences across essays.
max_sentlen: Maximum sentence length across essays.
"""
def read_essay_sets_word(essay_list, readability_features, normalized_features_df, vocab):
    out_data = {
        'essay_ids': [],
        'words': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_indices = []
        indices = []
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                for word in sent:
                    if is_number(word):
                        indices.append(vocab['<num>'])
                    elif word in vocab:
                        indices.append(vocab[word])
                    else:
                        indices.append(vocab['<unk>'])
                sent_indices.append(indices)
                indices = []
        out_data['words'].append(sent_indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        if out_data['max_sentnum'] < len(sent_indices):
            out_data['max_sentnum'] = len(sent_indices)
    assert(len(out_data['words']) == len(out_data['readability_x']))
    print(' word_x size: {}'.format(len(out_data['words'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data

# This function processes essays similarly but focuses on flattening the words into indices according to a vocabulary.
# Similar to read_essay_sets, but this function flattens each essay's words into a list of indices based on the vocabulary.
# Handles numerical values by assigning them a specific index (<num>) and uses <unk> for unknown words.
# The first function focuses on sentence-level POS tagging, while the second focuses on word-level indexing. 
# Both functions ensure that the extracted data is structured for further use, such as training machine learning models.
def read_essay_sets_single_score(essay_list, readability_features, normalized_features_df, pos_tags, attribute_name):
    out_data = {
        'pos_x': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        if attribute_name in essay.keys():
            y = int(essay[attribute_name])
            out_data['data_y'].append([y])
            item_index = np.where(readability_features[:, :1] == essay_id)
            item_row_index = item_index[0][0]
            item_features = readability_features[item_row_index][1:]
            out_data['readability_x'].append(item_features)
            feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
            feats_list = feats_df.values.tolist()[0][1:]
            out_data['features_x'].append(feats_list)
            sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
            sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

            sent_tag_indices = []
            tag_indices = []
            for sent in sent_tokens:
                length = len(sent)
                if length > 0:
                    if out_data['max_sentlen'] < length:
                        out_data['max_sentlen'] = length
                    tags = nltk.pos_tag(sent)
                    for tag in tags:
                        if tag[1] in pos_tags:
                            tag_indices.append(pos_tags[tag[1]])
                        else:
                            tag_indices.append(pos_tags['<unk>'])
                    sent_tag_indices.append(tag_indices)
                    tag_indices = []

            out_data['pos_x'].append(sent_tag_indices)
            out_data['prompt_ids'].append(essay_set)
            if out_data['max_sentnum'] < len(sent_tag_indices):
                out_data['max_sentnum'] = len(sent_tag_indices)
    assert(len(out_data['pos_x']) == len(out_data['readability_x']))
    print(' pos_x size: {}'.format(len(out_data['pos_x'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


# For each essay, the function extracts the ID, prompt, content, and specified score.
# It retrieves readability features and normalized features, then tokenizes the essay content into sentences and words.
# Each word is converted to an index based on the provided vocabulary, handling numbers and unknown words appropriately.
# The function keeps track of the maximum number of sentences and sentence length encountered.
def read_essay_sets_single_score_words(essay_list, readability_features, normalized_features_df, vocab, attribute_name):
    out_data = {
        'words': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id'])
        content = essay['content_text']
        if attribute_name in essay.keys():
            y = int(essay[attribute_name])
            out_data['data_y'].append([y])
            item_index = np.where(readability_features[:, :1] == essay_id)
            item_row_index = item_index[0][0]
            item_features = readability_features[item_row_index][1:]
            out_data['readability_x'].append(item_features)
            feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
            feats_list = feats_df.values.tolist()[0][1:]
            out_data['features_x'].append(feats_list)
            sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
            sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

            sent_indices = []
            indices = []
            for sent in sent_tokens:
                length = len(sent)
                if length > 0:
                    if out_data['max_sentlen'] < length:
                        out_data['max_sentlen'] = length
                    for word in sent:
                        if is_number(word):
                            indices.append(vocab['<num>'])
                        elif word in vocab:
                            indices.append(vocab[word])
                        else:
                            indices.append(vocab['<unk>'])
                    sent_indices.append(indices)
                    indices = []

            out_data['words'].append(sent_indices)
            out_data['prompt_ids'].append(essay_set)
            if out_data['max_sentnum'] < len(sent_indices):
                out_data['max_sentnum'] = len(sent_indices)
    assert(len(out_data['words']) == len(out_data['readability_x']))
    print(' words size: {}'.format(len(out_data['words'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


# This function retrieves readability features and linguistic features, normalizes the linguistic features, 
# and then loads training, validation, and test essays from specified paths.
# It calls read_essay_sets_word_flat for each dataset to process the essays and extract word-level features.
def read_essays_words_flat(read_configs, word_vocab):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_word_flat(
        train_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    dev_data = read_essay_sets_word_flat(
        dev_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    test_data = read_essay_sets_word_flat(
        test_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    return train_data, dev_data, test_data

# The process is essentially the same as in read_essays_words_flat, but it uses read_essay_sets_word to extract features.
def read_essays_words(read_configs, word_vocab):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_word(
        train_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    dev_data = read_essay_sets_word(
        dev_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    test_data = read_essay_sets_word(
        test_essays_list, readability_features, normalized_linguistic_features, word_vocab)
    return train_data, dev_data, test_data

# This function reads essays from configuration files and extracts linguistic and readability features.
# Retrieves readability and linguistic features from their respective paths.
# Normalizes the linguistic features.
# Loads training, development, and test essays from specified files.
# Calls read_essay_sets for each dataset to extract and structure the relevant features.
def read_essays(read_configs, pos_tags):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets(train_essays_list, readability_features, normalized_linguistic_features, pos_tags)
    dev_data = read_essay_sets(dev_essays_list, readability_features, normalized_linguistic_features, pos_tags)
    test_data = read_essay_sets(test_essays_list, readability_features, normalized_linguistic_features, pos_tags)
    return train_data, dev_data, test_data

# This function is similar to read_essays, but it focuses on extracting a single score attribute from the essays.
# The process is analogous to read_essays, but it calls read_essay_sets_single_score, which will process the essays to retrieve only the specified score.
def read_essays_single_score(read_configs, pos_tags, attribute_name):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_single_score(
        train_essays_list, readability_features, normalized_linguistic_features, pos_tags, attribute_name)
    dev_data = read_essay_sets_single_score(
        dev_essays_list, readability_features, normalized_linguistic_features, pos_tags, attribute_name)
    test_data = read_essay_sets_single_score(
        test_essays_list, readability_features, normalized_linguistic_features, pos_tags, attribute_name)
    return train_data, dev_data, test_data

# This function extends the functionality to extract word-level features along with a single score.
# The process mirrors the previous functions but calls read_essay_sets_single_score_words 
# to extract features at the word level and the specified score.
def read_essays_single_score_words(read_configs, word_vocab, attribute_name):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_single_score_words(
        train_essays_list, readability_features, normalized_linguistic_features, word_vocab, attribute_name)
    dev_data = read_essay_sets_single_score_words(
        dev_essays_list, readability_features, normalized_linguistic_features, word_vocab, attribute_name)
    test_data = read_essay_sets_single_score_words(
        test_essays_list, readability_features, normalized_linguistic_features, word_vocab, attribute_name)
    return train_data, dev_data, test_data
