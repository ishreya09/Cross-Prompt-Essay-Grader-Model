# read_data_pr.py is used to read the data for the prompt 
import pickle
import nltk
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils.general_utils import get_score_vector_positions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
MAX_SENTLEN = 50
MAX_SENTNUM = 100
pd.set_option('mode.chained_assignment', None)


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text

# Tokenizes the input string into words. It handles special cases like handling NER (starting with @).
def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens

# Shortens sentences into smaller segments if they exceed the max_sentlen. 
# It splits sentences based on specific keywords and length constraints.
def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
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

# Splits the input text into sentences and further tokenizes each sentence. It can also create a vocabulary if specified.
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

# This is the main function that combines URL replacement, text cleaning, and tokenization. 
# It processes the text to return either a list of sentences or 
# raises an error if sentence tokenization is not implemented.
def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
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

# Checks if a given token is a number using a regular expression.
def is_number(token):
    return bool(num_regex.match(token))

# Reads and constructs a vocabulary from the training essays. 
# It counts the frequency of each word and builds a vocabulary based on the specified size.
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

# Reads and constructs a vocabulary of part-of-speech (POS) tags from the training essays.
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

# Loads readability features from a specified file path.
def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features

# Loads linguistic features from a CSV file.
def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df

# Normalizes the linguistic features using Min-Max scaling.
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

# Reads prompts and associates each prompt with its part-of-speech (POS) tags.
# Steps:
# Initializes an output dictionary to store processed data.
# Iterates over each prompt in the provided prompt_list:
# Extracts the prompt ID and content.
# Tokenizes the content into sentences and words, converting them to lowercase.
# For each sentence, POS tags are generated using NLTK's pos_tag function.
# The indices of the corresponding POS tags (based on a provided pos_tags mapping) are stored in sent_tag_indices.
# Updates the maximum sentence length encountered.
# Appends the processed data (sentences and prompt IDs) to the output dictionary.
# Prints the size of the processed prompt POS data and returns it.
def read_pr_pos(prompt_list, pos_tags):
    out_data = {
        'prompt_pos': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for i in range(len(prompt_list)): 
        prompt_id = int(prompt_list['prompt_id'][i]) # prompt id
        content = prompt_list['prompt'][i]
        
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

        out_data['prompt_pos'].append(sent_tag_indices)
        out_data['prompt_ids'].append(prompt_id)
        if out_data['max_sentnum'] < len(sent_tag_indices):
            out_data['max_sentnum'] = len(sent_tag_indices)
    print(' prompt_pos size: {}'.format(len(out_data['prompt_pos'])))
    return out_data

# Reads prompts from a CSV file and processes them to obtain their POS tags.
def read_prompts_pos(prompt_file, pos_tags):
    prompt_list = pd.read_csv(prompt_file)
    prompt_data = read_pr_pos(prompt_list, pos_tags)
    return prompt_data

# Reads essays and extracts various features including readability, POS tags, and prompt-related information.
# Iterates over each essay in the essay_list:
# Extracts the essay ID, prompt ID, and content text.
# Constructs a target vector y_vector based on the scores present in the essay.
# Retrieves readability features associated with the essay ID.
# Normalizes features from the provided DataFrame.
# Obtains prompt words and POS data based on the associated prompt ID.
# Tokenizes the content of the essay into sentences and words, generating POS tags for each tokenized sentence.
# Updates maximum sentence number and length statistics.
def read_essay_sets_with_prompt_only_word_emb(essay_list, readability_features, normalized_features_df, prompt_data, prompt_pos_data, pos_tags):
    out_data = {
        'essay_ids': [],
        'pos_x': [],
        'prompt_words':[],
        'prompt_pos':[],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])
        essay_set = int(essay['prompt_id']) # prompt id
        content = essay['content_text']
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data['data_y'].append(y_vector)
        item_index = np.where(readability_features[:, :1] == essay_id)
        print("item_index:", item_index)

        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]
        out_data['features_x'].append(feats_list)
        
        prompt_index = prompt_data['prompt_ids'].index(essay_set)
        prompt = prompt_data['prompt_words'][prompt_index]
        out_data['prompt_words'].append(prompt)
        
        prompt_pos = prompt_pos_data['prompt_pos'][prompt_index]
        out_data['prompt_pos'].append(prompt_pos)
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
    assert(len(out_data['pos_x']) == len(out_data['prompt_words']))
    assert(len(out_data['pos_x']) == len(out_data['prompt_pos']))
    print(' pos_x size: {}'.format(len(out_data['pos_x'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data

# Reads essays and their associated prompts, extracting features for training, validation, and testing datasets.
# Process:
# Calls helper functions to obtain readability features, linguistic features, and normalized linguistic features.
# Loads training, validation (dev), and test essays from specified file paths in the read_configs dictionary.
# Utilizes the previously defined read_essay_sets_with_prompt_only_word_emb function to extract features for training, dev, and test sets.
# Returns the processed data for training, validation, and testing.
def read_essays_prompts(read_configs, prompt_data, prompt_pos_data, pos_tags):
    readability_features = get_readability_features(read_configs['readability_path'])
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    normalized_linguistic_features = get_normalized_features(linguistic_features)
    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_with_prompt_only_word_emb(train_essays_list, readability_features, normalized_linguistic_features, prompt_data, prompt_pos_data, pos_tags)
    dev_data = read_essay_sets_with_prompt_only_word_emb(dev_essays_list, readability_features, normalized_linguistic_features, prompt_data, prompt_pos_data, pos_tags)
    test_data = read_essay_sets_with_prompt_only_word_emb(test_essays_list, readability_features, normalized_linguistic_features, prompt_data, prompt_pos_data, pos_tags)
    return train_data, dev_data, test_data

# Reads prompts and tokenizes their content, converting words to their corresponding indices based on a provided vocabulary.
# Process:
# Initializes an output dictionary to store prompt words and IDs, along with maximum sentence number and length.
# Iterates over each prompt in the prompt_list:
# Extracts the prompt ID and content.
# Tokenizes the content into sentences and words, converting them to lowercase.
# For each tokenized sentence, converts words into indices using the provided vocabulary:
# Uses a special index for numbers, checks for known words, and assigns an unknown token index for any words not in the vocabulary.
# Appends the processed data (sentence indices and prompt IDs) to the output dictionary.
# Updates the maximum sentence number and length.
# Prints the size of the processed prompt words and returns the output dictionary.
def read_prompts_word(prompt_list, vocab):
    out_data = {
        'prompt_words': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for i in range(len(prompt_list)): 
        prompt_id = int(prompt_list['prompt_id'][i]) # prompt id
        content = prompt_list['prompt'][i]
        
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

        out_data['prompt_words'].append(sent_indices)
        out_data['prompt_ids'].append(prompt_id)
        if out_data['max_sentnum'] < len(sent_indices):
            out_data['max_sentnum'] = len(sent_indices)
    print(' prompt_words size: {}'.format(len(out_data['prompt_words'])))
    return out_data

# Purpose: Reads prompts from a CSV file and processes them to obtain their word indices based on the vocabulary.
# Process:
# Loads the prompts from the specified CSV file into a DataFrame.
# Calls read_prompts_word to process the DataFrame and return the prompt data with word indices.
def read_prompts_we(prompt_file, word_vocab):
    prompt_list = pd.read_csv(prompt_file)
    prompt_data = read_prompts_word(prompt_list, word_vocab)
    return prompt_data
