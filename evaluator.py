import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import spacy
import csv
import nltk
from nltk.corpus import brown, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import time
import textstat
from utils.read_data import text_tokenizer
import tensorflow as tf

import numpy as np
import spacy

# Load Spacy model (with POS tagging capabilities)
nlp = spacy.load("en_core_web_sm")
import readability
import numpy as np

nltk.download('brown')
set_words = set(brown.words())
nlp = spacy.load('en_core_web_sm')


class FeatureSet:
    def __init__(self, text, id, prompt_number, score):
        self.id = id
        self.prompt_number = prompt_number
        self.score = score
        self.raw_text = text
        self.raw_sentences = nltk.sent_tokenize(text)
        self.sentences = []
        self.words = []
        for sentence in self.raw_sentences:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            self.sentences.append(sentence)
            sent_words = nltk.word_tokenize(sentence)
            self.words.extend(sent_words)
        self.p = []
        self.p2 = []
        self.word_count = len(self.words)
        self.char_count = 0
        self.mean_word_length = 0
        self.word_length_variance = 0
        self.mean_sentence_length = 0
        self.sentence_length_variance = 0
        self.comma_and_prep = 0
        self.unique_words = 0
        self.spacy_clause_number = 0
        self.spacy_max_clauses_in_sentence = 0
        self.spacy_mean_clause_length = 0
        self.spacy_mean_clauses_per_sent = 0
        self.spelling_mistake_count = 0
        self.average_sentence_depth = 0
        self.average_leaf_depth = 0
        self.spacy_average_sentence_depth = 0
        self.spacy_average_leaf_depth = 0

        # readability
        self.syllable_count = 0
        self.flesch_reading_ease = 0
        self.flesch_kincaid_grade = 0
        self.fog_scale = 0
        self.smog = 0
        self.automated_readability = 0
        self.coleman_liau = 0
        self.linsear_write = 0
        self.dale_chall_readability = 0
        self.text_standard = 0

        # additional features
        self.stop_prop = 0
        self.punc_pos_proportions = {}
        self.positive_sentence_prop = 0
        self.negative_sentence_prop = 0
        self.neutral_sentence_prop = 0
        self.overall_positivity_score = 0
        self.overall_negativity_score = 0


    def get_readability_features(self):
        sent_tokens = text_tokenizer(self.raw_text, replace_url_flag=True, tokenize_sent_flag=True)
        sentences = [' '.join(sent) + '\n' for sent in sent_tokens]
        sentences = ''.join(sentences)
        self.syllable_count = textstat.syllable_count(sentences)
        self.flesch_reading_ease = textstat.flesch_reading_ease(sentences)
        self.flesch_kincaid_grade = textstat.flesch_kincaid_grade(sentences)
        self.fog_scale = textstat.gunning_fog(sentences)
        self.smog = textstat.smog_index(sentences)
        self.automated_readability = textstat.automated_readability_index(sentences)
        self.coleman_liau = textstat.coleman_liau_index(sentences)
        self.linsear_write = textstat.linsear_write_formula(sentences)
        self.dale_chall_readability = textstat.dale_chall_readability_score(sentences)
        self.text_standard = textstat.text_standard(sentences)


    def get_stopword_proportion(self):
        total_words = self.word_count
        removed = [word for word in self.words if word.lower() not in stopwords.words('english')]
        filtered_count = len(removed)
        self.stop_prop = filtered_count/total_words


    def get_word_sentiment_proportions(self):
        sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
        sentence_count = len(self.sentences)
        positive_sentences = 0
        negative_sentences = 0
        neutral_sentences = 0
        accumulative_sentiment = 0
        for sentence in self.sentences:
            ss = sentiment_intensity_analyzer.polarity_scores(sentence)
            if ss['compound'] > 0:
                positive_sentences += 1
            elif ss['compound'] < 0:
                negative_sentences += 1
            else:
                neutral_sentences += 1
            accumulative_sentiment += ss['compound']
        average_accumulative_sentiment = accumulative_sentiment / sentence_count

        self.positive_sentence_prop = positive_sentences / sentence_count
        self.negative_sentence_prop = negative_sentences / sentence_count
        self.neutral_sentence_prop = neutral_sentences / sentence_count
        if average_accumulative_sentiment > 0:
            self.overall_positivity_score = 1 - average_accumulative_sentiment
        elif average_accumulative_sentiment < 0:
            self.overall_negativity_score = 0 - average_accumulative_sentiment


    def spacy_parse(self):
        sentences = self.raw_sentences
        for sentence in sentences:
            self.p2.append(nlp(sentence))


    def calculate_mean_word_length(self):
        for word in self.words:
            self.char_count += len(word)
        self.mean_word_length = self.char_count/self.word_count


    def calculate_word_length_variance(self):
        squared_diff_sum = 0
        for word in self.words:
            diff = len(word) - self.mean_word_length
            squared_diff = diff * diff
            squared_diff_sum += squared_diff
        self.word_length_variance = squared_diff_sum / self.word_count


    def calculate_mean_sentence_length(self):
        self.mean_sentence_length = len(self.words) / len(self.sentences)


    def calculate_sentence_length_variance(self):
        squared_diff_sum = 0
        for sentence in self.sentences:
            sent_length = len(nltk.word_tokenize(sentence))
            diff = sent_length - self.mean_sentence_length
            squared_diff = diff * diff
            squared_diff_sum += squared_diff
        self.sentence_length_variance = squared_diff_sum / len(self.sentences)


    def count_punctuation_and_pos(self):
        punc_and_pos_count = \
        {
            ',': 0,
            '.': 0,
            'VB': 0,
            'JJR': 0,
            'WP': 0,
            'PRP$': 0,
            'VBN': 0,
            'VBG': 0,
            'IN': 0,
            'CC': 0,
            'JJS': 0,
            'PRP': 0,
            'MD': 0,
            'WRB': 0,
            'RB': 0,
            'VBD': 0,
            'RBR': 0,
            'VBZ': 0,
            'NNP': 0,
            'POS': 0,
            'WDT': 0,
            'DT': 0,
            'CD': 0,
            'NN': 0,
            'TO': 0,
            'JJ': 0,
            'VBP': 0,
            'RP': 0,
            'NNS': 0
        }
        tag_count = 0
        sentences = self.raw_sentences
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tags = nltk.pos_tag(words)
            for tag in tags:
                tag_count += 1
                if tag[1] in punc_and_pos_count.keys():
                    punc_and_pos_count[tag[1]] += 1
        self.comma_and_prep = punc_and_pos_count['IN'] + punc_and_pos_count[',']
        for key in punc_and_pos_count:
            self.punc_pos_proportions[key] = punc_and_pos_count[key] / tag_count


    def unique_word_count(self):
        word_counts = {}
        self.unique_words = 0
        for word in self.words:
            if word not in word_counts.keys():
                word_counts[word] = 1
            else:
                word_counts[word] += 1
        for w in word_counts:
            if word_counts[w] == 1:
                self.unique_words += 1


    def spacy_clause_count(self):
        clause_word_count = 0
        for parsed_sentence in self.p2:
            sentence_clause_count = 0
            for token in parsed_sentence:
                if token.dep_ == 'relcl':
                    self.spacy_clause_number += 1
                    sentence_clause_count += 1
                    this_clause = list(w.text_with_ws for w in token.subtree)
                    clause_word_count += len(this_clause)
            if sentence_clause_count > self.spacy_max_clauses_in_sentence:
                self.spacy_max_clauses_in_sentence = sentence_clause_count
        try:
            self.spacy_mean_clause_length = clause_word_count / self.spacy_clause_number
        except ZeroDivisionError:
            self.spacy_mean_clause_length = 0
        try:
            self.spacy_mean_clauses_per_sent = self.spacy_clause_number / len(self.sentences)
        except ZeroDivisionError:
            self.spacy_mean_clauses_per_sent = 0


    def spelling_mistakes(self):
        punctuation = set(string.punctuation)
        text = ''.join([w for w in self.raw_text.lower() if w not in punctuation])
        tokens = nltk.word_tokenize(text)
        self.spelling_mistake_count = len([word for word in tokens if word not in set_words and '@' not in word])


    def spacy_parser_depth(self):
        parser_depth_count = 0
        leaf_count = 0
        leaf_depth_count = 0
        for parsed_sentence in self.p2:
            root = []
            word_and_head = {}
            sentence_deepest_node = -1
            for token in parsed_sentence:
                word_and_head[token.idx] = token.head.idx
                if token.idx == token.head.idx:
                    root.append(token.idx)
            for word in word_and_head:
                leaf_count += 1
                current_word = word
                count = 0
                while current_word not in root:
                    count += 1
                    current_word = word_and_head[current_word]
                if count > sentence_deepest_node:
                    sentence_deepest_node = count
                leaf_depth_count += count
            parser_depth_count += sentence_deepest_node
            self.spacy_average_sentence_depth = parser_depth_count / len(self.sentences)
            self.spacy_average_leaf_depth = leaf_depth_count / leaf_count


def generate_linguistic_features(essay_content, essay_set):
    # Instantiate a FeatureSet object for a single essay
    feature_set = FeatureSet(essay_content, 0, essay_set, 0)
    
    # Run all feature calculations
    feature_set.get_readability_features()
    feature_set.calculate_mean_word_length()
    feature_set.calculate_word_length_variance()
    feature_set.calculate_mean_sentence_length()
    feature_set.calculate_sentence_length_variance()
    feature_set.count_punctuation_and_pos()
    feature_set.unique_word_count()
    feature_set.spacy_parse()
    feature_set.spacy_clause_count()
    feature_set.spelling_mistakes()
    feature_set.spacy_parser_depth()
    feature_set.get_stopword_proportion()
    feature_set.get_word_sentiment_proportions()

    # Collect features in a dictionary
    essay_features = {
        'item_id': feature_set.id,
        'prompt_id': feature_set.prompt_number,
        'mean_word': feature_set.mean_word_length,
        'word_var': feature_set.word_length_variance,
        'mean_sent': feature_set.mean_sentence_length,
        'sent_var': feature_set.sentence_length_variance,
        'ess_char_len': feature_set.char_count,
        'word_count': feature_set.word_count,
        'prep_comma': feature_set.comma_and_prep,
        'unique_word': feature_set.unique_words,
        'clause_per_s': feature_set.spacy_mean_clauses_per_sent,
        'mean_clause_l': feature_set.spacy_mean_clause_length,
        'max_clause_in_s': feature_set.spacy_max_clauses_in_sentence,
        'spelling_err': feature_set.spelling_mistake_count,
        'sent_ave_depth': feature_set.spacy_average_sentence_depth,
        'ave_leaf_depth': feature_set.spacy_average_leaf_depth,
        'automated_readability': feature_set.automated_readability,
        'linsear_write': feature_set.linsear_write,
        'stop_prop': feature_set.stop_prop,
        'positive_sentence_prop': feature_set.positive_sentence_prop,
        'negative_sentence_prop': feature_set.negative_sentence_prop,
        'neutral_sentence_prop': feature_set.neutral_sentence_prop,
        'overall_positivity_score': feature_set.overall_positivity_score,
        'overall_negativity_score': feature_set.overall_negativity_score,
        'score': feature_set.score
    }

    essay_features.update(feature_set.punc_pos_proportions)

    # Print the features and score for the single essay
    # print(essay_features)
    
    return essay_features

from utils.read_data import text_tokenizer

def create_readability_features(content):
    """
    Generates readability features for a single essay.
    
    Parameters:
    - content: str, the text content of the essay
    
    Returns:
    - dict, a dictionary containing the computed readability features
    """
    unwanted_features = [
        'paragraphs',
        'words',
        'characters',
        'sentences_per_paragraph',
        'words_per_sentence',
    ]

    # Tokenize sentences in the content
    sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
    sentences = [' '.join(sent) + '\n' for sent in sent_tokens]
    sentences = ''.join(sentences)
    
    # Calculate readability scores
    readability_scores = readability.getmeasures(sentences, lang='en')
    
    # Collect features in a dictionary, filtering out unwanted features
    features = {}
    for cat in readability_scores.keys():
        for subcat in readability_scores[cat].keys():
            if subcat not in unwanted_features:
                ind_score = readability_scores[cat][subcat]
                features[subcat] = ind_score
    
    return features


class Evaluator():
    def _init_(self):
        model_path = "model.h5"
        # Load the model from the specified path
        self.model = tf.keras.models.load_model(model_path)
        

    def load_glove_embeddings(glove_file_path):
        glove_embeddings = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vector = np.array(parts[1:], dtype='float32')
                glove_embeddings[word] = vector
        return glove_embeddings



    def generate_glove_embedding(text, glove_dim=50):
        """
        Generates the GloVe embedding for a given text.
        """
        glove_embeddings = load_glove_embeddings("glove.6B.50d.txt")  
        
        words = text.split()
        embedding = np.zeros(glove_dim)  # Initialize with zeros of GloVe dimension
        
        word_count = 0
        for word in words:
            if word in glove_embeddings:
                embedding += glove_embeddings[word]
                word_count += 1
        
        # Avoid division by zero; normalize if word_count > 0
        if word_count > 0:
            embedding /= word_count
        
        return embedding


    def generate_pos_embedding(text):
        """
        Generates a POS-tagging-based embedding for the given text.
        """
        doc = nlp(text)
        pos_counts = np.zeros(len(nlp.pipe_labels['tagger']))  # Zero array for each POS type

        # Count occurrences of each POS tag
        for token in doc:
            pos_idx = nlp.pipe_labels['tagger'].index(token.tag_) if token.tag_ in nlp.pipe_labels['tagger'] else None
            if pos_idx is not None:
                pos_counts[pos_idx] += 1

        # Normalize counts by total words
        total_words = len(doc)
        if total_words > 0:
            pos_counts /= total_words

        return pos_counts

        
    
        
    def evaluate(self, essay, essay_prompt):
        # Generate linguistic features for the essay
        linguistic_features = generate_linguistic_features(essay, essay_prompt)
        readability_features = create_readability_features(essay)
        
    
        # Generate GloVe and POS embeddings
        prompt_glove_embedding = generate_glove_embedding(essay_prompt)
        prompt_pos_embedding = generate_pos_embedding(essay)
        essay_pos_embedding = generate_pos_embedding(essay)
        
        # Combine all features
        combined_features = {
            'linguistic': linguistic_features,
            'readability': readability_features,
            'essay': essay_pos_embedding,
            'prompt_glove':prompt_glove_embedding,
            'prompt_pos':prompt_pos_embedding
        }
        
        # Format combined features as model input
        model_input = self.prepare_model_input(combined_features)
        
        # Run the model prediction
        prediction = self.model.predict(model_input)
        
        # Return the model's prediction
        return prediction
        
    def prepare_model_input(self, combined_features):
        # Extract individual feature arrays from the combined dictionary
        linguistic_features = combined_features['linguistic']
        readability_features = combined_features['readability']
        glove_embedding = combined_features['glove']
        pos_embedding = combined_features['pos']
        
        # Ensure all features are numpy arrays and have the correct shape
        # This example assumes features are 1D arrays; reshape as needed for your model.
        linguistic_features = np.array(linguistic_features).flatten()
        readability_features = np.array(readability_features).flatten()
        glove_embedding = np.array(glove_embedding).flatten()
        pos_embedding = np.array(pos_embedding).flatten()
        
        # Concatenate all features into a single input array
        model_input = np.concatenate([linguistic_features, readability_features, glove_embedding, pos_embedding])
        
        # Reshape the input if the model expects a specific input shape (e.g., (1, n_features) for batch dimension)
        model_input = model_input.reshape(1, -1)  # Shape (1, n_features) for single prediction
        
        return model_input


