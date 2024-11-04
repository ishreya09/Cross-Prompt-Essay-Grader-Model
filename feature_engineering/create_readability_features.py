from utils.read_data import text_tokenizer
import readability
import pickle
import numpy as np

"""
Common Readability Scores

Flesch Reading Ease: A score that indicates how easy a text is to read, with higher scores indicating easier readability.
Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

Flesch-Kincaid Grade Level: Indicates the US school grade level required to understand the text.
Formula: 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59

Gunning Fog Index: Estimates the years of formal education needed to understand the text.
Formula: 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))

SMOG Index: Calculates the years of education needed to understand a piece of writing.
Formula: 1.0430 * sqrt(complex_sentences * (30 / total_sentences)) + 3.1291

Coleman-Liau Index: Estimates the school grade level needed to understand the text based on letter counts and sentence counts.
Formula: 0.0588 * L - 0.296 * S - 15.8, where L is the average number of letters per 100 words, and 
S is the average number of sentences per 100 words.

Dale-Chall Readability Score: Based on the familiarity of words and the number of difficult words used in the text.
Formula: Uses a specific list of familiar words and compares them to the overall vocabulary.

Additional Metrics

Total Words: The total number of words in the text.
Total Sentences: The total number of sentences.
Total Syllables: The total number of syllables in the text.
Average Words per Sentence: Calculated as total words divided by total sentences.
Percentage of Complex Words: The percentage of words with three or more syllables.

"""

def main():
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


if __name__ == "__main__":
    main()