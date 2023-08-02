import nltk
import sys
import os
import string
import math
nltk.download('punkt')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")

    file_contents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                file_contents[filename] = file.read()

    return file_contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenize the document using nltk's word_tokenize
    words = nltk.tokenize.word_tokenize(document)

    # Lowercase all the words
    words = [word.lower() for word in words]

    # Filter out punctuation
    words = [word for word in words if word not in string.punctuation]

    # Filter out stopwords
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Count the number of documents
    num_documents = len(documents)

    # Initialize a dictionary to store the document frequency for each word
    doc_frequency = {}

    # Iterate through each document and update the document frequency
    for doc_words in documents.values():
        unique_words = set(doc_words)
        for word in unique_words:
            doc_frequency[word] = doc_frequency.get(word, 0) + 1

    # Calculate the IDF for each word
    idfs = {}
    for word, frequency in doc_frequency.items():
        idf = math.log(num_documents / frequency)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Calculate the TF-IDF scores for each file
    file_scores = {}
    for filename, file_words in files.items():
        tf_idf_sum = 0
        for word in query:
            if word in file_words:
                tf_idf_sum += file_words.count(word) * idfs.get(word, 0)
        file_scores[filename] = tf_idf_sum

    # Sort files by their TF-IDF scores in descending order
    sorted_files = sorted(file_scores.items(), key=lambda item: item[1], reverse=True)

    # Return the top n filenames
    top_n_files = [filename for filename, _ in sorted_files[:n]]
    return top_n_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Calculate the matching word measure and query term density for each sentence
    sentence_scores = {}
    for sentence, words in sentences.items():
        matching_word_measure = sum(idfs.get(word, 0) for word in query if word in words)
        query_term_density = sum(1 for word in words if word in query) / len(words)
        sentence_scores[sentence] = (matching_word_measure, query_term_density)

    # Sort sentences by their matching word measure and query term density in descending order
    sorted_sentences = sorted(
        sentence_scores.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True
    )

    # Return the top n sentences
    top_n_sentences = [sentence for sentence, _ in sorted_sentences[:n]]
    return top_n_sentences


if __name__ == "__main__":
    main()
