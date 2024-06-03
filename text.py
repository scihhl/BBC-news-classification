import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


class DataAnalyzer:
    def __init__(self):
        self.train_data = None
        self.test_data = None

    def load_data(self, file_path, is_train=True):
        """
        Load data from a CSV file.

        :param file_path: str, path to the CSV file
        :param is_train: bool, True if loading training data, False otherwise
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(file_path)
        if is_train:
            self.train_data = data
        else:
            self.test_data = data
        return data

    @staticmethod
    def show_basic_info(data):
        """
        Display basic information and statistics of the dataset.

        :param data: DataFrame, input data
        """
        print("Basic Information:")
        print(data.info())
        print("\nStatistical Summary:")
        print(data.describe())
        print("\nCategory Distribution:")
        print(data['Category'].value_counts())

    @staticmethod
    def plot_category_distribution(data):
        """
        Plot the distribution of article categories.

        :param data: DataFrame, input data
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Category', data=data, order=data['Category'].value_counts().index)
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_text_length_distribution(data, target_column='Text'):
        """
        Plot the distribution of text lengths in the dataset.
        :param target_column: targeted column name
        :param data: DataFrame, input data
        """
        data['Text_Length'] = data[target_column].apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Text_Length'], bins=50, kde=True)
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def show_word_frequencies(data, target_column='Text'):
        """
        Display the most common words and generate a word cloud.
        :param target_column: targeted column name
        :param data: DataFrame, input data
        """
        all_text = " ".join(data[target_column])
        words = all_text.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(20)

        print("\nMost Common 20 Words:")
        for word, count in common_words:
            print(f"{word}: {count}")

        wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate_from_frequencies(
            word_counts)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud')
        plt.axis('off')
        plt.show()

    def perform_eda(self, data, target_column='Text'):
        """
        Perform exploratory data analysis by calling individual EDA functions.
        :param target_column: targeted column name
        :param data: DataFrame, input data
        """
        self.show_basic_info(data)
        self.plot_category_distribution(data)
        self.plot_text_length_distribution(data, target_column=target_column)
        self.show_word_frequencies(data, target_column=target_column)


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# redefine the class DataCleaner
import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


class DataCleaner(DataAnalyzer):
    def __init__(self, analyzer):
        super().__init__()
        self.__dict__.update(analyzer.__dict__)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def clean_data(data):
        """
        Clean the data by removing duplicates and missing values.

        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data_cleaned = data.dropna().drop_duplicates()
        return data_cleaned

    def preprocess_text(self, text):
        """
        Preprocess the text by converting to lowercase, removing stop words, punctuation, high-frequency non-informative words, and applying lemmatization.

        :param text: str, input text
        :return: str, processed text
        """
        high_freq_words = {'said', 'mr', 'also', 'would', 'could', 'u', 'say', 'year', 'you', 'go', 'come', 'last',
                           'first', 'time', 'make', 'use', 'take', 'get', 'new', 'people', 'good', 'one', 'two',
                           'three', 'best', 'tell', 'show', 'work', 'want'}
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(tag)) for token, tag in pos_tags]

        tokens = [word for word in tokens if word not in self.stop_words and word not in high_freq_words]
        return " ".join(tokens)

    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess_texts(self, data):
        """
        Apply text preprocessing to each text in the DataFrame.

        :param data: DataFrame, input data
        :return: DataFrame, data with processed texts
        """
        data['Processed_Text'] = data['Text'].apply(self.preprocess_text)
        return data

    def data_process(self):
        """
        Perform data cleaning and text preprocessing.
        """
        # Clean the data
        self.train_data = self.clean_data(self.train_data)
        # Preprocess the texts
        self.train_data = self.preprocess_texts(self.train_data)
        if self.test_data is not None:
            self.test_data = self.clean_data(self.test_data)
            self.test_data = self.preprocess_texts(self.test_data)


# Basic class
import numpy as np
import spacy
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest


class FeatureExtractor:
    def __init__(self, data):
        self.data = data
        self.features = None

    def extract_features(self):
        raise NotImplementedError("Subclasses should implement this method!")

    def visualize_features(self):
        if self.features is not None:
            plt.figure(figsize=(10, 6))
            plt.spy(self.features)
            plt.title('Feature Matrix Visualization')
            plt.xlabel('Features')
            plt.ylabel('Documents')
            plt.show()
        else:
            print("No features to visualize.")

    def feature_selection(self, labels, num_features=500):
        selector = SelectKBest(chi2, k=num_features)
        selected_features = selector.fit_transform(self.features, labels)
        return selected_features

    def apply_matrix_factorization(self, n_components=5, method='NMF'):
        if method == 'NMF':
            model = NMF(n_components=n_components, init='random', random_state=0)
        elif method == 'SVD':
            model = TruncatedSVD(n_components=n_components)
        else:
            raise ValueError("Unsupported factorization method.")

        W = model.fit_transform(self.features)
        H = model.components_
        return W, H

    # Subclass
class TFIDFFeatureExtractor(FeatureExtractor):
    def __init__(self, data):
        super().__init__(data)
        self.vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, stop_words='english')

    def extract_features(self):
        self.features = self.vectorizer.fit_transform(self.data['Processed_Text'])
        return self.features

    # 子类 - Embedding
class EmbeddingFeatureExtractor(FeatureExtractor):
    def __init__(self, data):
        super().__init__(data)
        self.nlp = spacy.load('en_core_web_lg')

    def document_vector(self, doc):
        """Generate document vectors by averaging word vectors, handling cases where no word has a vector."""
        doc = self.nlp(doc)
        vectors = [word.vector for word in doc if not word.is_stop and word.has_vector]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            # Return a zero vector if no word has a vector
            return np.zeros((300,))  # 300 is the dimensionality of GloVe vectors in 'en_core_web_lg'

    def extract_features(self):
        """Use document_vector method to create feature matrix."""
        self.features = np.array([self.document_vector(text) for text in self.data['Processed_Text']])
        return self.features


if __name__ == '__main__':
    analyzer = DataAnalyzer()
    train_data = analyzer.load_data('data/BBC News Train.csv', is_train=True)
    test_data = analyzer.load_data('data/BBC News Test.csv', is_train=False)
    analyzer.perform_eda(train_data)

    cleaner = DataCleaner(analyzer)
    cleaner.data_process()
    cleaner.perform_eda(cleaner.train_data, target_column='Processed_Text')

    tfidf_extractor = TFIDFFeatureExtractor(cleaner.train_data)
    tfidf_features = tfidf_extractor.extract_features()

    tfidf_extractor.visualize_features()
    selected_features = tfidf_extractor.feature_selection(cleaner.train_data['Category'])
    W, H = tfidf_extractor.apply_matrix_factorization()

    print("Matrix W (document-topic weights):", W.shape)
    print("Matrix H (topic-feature weights):", H.shape)

    embedding_extractor = EmbeddingFeatureExtractor(cleaner.train_data)
    embedding_features = embedding_extractor.extract_features()
    embedding_extractor.visualize_features()