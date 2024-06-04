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


analyzer = DataAnalyzer()
train_data = analyzer.load_data('data/BBC News Train.csv', is_train=True)
test_data = analyzer.load_data('data/BBC News Test.csv', is_train=False)
analyzer.perform_eda(train_data)


#%%
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
                           'three', 'best', 'tell', 'show', 'work', 'want', 'give', 'like', 'many', 'number'}

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
        # Preprocess the texts
        self.train_data = self.clean_data(self.train_data)
        self.train_data = self.preprocess_texts(self.train_data)
        if self.test_data is not None:
            self.test_data = self.clean_data(self.test_data)
            self.test_data = self.preprocess_texts(self.test_data)

cleaner = DataCleaner(analyzer)
cleaner.data_process()
cleaner.perform_eda(cleaner.train_data, target_column='Processed_Text')

# Basic class
import numpy as np
import spacy
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2, SelectKBest
from scipy.sparse import csr_matrix
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_regression

class FeatureExtractor:
    def __init__(self, cleaner):
        self.data = cleaner.train_data.copy()
        self.test_data = cleaner.test_data.copy()
        self.features = None
        self.W = None
        self.H = None
        self.model = None
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.create_label_mapping()

    def extract_features(self):
        raise NotImplementedError("Subclasses should implement this method!")

    def create_label_mapping(self):
        unique_labels = self.data['Category'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_mapping = {idx: label for idx, label in enumerate(unique_labels)}
        self.data['Category'] = self.data['Category'].map(self.label_mapping).astype(int)

    @staticmethod
    def visualize_features(features):
        if isinstance(features, csr_matrix):
            features = features.toarray()

        plt.figure(figsize=(10, 6))

        cmap = plt.cm.seismic

        norm = plt.Normalize(vmin=-np.max(np.abs(features)), vmax=np.max(np.abs(features)))
        plt.imshow(features, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
        plt.colorbar()
        plt.title('Feature Matrix Visualization')
        plt.xlabel('Features')
        plt.ylabel('Documents')
        plt.show()

    def feature_selection(self, labels, num_features=300):
        # Ensure there are no NaN or infinite values in labels
        labels = labels.dropna()


        num_features = min(num_features, self.features.shape[1])  # Ensure k does not exceed n_features
        selector = SelectKBest(f_classif, k=num_features)
        self.features = selector.fit_transform(self.features, labels)
        return self.features

    def apply_matrix_factorization(self, n_components=5, method='NMF'):
        if method == 'NMF':
            model = NMF(n_components=n_components, init='random', random_state=0)
        elif method == 'SVD':
            model = TruncatedSVD(n_components=n_components)
        else:
            raise ValueError("Unsupported factorization method.")

        self.W = model.fit_transform(self.features)
        self.H = model.components_
        self.model = model
        return self.W, self.H

# Subclass-TF-IDF
class TFIDFFeatureExtractor(FeatureExtractor):
    def __init__(self, data):
        super().__init__(data)
        self.vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, stop_words='english')

    def extract_features(self):
        self.features = self.vectorizer.fit_transform(self.data['Processed_Text'])
        return self.features

# Subclass-Embedding
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


tfidf_extractor = TFIDFFeatureExtractor(cleaner)
tfidf_features = tfidf_extractor.extract_features()
tfidf_extractor.visualize_features(tfidf_features)
#selected_features_tfidf = tfidf_extractor.feature_selection(tfidf_extractor.data['Category'])
#tfidf_extractor.visualize_features(selected_features_tfidf)
embedding_extractor = EmbeddingFeatureExtractor(cleaner)
embedding_features = embedding_extractor.extract_features()
embedding_extractor.visualize_features(embedding_features)

# selected_features_embedding = embedding_extractor.feature_selection(cleaner.train_data['Category'])
W1, H1 = tfidf_extractor.apply_matrix_factorization(method='SVD')

print("Matrix W1 (document-topic weights):", W1.shape)
print("Matrix H1 (topic-feature weights):", H1.shape)

tfidf_extractor.visualize_features(W1)
tfidf_extractor.visualize_features(H1)

W2, H2 = embedding_extractor.apply_matrix_factorization(method='SVD')

print("Matrix W2 (document-topic weights):", W2.shape)
print("Matrix H2 (topic-feature weights):", H2.shape)

embedding_extractor.visualize_features(W2)
embedding_extractor.visualize_features(H2)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import permutations
from kaggle.api.kaggle_api_extended import KaggleApi


class ModelTrainer:
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.model = self.extractor.model
        self.W = self.extractor.W
        self.H = self.extractor.H
        self.cluster_model = None
        self.label_mapping = self.extractor.label_mapping
        self.inverse_label_mapping = self.extractor.inverse_label_mapping
        self.cluster_to_label_mapping = {}  # 新增的映射变量
        self.performance_log = []

    def cluster_documents(self, n_clusters=5):
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_model.fit(self.W)
        return self.cluster_model.labels_

    @staticmethod
    def find_best_label_mapping(true_labels, predicted_labels):
        unique_labels = np.unique(true_labels)
        best_mapping = None
        best_accuracy = 0
        best_mapped_labels = None

        for perm in permutations(unique_labels):
            label_mapping = {i: perm[i] for i in range(len(unique_labels))}
            mapped_labels = [label_mapping[label] for label in predicted_labels]
            accuracy = accuracy_score(true_labels, mapped_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_mapping = label_mapping
                best_mapped_labels = mapped_labels

        return best_mapping, best_mapped_labels, best_accuracy

    @staticmethod
    def submit_to_kaggle_and_get_score(competition, filename, message):
        api = KaggleApi()
        api.authenticate()
        api.competition_submit(file_name=filename, competition=competition, message=message)

    def evaluate_performance(self, true_labels, plot=True):
        if true_labels is None:
            raise ValueError("True labels are not provided.")

        true_labels = true_labels.astype(int)
        predicted_labels = self.cluster_documents(n_clusters=5)
        self.cluster_to_label_mapping, mapped_labels, best_accuracy = self.find_best_label_mapping(true_labels,
                                                                                                   predicted_labels)

        conf_matrix = confusion_matrix(true_labels, mapped_labels)
        if plot:
            print("Best Accuracy:", best_accuracy)
            print("Confusion Matrix:\n", conf_matrix)
            print("Num of features:\n", self.H.shape[1])
            self.plot_confusion_matrix(conf_matrix, np.unique(true_labels))

        self.performance_log.append({
            'num_features': self.H.shape[1],
            'accuracy': best_accuracy
        })
        self.predict_new()
        return best_accuracy, conf_matrix

    def predict_batch(self, new_data):
        """Transform and predict labels for a batch of new data."""
        new_features = self.model.transform(new_data)
        predicted_labels = self.cluster_model.predict(new_features)
        return predicted_labels

    def predict_new(self):
        if self.extractor.test_data is None:
            raise ValueError("Test data is not available.")

        if isinstance(self.extractor, TFIDFFeatureExtractor):
            test_features = self.extractor.vectorizer.transform(self.extractor.test_data['Processed_Text'])
        elif isinstance(self.extractor, EmbeddingFeatureExtractor):
            test_features = np.array(
                [self.extractor.document_vector(text) for text in self.extractor.test_data['Processed_Text']])
        else:
            raise ValueError("Unsupported feature extractor type.")

        predicted_labels = self.predict_batch(test_features)

        numeric_labels = [self.cluster_to_label_mapping[label] for label in predicted_labels]

        predicted_categories = [self.extractor.inverse_label_mapping[label] for label in numeric_labels]

        submission = pd.DataFrame({
            'ArticleId': self.extractor.test_data['ArticleId'],
            'Category': predicted_categories
        })
        submission.to_csv('submission.csv', index=False)
        print("Submission file created: submission.csv")
        # self.submit_to_kaggle_and_get_score(competition='learn-ai-bbc', filename='submission.csv', message='Message')

    @staticmethod
    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def visualize_feature_matrices(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(self.W, cmap='viridis')
        plt.title('W Matrix (Document-Topic)')

        plt.subplot(1, 2, 2)
        sns.heatmap(self.H, cmap='viridis')
        plt.title('H Matrix (Topic-Term)')

        plt.show()

    def tune_hyperparameters(self, true_labels, num_features_list):
        best_accuracy = 0
        best_params = None

        for num_features in num_features_list:
            # Feature selection
            self.extractor.feature_selection(true_labels, num_features)

            # Apply matrix factorization
            self.W, self.H = self.extractor.apply_matrix_factorization(n_components=5, method='SVD')

            # Evaluate performance
            accuracy, _ = self.evaluate_performance(true_labels, n_clusters=5, plot=False)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = num_features

        print(f"Best params: {best_params} with accuracy: {best_accuracy}")
        return best_params, best_accuracy

    def plot_performance_log(self):
        num_features = [log['num_features'] for log in self.performance_log if 'num_features' in log]
        accuracies = [log['accuracy'] for log in self.performance_log if 'accuracy' in log]
        plt.figure(figsize=(10, 7))
        plt.plot(num_features, accuracies, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.title('Performance Across Different Number of Features')
        plt.grid(True)
        plt.show()

tfidf_trainer = ModelTrainer(tfidf_extractor)
true_labels = tfidf_extractor.data['Category']  # Accessing the true categories directly
tfidf_trainer.evaluate_performance(true_labels)
tfidf_trainer.predict_new()
embedding_trainer = ModelTrainer(embedding_extractor)
embedding_trainer.evaluate_performance(true_labels)
embedding_trainer.predict_new()

# Hyperparameter tuning
num_features_list = [4000, 3000, 2000, 1000, 500]
num_components_list = [5, 10, 15, 20]
best_params, best_accuracy = tfidf_trainer.tune_hyperparameters(true_labels, num_features_list, num_components_list)
tfidf_trainer.plot_performance_log()