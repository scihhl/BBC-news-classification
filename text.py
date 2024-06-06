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

# redefine the class DataCleaner
import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import nltk

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
print(cleaner.test_data)
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.sparse import csr_matrix


class FeatureExtractor:
    def __init__(self, cleaner, trim_features=None):
        self.data = cleaner.train_data.copy()
        self.test_data = cleaner.test_data.copy()
        self.trim_features = trim_features
        self.features = None
        self.W = None
        self.H = None
        self.model = None
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.create_label_mapping()
        self.selector = None
        self.test_features = None

    @staticmethod
    def visualize_features(features):
        if isinstance(features, csr_matrix):
            features = features.toarray()

        plt.figure(figsize=(10, 6))

        cmap = plt.cm.seismic

        norm = plt.Normalize(vmin=-np.max(np.abs(features)), vmax=np.max(np.abs(features)))
        plt.imshow(features, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
        plt.colorbar()
        plt.title('Matrix Visualization')
        if features.shape[0] == 5:
            plt.xlabel('Features')
            plt.ylabel('Categories')
        elif features.shape[1] == 5:
            plt.xlabel('Categories')
            plt.ylabel('Documents')
        else:
            plt.xlabel('Features')
            plt.ylabel('Documents')
        plt.show()

    def extract_features(self):
        raise NotImplementedError("Subclasses should implement this method!")

    def trim(self):
        if self.features.shape[1] > self.trim_features:
            self.selector = SelectKBest(f_classif, k=self.trim_features)
            self.features = self.selector.fit_transform(self.features, self.data['Category'])

    def process_test_data(self):
        if self.trim_features:
            self.test_features = self.selector.transform(self.test_features)
        self.W_test = self.model.transform(self.test_features)

    def create_label_mapping(self):
        unique_labels = self.data['Category'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_mapping = {idx: label for idx, label in enumerate(unique_labels)}
        self.data['Category'] = self.data['Category'].map(self.label_mapping).astype(int)

    def SVD_fit(self, features, n_components=5):
        self.model = TruncatedSVD(n_components=n_components)
        self.W = self.model.fit_transform(features)
        self.H = self.model.components_


# Subclass-TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor(FeatureExtractor):
    def __init__(self, cleaner, trim_features=None):
        super().__init__(cleaner, trim_features=trim_features)

    def extract_features(self):
        self.vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, stop_words='english')
        self.features = self.vectorizer.fit_transform(self.data['Processed_Text']).toarray()
        self.test_features = self.vectorizer.transform(self.test_data['Processed_Text']).toarray()
        if self.trim_features is not None:  # 根据 trim 参数决定是否裁剪特征
            self.trim()
        self.SVD_fit(self.features)
        self.process_test_data()
        return self.features


# Subclass-Embedding
import spacy


class EmbeddingFeatureExtractor(FeatureExtractor):
    def __init__(self, cleaner, trim_features=None):
        super().__init__(cleaner, trim_features=trim_features)
        self.nlp = spacy.load('en_core_web_lg')

    def document_vector(self, doc):
        """Generate document vectors by averaging word vectors, handling cases where no word has a vector."""
        doc = self.nlp(doc)
        vectors = [word.vector for word in doc if not word.is_stop and word.has_vector]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros((300,))

    def extract_features(self):
        """Use document_vector method to create feature matrix."""
        self.features = np.array([self.document_vector(text) for text in self.data['Processed_Text']])
        self.test_features = np.array([self.document_vector(text) for text in self.test_data['Processed_Text']])
        if self.trim_features is not None:
            self.trim()
        self.SVD_fit(self.features)
        self.process_test_data()
        return self.features


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi


class ModelTrainer:
    def __init__(self, feature_extractor):
        self.features = feature_extractor.features
        self.W = feature_extractor.W
        self.H = feature_extractor.H
        self.svd_model = feature_extractor.model
        self.label_mapping = feature_extractor.label_mapping
        self.inverse_label_mapping = feature_extractor.inverse_label_mapping
        self.true_labels = feature_extractor.data['Category']
        self.data = feature_extractor.data
        self.test_data = feature_extractor.test_data
        self.vectorizer = getattr(feature_extractor, 'vectorizer', None)
        self.document_vector = getattr(feature_extractor, 'document_vector', None)
        self.W_test = feature_extractor.W_test

        self.cluster_model = None
        self.performance_log = []
        self.mapping = None
        self.accuracy = None
        self.test_pred = None
        self.train_pred = None

    def train_model(self, plot=False):
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
        self.cluster_model.fit(self.W)

        predicted_labels_train = self.cluster_model.labels_
        self.mapping, mapped_labels, self.accuracy = self.find_best_label_mapping(self.true_labels,
                                                                                  predicted_labels_train)
        self.train_pred = [self.inverse_label_mapping[label] for label in mapped_labels]

        if plot:
            self.plot_confusion_matrix(confusion_matrix(self.true_labels, mapped_labels), np.unique(self.true_labels))
            print(f'accuracy is {self.accuracy}')

    def predict_test_data(self):
        predicted_labels_test = self.cluster_model.predict(self.W_test)

        mapped_labels_test = [self.mapping[label] for label in predicted_labels_test]
        predicted_categories = [self.inverse_label_mapping[label] for label in mapped_labels_test]

        self.test_pred = pd.DataFrame({
            'ArticleId': self.test_data['ArticleId'],
            'Category': predicted_categories
        })

    @staticmethod
    def find_best_label_mapping(true_labels, predicted_labels):
        from itertools import permutations
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
    def plot_confusion_matrix(cm, labels):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def submit_to_kaggle(self, competition, filename, message):
        submission = self.test_pred
        submission.to_csv(filename, index=False)
        api = KaggleApi()
        api.authenticate()
        api.competition_submit(file_name=filename, competition=competition, message=message)

    @staticmethod
    def plot_performance_log(num_features, accuracies):
        plt.figure(figsize=(10, 7))
        plt.plot(num_features, accuracies, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.title('Performance Across Different Number of Features')
        plt.grid(True)
        plt.show()

    @staticmethod
    def hyperparameter_tuning(features_list, cleaner, mode='tfidf'):
        import gc
        best_trainer = None
        best_accuracy = 0
        accuracies = []
        best_num = 0
        test_pred = []
        train_pred = []
        for num in features_list:
            if mode == 'tfidf':
                extractor = TFIDFFeatureExtractor(cleaner, trim_features=num)
            elif mode == 'embedding':
                extractor = EmbeddingFeatureExtractor(cleaner, trim_features=num)
            else:
                raise TypeError
            extractor.extract_features()
            trainer = ModelTrainer(extractor)
            trainer.train_model()
            accuracy = trainer.accuracy
            accuracies.append(accuracy)

            trainer.predict_test_data()
            # trainer.submit_to_kaggle('learn-ai-bbc', 'submission.csv', 'Message')
            test_pred.append(trainer.test_pred)
            train_pred.append(trainer.train_pred)
            if best_accuracy < accuracy:
                best_trainer = trainer
                best_num = num
                best_accuracy = accuracy
            else:
                del trainer
                del extractor
            gc.collect()

        ModelTrainer.plot_performance_log(features_list, accuracies)
        print(f'The best trainer has {best_num} hyperparameters, with accuracy of {best_accuracy}')
        return best_trainer, best_accuracy, best_num, accuracies, test_pred, train_pred


# Hyperparameter tuning
(best_embedding_trainer, best_embedding_accuracy, best_embedding_num, embedding_accuracies,
 embedding_test_pred, embedding_train_pred) = (
    ModelTrainer.hyperparameter_tuning([5 * i for i in range(59, 0, -1)], cleaner, mode='embedding'))
