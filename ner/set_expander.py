from entity import Entity
from set_expansion import prepare_data
from entity_extraction import EntityExtraction
from set_expansion.set_expand import SetExpand
from nlp_architect.models.np2vec import NP2vec


class SetExpander(EntityExtraction):
    """
    Entity extraction using SetExpan
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def train(self):

        '''
        path to the corpus. path to the input corpus. Compressed files (gz) are also supported.
        '''
        #corpus = ""
        if self.config.get('corpus') is None:
            raise Exception('corpus path is missing.')
        else:
            corpus = self.config['corpus']

        marked_corpus = "processed_corpus.txt"
        if self.config.get('marked_corpus') is not None:
            marked_corpus = self.config['marked_corpus']

        '''
        special character that marks word separator and NP suffix
        '''
        mark_char = "_"
        if self.config.get('mark_char') is not None:
            mark_char = self.config['mark_char']

        '''
        perform noun-phrase grouping
        '''
        grouping = False
        if self.config.get('grouping') is not None:
            grouping = self.config['grouping']

        '''
        chunker to use for detecting noun phrases. 'spacy' for using spacy built-in
        chunker or 'nlp_arch' for NLP Architect NP Extractor
        '''
        chunker = "spacy"
        if self.config.get('chunker') is not None:
            chunker = self.config['chunker']

        prepare_data.prepare_data(corpus, marked_corpus=marked_corpus, mark_char=mark_char, grouping=grouping,
                                  chunker=chunker)

        '''
        Using corpus_format as txt as the output of prepare data is marked corpus in txt format.
        '''
        corpus_format = "txt"

        '''
        word embedding model type; word2vec and fasttext are supported.
        '''
        word_embedding_type = "word2vec"
        if self.config.get('word_embedding_type') is not None:
            word_embedding_type = self.config['word_embedding_type']

        '''
        path to the file where the trained np2vec model has to be stored.
        '''
        np2vec_model_file = self.config['np2vec_model_file']
        if self.config.get('np2vec_model_file') is not None:
            np2vec_model_file = self.config['np2vec_model_file']

        print('np2vec', np2vec_model_file)

        '''
        boolean indicating whether the model is stored in binary format; if
        word_embedding_type is fasttext and word_ngrams is 1, 
        binary should be set to True.
        '''
        binary = True
        if self.config.get('binary') is not None:
            binary = self.config['binary']

        '''
        model training hyperparameter, skip-gram. Defines the training algorithm. If 1,
        CBOW is used, otherwise, skip-gram is employed.
        '''
        sg = 0
        if self.config.get('sg') is not None:
            sg = self.config['sg']

        '''
        model training hyperparameter, size of the feature vectors.
        '''
        size = 100
        if self.config.get('size') is not None:
            size = self.config['size']

        '''
        model training hyperparameter, maximum distance
        between the current and predicted word within a
        sentence.
        '''
        window = 10
        if self.config.get('window') is not None:
            window = self.config['window']

        '''
        model training hyperparameter. The initial learning rate.
        '''
        alpha = 0.025
        if self.config.get('alpha') is not None:
            alpha = self.config['alpha']

        '''
        model training hyperparameter. Learning rate will linearly drop to `min_alpha` as
        training progresses.
        '''
        min_alpha = 0.0001
        if self.config.get('min_alpha') is not None:
            min_alpha = self.config['min_alpha']

        '''
        model training hyperparameter, ignore all words
        with total frequency lower than this.
        '''
        min_count = 2
        if self.config.get('min_count') is not None:
            min_count = self.config['min_count']

        '''
        model training hyperparameter, threshold for
        configuring which higher-frequency words are
        randomly downsampled, useful range is (0, 1e-5)
        '''
        sample = 1e-5
        if self.config.get('sample') is not None:
            sample = self.config['sample']

        '''
        model training hyperparameter, number of worker threads.
        '''
        workers = 20
        if self.config.get('workers') is not None:
            workers = self.config['workers']

        '''
        model training hyperparameter, hierarchical softmax. If set to 1, hierarchical
        softmax will be used for model training.
        If set to 0, and `negative` is non-zero, negative sampling will be used.
        '''
        hs = 0
        if self.config.get('hs') is not None:
            hs = self.config['hs']

        '''
        model training hyperparameter, negative sampling. If > 0, negative sampling will be
        used, the int for negative specifies how many "noise words" should be drawn (
        usually between 5-20). If set to 0, no negative sampling is used.
        '''
        negative = 25
        if self.config.get('negative') is not None:
            negative = self.config['negative']

        '''
        model training hyperparameter.  If 0, use the sum of the context word vectors.
        If 1, use the mean, only applies when cbow is used.
        '''
        cbow_mean = 1
        if self.config.get('cbow_mean') is not None:
            cbow_mean = self.config['cbow_mean']

        '''
        model training hyperparameter, number of iterations.
        '''
        iter = 15
        if self.config.get('iter') is not None:
            iter = self.config['iter']

        '''
        fasttext training hyperparameter. Min length of char ngrams to be used for training
        word representations.
        '''
        min_n = 3
        if self.config.get('min_n') is not None:
            min_n = self.config['min_n']

        '''
        fasttext training hyperparameter. Max length of char ngrams to be used for training
        word representations.
        Set `max_n` to be lesser than `min_n` to avoid char ngrams being used.
        '''
        max_n = 6
        if self.config.get('max_n') is not None:
            max_n = self.config['max_n']

        '''
        fasttext training hyperparameter. If 1, uses enrich word vectors with subword (
        ngrams) information. If 0, this is equivalent to word2vec training.
        '''
        word_ngrams = 1
        if self.config.get('word_ngrams') is not None:
            word_ngrams = self.config['word_ngrams']
        print('preprocess done!')
        np2vec = NP2vec(
            marked_corpus,
            corpus_format,
            mark_char,
            word_embedding_type,
            sg,
            size,
            window,
            alpha,
            min_alpha,
            min_count,
            sample,
            workers,
            hs,
            negative,
            cbow_mean,
            iter,
            min_n,
            max_n,
            word_ngrams,
        )

        np2vec.save(np2vec_model_file, False)

        print('word2vec done')

    def expand_single_class(self, entities):
        # it will just be current get_entities class
        pass

    def get_entities(self, text):

        if isinstance(text, list):
            # input is as before, list of string
            # just use expand_single_class above
            # output same as before
            pass
        elif isinstance(text, dict):
            # input is dictionary like this {'apple': 'fruit', 'mango': 'fruit', 'rose': 'flower', 'lily': 'flower}
            # key: text string, value: class name string
            # output will be list of entities like above
            # you will need to run expand_single_class separately for each class and append them in larger list
            pass
        else:
            raise TypeError('Unsupported data type for set expansion')

        '''
        boolean indicating whether the np2vec model to load is in binary format
        '''
        binary = False
        if self.config.get('binary') is not None:
            binary = self.config['binary']

        '''
        If 1, np2vec model to load uses word vectors with subword (
        ngrams) information.
        '''
        word_ngrams = 0
        if self.config.get('word_ngrams') is not None:
            word_ngrams = self.config['word_ngrams']

        '''
        grouping mode
        '''
        grouping = False
        if self.config.get('grouping') is not None:
            grouping = self.config['grouping']

        '''
        boolean indicating whether to load all maps for grouping.
        '''
        light_grouping = False
        if self.config.get('light_grouping') is not None:
            light_grouping = self.config['light_grouping']

        '''
        path to the directory containing maps for grouping.
        '''
        grouping_map_dir = False
        if self.config.get('grouping_map_dir') is not None:
            grouping_map_dir = self.config['grouping_map_dir']

        self.expander = SetExpand(np2vec_model_file=self.config['model'],
                                  grouping=grouping,
                                  grouping_map_dir=grouping_map_dir,
                                  binary=binary,
                                  word_ngrams=word_ngrams,
                                  light_grouping=light_grouping
                                  )

        '''
        Number of times the seed terms should be expanded
        '''
        iterations = 1
        if self.config.get('num_iteration') is not None:
            iterations = self.config['num_iteration']

        '''
        maximal number of expanded terms to return
        '''
        topn = 500
        if self.config.get('topn') is not None:
            topn = self.config['topn']

        '''
        minimum similarity threshold for the expanded terms
        '''
        threshold = 0.5
        if self.config.get('minimum_similarity_score') is not None:
            threshold = self.config['minimum_similarity_score']

        expanded_terms_set = set(text)
        expanded_entities_list = list()

        maxTermsCount =topn+len(expanded_terms_set)
        for iter in range(iterations):
            expanded_terms_list = self.expander.expand(expanded_terms_set,
                                                   topn=topn,
                                                   threshold=threshold
                                                   )
            if (len(expanded_terms_set)<maxTermsCount) and expanded_terms_list is not None:
                for term, similarity in expanded_terms_list:
                    if(len(expanded_terms_set)==maxTermsCount):
                        break
                    expanded_entities_list.append(Entity(-1, -1, term, "Expanded term", similarity))
                    expanded_terms_set.add(term)
            else:
                break
        return expanded_entities_list
