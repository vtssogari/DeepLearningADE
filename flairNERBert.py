import apex
from flair.data import TaggedCorpus 
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings
from typing import List
from flair.embeddings import FlairEmbeddings, BertEmbeddings




def train(data_folder, model_output_folder):
    
    corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03, base_path=data_folder)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # init Flair embeddings
    flair_forward_embedding = FlairEmbeddings('multi-forward')
    flair_backward_embedding = FlairEmbeddings('multi-backward')

    # init multilingual BERT
    bert_embedding = BertEmbeddings('bert-base-multilingual-cased')

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        flair_forward_embedding,
        flair_backward_embedding,
        bert_embedding
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)
    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(model_output_folder, 
                  mini_batch_size=256,
                  max_epochs=150)

    # 8. plot training curves (optional)
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_training_curves(model_output_folder + '/loss.tsv')
    plotter.plot_weights(model_output_folder + '/weights.txt')