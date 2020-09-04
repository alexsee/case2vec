import gensim as gs

import numpy as np
import pandas as pd

import string

class Doc2VecRepresentation(object):
    def __init__(self,
                event_log):
        self._event_log = event_log
        
        # properties
        self._model = None
        self._tagged_docs = None
        
    
    def build_model(self, vector_size=32, window=5, min_count=0, append_event_attr=False, append_case_attr=False, concat=False, epochs=100):
        # generate tagged documents
        self._tagged_docs = []
        
        for index, trace in enumerate(self._event_log.get_event_log()):
            words = []
            words.append('start')
            
            case_attr = ('+' + '+'.join(['' if feature not in trace.attributes else trace.attributes[feature] for feature in self._event_log.case_attributes])) if append_case_attr else ''
   
            if concat:
                for event in trace:
                    event_attr = '+'.join(filter(None, [None if event_attribute != 'concept:name' and append_event_attr == False else str(event[event_attribute]) for event_attribute in set(self._event_log.event_attributes).intersection(event.keys())]))
                    words.append(event_attr + str(case_attr))
            else:
                for event in trace:
                    for event_attribute in self._event_log.event_attributes:
                        if event_attribute != 'concept:name' and append_event_attr == False:
                            continue
                        
                        if event_attribute in event:
                            words.append(str(event[event_attribute]) + case_attr)
            
            words.append('end')
            
            td = gs.models.doc2vec.TaggedDocument(words=words, tags=[index])
            self._tagged_docs.append(td)
        
        # generate document model
        self._model = gs.models.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, dm = 0, alpha=0.025, min_alpha=0.025)
        self._model.build_vocab(self._tagged_docs)
    
    
    def fit(self):
        self._model.train(self._tagged_docs, total_examples=self._model.corpus_count, epochs=self._model.epochs)
        
        
    def predict(self, epochs=None):
        return [self._model.infer_vector(doc.words, epochs=epochs) for doc in self._tagged_docs]
