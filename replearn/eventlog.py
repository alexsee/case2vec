import gzip
import json
import numpy as np

from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects import log as log_lib

from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter

class EventLog(object):
    def __init__(self, 
                 name,
                 event_attributes=['concept:name'],
                 case_attributes=[],
                 true_cluster_label='cluster',
                 max_length=50,
                 max_traces=None):
        self._event_log = None
        self.name = name
        self.event_attributes = event_attributes
        self.case_attributes = case_attributes
        self.true_cluster_label = true_cluster_label
        self.max_length = max_length
        self.max_traces = max_traces
        
        # properties
        self._case_lens = None
        self._case_ids = None
        self._case_idx = None
        
        self._true_cluster_labels = None
        self._event_attributes_features = None
        self._case_attributes_features = None
        
        self._event_attribute_encodes = None
        self._event_attribute_encoders = None
        self._case_attribute_encodes = None
        self._case_attribute_encoders = None
    
    def load(self, file_name, sort=True, filter_start_activities=False, filter_end_activities=False):
        """
        Load an event log from disk.
        
        :file_name:
        """
        if file_name.endswith('.json.gz'):
            self.load_from_json(file_name, filter_start_activities, filter_end_activities)
        else:
            self.load_from_xes(file_name, sort, filter_start_activities, filter_end_activities)
        
    def load_from_xes(self, file_name, sort=True, filter_start_activities=False, filter_end_activities=False):
        # load file
        parameters = {'timestamp_sort': sort}
        self._event_log = xes_import_factory.apply(file_name, parameters=parameters)
        
        if filter_end_activities:
            self._event_log = end_activities_filter.apply_auto_filter(self._event_log, parameters={"decreasingFactor": 0.6})
        
        if filter_start_activities:
            self._event_log = start_activities_filter.apply_auto_filter(self._event_log, parameters={"decreasingFactor": 0.6})
    
    def load_from_json(self, file_name, filter_start_activities=False, filter_end_activities=False):
        """
        Load an event log stored as a json file from disk.
        
        :file_name:
        """
        with gzip.open(file_name, 'rb') as f:
            json_eventlog = json.load(f)

        log = log_lib.log.EventLog()

        # read json file
        for tr in json_eventlog['cases']:
            attr_dict = tr['attributes']
            evnt_list = tr['events']

            trace = log_lib.log.Trace()
            trace.attributes['concept:name'] = tr['id']

            # attach attributes
            for key in attr_dict.keys():
                trace.attributes[key] = attr_dict[key]

            for evnt in evnt_list:
                event = log_lib.log.Event()
                event['concept:name'] = evnt['name']

                if evnt['timestamp'] is not None:
                    event['time:timestamp'] = ciso8601.parse_datetime(evnt['timestamp'])

                # attach other event attributes
                for key in evnt['attributes'].keys():
                    event[key] = evnt['attributes'][key]

                trace.append(event)

            log.append(trace)

        self._event_log = log
        
        if filter_end_activities:
            self._event_log = end_activities_filter.apply_auto_filter(self._event_log, parameters={"decreasingFactor": 0.6})
        
        if filter_start_activities:
            self._event_log = start_activities_filter.apply_auto_filter(self._event_log, parameters={"decreasingFactor": 0.6})
    
    def preprocess(self):
        self._case_lens = []
        self._case_ids = []
        self._case_idx = []
        
        self._true_cluster_labels = []
        self._event_attributes_features = {}
        self._case_attributes_features = []
        
        # load case attributes with heuristic
        if self.case_attributes == None:
            self.case_attributes = self.get_case_attributes_heuristic()
        
        # iterate over all cases
        for case_index, case in enumerate(self._event_log):
            # max traces
            if self.max_traces != None and case_index > self.max_traces:
                break
            
            self._case_lens.append(min(len(case), self.max_length))
            self._case_ids.append(case.attributes['concept:name'])
            self._case_idx.append(case_index)

            # preprocess case attributes
            case_attribute = []

            for attribute in self.case_attributes:
                if attribute in case.attributes.keys():
                    case_attribute.append(str(case.attributes[attribute]))
                else:
                    case_attribute.append('')

            self._case_attributes_features.append(case_attribute)

            # obtain true cluster label
            self._true_cluster_labels.append(case.attributes[self.true_cluster_label])

            # load all events
            for event_index, event in enumerate(case):
                # throw away events more than max_length
                if event_index >= self.max_length:
                    break

                for attr_index, attribute in enumerate(self.event_attributes):
                    # get value
                    if attribute not in event:
                        attribute_value = 0
                    else:
                        attribute_value = event[attribute]

                    # load all attribute values
                    if attribute not in self._event_attributes_features:
                        self._event_attributes_features[attribute] = []

                    self._event_attributes_features[attribute].append(attribute_value)

        self._case_attributes_features = np.array(self._case_attributes_features)
        self._case_lens = np.array(self._case_lens)
        
        # event attribute encodes
        event_attribute_encoders = {}

        for key in self._event_attributes_features:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()

            self._event_attributes_features[key] = encoder.fit_transform(self._event_attributes_features[key]) + 1
            event_attribute_encoders[key] = encoder

        # reshape features vector
        offsets = np.concatenate(([0], np.cumsum(self._case_lens)[:-1]))
        event_attribute_encodes = [np.zeros((self._case_lens.shape[0], self._case_lens.max())) for _ in range(len(self._event_attributes_features))]

        for i, (offset, case_len) in enumerate(zip(offsets, self._case_lens)):
            for k, key in enumerate(self._event_attributes_features):
                x = self._event_attributes_features[key]
                event_attribute_encodes[k][i, :case_len] = x[offset:offset + case_len]

        self._event_attribute_encodes = event_attribute_encodes
        self._event_attribute_encoders = event_attribute_encoders
        
        # case attribute encodes
        case_attribute_encodes = []
        case_attribute_encoders = []

        for i in range(self._case_attributes_features.shape[1]):
            # integer encode targets
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            encodes = encoder.fit_transform(self._case_attributes_features[:,i])

            case_attribute_encodes.append(encodes)
            case_attribute_encoders.append(encoder)
        
        self._case_attribute_encodes = case_attribute_encodes
        self._case_attribute_encoders = case_attribute_encoders
    
    def encode_event_attributes(self):
        return self._event_attribute_encodes, self._event_attribute_encoders
    
    def encode_case_attributes(self):    
        return self._case_attribute_encodes, self._case_attribute_encoders

    def get_event_log(self):
        return self._event_log
    
    def get_case_attributes_heuristic(self):
        ignore = ['concept:name', 'cluster', 'label', self.true_cluster_label]
        return [key for key in self._event_log[0].attributes.keys() if key not in ignore]
    
    @property
    def event_attributes_features(self):
        event_attribute_encodes, event_attribute_encoders = self.encode_event_attributes()
        return event_attribute_encodes
    
    @property
    def event_attributes_onehot_features(self):
        from tensorflow.keras.utils import to_categorical
        return [to_categorical(f)[:, :, 1:] for f in self.event_attributes_features]
    
    @property
    def event_attributes_flat_onehot_features(self):
        return np.concatenate(self.event_attributes_onehot_features, axis=2)
    
    @staticmethod
    def remove_time_dimension(x):
        return x.reshape((x.shape[0], np.product(x.shape[1:])))

    @property
    def event_attributes_flat_onehot_features_2d(self):
        return self.remove_time_dimension(self.event_attributes_flat_onehot_features)
    
    @property
    def case_attributes_features(self):
        return self._case_attributes_features
    
    @property
    def true_cluster_labels(self):
        return self._true_cluster_labels
    
    @property
    def case_lens(self):
        return self._case_lens
    
    @property
    def case_ids(self):
        return self._case_ids