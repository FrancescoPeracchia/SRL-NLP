import json
import random

import numpy as np
from typing import List, Tuple
from transformers import AutoModel,AutoTokenizer
from model import Model
import torch.nn as nn
import torch
from .dataset import SRL
from .arg import Arg_Classifier

def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return Task34(language=language)


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class Task34(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str):
        # load the specific model for the input language
        self.language = language
        #only for initialization
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.auto_model = torch.load("hw2/stud/saved/bert.pth")
        #self.auto_model = torch.load("stud/saved/bert.pth")
        self.auto_model.eval()

        with open('hw2/stud/saved/vocabulary.json') as json_file:
            data = json.load(json_file)

            
        self.pos_list = data['pos_list']
        self.args_roles = data['args_roles']
        self.predicate_dis = data['predicate_dis']



        self.SRL = SRL("EN",self.tokenizer,"train",self.args_roles,self.pos_list,self.predicate_dis)
        


        #-----------------------------------------
        embeddings = dict()

        embeddings["predicate_flag_embedding_output_dim"] = 32
        #defined in initial exploration of the dataset
        embeddings["pos_embedding_input_dim"] = 0
        embeddings["pos_embedding_output_dim"] = 100
        #-------------------------------------------------
        embeddings["predicate_embedding_input_dim"] = 0
        embeddings["predicate_embedding_output_dim"] = False
        #defined in initial exploration of the dataset
        n_classes = 0



        bilstm = dict()
        bilstm["n_layers"] = 2
        bilstm["output_dim"] = 50
        dropouts = [0.4,0.3,0.3]

        language_portable = True
        predicate_meaning = True
        pos = True

        cfg = dict()
        cfg["embeddings"] = embeddings
        cfg["n_classes"] = n_classes
        cfg["bilstm"] = bilstm
        cfg["language_portable"] = language_portable
        cfg["dropouts"] = dropouts
        #-----------------------------------------
        cfg["embeddings"]["pos_embedding_input_dim"] = len(self.SRL.pos_list)
        cfg["embeddings"]["predicate_embedding_input_dim"] = len(self.SRL.predicate_dis)
        cfg["n_classes"] = len(self.SRL.args_roles)
        #-----------------------------------------



        self.model = Arg_Classifier("EN",cfg)
        PATH = "hw2/stud/saved/model_2022_12_24_17_43_42.pth"
        #PATH = "stud/saved/model_2022_12_24_17_43_42.pth"
        self.model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence.
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                    }
        """

        sample_batched,flag = self.SRL.prepare_batch(sentence)

        if flag :
    
        
        #----------------------PREPARE INPUT/OUTPUT-------------------------------
            input_bert = sample_batched["BERT_input"]
            input_bert['input_ids'] = input_bert['input_ids']
            input_bert['token_type_ids'] = input_bert['token_type_ids']
            input_bert['attention_mask'] = input_bert['attention_mask']
            sample_batched["positional_encoding"] = sample_batched["positional_encoding"]
            sample_batched["pos_index"] = sample_batched["pos_index"]
            sample_batched["predicate_meaning_idx"] = sample_batched["predicate_meaning_idx"]
            #prepare gt
            offset = sample_batched["offset_mapping"]
            #-----------------BERT EMBEDDING---------------------------
            with torch.no_grad():
                output = self.auto_model(**input_bert)
                output_hidden_states_sum = torch.stack(output.hidden_states[-4:], dim=0).sum(dim=0)
                b,n,h = output_hidden_states_sum.size()
        
                #------------------FILTERING SUB-WORDS----------------------
                subtoken_mask = torch.unsqueeze(offset[:,:, 0] != 0,dim =-1)
                word_emebedding = []
                for i in range(n): 
                    subwords_embedding = torch.unsqueeze(output_hidden_states_sum[:,i,:],dim = 1)
                    flag = subtoken_mask[0,i,0]
                    if flag :
                        continue
                    else :
                        word_emebedding.append(subwords_embedding)
                word_emebedding = torch.cat(word_emebedding,dim = 1)
                #-------------------------FORWARD----------------------------------
                x =self.model.forward(subwords_embeddings = output_hidden_states_sum,
                            perdicate_positional_encoding = sample_batched["positional_encoding"],
                            predicate_index = sample_batched["predicate_index"],
                            pos_index_encoding = sample_batched["pos_index"],
                            predicate_meaning_encoding = sample_batched["predicate_meaning_idx"])   


                b,n = sample_batched["gt"]["arg_gt"].size()
                #-------------------------RESULT STORING----------------------------------
                predicted = torch.argmax(x, dim=1)
                predicted = predicted.view(b,n)
                predicted_list = predicted.tolist()
                labels_list = []
                pre = dict()

                for i,p in enumerate(predicted_list) :
                    role = list(sentence["roles"].keys())
                    labels = [self.SRL.args_roles[i] for i in p]
                    pre[role[i]] = labels
                
                output = dict()
                output["roles"] = pre

        else :
            output = sample_batched
        


        return output









        