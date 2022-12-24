import os
import json
import logging
import torch
from torch.utils.data import DataLoader,Dataset
import random
from typing import Dict
from transformers import AutoModel,AutoTokenizer
import time
class SRL(Dataset):
 
    def __init__(self,language,tokenizer,path,args_roles = None,pos_list = None,predicate_dis = None) -> None:
        #train
        #self.path_root = 'data'
        #inference 
        self.path_root = 'hw2/stud/data'
        #self.path_root = 'stud/data'
        self.load_data(language,path)
        if args_roles is None :
            self.args_roles,self.list_broken_id = self.list_arg_roles()
            self.predicate_dis.append("UNK")
        else : 
            self.args_roles = args_roles
            _,self.list_broken_id = self.list_arg_roles()
        

        if pos_list is None :
            self.pos_list,_ = self.list_pos()
            self.pos_list.append("Nothing")
            self.pos_list.append("UNK")
        else : 
            self.pos_list = pos_list
        


        if predicate_dis is None :
            self.predicate_dis,_ = self.list_predicate_roles()
            self.predicate_dis.append("Nothing")
            self.predicate_dis.append("UNK")
        else : 
            self.predicate_dis = predicate_dis
        
        
        


        self.tokenizer = tokenizer

    def load_data(self,language,mode):
        
        mode = mode+".json"
        path = os.path.join(self.path_root,language,mode)
        data_file = open(path)
       
        data_ = json.load(data_file)

        list_data = []

        for data in data_:
            list_data.append(data_[data])
        

        self.data = list_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, id : int):

        flag = False
        if id in self.list_broken_id :
            flag = True
            while flag == True:

                rand_id = random.randint(0, len(self.data)-1)
                
                if rand_id in self.list_broken_id :
                    pass
                else :
                    flag = False
                    id = rand_id        


        data = self.pre_processing(self.data[id])
        data = self.processig(data)
        return data
        
    def pre_processing(self, data:dict):
        data_list = []
        for role in data["roles"]:
            dictionary = dict()
            dictionary["words"] = data["words"]
            dictionary["role"] = data["roles"][role]
            dictionary["pre_idx"] = role
            dictionary["pos_tags"] = data["pos_tags"]
            dictionary["predicate_meaning"] = data["predicates"]
            data_list.append(dictionary)    
        return data_list
    
    def processig(self,data_list:list):
        
        for dictionary in data_list:

            #dictionary["words"] = data["words"]
            dictionary["gt_arg_identification"] = self.arg_id(dictionary["role"])
            dictionary["gt_arg_classification"] = self.arg_class(dictionary["role"])
            dictionary["pos_idx"] = self.pos_idx(dictionary["pos_tags"])
            dictionary["predicate_meaning_idx"] = self.predicate_meaning_idx(dictionary["predicate_meaning"])
        
        return data_list
   
    def list_arg_roles(self):
        list_roles = []
        list_broken_id = []
        for i,element in enumerate(self.data):
            flag = True
            try : roles = element["roles"]
            except : flag = False
            if flag :
                for e in roles:
                    sentence = element["roles"][e]

                    for word in sentence:
                        
                        list_roles.append(word)
                list_roles = list(set(list_roles))
            else : 
                list_broken_id.append(i)
        return list_roles,list_broken_id

    def list_predicate_roles(self):
        list_predicate_roles = []
        list_broken_id = []
        for i,element in enumerate(self.data):
            flag = True
            try : predicates = element["predicates"]
            except : flag = False
            if flag :
                for pre in predicates:
                    list_predicate_roles.append(pre)
                list_predicate_roles = list(set(list_predicate_roles))
            else : 
                list_broken_id.append(i)
        return list_predicate_roles,list_broken_id

    def list_pos(self):
        list_pos = []
        list_broken_id = []
        for i,element in enumerate(self.data):
            flag = True
            try : pos = element["pos_tags"]
            except : flag = False
            if flag :
                for e in pos:
                    list_pos.append(e)
                list_pos = list(set(list_pos))
            else : 
                list_broken_id.append(i)
        return list_pos,list_broken_id
  
    def arg_class(self,role:list):
        list_idxs = []
        for element in role:
            try : list_idxs.append(self.args_roles.index(element))
            except : list_idxs.append(self.args_roles.index("UNK"))
        

        return torch.tensor(list_idxs, dtype=torch.int64)

    def arg_id(self,role:dict):
        list_idxs = []
        for element in role:
            if element == "_":
                list_idxs.append(0)
            else :
                list_idxs.append(1)

        

        return torch.tensor(list_idxs, dtype=torch.int64)

    def pos_idx(self,pos_tags:dict):
        list_idxs = []
        list_idxs.append(self.pos_list.index("Nothing"))

        for element in pos_tags:
            try :list_idxs.append(self.pos_list.index(element))
            except :list_idxs.append(self.pos_list.index("UNK"))
        
        list_idxs.append(self.pos_list.index("Nothing"))
        return torch.tensor(list_idxs, dtype=torch.int64)
    
    def predicate_meaning_idx(self,predicate_meaning_tags:dict):
        list_idxs = []
        list_idxs.append(self.predicate_dis.index("Nothing"))

        for element in predicate_meaning_tags:
            try : list_idxs.append(self.predicate_dis.index(element))
            except : list_idxs.append(self.predicate_dis.index("UNK"))
            
        
        list_idxs.append(self.predicate_dis.index("Nothing"))
        return torch.tensor(list_idxs, dtype=torch.int64)
   
    def role_gen(self,sentence):

        base = ["_"]*len(sentence["predicates"])
        roles_dict = dict()
        counter = 0
        for i,item in enumerate(sentence["predicates"]):

            if item != "_":
                base = ["_"]*len(sentence["predicates"])
                sentence["roles"] = 10
                roles_dict[str(i)] = base
                counter += 1
        
        if counter == 0:
            sentence["roles"] = { }
            flag = False
            
                
        else :
            sentence["roles"] = roles_dict
            flag = True

        return sentence,flag
        
    def prepare_batch(self,sentence):

        sentence,flag = self.role_gen(sentence)
        
        if flag :

            data = self.pre_processing(sentence)
            data = self.processig(data)
            data = [data]
            
            
            input = dict() 
            gt = dict()
            batch_sentence = [] 
            
            for period in data:
                for sentence in period :

                    
                
                    #print(len(sentence[0]["words"]))
                    pre_idx = int(sentence["pre_idx"])
                    

                    predicate = sentence["words"][pre_idx]

                    text = " ".join(sentence["words"])
                    tokens: list[str] = text.split()
                    predicate: list[str] = predicate.split()

                    #text = sentence[0]["words"]
                    
                    t = (tokens,predicate)

                    batch_sentence.append(t)
                
                
            
        
        

            batch_output = self.tokenizer.batch_encode_plus(batch_sentence,padding=True,is_split_into_words=True, truncation=True,return_offsets_mapping=True, return_tensors="pt")
            


            for period in data:

                list_positional_predicate_encoding = []
                list_predicate_index = [] 
                list_pos_index = [] 
                list_arg_gt = []
                list_predicate_meaning_index = []

                for sentence in period:
                    #positional_encoding
                    #+2 per il CLS iniziale ad SEP finale
                    sentence_words_lenght =  len(sentence["words"])
                    positional_predicate_encoding = torch.zeros(1,sentence_words_lenght+2)
                    #+1 per il CLS iniziale
                    pre_idx = int(sentence["pre_idx"])
                    positional_predicate_encoding[:,pre_idx+1] = 1
                    list_positional_predicate_encoding.append(positional_predicate_encoding)
                    #print("positional_prefix_encoding",positional_predicate_encoding)
                    list_predicate_index.append(pre_idx)
                    

                    pos = torch.unsqueeze(sentence["pos_idx"],dim = 0)
                    list_pos_index.append(pos)
                    predicate_meaning_idxs = torch.unsqueeze(sentence["predicate_meaning_idx"],dim = 0)
                    list_predicate_meaning_index.append(predicate_meaning_idxs)


                    arg_gt = torch.unsqueeze(sentence["gt_arg_classification"],dim = 0)
                    list_arg_gt.append(arg_gt)


            list_arg_gt = torch.cat(list_arg_gt,dim = 0)
            list_pos_index = torch.cat(list_pos_index,dim = 0)
            list_predicate_meaning_index = torch.cat(list_predicate_meaning_index,dim = 0)
            list_positional_predicate_encoding = torch.cat(list_positional_predicate_encoding,dim = 0)
            gt["arg_gt"] = list_arg_gt
            input["predicate_index"] = list_predicate_index
            input["pos_index"] = list_pos_index.long()
            input["predicate_meaning_idx"] = list_predicate_meaning_index.long()
            offset = batch_output.pop("offset_mapping")
            input["BERT_input"] = batch_output
            input["positional_encoding"] = list_positional_predicate_encoding.long()
            input["offset_mapping"] = offset
            input["gt"] = gt
        
        else :
            input = sentence






        return input,flag

    




    


