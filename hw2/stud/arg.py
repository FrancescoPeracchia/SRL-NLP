import torch



class Arg_Classifier(torch.nn.Module):

    def __init__(self, cfg: dict):
        super(Arg_Classifier, self).__init__()


        #it could be only 0 or 1
        #predicate embedding
        self.predicate_flag_embedding_input_dim = 2
        self.predicate_flag_embedding_output_dim = cfg["embeddings"]["predicate_flag_embedding_output_dim"]
        
        #pos embedding
        self.pos_embedding_input_dim = cfg["embeddings"]["pos_embedding_input_dim"]
        self.pos_embedding_output_dim = cfg["embeddings"]["pos_embedding_output_dim"]
        
        #predicate embedding
        self.predicate_embedding_input_dim = cfg["embeddings"]["predicate_embedding_input_dim"]
        self.predicate_embedding_output_dim = cfg["embeddings"]["predicate_embedding_output_dim"]



        #bi-lstm
        self.bilstm_n_layers = cfg["bilstm"]["n_layers"]
        self.bilstm_output_dim = cfg["bilstm"]["output_dim"]


        self.language_portable = cfg["language_portable"]

        #list position 0: pre-classifier branch A , 1: in-classifier 2: pre-classifier branch B 
        self.dropouts = cfg["dropouts"]



        self.bert_output_dim = 768


       




        #predicate flag embedding is determinant while pos embedding could be added or not

        self.bilstm_input_dim = self.bert_output_dim + self.predicate_flag_embedding_output_dim

        if self.pos_embedding_output_dim :
            self.bilstm_input_dim = self.bilstm_input_dim + self.pos_embedding_output_dim
        else : 
            self.bilstm_input_dim = self.bilstm_input_dim
        
        if self.predicate_embedding_output_dim :
            self.bilstm_input_dim = self.bilstm_input_dim + self.predicate_embedding_output_dim
        else : 
            self.bilstm_input_dim = self.bilstm_input_dim
        

        if self.language_portable and self.pos_embedding_output_dim:

            self.bi_lstm_portable_input_dim = self.pos_embedding_output_dim + self.predicate_flag_embedding_output_dim
            self.bi_lstm_portable = torch.nn.LSTM(self.bi_lstm_portable_input_dim, self.bilstm_output_dim,self.bilstm_n_layers,dropout = 0.3,batch_first=True, bidirectional = True)
            #self.dropout_pre_B = torch.nn.Dropout(p=self.dropouts[2])
            #bi-directional ---> *2  Token hidden state,predicate hidden state, Token hidden state structurale inter-language information
            self.linear0_dim = self.bilstm_output_dim*2+self.bilstm_output_dim*2+self.bilstm_output_dim*2
        else :
            #bi-directional ---> *2  Token hidden state,predicate hidden state
            self.linear0_dim = self.bilstm_output_dim*2+self.bilstm_output_dim*2

        self.linear1_dim = cfg["n_classes"]*25
        self.linear2_dim = cfg["n_classes"]*5
        self.output_classes = cfg["n_classes"]
        
        

        self.embedding_predicate_flag = torch.nn.Embedding(self.predicate_flag_embedding_input_dim, self.predicate_flag_embedding_output_dim, max_norm=True)
        self.embedding_predicate = torch.nn.Embedding(self.predicate_embedding_input_dim, self.predicate_embedding_output_dim, max_norm=True)
        self.embedding_pos = torch.nn.Embedding(self.pos_embedding_input_dim, self.pos_embedding_output_dim, max_norm=True)

        self.bi_lstm = torch.nn.LSTM(self.bilstm_input_dim, self.bilstm_output_dim,self.bilstm_n_layers,dropout = 0.3,batch_first=True, bidirectional = True)
        #self.dropout_pre_A = torch.nn.Dropout(p=self.dropouts[1])



        self.flag_dropout = False
        self.dropout_language_constraint = torch.nn.Dropout(p=0.6)



        self.dropout_in_classifier = torch.nn.Dropout(p=self.dropouts[0])
        self.Relu = torch.nn.ReLU()
        self.Sigmoid  = torch.nn.Sigmoid()
        self.linear0 = torch.nn.Linear(self.linear0_dim, self.linear1_dim)
        self.linear1 = torch.nn.Linear(self.linear1_dim, self.linear2_dim)
        self.linear2 = torch.nn.Linear(self.linear2_dim, self.output_classes)

    def forward(self, subwords_embeddings :torch.tensor, perdicate_positional_encoding : torch.tensor, predicate_index:list, pos_index_encoding:torch.tensor, predicate_meaning_encoding:torch.tensor):
        
        #-------------------Emdedding and recombining----------------------------- 
        flag_embedding = self.embedding_predicate_flag(perdicate_positional_encoding)
        b,n,h = flag_embedding.size()
        subwords_embeddings = subwords_embeddings[:,:n,:]


        input_bilstm = torch.cat((subwords_embeddings,flag_embedding),2)
        
        if self.pos_embedding_output_dim  :
            #embedd pos
            pos_embedding = self.embedding_pos(pos_index_encoding)
            input_bilstm = torch.cat((input_bilstm,pos_embedding),2)
        
        
        if self.predicate_embedding_output_dim :
            #embedd disambiguate predicate
            predicate_meaning_embedding = self.embedding_predicate(predicate_meaning_encoding)
            input_bilstm = torch.cat((input_bilstm,predicate_meaning_embedding),2)




        #-------------------Bi-LSTM Structurale information----------------------------- 
        if self.language_portable and self.pos_embedding_output_dim:
            predicate__embedding = self.embedding_pos(pos_index_encoding)
            
            #embedd all strucural informations that are similar to different language
            input_bilstm_language_portable = torch.cat((pos_embedding,flag_embedding),2)
            output_bilstm_portable,_ = self.bi_lstm_portable(input_bilstm_language_portable)


 
        #-------------------Bi-LSTM----------------------------- 
        output_bilstm,_ = self.bi_lstm(input_bilstm)
        if self.flag_dropout :
            output_bilstm = self.dropout_language_constraint(output_bilstm)


        #-------------------POST Bi-LSTM----------------------------- 
        #eliminate [CLS] and [SEP] hidden state output
        output_bilstm = output_bilstm[:,1:-1,:]

        #Extract predicate embedding from bilstm output 
        predicate_embedding = []
        for i,j in enumerate(predicate_index):
            predicate_embedding.append(torch.unsqueeze(output_bilstm[i,j,:],dim = 0))
        
        predicate_embedding = torch.cat(predicate_embedding,dim = 0)

        b,n,h = output_bilstm.size()
        predicate_embedding = predicate_embedding.unsqueeze(1).repeat(1,n,1)



        #output_bilstm = self.dropout_pre_A(output_bilstm)


        if self.language_portable and self.pos_embedding_output_dim:
            #output_bilstm_portable = self.dropout_pre_B(output_bilstm_portable)
            output_bilstm_portable = output_bilstm_portable[:,1:-1,:]
            x = torch.cat((output_bilstm,output_bilstm_portable,predicate_embedding),2)
        else :
            x = torch.cat((output_bilstm,predicate_embedding),2)
        
        
        
            

            


        b,n,h = x.size()
        


        #-------------------Classifier----------------------------- 
        x = x.reshape(b*n,h)
        x = self.linear0(x)
        x = self.Relu(x)
        x = self.dropout_in_classifier(x)
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.dropout_in_classifier(x)
        x = self.linear2(x)

        return x

    def set_language_constrains(self):
        self.flag_dropout = True
    
    def freeze_parts(self):
        # Freezing backbone and FPN
        if self.language_portable and self.pos_embedding_output_dim:
            self.bi_lstm_portable.backbone.requires_grad_(False)


    def freeze_parts(self):
        # Freezing backbone and FPN
        if self.language_portable and self.pos_embedding_output_dim:
            print("Freezed layer : bi_lstm_portable and embedding ")
            self.bi_lstm_portable.requires_grad_(False)
            self.embedding_predicate_flag.requires_grad_(False)
            self.embedding_predicate.requires_grad_(False)
            self.embedding_pos.requires_grad_(False)



class Arg_Classifier_from_paper(torch.nn.Module):

    def __init__(self,language: str, cfg: dict):
        super(Arg_Classifier_from_paper, self).__init__()


        self.language = language


        #it could be only 0 or 1
        #predicate embedding
        self.predicate_flag_embedding_input_dim = 2
        self.predicate_flag_embedding_output_dim = 10
        

 

        #bi-lstm
        self.bilstm_n_layers = 1
        self.bilstm_output_dim = 150


        self.linear1_dim = cfg["n_classes"]*25
        self.linear2_dim = cfg["n_classes"]*5
        self.output_classes = cfg["n_classes"]
        
        


        self.embedding_predicate_flag = torch.nn.Embedding(self.predicate_flag_embedding_input_dim, self.predicate_flag_embedding_output_dim, max_norm=True)
        self.bi_lstm = torch.nn.LSTM(768+10, self.bilstm_output_dim,self.bilstm_n_layers,batch_first=True, bidirectional = True)
        self.linear0 = torch.nn.Linear(300, self.output_classes)

    def forward(self, subwords_embeddings :torch.tensor, perdicate_positional_encoding : torch.tensor, predicate_index:list, pos_index_encoding:torch.tensor, predicate_meaning_encoding:torch.tensor):
        
        #-------------------Emdedding and recombining----------------------------- 
        flag_embedding = self.embedding_predicate_flag(perdicate_positional_encoding)
        b,n,h = flag_embedding.size()
        subwords_embeddings = subwords_embeddings[:,:n,:]
        input_bilstm = torch.cat((subwords_embeddings,flag_embedding),2)
   
        #-------------------Bi-LSTM----------------------------- 
        output_bilstm,_ = self.bi_lstm(input_bilstm)


        #-------------------POST Bi-LSTM----------------------------- 
        #eliminate [CLS] and [SEP] hidden state output
        output_bilstm = output_bilstm[:,1:-1,:]

        #Extract predicate embedding from bilstm output 
        predicate_embedding = []
        for i,j in enumerate(predicate_index):
            predicate_embedding.append(torch.unsqueeze(output_bilstm[i,j,:],dim = 0))
        
        predicate_embedding = torch.cat(predicate_embedding,dim = 0)

        b,n,h = output_bilstm.size()
        predicate_embedding = predicate_embedding.unsqueeze(1).repeat(1,n,1)



        b,n,h = predicate_embedding.size()
        
        #-------------------Classifier----------------------------- 
        predicate_embedding = predicate_embedding.reshape(b*n,h)
        predicate_embedding = self.linear0(predicate_embedding)


        return predicate_embedding





        









