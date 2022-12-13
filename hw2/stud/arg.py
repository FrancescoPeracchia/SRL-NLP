import torch



class Arg_Classifier(torch.nn.Module):

    def __init__(self,language: str, cfg: dict):
        super(Arg_Classifier, self).__init__()
        self.language = language

        #it could be only 0 or 1
        #predicate embedding
        self.predicate_flag_embedding_input_dim = 2
        self.predicate_flag_embedding_output_dim = cfg.embeddings.predicate_flag_embedding_output_dim
        

        #pos embedding
        self.pos_embedding_input_dim = cfg.embeddings.pos_embedding_input_dim
        self.pos_embedding_output_dim = cfg.embeddings.pos_embedding_output_dim
      


        #bi-lstm
        self.bilstm_n_layers = cfg.bilstm.n_layers
        self.bilstm_output_dim = cfg.bilstm.output_dim


        self.language_portable = cfg.language_portable

        #list position 0: pre-classifier branch A , 1: in-classifier 2: pre-classifier branch B 
        self.dropouts = cfg.dropouts



        self.bert_output_dim = 768


       




        #predicate flag embedding is determinant while pos embedding could be added or not
        if self.pos_embedding_output_dim :
            self.bilstm_input_dim = self.bert_output_dim + self.pos_embedding_output_dim + self.predicate_flag_embedding_output_dim
        else : 
            self.bilstm_input_dim = self.bert_output_dim + self.predicate_flag_embedding_output_dim
        

        if self.language_portable and self.pos_embedding_output_dim:

            self.bi_lstm_portable_input_dim = self.pos_embedding_output_dim + self.predicate_flag_embedding_output_dim
            self.bi_lstm_portable = torch.nn.LSTM(self.bi_lstm_portable_input_dim, self.bilstm_output_dim,self.bilstm_n_layers,dropout = 0.2,batch_first=True, bidirectional = True)
            self.dropout_pre_B = torch.nn.Dropout(p=self.dropouts[2])
            #bi-directional ---> *2  Token hidden state,predicate hidden state, Token hidden state structurale inter-language information
            self.linear0_dim = self.bilstm_output_dim*2+self.bilstm_output_dim*2+self.bilstm_output_dim*2
        else :
            #bi-directional ---> *2  Token hidden state,predicate hidden state
            self.linear0_dim = self.bilstm_output_dim*2+self.bilstm_output_dim*2

        self.linear1_dim = 100
        self.output_classes = cfg.n_classes
        
        

        self.embedding_predicate = torch.nn.Embedding(self.predicate_flag_embedding_input_dim, self.predicate_flag_embedding_output_dim, max_norm=True)
        self.embedding_pos = torch.nn.Embedding(self.pos_embedding_input_dim, self.pos_embedding_output_dim, max_norm=True)

        self.bi_lstm = torch.nn.LSTM(self.bilstm_input_dim, self.bilstm_output_dim,self.bilstm_n_layers,dropout = 0.2,batch_first=True, bidirectional = True)
        self.dropout_pre_A = torch.nn.Dropout(p=self.dropouts[1])



        self.dropout_in_classifier = torch.nn.Dropout(p=self.dropouts[0])
        self.Relu = torch.nn.ReLU()
        self.Sigmoid  = torch.nn.Sigmoid()
        self.linear0 = torch.nn.Linear(self.linear0_dim, self.linear1_dim)
        self.linear1 = torch.nn.Linear(self.linear1_dim, self.output_classes)

        

    def forward(self, subwords_embeddings :torch.tensor, perdicate_positional_encoding : torch.tensor, predicate_index:list, pos_index_encoding:torch.tensor):
        
        #-------------------Emdedding and recombining----------------------------- 
        perdicate_positional_encoding = perdicate_positional_encoding
        flag_embedding = self.embedding_predicate(perdicate_positional_encoding)
        b,n,h = flag_embedding.size()
        subwords_embeddings = subwords_embeddings[:,:n,:]
        
        if self.pos_embedding_output_dim :
            #embedd pos
            pos_embedding = self.embedding_pos(pos_index_encoding)
            input_bilstm = torch.cat((subwords_embeddings,flag_embedding,pos_embedding),2)
        
        else :
            input_bilstm = torch.cat((subwords_embeddings,flag_embedding),2)


        #-------------------Bi-LSTM Structurale information----------------------------- 
        if self.language_portable and self.pos_embedding_output_dim:
            
            #embedd all strucural informations that are similar to different language
            input_bilstm_language_portable = torch.cat((pos_embedding,flag_embedding),2)
            output_bilstm_portable,_ = self.bi_lstm_portable(input_bilstm_language_portable)


 
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



        output_bilstm = self.dropout_pre_A(output_bilstm)


        if self.language_portable and self.pos_embedding_output_dim:
            output_bilstm_portable = self.dropout_pre_B(output_bilstm_portable)
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

        return x





        














