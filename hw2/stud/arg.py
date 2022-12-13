import torch



class Arg_Classifier(torch.nn.Module):

    def __init__(self,language: str):
        super(Arg_Classifier, self).__init__()
        self.language = language

        #flag 0,1 embedder
        #cat 768 with 32 to obtain an embedding 800 
        self.embedding = torch.nn.Embedding(2, 32, max_norm=True)
        self.bi_lstm = torch.nn.LSTM(800, 50, 2,dropout = 0.2,batch_first=True, bidirectional = True)

        self.dropout = torch.nn.Dropout(p=0.3)
        self.Relu = torch.nn.ReLU()
        self.Sigmoid  = torch.nn.Sigmoid()
        N = 27

        self.linear0 = torch.nn.Linear(200, 100)
        self.linear1 = torch.nn.Linear(100, N)

    def forward(self, subwords_embeddings :torch.tensor, perdicate_positional_encoding : torch.tensor, predicate_index:list):
        perdicate_positional_encoding.cuda()
        flag_embedding = self.embedding(perdicate_positional_encoding)

        if self.pos_embedding :
            #embedd pos
        
        if self.language_portable :
            #embedd all strucural informations that are similar to


            #train en model with it

            #modify its dorpout lower and word embedder higher 
            #the model would rely more in this embedding 

            #use it in another language

        

        

        b,n,h = flag_embedding.size()

        subwords_embeddings = subwords_embeddings[:,:n,:].cuda()

        #print("subwords_embeddings",subwords_embeddings.size())
        input_bilstm = torch.cat((subwords_embeddings,flag_embedding),2)
        #print(input_bilstm.size())

        output_bilstm,_ = self.bi_lstm(input_bilstm)
        #print(output_bilstm.size())

        output_bilstm = output_bilstm[:,1:-1,:]
        



        #Extract predicate embedding from bilstm output 
        predicate_embedding = []
        for i,j in enumerate(predicate_index):
            predicate_embedding.append(torch.unsqueeze(output_bilstm[i,j,:],dim = 0))
        
        predicate_embedding = torch.cat(predicate_embedding,dim = 0)

        #print('preidcate emb',predicate_embedding.size())
        b,n,h = output_bilstm.size()
        predicate_embedding = predicate_embedding.unsqueeze(1).repeat(1,n,1)
        #print('preidcate emb',predicate_embedding.size())

        output_bilstm = torch.cat((output_bilstm,predicate_embedding),2)
        #print('output_bilstm final',output_bilstm.size())
        b,n,h = output_bilstm.size()


        x = output_bilstm.reshape(b*n,h)
     
        

        x = self.linear0(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.linear1(x)

        return x





        














