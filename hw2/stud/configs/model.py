


embeddings = dict()

embeddings["predicate_flag_embedding_output_dim"] = 32
#defined in initial exploration of the dataset
embeddings["pos_embedding_input_dim"] = 0
embeddings["pos_embedding_output_dim"] = 100


bilstm = dict()

bilstm["n_layers"] = 2
bilstm["output_dim"] = 50


language_portable = False



dropouts = [0.3,0.3,0.3]