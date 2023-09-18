
ciao={
         "dataset":{
            "train":560000,
            "dev":38000,
            "test":38000,
         },
         "model":{
            "num_user": 7317,
            "max_degree_user": 804,
            "num_item": 105114,
            "max_degree_item": 915,
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim 
            "d_ffn": 32,            # FFN dim
            "num_heads": 2,
            "dropout": 0.2,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers": 2
         },
         ## FIXME: fixed batch_size & num_epochs
         "training":{
            "batch_size":64,       
            "optimizer":"adam",
            "learning_rate":0.0001,
            "warmup":80, 
            "lr_decay":"linear",
            "weight_decay":0,
            "eval_frequency":200, 
            "num_epochs":10,
            "num_eval_steps":100, 
            "patience":10, 
         },
     }

epinions={
         "dataset":{
             "train":560000,
             "dev":38000,
             "test":38000,
         },
         "model":{
             "embedding_dim":64, 
             "transformer_dim":64, 
             "transformer_hidden_dim":128, 
             "head_dim":32, 
             "num_head":2, 
             "num_layers":2,
             "vocab_size":512,
             "encoder_seq_length":20, 
             "decoder_seq_length":250, 
             "dropout_prob":0.1,
             "attention_dropout":0.1,
             "num_classes": 5,
         },
         "training":{
             "batch_size":128,
             "optimizer":"adam",
             "learning_rate":0.0001,
             "warmup":80, 
             "lr_decay":"linear",
             "weight_decay":0,
             "eval_frequency":200, 
             "num_epochs":10,
             "num_eval_steps":100, 
             "patience":10, 
         },
     }

Config = {
    "ciao":ciao,
    "epinions":epinions,
}