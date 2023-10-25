
ciao={  
         # dataset의 전체 길이 (기록 및 확인용) (train:val:test = 8:1:1)
         "dataset":{
            "train":53393,
            "dev":8489,
            "test":8428,
         },
         "model":{
            "num_user": 7317,       # rating(7375)에서 social에 없는 사용자 제외 -> 7317
            "max_degree_user": 804,
            "num_item": 104975,     # 105114 -> id re-indexing 하고 나서 count(rating.csv로 확인) -> 104975
            "max_degree_item": 969, # 915 -> id re-indexing 하고 나서 rating.csv로 다시 확인하니 969임. -> 969 뭘 확인한걸까.
            # "max_spd_value": 15,    # (from _test16.py) unreachable distance를 `max(num_user) + 1` 가 아닌 실제 SPD값 보다 조금 더 큰 값으로 설정. (ciao의 spd: 1 ~ 10)
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim 
            "d_ffn": 256,            # FFN dim
            "num_heads": 2,
            "dropout": 0.2,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 2,
            "num_layers_dec": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.0001,
            "warmup":80, 
            "lr_decay":"linear",
            "weight_decay":0,
            "eval_frequency":400, 
            "num_epochs":50,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
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
            "num_user": 18098,
            "max_degree_user": 2026,
            "num_item": 261679,
            "max_degree_item": 1510,
            # "max_spd_value": 15,    # (from _test16.py) unreachable distance를 `max(num_user) + 1` 가 아닌 실제 SPD값 보다 조금 더 큰 값으로 설정. (epinions의 spd: 1~9)
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim 
            "d_ffn": 256,            # FFN dim
            "num_heads": 2,
            "dropout": 0.2,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 2,
            "num_layers_dec": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.0001,
            "warmup":80, 
            "lr_decay":"linear",
            "weight_decay":0,
            "eval_frequency":400, 
            "num_epochs":50,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
            "patience":10, 
         },
     }

Config = {
    "ciao":ciao,
    "epinions":epinions,
}