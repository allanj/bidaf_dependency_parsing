local bert_model = "bert-base-uncased";

{
    "dataset_reader":{
        "type":"ontonotes_dependency",
        "token_indexers": {
          "bert": {
            "type": "bert-pretrained",
            "pretrained_model": bert_model,
            "do_lowercase": true,
            "use_starting_offsets": true,
            "truncate_long_sequences": false
          }
        }
    },
    "train_data_path": "datasets/ontonotes/train.sd.conllx",
    "validation_data_path": "datasets/ontonotes/dev.sd.conllx",
    "test_data_path": "datasets/ontonotes/test.sd.conllx",
    "model": {
      "type": "biaffine_parser",
      "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
          "bert": ["bert", "bert-offsets", "bert-type-ids"]
      },
      "token_embedders": {
        "bert":{
            "type": "bert-pretrained",
            "pretrained_model": bert_model,
            "top_layer_only": true,
             "requires_grad": true
        }
      }
     },
      "pos_tag_embedding":{
        "embedding_dim": 100,
        "vocab_namespace": "pos",
        "sparse": true
      },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 868,
        "hidden_size": 400,
        "num_layers": 3,
        "recurrent_dropout_probability": 0.3,
        "use_highway": true
      },
      "use_mst_decoding_for_validation": false,
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
      "initializer": [
        [".*projection.*weight", {"type": "xavier_uniform"}],
        [".*projection.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]]
    },

    "iterator": {
      "type": "bucket",
      "sorting_keys": [["words", "num_tokens"]],
      "batch_size" : 128
    },
    "trainer": {
      "num_epochs": 50,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
  }
