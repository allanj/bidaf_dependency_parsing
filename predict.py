from allennlp.predictors.predictor import Predictor

from src.predictors import DependencyParserPredictor


cuda_device = -1
# language = "english"
# dataset = "ontonotes"

languages = ["english", "chinese", "catalan", "spanish"]
datasets = ["ontonotes", "ontonotes_chinese", "semca", "semes"]

dataset_types = ["dev","test"]

for language, dataset in zip(languages, datasets):

    model_path = f"https://allan-dependency.s3-us-west-2.amazonaws.com/{language}/model.tar.gz"
    predictor = DependencyParserPredictor.from_path(archive_path=model_path, predictor_name="dep_parser", cuda_device=cuda_device)

    for dataset_type in dataset_types:

        file_input = f"datasets/{dataset}/{dataset_type}.sd.conllx"
        output = f"datasets/{dataset}/{dataset_type}.predsd.conllx"

        f = open(output, "w", encoding="utf-8")

        for inst in predictor._dataset_reader._read(file_input):
            result = predictor.predict_instance(inst)
            words = inst["metadata"]["words"]
            pos_tags = inst["metadata"]["pos"]
            entities = inst["metadata"]["entities"]
            pred_heads = result["predicted_heads"]
            dep_labels = result["predicted_dependencies"]

            for i in range(len(words)):
                f.write(f"{i+1}\t{words[i]}\t_\t{pos_tags[i]}\t{pos_tags[i]}\t_\t{pred_heads[i]}\t{dep_labels[i]}\t_\t_\t{entities[i]}\n")
            f.write("\n")


        f.close()
