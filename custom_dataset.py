from datasets import Dataset, DatasetDict


from datasets import load_dataset
from pathlib import Path


class customData(Dataset):
    def __init__(
            self,
            tokenizer,
            csv_name=None,
    ):

        try:
            #self.dataset = load_dataset(
            #    "csv",
            #    data_files={"train": [csv_name]},  # "eval": "grammar_validation.csv"},
            #    delimiter=",",
            #)

            import pandas
            df = pandas.read_csv(csv_name)
            tds = Dataset.from_pandas(df)
            ds = DatasetDict()
            ds['train'] = tds
            ds['validation'] = tds
            self.dataset = ds

        except Exception as e:
            print(
                "Loading of custom dataset failed! Please run prepare_and_load_data.ipynb to generate the dataset.")
            raise e

        # self.dataset = load_dataset("wikihow", "all", data_dir="data/", split=type_path)
        # if num_samples:
        #    self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        input_ = example_batch["input"]
        target_ = example_batch["target"]

        prompt = f"Continue this document. Input: {input_}\n---\nContinuation: {target_}"
        sample = self.tokenizer(prompt)

        return sample

    def __getitem__(self, index):
        sample = self.convert_to_features(self.dataset["train"][index])
        source_ids = sample["input_ids"]

        src_mask = sample["attention_mask"]

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": source_ids.copy(),
        }




# dataset = load_dataset("csv", data_files="test_properly_formatted_data.csv")


if __name__ == "__main__":
    from custom_dataset import customData
    import datasets
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = 'daryl149/Llama-2-7b-hf'
    # Pull the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = customData(tokenizer=tokenizer, csv_name='test_properly_formatted_data.csv')

    ff = dataset[0]
