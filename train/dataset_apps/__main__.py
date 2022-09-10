import os
import json

import transformers

from train.dataset_apps.APPSBaseDataset import APPSBaseDataset

if __name__ == "__main__":
    # Do sanity checking
    with open(os.path.expanduser("~/apps/data_split/train.json")) as f:
        fnames = json.load(f)

    tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(
        "EleutherAI/gpt-neox-20b"
    )
    dataset = APPSBaseDataset(
        dataroot="~/apps/",
        problem_dirs=fnames,
        mode="EleutherAI/gpt-neox-20b",
        max_tokens=2048,
        sample_mode="uniform_prob",
    )

    e = dataset[0]
    print(e)
    print(
        "------- input_ids ------------------------------------------------------------------------------------"
    )
    print(tokenizer.decode(e["input_ids"]))
    print(
        "------- labels ------------------------------------------------------------------------------------"
    )
    labels = e["labels"]
    labels[labels == -100] = tokenizer.eos_token_id
    labels_str = tokenizer.decode(labels)
    print(labels_str)
