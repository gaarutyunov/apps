import io
import json
import os
import subprocess

from torch.utils.data import Dataset
from tqdm import tqdm

from train.dataset_lm.reindent import run as run_reindent


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False,
        },
    )

    return ret.getvalue()


class APPSBaseDataset(Dataset):
    def __init__(self, dataroot, problem_dirs):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs  # Loaded from train/test split json files

        self.samples = []  # Should be set in initialize()
        self.initialize()

    def initialize(self):
        """
        Assume self.dataroot is set to folderName/data
        """

        all_samples = []
        skipped_problems = []

        all_samples_dict = {}  # Mapping from question_fname to list of samples

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")

            # print(question_fname)

            if os.path.exists(starter_code):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue

            if os.path.isfile(starter_code):
                with open(starter_code, "r") as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Read the question description
            with open(question_fname, "r") as f:
                question_str = f.read()

            # Read all the solutions
            with open(sols_fname, "r") as f:
                sols_str_list = json.load(f)
                for sol_str in sols_str_list:
                    sol_str = reindent_code(sol_str)
                    sample = (question_str, starter_code, sol_str, answer_type)

                    all_samples.append(sample)
                    if question_str in all_samples_dict:
                        all_samples_dict[question_str].append(sample)
                    else:
                        all_samples_dict[question_str] = [sample]

        print(f"Loaded {len(all_samples)} saamples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        self.samples = all_samples
        self.samples_dict = all_samples_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        q_str, s_str, a_str, answer_type = self.samples[idx]

        return {
            "text": "Q:" + q_str + "\n" + s_str + "\n" + answer_type + "\nA:\n" + a_str
        }


with open("train.json") as f:
    fnames = json.load(f)

dataset = APPSBaseDataset(dataroot=".", problem_dirs=fnames)

iterator = iter(dataset)

with open("apps.jsonl", "a") as f:
    for n in tqdm(iterator):
        json.dump(n, f)
        f.write("\n")


subprocess.run(["zstd", "apps.jsonl"])
