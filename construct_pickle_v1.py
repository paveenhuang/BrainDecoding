# from google.colab import drive
# drive.mount('/content/drive')

import os
import pickle
from glob import glob
from tqdm import tqdm
import scipy.io as sio

import argparse

parser = argparse.ArgumentParser(
    description="Specify task name for converting ZuCo v1.0 Mat file to Pickle"
)
parser.add_argument(
    "-t",
    "--task_name",
    help="name of the task in /dataset/ZuCo, choose from {task1_SR,task2_NR,task3_TSR}",
    required=True,
)
args = vars(parser.parse_args())

task_name = args["task_name"]
# task_name = "task1_SR"

TASK1_NAME = "task1_SR"

print("##############################")
print(f"start processing ZuCo {task_name}...")

"""path definition"""
path = "/content/drive/MyDrive"
input_dir = path + f"/data/ZuCo2018/{task_name}"
# path = os.getcwd()
# input_dir = os.path.join(path, 'data')
print("input path:", input_dir)
output_dir = path + "/result"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
mat_files = glob(os.path.join(input_dir, "*.mat"))
print(f"number of {task_name} files:", len(mat_files))
mat_files = sorted(mat_files)
print(mat_files)

"""read and save dict"""
dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject_name = (
        os.path.basename(mat_file).split("_")[0].replace("results", "").strip()
    )
    print("handling the data of", subject_name)
    dataset_dict[subject_name] = []
    matdata = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)[
        "sentenceData"
    ]
    for sent in matdata:  # mat struct
        # print(sent._fieldnames)
        word_data = sent.word  # (22, ) mat struct
        if isinstance(word_data, float):
            print(
                f"missing sent: subj:{subject_name} content:{sent.content}, return None"
            )
            dataset_dict[subject_name].append(None)
        else:
            sent_obj = {
                "content": sent.content
            }  # sentence groud truth (English)
            sent_obj["sentence_level_EEG"] = {
                "mean_t1": sent.mean_t1,
                "mean_t2": sent.mean_t2,
                "mean_a1": sent.mean_a1,
                "mean_a2": sent.mean_a2,
                "mean_b1": sent.mean_b1,
                "mean_b2": sent.mean_b2,
                "mean_g1": sent.mean_g1,
                "mean_g2": sent.mean_g2,
            }  # (105,) each band
            if task_name == TASK1_NAME:  # answer only for task1
                sent_obj["answer_EEG"] = {
                    "answer_mean_t1": sent.answer_mean_t1,
                    "answer_mean_t2": sent.answer_mean_t2,
                    "answer_mean_a1": sent.answer_mean_a1,
                    "answer_mean_a2": sent.answer_mean_a2,
                    "answer_mean_b1": sent.answer_mean_b1,
                    "answer_mean_b2": sent.answer_mean_b2,
                    "answer_mean_g1": sent.answer_mean_g1,
                    "answer_mean_g2": sent.answer_mean_g2,
                }

            """word level data"""
            sent_obj["word"] = []
            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []  # word targets
            for word in word_data:
                word_obj = {"content": word.content}  # word target
                word_tokens_all.append(word.content)
                word_obj["nFixations"] = word.nFixations  # ET features
                if word.nFixations > 0:  # if fixation
                    word_obj["word_level_EEG"] = {
                        "FFD": {
                            "FFD_t1": word.FFD_t1,
                            "FFD_t2": word.FFD_t2,
                            "FFD_a1": word.FFD_a1,
                            "FFD_a2": word.FFD_a2,
                            "FFD_b1": word.FFD_b1,
                            "FFD_b2": word.FFD_b2,
                            "FFD_g1": word.FFD_g1,
                            "FFD_g2": word.FFD_g2,
                        }
                    }
                    word_obj["word_level_EEG"]["TRT"] = {
                        "TRT_t1": word.TRT_t1,
                        "TRT_t2": word.TRT_t2,
                        "TRT_a1": word.TRT_a1,
                        "TRT_a2": word.TRT_a2,
                        "TRT_b1": word.TRT_b1,
                        "TRT_b2": word.TRT_b2,
                        "TRT_g1": word.TRT_g1,
                        "TRT_g2": word.TRT_g2,
                    }
                    word_obj["word_level_EEG"]["GD"] = {
                        "GD_t1": word.GD_t1,
                        "GD_t2": word.GD_t2,
                        "GD_a1": word.GD_a1,
                        "GD_a2": word.GD_a2,
                        "GD_b1": word.GD_b1,
                        "GD_b2": word.GD_b2,
                        "GD_g1": word.GD_g1,
                        "GD_g2": word.GD_g2,
                    }
                    sent_obj["word"].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append("[MASK]")

            sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
            sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
            sent_obj["word_tokens_all"] = word_tokens_all
            dataset_dict[subject_name].append(sent_obj)

    print("num of sent:", len(dataset_dict[subject_name]))

    """output"""
output_name = f"{task_name}.pickle"
with open(os.path.join(output_dir, output_name), "wb") as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("write to:", os.path.join(output_dir, output_name))
