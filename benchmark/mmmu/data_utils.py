"""Utils for data load, save, and process (e.g., prompt construction)"""

import json
import os
import re

import yaml

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


CAT_SHORT2LONG = {
    "acc": "Accounting",
    "agri": "Agriculture",
    "arch": "Architecture_and_Engineering",
    "art": "Art",
    "art_theory": "Art_Theory",
    "bas_med": "Basic_Medical_Science",
    "bio": "Biology",
    "chem": "Chemistry",
    "cli_med": "Clinical_Medicine",
    "cs": "Computer_Science",
    "design": "Design",
    "diag_med": "Diagnostics_and_Laboratory_Medicine",
    "econ": "Economics",
    "elec": "Electronics",
    "ep": "Energy_and_Power",
    "fin": "Finance",
    "geo": "Geography",
    "his": "History",
    "liter": "Literature",
    "manage": "Manage",
    "mark": "Marketing",
    "mate": "Materials",
    "math": "Math",
    "mech": "Mechanical_Engineering",
    "music": "Music",
    "phar": "Pharmacy",
    "phys": "Physics",
    "psy": "Psychology",
    "pub_health": "Public_Health",
    "socio": "Sociology",
}


def load_yaml(file_path):
    with open(file_path, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data["question"]
    o_imgs_paths = []
    for option in data["options"]:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {
            "id": data["id"],
            "question": question,
            "options": data["options"],
            "answer": data["answer"],
            "image": None,
            "question_type": data["question_type"],
        }
    else:
        return {
            "id": data["id"],
            "question": question,
            "options": data["options"],
            "answer": data["answer"],
            "image": data["image_1"],
            "question_type": data["question_type"],
        }


# DATA SAVING
def save_json(filename, ds):
    print(f"answers saved to: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(ds, f, indent=4)


# DATA PROCESSING
def construct_prompt(sample, config):
    question = sample["question"]
    options = eval(sample["options"])
    example = ""
    if sample["question_type"] == "multiple-choice":
        start_chr = "A"
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config["multi_choice_example_format"]
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict["index2ans"] = index2ans
        res_dict["correct_choice"] = sample["answer"]
        res_dict["all_choices"] = prediction_range
        res_dict["empty_prompt"] = empty_prompt
        if config["task_instructions"]:
            res_dict["final_input_prompt"] = (
                config["task_instructions"].strip() + "\n\n" + empty_prompt
            )
        else:
            res_dict["final_input_prompt"] = empty_prompt

        res_dict["gt_content"] = options[ord(sample["answer"].upper()) - ord("A")]
    else:
        empty_prompt_sample_structure = config["short_ans_example_format"]
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict["empty_prompt"] = empty_prompt
        if config["task_instructions"]:
            res_dict["final_input_prompt"] = (
                config["task_instructions"].strip() + "\n\n" + empty_prompt
            )
        else:
            res_dict["final_input_prompt"] = empty_prompt
        res_dict["gt_content"] = sample["answer"]

    res_dict.update(sample)
    return res_dict
