####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project
####################################

"""
Document Summarization Modules and Models
"""

from __future__ import absolute_import, division, print_function

import json
import os
import os.path
from multiprocessing import Pool

from pyrouge import Rouge155

from flags import FLAGS

def _rouge(system_dir, gold_dir):
    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = gold_dir
    r.system_filename_pattern = "([a-zA-Z0-9]*).model"
    r.model_filename_pattern = "#ID#.gold"

    output = r.convert_and_evaluate(
        rouge_args="-e /Users/denaya/z/balebali/rouge/tools/ROUGE-1.5.5/data -a -c 95 -m -n 4 -w 1.2"
    )
    output_dict = r.output_to_dict(output)

    avg_rscore = (
        output_dict["rouge_1_f_score"]
        + output_dict["rouge_2_f_score"]
        + output_dict["rouge_l_f_score"]
    ) / 3.0

    return avg_rscore


class Reward_Generator:
    def __init__(self):
        self.rouge_dict = {}

        # Start a pool
        self.pool = Pool(10)

    def save_rouge_dict(self):
        with open(FLAGS.train_dir + "/rouge-dict.json", "w") as outfile:
            json.dump(self.rouge_dict, outfile)

    def restore_rouge_dict(self):
        self.rouge_dict = {}
        if os.path.isfile(FLAGS.train_dir + "/rouge-dict.json"):
            with open(FLAGS.train_dir + "/rouge-dict.json") as data_file:
                self.rouge_dict = json.load(data_file)

    def get_full_rouge(self, system_dir, datatype):
        # Gold Directory: Always use original files
        gold_summary_directory = (
            FLAGS.gold_summary_directory
            + "/gold-"
            + FLAGS.data_mode
            + "-"
            + datatype
            + "-orgcase"
        )

        rouge_score = _rouge(system_dir, gold_summary_directory)

        # Delete any tmp file
        os.system("rm -r " + FLAGS.tmp_directory + "/tmp*")

        return rouge_score
