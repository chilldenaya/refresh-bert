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

import os
import os.path
from multiprocessing import Pool

from pyrouge import Rouge155

from flags import FLAGS


class Reward_Generator:
    def __init__(self):
        # Start a pool
        self.pool = Pool(10)

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

def _rouge(system_dir, gold_dir):
    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = gold_dir
    r.system_filename_pattern = "([a-zA-Z0-9]*).model"
    r.model_filename_pattern = "#ID#.gold"

    output = r.convert_and_evaluate(
        rouge_args="-e /home/jupyter-23521027/pyrouge/rouge/tools/ROUGE-1.5.5/data -a -c 95 -m -n 4 -w 1.2"
    )
    output_dict = r.output_to_dict(output)
    print(
        "rouge_1_f_score: ",
        output_dict["rouge_1_f_score"],
        " rouge_2_f_score: ",
        output_dict["rouge_2_f_score"],
        " rouge_l_f_score ",
        output_dict["rouge_l_f_score"],
    )
    
    avg_rscore = (
        output_dict["rouge_1_f_score"]
        + output_dict["rouge_2_f_score"]
        + output_dict["rouge_l_f_score"]
    ) / 3.0

    return avg_rscore
