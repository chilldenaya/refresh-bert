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
import random

import numpy as np

from utils.model_utils import convert_logits_to_softmax, predict_topranked
from flags import FLAGS

# Special IDs
PAD_ID = 0
UNK_ID = 1


class Data:
    def __init__(self, vocab_dict, data_type):
        self.filenames = []
        self.docs = []
        self.titles = []
        self.images = []
        self.labels = []
        self.rewards = []
        self.weights = []
        self.sbert_vecs = []

        self.fileindices = []

        self.data_type = data_type

        # populate the data
        self.populate_data(data_type)

        # Write to files
        self.write_to_files(data_type)

    def write_prediction_summaries(self, pred_logits, modelname, session=None):
        print("Writing predictions and final summaries ...")

        # Convert to softmax logits
        pred_logits = convert_logits_to_softmax(pred_logits, session=session)

        # Save Output Logits
        np.save(
            FLAGS.train_dir + "/" + modelname + "." + self.data_type + "-prediction",
            pred_logits,
        )

        # Writing
        pred_labels = predict_topranked(pred_logits, self.weights, self.filenames)
        self.write_predictions(
            modelname + "." + self.data_type, pred_logits, pred_labels
        )
        self.process_predictions_topranked(modelname + "." + self.data_type)

    def write_predictions(self, file_prefix, np_predictions, np_labels):
        foutput = open(FLAGS.train_dir + "/" + file_prefix + ".predictions", "w")
        for fileindex in self.fileindices:
            filename = self.filenames[fileindex]
            foutput.write(filename + "\n")

            sentcount = 0
            for sentpred, sentlabel in zip(
                np_predictions[fileindex], np_labels[fileindex]
            ):
                one_prob = sentpred[0]
                label = sentlabel[0]

                if sentcount < len(self.weights[fileindex]):
                    foutput.write(str(int(label)) + "\t" + str(one_prob) + "\n")
                else:
                    break

                sentcount += 1
            foutput.write("\n")
        foutput.close()

    def process_predictions_topranked(self, file_prefix):
        predictiondata = (
            open(FLAGS.train_dir + "/" + file_prefix + ".predictions")
            .read()
            .strip()
            .split("\n\n")
        )
        # print len(predictiondata)

        summary_dirname = FLAGS.train_dir + "/" + file_prefix + "-summary-topranked"
        os.system("mkdir " + summary_dirname)

        for item in predictiondata:
            # print(item)

            itemdata = item.strip().split("\n")
            # print len(itemdata)

            filename = itemdata[0]
            # print filename

            # predictions file already have top three sentences marked
            final_sentids = []
            for sentid in range(len(itemdata[1:])):
                label_score = itemdata[sentid + 1].split()
                if label_score[0] == "1":
                    final_sentids.append(sentid)

            # Create final summary files
            fileid = filename.split("-")[-1]  # cnn-fileid, dailymail-fileid
            summary_file = open(summary_dirname + "/" + fileid + ".model", "w")

            # Read Sents in the document : Always use original sentences
            sent_filename = (
                FLAGS.doc_sentence_directory
                + "/"
                + self.data_type
                + "/mainbody/"
                + fileid
                + ".mainbody"
            )
            docsents = open(sent_filename).readlines()

            # Top Ranked three sentences
            selected_sents = [
                docsents[sentid] for sentid in final_sentids if sentid < len(docsents)
            ]

            summary_file.write("".join(selected_sents) + "\n")
            summary_file.close()

    def get_batch(self, startidx, endidx):
        """
        Generate batch informations including docs, labels, weights, oracle multisample, and reward multisample
        Flow:
            1. Initialize batch docnames, docs, label, weight, oracle, and reward variables
            2. For every file in the batch:
                3. Set batch docnames
                4. Set batch docs
                5. Set batch labels: Select the single best
                6. Set batch weights
                7. Set multiple oracle and rewards

        """
        # This is very fast if you keep everything in Numpy

        def process_to_chop_pad(orgids, requiredsize):
            if len(orgids) >= requiredsize:
                return orgids[:requiredsize]
            else:
                padids = [PAD_ID] * (requiredsize - len(orgids))
                return orgids + padids

        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32

        # 1. Initialize batch docnames, docs, label, weight, oracle, and reward variables
        batch_docnames = np.empty((endidx - startidx), dtype="S60")
        batch_docs = np.empty(
            (
                (endidx - startidx),
                FLAGS.max_doc_length,
                FLAGS.max_sent_length,
            ),
            dtype="int32",
        )
        batch_label = np.empty(
            ((endidx - startidx), FLAGS.max_doc_length, FLAGS.target_label_size),
            dtype=dtype,
        )  # Single best oracle, used for JP models or accuracy estimation
        batch_weight = np.empty(
            ((endidx - startidx), FLAGS.max_doc_length), dtype=dtype
        )
        batch_oracle_multiple = np.empty(
            ((endidx - startidx), 1, FLAGS.max_doc_length, FLAGS.target_label_size),
            dtype=dtype,
        )
        batch_reward_multiple = np.empty(((endidx - startidx), 1), dtype=dtype)
        batch_sbert_vec = np.empty(
            ((endidx - startidx), FLAGS.max_doc_length, FLAGS.sentembed_size), 
            dtype=dtype,
        )

        # 2. For every file in the batch:
        batch_idx = 0
        for fileindex in self.fileindices[startidx:endidx]:
            # 3. Set batch docnames
            batch_docnames[batch_idx] = self.filenames[fileindex]

            # 4. Set batch docs
            doc_wordids = (
                []
            )
            for idx in range(FLAGS.max_doc_length):
                thissent = []
                if idx < len(self.docs[fileindex]):
                    thissent = self.docs[fileindex][idx][:]
                thissent = process_to_chop_pad(
                    thissent, FLAGS.max_sent_length
                )  # [FLAGS.max_sent_length]
                doc_wordids.append(thissent)
            batch_docs[batch_idx] = np.array(doc_wordids[:], dtype="int32")

            # 5. Set batch labels: Select the single best
            labels_vecs = [
                [1, 0] if (item in self.labels[fileindex][0]) else [0, 1]
                for item in range(FLAGS.max_doc_length)
            ]
            batch_label[batch_idx] = np.array(labels_vecs[:], dtype=dtype)
            

            # 6. Set batch weights
            weights = process_to_chop_pad(
                self.weights[fileindex][:], FLAGS.max_doc_length
            )
            batch_weight[batch_idx] = np.array(weights[:], dtype=dtype)

            if FLAGS.is_use_sbert:
                sent_docids = []
                for idx in range(FLAGS.max_doc_length):
                    thissent = []
                    if idx < len(self.sbert_vecs[fileindex]):
                        thissent = self.sbert_vecs[fileindex][idx][:]
                    thissent = process_to_chop_pad(
                        thissent, FLAGS.sentembed_size
                    )  # [FLAGS.max_sent_length]
                    sent_docids.append(thissent)
                batch_sbert_vec[batch_idx] = np.array(sent_docids[:], dtype=dtype)

            # 7. Set multiple oracle and rewards
            labels_set = (
                []
            )  # FLAGS.num_sample_rollout, FLAGS.max_doc_length, FLAGS.target_label_size
            # labels_set = [ d1[[1,0], [0,1], ..., [0,1]], d2[[1,0], [0,1], ..., [0,1]] ]
            reward_set = (
                []
            )  # FLAGS.num_sample_rollout, FLAGS.max_doc_length, FLAGS.target_label_size

            for candidate_summary_idx in range(FLAGS.num_sample_rollout):
                thislabels = []
                # self.labels[fileindex] = d1[[1, 19, 25], [1 19]]
                if candidate_summary_idx < len(self.labels[fileindex]):
                    thislabels = [
                        [1, 0]
                        if (item in self.labels[fileindex][candidate_summary_idx])
                        else [0, 1]
                        for item in range(FLAGS.max_doc_length)
                    ]
                    reward_set.append(self.rewards[fileindex][candidate_summary_idx])
                else:
                    # Simply copy the best one
                    thislabels = [
                        [1, 0] if (item in self.labels[fileindex][0]) else [0, 1]
                        for item in range(FLAGS.max_doc_length)
                    ]
                    reward_set.append(self.rewards[fileindex][0])
                labels_set.append(thislabels)

            # Randomly Sample one oracle label
            randidx_oracle = random.randint(0, (FLAGS.num_sample_rollout - 1))
            batch_oracle_multiple[batch_idx][0] = np.array(
                labels_set[randidx_oracle][:], dtype=dtype
            )
            batch_reward_multiple[batch_idx] = np.array(
                [reward_set[randidx_oracle]], dtype=dtype
            )

            # increase batch count
            batch_idx += 1

        return (
            batch_docnames,
            batch_docs,
            batch_label,
            batch_weight,
            batch_oracle_multiple,
            batch_reward_multiple,
            batch_sbert_vec,
        )

    def shuffle_fileindices(self):
        self.fileindices = list(self.fileindices)
        random.shuffle(self.fileindices)

    def write_to_files(self, data_type):
        full_data_file_prefix = (
            FLAGS.train_dir + "/" + FLAGS.data_mode + "." + data_type
        )
        print(
            "Writing data files with prefix (.filename, .doc, .title, .image, .label, .weight, .rewards): %s"
            % full_data_file_prefix
        )

        ffilenames = open(full_data_file_prefix + ".filename", "w")
        fdoc = open(full_data_file_prefix + ".doc", "w")
        flabel = open(full_data_file_prefix + ".label", "w")
        fweight = open(full_data_file_prefix + ".weight", "w")
        freward = open(full_data_file_prefix + ".reward", "w")

        for filename, doc, title, image, label, weight, reward in zip(
            self.filenames,
            self.docs,
            self.titles,
            self.images,
            self.labels,
            self.weights,
            self.rewards,
        ):
            ffilenames.write(filename + "\n")
            fdoc.write(
                "\n".join(
                    [" ".join([str(item) for item in itemlist]) for itemlist in doc]
                )
                + "\n\n"
            )
            flabel.write(
                "\n".join(
                    [" ".join([str(item) for item in itemlist]) for itemlist in label]
                )
                + "\n\n"
            )
            fweight.write(" ".join([str(item) for item in weight]) + "\n")
            freward.write(" ".join([str(item) for item in reward]) + "\n")

        ffilenames.close()
        fdoc.close()
        flabel.close()
        fweight.close()
        freward.close()

    def populate_data(self, data_type):
        """
        Populate docs, labels, and multiple oracle from dataset
        Flow
            1. Open and read files for all documents
            2. For every document in documents:
                3. Get all sentences in a document
                4. Run if only the doc's id match each other
                5. Save doc's sentences to a matrix form
                6. Set weight for non-padded sentences
                7. Save top ranked collective ROUGE to labels and rewards
        """
        # 1. Open and read files for all documents
        full_data_file_prefix = (
            FLAGS.preprocessed_data_directory + "/" + FLAGS.data_mode + "." + data_type
        )
        doc_data_list = (
            open(full_data_file_prefix + ".doc").read().strip().split("\n\n")
        )
        label_data_list = (
            open(full_data_file_prefix + ".label.multipleoracle")
            .read()
            .strip()
            .split("\n\n")
        )
        sbert_data_list = (
            open(full_data_file_prefix + ".sbert").read().strip().split("\n\n")
        )

        # 2. For every document in documents:
        doccount = 0
        for doc_data, label_data, sbert_data in zip(
            doc_data_list, label_data_list, sbert_data_list
        ):
            # 3. Get all sentences in a document
            doc_lines = doc_data.strip().split("\n") # line sentence
            label_lines = label_data.strip().split("\n")
            sbert_lines = sbert_data.strip().split("\n")

            filename = doc_lines[0].strip()

            # 4. Run if only the doc's id match each other
            if (
                (filename == label_lines[0].strip())
                and (filename == sbert_lines[0].strip())
            ):
                self.filenames.append(filename)

                # 5. Save doc's sentences to a matrix form
                # self.docs = [[d1s1, d1s2, ... , d1sn], [d2s1, d2s2, ... , d2sn]]
                thisdoc = []
                for line in doc_lines[1 : FLAGS.max_doc_length + 1]:
                    thissent = [int(item) for item in line.strip().split()] # sentence [w1, w2, ...]
                    thisdoc.append(thissent)
                self.docs.append(thisdoc)

                # 6. Save weight for non-padded sentences
                originaldoclen = int(label_lines[1].strip())
                thisweight = [1 for item in range(originaldoclen)][
                    : FLAGS.max_doc_length
                ]
                self.weights.append(thisweight)

                # 7. Save top ranked collective ROUGE to labels and rewards
                thislabel = []
                thisreward = []
                for line in label_lines[2 : FLAGS.num_sample_rollout + 2]:
                    thislabel.append([int(item) for item in line.split()[:-1]])
                    thisreward.append(float(line.split()[-1]))
                self.labels.append(thislabel)  # [[1, 19, 25], [1 19]]
                self.rewards.append(thisreward)  # [0.555, 0.0508]


                if FLAGS.is_use_sbert:
                    thissbert = []
                    for embedding_per_sentence_in_docs_str in sbert_lines[1 : FLAGS.max_doc_length + 1]:
                        thissent_vec = [float(item) for item in embedding_per_sentence_in_docs_str.strip().split()] # 1 sentence
                        thissbert.append(thissent_vec) # semua sentence dalam dokumen
                    self.sbert_vecs.append(thissbert) # semua dokumen

            else:
                print("Some problem with %s.* files. Exiting!" % full_data_file_prefix)
                exit(0)

            if doccount % 10000 == 0:
                print("%d ..." % doccount)

            doccount += 1
            if doccount == FLAGS.doc_num:
                break

        # Set Fileindices
        self.fileindices = range(len(self.filenames))


class DataProcessor:
    def prepare_news_data(self, vocab_dict, data_type="training"):
        data = Data(vocab_dict, data_type)
        return data

    def prepare_vocab_embeddingdict(self):
        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32

        vocab_dict = {}
        word_embedding_array = []

        # Add padding
        vocab_dict["_PAD"] = PAD_ID
        # Add UNK
        vocab_dict["_UNK"] = UNK_ID

        # Read word embedding file
        wordembed_filename = FLAGS.pretrained_wordembedding
        print("Reading pretrained word embeddings file: %s" % wordembed_filename)

        linecount = 0
        with open(wordembed_filename, "r") as fembedd:
            for line in fembedd:
                if linecount == 0:
                    vocabsize = int(line.split()[0])
                    # Initiate fixed size empty array
                    word_embedding_array = np.empty(
                        (vocabsize, FLAGS.wordembed_size), dtype=dtype
                    )
                else:
                    linedata = line.split()
                    vocab_dict[linedata[0]] = linecount + 1
                    embeddata = [float(item) for item in linedata[1:]][
                        0 : FLAGS.wordembed_size
                    ]
                    word_embedding_array[linecount - 1] = np.array(
                        embeddata, dtype=dtype
                    )

                if linecount % 100000 == 0:
                    print(str(linecount) + " ...")
                linecount += 1
        print("Read pretrained embeddings: %s" % str(word_embedding_array.shape))

        print("Size of vocab: %d (_PAD:0, _UNK:1)" % len(vocab_dict))
        vocabfilename = FLAGS.train_dir + "/vocab.txt"
        print("Writing vocab file: %s" % vocabfilename)

        foutput = open(vocabfilename, "w")
        vocab_list = [(vocab_dict[key], key) for key in vocab_dict.keys()]
        vocab_list.sort()
        vocab_list = [item[1] for item in vocab_list]
        foutput.write("\n".join(vocab_list) + "\n")
        foutput.close()
        return vocab_dict, word_embedding_array
