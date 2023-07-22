####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project
# Comments: Jan 2017
# Improved for Reinforcement Learning
####################################

"""
Document Summarization System
"""

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import matplotlib.pyplot as plt
import random

from utils.data_utils import DataProcessor, Data
from flags import FLAGS
from model.refresh import Refresh
from utils.reward_utils import Reward_Generator
from datetime import datetime


def train():
    """
    Training Mode: Create a new model and train the network
    Within a session, do followings:
        1. Prepare vocab dict (map from word to vector)
        2. Prepare data for training and validation
        3. Create ROUGE reward generator
        4. Create MODEL object
        5. Assign word embedding to model using vocab dict created in step 1
        6. Run epoch:
            7. Create new or read existing rouge dict
            8. For every epoch, shuffle train data
            9. Create data batch and start batch training.
                10. Run optimizer: optimize policy network for every data batch
            11. Save model checkpoint
            12. Save rouge dict
            13. Get performance on this epoch to validation set
            14. Print rouge score for this epoch's summary
    """
    with tf.Graph().as_default() and tf.device(FLAGS.use_gpu):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            # 1. Prepare vocab dict (map from word to vector)

            vocab_dict = {}
            if not FLAGS.is_use_sbert:
                (
                    vocab_dict,
                    word_embedding_array,
                ) = DataProcessor().prepare_vocab_embeddingdict()

            # 2. Prepare data for training and validation
            train_data = DataProcessor().prepare_news_data(data_type="training")
            validation_data = DataProcessor().prepare_news_data(data_type="validation")

            # 3. Prepare ROUGE reward generator
            rouge_generator = Reward_Generator()

            # 4. Create MODEL object
            model = Refresh(sess, len(vocab_dict) - 2)

            # 5. Assign word embedding to model using vocab dict created in step 1
            if not FLAGS.is_use_sbert:
                sess.run(model.vocab_embed_variable.assign(word_embedding_array))

            # 6. Run epoch:
            start_epoch = 1
            validation_rouge_scores = []
            training_rouge_scores = []
            now = datetime.now()

            for epoch in range(start_epoch, FLAGS.train_epoch_wce + 1):
                #                 var_name = "PolicyNetwork/SentExt/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read:0"
                #                 print("Variable name:", var_name)

                #                 var_value = sess.run(
                #                     tf.get_default_graph().get_tensor_by_name(var_name)
                #                 )
                #                 print("Initial Variable value:", var_value)

                batch_losses = []

                # 8. For every epoch, shuffle train data
                random.seed(start_epoch)
                train_data.shuffle_fileindices()

                # 9. Create batch data and start batch training
                step = 1
                while (step * FLAGS.batch_size) <= len(train_data.fileindices):
                    (
                        _,
                        batch_docs,
                        batch_label,
                        batch_weight,
                        batch_oracle_multiple,
                        batch_reward_multiple,
                        batch_sbert_vec,
                    ) = train_data.get_batch(
                        ((step - 1) * FLAGS.batch_size), (step * FLAGS.batch_size)
                    )

                    # Print the progress
                    if (step % FLAGS.training_checkpoint) == 0:
                        feed_dict = {
                            model.document_placeholder: batch_docs,
                            model.predicted_multisample_label_placeholder: batch_oracle_multiple,
                            model.actual_reward_multisample_placeholder: batch_reward_multiple,
                            model.label_placeholder: batch_label,
                            model.weight_placeholder: batch_weight,
                            model.sbert_placeholder: batch_sbert_vec,
                        }

                        ce_loss_val, ce_loss_sum, acc_val, acc_sum = sess.run(
                            [
                                model.rewardweighted_cross_entropy_loss_multisample,  # train_op to be optimized
                                model.rewardweighted_ce_multisample_loss_summary,  # train_op summary
                                model.accuracy,  # return accuracy / model performance in this batch
                                model.taccuracy_summary,  # return accuracy / model performance summary in this batch
                            ],
                            feed_dict=feed_dict,
                        )

                        batch_losses.append(ce_loss_val)

                        print(
                            "MRT: Epoch "
                            + str(epoch)
                            + "/"
                            + str(FLAGS.train_epoch_wce)
                            + " : Covered "
                            + str(step * FLAGS.batch_size)
                            + "/"
                            + str(len(train_data.fileindices))
                            + " : Minibatch Reward Weighted Multisample CE Loss= {:.6f}".format(
                                ce_loss_val
                            )
                            + " : Minibatch training accuracy= {:.6f}".format(acc_val)
                            + " "
                            + str(datetime.now() - now)
                        )

                    # 10. Run optimizer: optimize policy network for the data batch
                    sess.run(
                        [model.train_op_policynet_expreward],
                        feed_dict={
                            model.document_placeholder: batch_docs,
                            model.predicted_multisample_label_placeholder: batch_oracle_multiple,
                            model.actual_reward_multisample_placeholder: batch_reward_multiple,
                            model.weight_placeholder: batch_weight,
                            model.sbert_placeholder: batch_sbert_vec,
                        },
                    )

                    #                     var_value = sess.run(
                    #                         tf.get_default_graph().get_tensor_by_name(var_name)
                    #                     )
                    #                     print("Updated Variable value:", var_value)

                    # Increase step
                    step += 1

                # Plot the batch losses
                epoch_loss = sum(batch_losses) / len(batch_losses)
                print("Epoch:", epoch, "Loss:", epoch_loss)
                plt.plot(range(step - 1), batch_losses)
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.title("Batch Losses - Epoch ")
                plt.savefig(
                    FLAGS.train_dir + "/losses_" + str(epoch) + ".png",
                )
                plt.clf()

                # 11. Save model checkpoint
                checkpoint_path = os.path.join(
                    FLAGS.train_dir, "model.ckpt.epoch-" + str(epoch)
                )
                plt.clf()

                # 11. Save model checkpoint
                model.saver.save(
                    sess,
                    os.path.join(FLAGS.train_dir, "model.ckpt.epoch-" + str(epoch)),
                )

                # 13. Get performance on this epoch to validation set
                (
                    validation_logits,
                    validation_labels,
                    validation_weights,
                ) = _batch_predict_with_a_model(validation_data, model, session=sess)
                (
                    training_logits,
                    training_labels,
                    training_weights,
                ) = _batch_predict_with_a_model(train_data, model, session=sess)

                validation_acc, validation_sum = sess.run(
                    [model.final_accuracy, model.vaccuracy_summary],
                    feed_dict={
                        model.logits_placeholder: validation_logits.eval(session=sess),
                        model.label_placeholder: validation_labels.eval(session=sess),
                        model.weight_placeholder: validation_weights.eval(session=sess),
                    },
                )

                training_acc, training_sum = sess.run(
                    [model.final_accuracy, model.vaccuracy_summary],
                    feed_dict={
                        model.logits_placeholder: training_logits.eval(session=sess),
                        model.label_placeholder: training_labels.eval(session=sess),
                        model.weight_placeholder: training_weights.eval(session=sess),
                    },
                )

                # 14. Print rouge score for this epoch's summary
                # The output of the convert_and_evaluate function when used on multiple documents
                # is a dictionary containing the average ROUGE scores across all documents.
                validation_data.write_prediction_summaries(
                    validation_logits, "model.ckpt.epoch-" + str(epoch), session=sess
                )
                train_data.write_prediction_summaries(
                    training_logits, "model.ckpt.epoch-" + str(epoch), session=sess
                )

                (
                    validation_rouge_score,
                    validation_output_dict,
                ) = rouge_generator.get_full_rouge(
                    FLAGS.train_dir
                    + "/model.ckpt.epoch-"
                    + str(epoch)
                    + ".validation-summary-topranked",
                    "validation",
                )
                (
                    training_rouge_score,
                    training_output_dict,
                ) = rouge_generator.get_full_rouge(
                    FLAGS.train_dir
                    + "/model.ckpt.epoch-"
                    + str(epoch)
                    + ".training-summary-topranked",
                    "training",
                )

                validation_rouge_scores.append(
                    {
                        "rouge_1_f_score": validation_output_dict["rouge_1_f_score"],
                        "rouge_2_f_score": validation_output_dict["rouge_2_f_score"],
                        "rouge_l_f_score": validation_output_dict["rouge_l_f_score"],
                    }
                )
                training_rouge_scores.append(
                    {
                        "rouge_1_f_score": training_output_dict["rouge_1_f_score"],
                        "rouge_2_f_score": training_output_dict["rouge_2_f_score"],
                        "rouge_l_f_score": training_output_dict["rouge_l_f_score"],
                    }
                )

                now = datetime.now()

            print(
                "TRAINING ROUGE scores across all epochs:",
                training_rouge_scores,
            )
            print(
                "VALIDATION ROUGE scores across all epochs:",
                validation_rouge_scores,
            )

            total_trainable_params = 0
            trainable_vars = tf.trainable_variables()

            for variable in trainable_vars:
                shape = variable.get_shape()
                variable_params = 1
                for dim in shape:
                    variable_params *= dim.value
                total_trainable_params += variable_params
            print("Total trainable parameters:", total_trainable_params)


def test():
    """
    Test Mode: Loads an existing model and test it on the test set
    """

    # Training: use the tf default graph

    with tf.Graph().as_default() and tf.device(FLAGS.use_gpu):
        config = tf.ConfigProto(allow_soft_placement=True)

        # Start a session
        with tf.Session(config=config) as sess:
            ### Prepare data for training
            print("Prepare vocab dict and read pretrained word embeddings ...")
            (
                vocab_dict,
                word_embedding_array,
            ) = DataProcessor().prepare_vocab_embeddingdict()
            # vocab_dict contains _PAD and _UNK but not word_embedding_array

            print("Prepare test data ...")
            test_data = DataProcessor().prepare_news_data(data_type="test")

            # Create Model with various operations
            model = Refresh(sess, len(vocab_dict) - 2)

            # Select the model
            if os.path.isfile(
                FLAGS.train_dir + "/model.ckpt.epoch-" + str(FLAGS.model_to_load)
            ):
                selected_modelpath = (
                    FLAGS.train_dir + "/model.ckpt.epoch-" + str(FLAGS.model_to_load)
                )
            else:
                print("Model not found in checkpoint folder.")
                exit(0)

            # Reload saved model and test
            print("Reading model parameters from %s" % selected_modelpath)
            model.saver.restore(sess, selected_modelpath)
            print("Model loaded.")

            # Initialize word embedding before training
            print("Initialize word embedding vocabulary with pretrained embeddings ...")
            sess.run(model.vocab_embed_variable.assign(word_embedding_array))

            # Test Accuracy and Prediction
            print("Performance on the test data:")
            FLAGS.authorise_gold_label = False
            test_logits, test_labels, test_weights = _batch_predict_with_a_model(
                test_data, model, session=sess
            )
            test_acc = sess.run(
                model.final_accuracy,
                feed_dict={
                    model.logits_placeholder: test_logits.eval(session=sess),
                    model.label_placeholder: test_labels.eval(session=sess),
                    model.weight_placeholder: test_weights.eval(session=sess),
                },
            )

            # Print Test Summary
            print(
                "Test ("
                + str(len(test_data.fileindices))
                + ") accuracy= {:.6f}".format(test_acc)
            )

            # Writing test predictions and final summaries
            test_data.write_prediction_summaries(
                test_logits,
                "model.ckpt.epoch-" + str(FLAGS.model_to_load),
                session=sess,
            )


def _batch_predict_with_a_model(data: Data, model: Refresh, session=None):
    data_logits = []
    data_labels = []
    data_weights = []

    step = 1
    while (step * FLAGS.batch_size) <= len(data.fileindices):
        # Get batch data as Numpy Arrays : Without shuffling
        (
            batch_docnames,
            batch_docs,
            batch_label,
            batch_weight,
            batch_oracle_multiple,
            batch_reward_multiple,
            batch_sbert_vec,
        ) = data.get_batch(((step - 1) * FLAGS.batch_size), (step * FLAGS.batch_size))

        batch_logits = session.run(
            model.logits,
            feed_dict={
                model.document_placeholder: batch_docs,
                model.sbert_placeholder: batch_sbert_vec,
            },
        )

        data_logits.append(batch_logits)
        data_labels.append(batch_label)
        data_weights.append(batch_weight)

        # Increase step
        step += 1

    # Check if any data left
    if len(data.fileindices) > ((step - 1) * FLAGS.batch_size):
        # Get last batch as Numpy Arrays
        (
            batch_docnames,
            batch_docs,
            batch_label,
            batch_weight,
            batch_oracle_multiple,
            batch_reward_multiple,
            batch_sbert_vec,
        ) = data.get_batch(((step - 1) * FLAGS.batch_size), len(data.fileindices))
        batch_logits = session.run(
            model.logits,
            feed_dict={
                model.document_placeholder: batch_docs,
                model.sbert_placeholder: batch_sbert_vec,
            },
        )

        data_logits.append(batch_logits)
        data_labels.append(batch_label)
        data_weights.append(batch_weight)

    # Convert list to tensors
    data_logits = tf.concat(0, data_logits)
    data_lables = tf.concat(0, data_labels)
    data_weights = tf.concat(0, data_weights)

    return data_logits, data_lables, data_weights


def main(_):
    if FLAGS.exp_mode == "train":
        train()
    else:
        test()


if __name__ == "__main__":
    tf.app.run()
