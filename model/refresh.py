"""
Document Summarization Final Model
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

import model.model_docsum as model_docsum
import utils.model_utils as model_utils
from flags import FLAGS


class Refresh:
    def __init__(self, sess, vocab_size):
        """
        Initialize model with several class variables.
            1. Word embeddings
            2. Document placeholder
            3. Weight placeholder
            4. Reward placeholder
            5. Predicted label placeholder
            6. Logit placeholder for validation and test

        Define network and related training purpose
            7. Define Policy Core Network: Consists of Encoder, Decoder and Convolution
            8. Define Reward-Weighted Cross Entropy Loss
            9. Define training operator
            10. Define accuracy operator
            11. Define train saver
            12. Define scalar summary operations
            13. Initializations
            14. Create summary graph for Tensorboard
        """
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

        # 1. Word embeddings
        self.vocab_embed_variable = model_utils.get_vocab_embed_variable(vocab_size)

        # 2. Document placeholder
        self.document_placeholder = tf.placeholder(
            "int32",
            [
                None,
                (
                    FLAGS.max_doc_length
                    + FLAGS.max_title_length
                    + FLAGS.max_image_length
                ),
                FLAGS.max_sent_length,
            ],
            name="doc-ph",
        )
        self.label_placeholder = tf.placeholder(
            dtype,
            [None, FLAGS.max_doc_length, FLAGS.target_label_size],
            name="label-ph",
        )

        self.sbert_placeholder = tf.placeholder(
            "float32",
            [
                None,
                FLAGS.max_doc_length,
                FLAGS.sentembed_size,
            ],
            name="sbertvec-ph",
        )

        # 7. Define Policy Core Network: Consists of Encoder, Decoder and Convolution
        self.extractor_output, self.logits = model_docsum.policy_network(
            self.vocab_embed_variable, 
            self.document_placeholder, 
            self.sbert_placeholder,
        )

        # 5. Predicted label placeholder
        self.predicted_multisample_label_placeholder = tf.placeholder(
            dtype,
            [None, 1, FLAGS.max_doc_length, FLAGS.target_label_size],
            name="pred-multisample-label-ph",
        )

        # 4. Reward placeholder
        self.actual_reward_multisample_placeholder = tf.placeholder(
            dtype, [None, 1], name="actual-reward-multisample-ph"
        )

        # 3. Weight placeholder
        self.weight_placeholder = tf.placeholder(
            dtype, [None, FLAGS.max_doc_length], name="weight-ph"
        )

        # 8. Define Reward-Weighted Cross Entropy Loss
        # bagian ini harusnya gak perlu diubah lagi
        self.rewardweighted_cross_entropy_loss_multisample = (
            model_docsum.reward_weighted_cross_entropy_loss_multisample(
                self.logits, # ini akan teradjust dengan menggunakan SBERT
                self.predicted_multisample_label_placeholder,
                self.actual_reward_multisample_placeholder,
                self.weight_placeholder,
            )
        )

        # 9. Define training operator
        self.train_op_policynet_expreward = model_docsum.train_neg_expectedreward(
            self.rewardweighted_cross_entropy_loss_multisample
        )

        # 10. Define accuracy operator
        self.accuracy = model_docsum.accuracy(
            self.logits, self.label_placeholder, self.weight_placeholder
        )

        # 6. Logit placeholder for validation and test
        self.logits_placeholder = tf.placeholder(
            dtype,
            [None, FLAGS.max_doc_length, FLAGS.target_label_size],
            name="logits-ph",
        )

        self.final_accuracy = model_docsum.accuracy(
            self.logits_placeholder, self.label_placeholder, self.weight_placeholder
        )

        # 11. Define train saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # 12. Define scalar summary operations
        self.rewardweighted_ce_multisample_loss_summary = tf.scalar_summary(
            "rewardweighted-cross-entropy-multisample-loss",
            self.rewardweighted_cross_entropy_loss_multisample,
        )
        self.taccuracy_summary = tf.scalar_summary("training_accuracy", self.accuracy)
        self.vaccuracy_summary = tf.scalar_summary(
            "validation_accuracy", self.final_accuracy
        )

        # 13. Initializations
        init = tf.global_variables_initializer()
        sess.run(init)

        # 14. Create summary graph for Tensorboard
        self.summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
