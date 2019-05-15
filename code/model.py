import sys
import os
import math
import random
import yaml
import argparse

import numpy as np
import tensorflow as tf

random.seed(41)
from data_reader import DataReader
config = yaml.safe_load(open("config.yml"))

log_2_e = 1.44269504089 # Constant to convert to binary entropies

class WordModel(tf.keras.Model):
        def __init__(self, embed_dim, hidden_dim, num_layers, vocab_dim):
            super(WordModel, self).__init__()
            random_init = tf.random_normal_initializer(stddev=0.1)
            self.embed = tf.Variable(random_init([vocab_dim, embed_dim]), dtype=tf.float32)
            self.dropout = tf.keras.layers.Dropout(rate=0.5)
            self.rnns = [tf.keras.layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)]
            self.project = tf.keras.layers.Dense(vocab_dim)

        # Very basic RNN-based language model: embed inputs, encode through several layers and project back to vocabulary
        def call(self, indices, training=True):
                states = tf.nn.embedding_lookup(self.embed, indices)
                for ix, rnn in enumerate(self.rnns):
                        states = rnn(states)
                        states = self.dropout(states)
                preds = self.project(states)
                input_length = []
                for batch in preds:
                    input_length.append(len(batch))
                results = tf.keras.backend.ctc_decode(preds, input_length, False, 5)
                return results

# Source: https://github.com/mmehdig/lm_beam_search
def beam_search(model, src_input, k=1, sequence_max_len=25):
        # (log(1), initialize_of_zeros)
        k_beam = [(0, [0]*(sequence_max_len+1))]

        # l : point on target sentence to predict
        for l in range(sequence_max_len):
            all_k_beams = []
            for prob, sent_predict in k_beam:
                predicted = model.predict([np.array([src_input]), np.array([sent_predict])])[0]
                # top k!
                possible_k = predicted[l].argsort()[-k:][::-1]

                # add to all possible candidates for k-beams
                all_k_beams += [
                        (
                            sum(np.log(predicted[i][sent_predict[i+1]]) for i in range(l)) \
                                    + np.log(predicted[l][next_wid]), \
                                    list(sent_predict[:l+1]) + [next_wid] + [0] * (sequence_max_len-l-1)
                        )   
                        for next_wid in possible_k
                    ]

            # top k
            k_beam = sorted(all_k_beams)[-k:]

        return k_beam


def eval(model, data):
        mbs = 0
        count = 0
        entropy = 0.0
        for indices, masks in data.batcher(data.valid_data, is_training=False):
                mbs += 1
                samples = int(tf.reduce_sum(masks[:, 1:]).numpy())
                count += samples
                preds = model(indices[:, :-1])
                loss = masked_ce_loss(indices, masks, preds)
                entropy += log_2_e * float(samples*loss.numpy())
        entropy = entropy / count
        return entropy, count

# Compute cross-entropy loss, making sure not to include "masked" padding tokens
def masked_ce_loss(indices, masks, preds):
        samples = tf.reduce_sum(masks[:, 1:])
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices[:, 1:], preds[-1][0]), logits=preds)
        loss *= masks[:, 1:]
        loss = tf.reduce_sum(loss) / samples
        return loss

def train(model, data):
        # Declare the learning rate as a variable to include it in the saved state
        learning_rate = tf.Variable(config["training"]["lr"], name="learning_rate")
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        is_first = True
        for epoch in range(config["training"]["num_epochs"]):
                print("Epoch:", epoch + 1)
                mbs = 0
                words = 0
                avg_loss = 0
                # Batcher returns a square index array and a binary mask indicating which words are padding (0) and real (1)
                for indices, masks in data.batcher(data.train_data):
                        mbs += 1
                        samples = tf.reduce_sum(masks[:, 1:])
                        words += int(samples.numpy())
                        
                        # Run through one batch to init variables
                        if is_first:
                                model(indices[:, :-1])
                                is_first = False
                        
                        # Compute loss in scope of gradient-tape (can also use implicit gradients)
                        with tf.GradientTape(watch_accessed_variables=False) as tape:
                                tape.watch(model.variables)
                                preds = model(indices[:, :-1])
                                loss = masked_ce_loss(indices, masks, preds)
                        
                        # Collect gradients, clip and apply
                        grads = tape.gradient(loss, model.variables)
                        grads, _ = tf.clip_by_global_norm(grads, 0.25)
                        
                        optimizer.apply_gradients(zip(grads, model.variables))
                        # Update average loss and print if applicable
                        avg_loss += log_2_e * loss
                        if mbs % config["training"]["print_freq"] == 0:
                                avg_loss = avg_loss.numpy()/config["training"]["print_freq"]
                                print("MB: {0}: words: {1}, entropy: {2:.3f}".format(mbs, words, avg_loss))
                                avg_loss = 0.0
                
                # Run a validation pass at the end of every epoch
                entropy, count = eval(model, data)
                print("Validation: tokens: {0}, entropy: {1:.3f}, perplexity: {2:.3f}".format(count, entropy, 0.0 if entropy > 100 else math.pow(2, entropy)))

def main():
        # Extract arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("train_data", help="Path to training data")
        ap.add_argument("-v", "--valid_data", required=False, help="(optional) Path to held-out (validation) data")
        ap.add_argument("-t", "--test_data", required=False, help="(optional) Path to test data")
        args = ap.parse_args()
        print("Using configuration:", config)
        data = DataReader(args.train_data, args.valid_data, args.test_data)
        model = WordModel(config["model"]["embed_dim"], config["model"]["hidden_dim"], config["model"]["num_layers"], data.vocab_dim)
        random.shuffle(data.valid_data) # Shuffle just once
        train(model, data)

if __name__ == '__main__':
        main()
