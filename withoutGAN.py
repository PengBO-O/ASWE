import codecs
import collections
import math
import pickle
import random
import time

import numpy as np
import tensorflow as tf


def has_letter(word):
    """
        Checks if input has a single letter. Used to remove punctuation.

        INPUT
        text: 		(string) Input String

        OUTPUT
        unnamed:	(boolean) True if at least one alphabetical letter in string.
    """
    for char in word:
        if char.isalpha():
            return True
    return False


def read_twitter(filename):
    """
        Read Twitter file. Counts number of words, creates dictionary, reverse dictionary.

        INPUTS:
        filename:	(string) The name of the wikipedia file

        OUTPUTS:
        data:			(list) All words in File
    """
    data = []
    counter = 0
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            stuff = [x for x in line.lower().split() if (has_letter(x) and len(x) > 1)]
            data += stuff
            counter += 1
    return data


def keep_word(word, num_words, count_dict):
    """
    (Undersampling) - Determine if a word should be kept in the data.

    INPUTS
    word:		(string)
    num_words:	(int) Total number of words in corpus
    count_dict: (dict) Dictionary of word:num occurences

    OUTPUT
                (boolean) If the word should be kept
    """
    try:
        z_wi = float(count_dict[word]) / (num_words)
        part1 = math.sqrt((z_wi / 0.001)) + 1
        part2 = 0.001 / z_wi
        total_prob = part1 * part2
        return random.uniform(0, 1) < total_prob
    except:
        return False


def build_dataset(words):
    """
    Build dictionary and reverse_dictionary of words.
    Do not include words that appear less than occurence_lower_limit.

    INPUT
    words:		(list)  All words in File

    OUTPUT
    count: 				(list) Storing Counts of each word
    dictionary:			(dict) Dictionary of word:index (to be used for embedding)
    reverse_dictionary:	(dict) Dictionary of index:word (used for decoding)
    """
    occurence_lower_limit = 1
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    dictionary = {'UNK': 0}
    for word, number in count:
        if number < occurence_lower_limit:
            continue
        dictionary[word] = len(dictionary)
    unk_count = 0
    for word in words:
        if word in dictionary:
            pass
        else:
            unk_count += 1
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary


def twitter_data(filename, dictionary):
    """
        Create a list of numbers represnted by each word.
        [The big dog] + {"the":1, "big":4, "dog":8} == [1, 4,8]
    """
    new_data = []
    with codecs.open(filename, 'r', 'utf8') as f:
        for line in f:
            new_line = []
            stuff = [x for x in line.lower().split() if
                     ((has_letter(x) or len(x) >= 1) and keep_word(x, num_words, count_dict))]
            for word in stuff:
                new_line.append(dictionary.get(word, 1))
            if len(new_line) > 0:
                new_data.append(new_line)
    return new_data


def num_sen_to_input(sent, index, context_window):
    """
    Given a sentence prepare for batch input by grabbing context

    INPUT
    sent:			(string) Sentence
    index: 			(int) Current index in Sentence
    CONTEXT_WINDOW_ONE:	(int) Size of context

    OUTPUT
    return_array:		(list) list of ints each int representing a word
    """
    return_array = []

    # look back context
    for j in range(context_window, 0, -1):
        if (index - j) < 0:
            return_array.append(0)
        else:
            return_array.append(sent[index - j])

    # look forward context
    for j in range(1, context_window + 1):
        if (index + j) >= len(sent):
            return_array.append(0)
        else:
            return_array.append(sent[index + j])

    return return_array


# So we can do 5 full iterations over the data


def new_generate_batch(batch_size, context_window):
    """
    Generates new batch for TF.

    INPUT
    BATCH_SIZE: 		(int) Number of examples in batch
    CONTEXT_WINDOW_ONE:		(int) Context Window

    OUTPUT
    batch:				(numpy array)
    labels:				(numpy array)
    senti_labels:		(numpy array)
    """
    global positive_index
    global negative_index

    global positive_word_index
    global negative_word_index

    global positive_sentence
    global negative_sentence

    pos_loops, neg_loops = 0, 0
    context_size = 2 * context_window
    batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    sentiment_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    batch_index = 0
    while batch_index < batch_size:
        # do positive index stuff
        if positive_word_index == len(positive_sentence):
            positive_index = (positive_index + 1) % len(positive_data)
            if (positive_index == 0):
                pos_loops += 1
            positive_word_index = 0
            positive_sentence = positive_data[positive_index]
        # do work
        labels[batch_index, 0] = positive_sentence[positive_word_index]

        batch_input = num_sen_to_input(positive_sentence, positive_word_index, context_window)
        for i2 in range(len(batch_input)):
            batch[batch_index, i2] = batch_input[i2]
        sentiment_labels[batch_index] = [1]

        positive_word_index += 1
        batch_index += 1

        if (batch_index) == batch_size // 2:
            break

    # TIME TO DO NEGATIVE
    while batch_index < batch_size:
        # do negative stuff
        if negative_word_index == len(negative_sentence):
            negative_index = (negative_index + 1) % len(negative_data)
            if (negative_index == 0):
                neg_loops += 1
            negative_word_index = 0
            negative_sentence = negative_data[negative_index]

        # do work
        labels[batch_index, 0] = negative_sentence[negative_word_index]

        batch_input = num_sen_to_input(negative_sentence, negative_word_index, context_window)
        for i2 in range(len(batch_input)):
            batch[batch_index, i2] = batch_input[i2]

        sentiment_labels[batch_index] = [0]

        negative_word_index += 1
        batch_index += 1
        if batch_index == batch_size:
            break

    zipped = zip(batch, labels, sentiment_labels)
    shuffled = []
    batch_, labels_, sl_ = [], [], []
    for i in zipped:
        shuffled.append(i)

    np.random.shuffle(shuffled)

    for j in shuffled:
        batch_.append(j[0])
        labels_.append(j[1])
        sl_.append(j[2])

    batch_ = np.array(batch_)
    labels_ = np.array(labels_)
    sl_ = np.array(sl_)

    return batch_, labels_, sl_, pos_loops, neg_loops


def hTanh(input):
    '''
    hTanh activation function

    '''
    hidden = tf.maximum(input, -1.)
    hidden = tf.minimum(hidden, 1.)
    return hidden


# The multi-head mechanism refer to https://github.com/Kyubyong/transformer
def split_heads(q, k, v):
    def split_last_dimension_then_transpose(tensor, num_heads, dim):
        t_shape = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
        return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

    qs = split_last_dimension_then_transpose(q, 2, EMBED_SIZE)
    ks = split_last_dimension_then_transpose(k, 2, EMBED_SIZE)
    vs = split_last_dimension_then_transpose(v, 2, EMBED_SIZE)

    return qs, ks, vs


def scaled_dot(qs, ks, vs, num_heads):
    key_dim_per_head = EMBED_SIZE // num_heads
    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (key_dim_per_head ** 0.5)
    o3 = tf.nn.softmax(o2)
    return tf.matmul(o3, vs)


def concat_heads(outputs):
    def transpose_then_concat_last_two_dimenstion(tensor):
        tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
        t_shape = tensor.get_shape().as_list()
        num_heads, dim = t_shape[-2:]
        return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    return transpose_then_concat_last_two_dimenstion(outputs)


def generator(context_input, name):
    with tf.variable_scope("G_" + name, reuse=tf.AUTO_REUSE):
        q = tf.layers.dense(context_input, EMBED_SIZE, use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        k = tf.layers.dense(context_input, EMBED_SIZE, use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        v = tf.layers.dense(context_input, EMBED_SIZE, use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        qs, ks, vs = split_heads(q, k, v)
        outputs = scaled_dot(qs, ks, vs, num_heads=2)
        output = concat_heads(outputs)
        output = tf.contrib.layers.layer_norm(output)
        logits = tf.reduce_sum(output, axis=1)
        hidden = tf.layers.conv1d(tf.reshape(logits, [BATCH_SIZE, EMBED_SIZE, 1]), filters=2, kernel_size=5, strides=5,
                                  padding='same', activation=tf.nn.leaky_relu)
        hidden = tf.reduce_mean(hidden, axis=-1)
        senti = tf.layers.dense(hidden, 2)
    return logits, senti


def cosine_similarity(x, y):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(x * x, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(y * y, 1))
    pooled_mul_12 = tf.reduce_sum(x * y, 1)
    scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    return scores


def inputs(batch_size, context_size, embeddings, name):
    input_holder = tf.placeholder(tf.int32, shape=[batch_size, context_size], name="input_" + name)
    label_holder = tf.placeholder(tf.int32, shape=[batch_size, 1], name="label_" + name)
    sentiment_holder = tf.placeholder(tf.int32, shape=[batch_size, 1], name="sentiment" + name)
    embed = tf.nn.embedding_lookup(embeddings, input_holder)

    return input_holder, label_holder, sentiment_holder, embed


def train():
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size - 1, EMBED_SIZE], -0.01, 0.01))
    pad_tensor = tf.Variable(tf.zeros([1, EMBED_SIZE]))
    embeddings = tf.concat([pad_tensor, embeddings], 0)

    context_size_1 = CONTEXT_WINDOW_1 * 2
    context_size_2 = CONTEXT_WINDOW_2 * 2

    # Place_holders for different context size inputs
    input_1, label_1, senti_1, embed_1 = inputs(BATCH_SIZE, context_size_1, embeddings, "1")
    input_2, label_2, senti_2, embed_2 = inputs(BATCH_SIZE, context_size_2, embeddings, "2")

    onehot_1 = tf.reshape(tf.one_hot(indices=senti_1, depth=2, on_value=1.0, off_value=0.),
                          [BATCH_SIZE, 2])
    onehot_2 = tf.reshape(tf.one_hot(indices=senti_2, depth=2, on_value=1.0, off_value=0.),
                          [BATCH_SIZE, 2])
    sentiments_onehot = tf.concat([onehot_1, onehot_2], axis=0)

    # Generating word embeddings and sentiments by different context input
    # We employ different generator for different context-size input and then concatenate their outputs
    gen_embed_1, pred_senti_1 = generator(embed_1, "1")
    gen_embed_2, pred_senti_2 = generator(embed_2, "2")

    gen_embed = tf.concat([gen_embed_1, gen_embed_2], axis=0)
    pred_senti = tf.concat([pred_senti_1, pred_senti_2], axis=0)

    ### softmax cross-entropy loss ###
    # s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sentiments_onehot, logits=pred_senti),
    #                         name="s_loss")
    ### hinge loss ###
    cost = tf.reduce_sum(1.0 - pred_senti * sentiments_onehot, axis=1)
    s_loss = tf.reduce_mean(tf.maximum(cost, 0.))
    ###

    label_embed_1 = tf.reshape(tf.nn.embedding_lookup(embeddings, label_1), [BATCH_SIZE, EMBED_SIZE],
                               name="label_embed_1")
    label_embed_2 = tf.reshape(tf.nn.embedding_lookup(embeddings, label_1), [BATCH_SIZE, EMBED_SIZE],
                               name="label_embed_2")
    label_embed = tf.concat([label_embed_1, label_embed_2], axis=0)

    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("G_1") or var.name.startswith("G_2")]

    W_neg = tf.Variable(tf.truncated_normal([vocabulary_size, EMBED_SIZE], stddev=1.0 / math.sqrt(EMBED_SIZE)))
    b_neg = tf.Variable(tf.truncated_normal([vocabulary_size], stddev=1.0 / math.sqrt(vocabulary_size)))

    c_loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=W_neg,
                                   biases=b_neg,
                                   inputs=gen_embed,
                                   labels=tf.concat([label_1, label_2], axis=0),
                                   num_sampled=NUM_SAMPLED,
                                   num_classes=vocabulary_size))

    joint_loss = ALPHA * s_loss + (1. - ALPHA) * c_loss
    joint_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(joint_loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        init.run()
        num_pos_loops = 0
        num_neg_loops = 0
        print("Initialized", num_neg_loops, num_pos_loops)

        average_loss_j = 0

        record_loss_j = []
        step = -1
        L = 0

        start = time.time()
        while (num_neg_loops < 2 * NUM_ITERATIONS) or (num_pos_loops < 2 * NUM_ITERATIONS):
            step += 1
            batch_inputs_1, batch_labels_1, batch_sentiment_1, pos_loop_1, neg_loop_1 = new_generate_batch(
                BATCH_SIZE, CONTEXT_WINDOW_1)
            num_pos_loops += pos_loop_1
            num_neg_loops += neg_loop_1

            batch_inputs_2, batch_labels_2, batch_sentiment_2, pos_loop_2, neg_loop_2 = new_generate_batch(
                BATCH_SIZE, CONTEXT_WINDOW_2)
            num_pos_loops += pos_loop_2
            num_neg_loops += neg_loop_2

            feed_dict = {input_1: batch_inputs_1, label_1: batch_labels_1, senti_1: batch_sentiment_1,
                         input_2: batch_inputs_2, label_2: batch_labels_2, senti_2: batch_sentiment_2}

            opt_j, loss_j_val = session.run([joint_optimizer, joint_loss], feed_dict=feed_dict)
            average_loss_j += loss_j_val

            if step % 2000 == 0:
                end = time.time()
                if step > 0:
                    average_loss_j /= 2000

                print(
                    "-------------------------------------------------------------------------------------------------")

                print("Cost {:.2f} seconds".format(end - start))
                print("Average loss at {} step".format(step))
                print("Average joint loss: {:.4f}".format(average_loss_j))
                print("Num Pos Loops", num_pos_loops, "Num Neg Loops", num_neg_loops)
                print(
                    "-------------------------------------------------------------------------------------------------")

                record_loss_j.append(average_loss_j)

                L = L + 1
                average_loss_j = 0

                start = time.time()

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(VALID_SIZE):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
            if (num_neg_loops == 2 * NUM_ITERATIONS) and (num_pos_loops == 2 * NUM_ITERATIONS):
                print("Reached", 2 * NUM_ITERATIONS, " iterations. Time to break.")
                break

        final_embeddings = normalized_embeddings.eval()
        return final_embeddings


if __name__ == '__main__':

    BATCH_SIZE = 200
    EMBED_SIZE = 300
    CONTEXT_WINDOW_1 = 2
    CONTEXT_WINDOW_2 = 3

    LAMBDA = 10
    ALPHA = 0.4

    twitter_data_pos = "data/positive_tweets.txt"
    twitter_data_neg = "data/negative_tweets.txt"

    words = read_twitter(twitter_data_pos) + read_twitter(twitter_data_neg)
    num_words = len(words)
    print("Read Twitter Data Successfully")
    print('Data size', num_words)

    count, dictionary, reverse_dictionary = build_dataset(words)
    print("Number of words", len(dictionary))
    print('Most common words (+UNK)', count[:10])
    vocabulary_size = len(dictionary)

    count_dict = dict(count)

    positive_data = twitter_data(twitter_data_pos, dictionary)
    negative_data = twitter_data(twitter_data_neg, dictionary)

    count_0 = 0
    for sen in positive_data:
        if len(sen) == 0:
            count_0 += 1
    print("Positive 0's Sentences Count 0", count_0)

    count_0 = 0
    for sen in negative_data:
        if len(sen) == 0:
            count_0 += 1
    print("Negative 0's Sentences Count 0", count_0)

    print('Sample data', positive_data[0][:10], [reverse_dictionary[i] for i in positive_data[0][:10]])

    positive_index = 0
    positive_word_index = 0
    negative_index = 0
    negative_word_index = 0

    positive_sentence = positive_data[0]
    negative_sentence = negative_data[0]

    VALID_SIZE = 15
    VALID_WORDS = ['upset', 'hurt', 'sad',
                   'cry', 'hate', 'happy',
                   'great', 'smile', 'beautiful',
                   'interesting', 'water',
                   'people', 'world',
                   'weather', 'man']
    valid_examples = np.array([dictionary[w] for w in VALID_WORDS])
    NUM_SAMPLED = 20  # Number of negative examples to sample.
    NUM_ITERATIONS = 2  # Number of times to go over corpus
    CONTEXT_ITERATIONS = 1

    fe = train()
    print("training over")

    with open('vector/noGAN.pickle', 'wb') as handle:
        pickle.dump((dictionary, fe), handle, protocol=2)
