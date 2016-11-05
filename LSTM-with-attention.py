import tensorflow as tf
import numpy as np
import json

glove_dimension = 50
def glove(word, glove_d):
    # select appropriate glove file for glove dimension given
    glove_vectors = open("glove.6B/glove.6B.50d.txt")

    length = 0
    vocab = []
    for line in glove_vectors:
        item = line.split(" ")
        vocab.append(item[0])


        if item[0] == word:
            word_vector = [float(n) for n in item[1:]]
            glove_vectors.close()
            return word_vector

    if word not in vocab:
        #values taken from paper
        return np.random.uniform(-0.05, 0.05, glove_d)

glove_dimension = 50
def glove(word, glove_d):
    # select appropriate glove file for glove dimension given
    glove_vectors = open("glove.6B/glove.6B.50d.txt")

    length = 0
    vocab = []
    for line in glove_vectors:
        item = line.split(" ")
        vocab.append(item[0])


        if item[0] == word:
            word_vector = [float(n) for n in item[1:]]
            glove_vectors.close()
            return word_vector

    if word not in vocab:
        #values taken from paper
        return np.random.uniform(-0.05, 0.05, glove_d)


def get_data(file, max_num):
    # I reccomend that the max_num is kept very small as my padding function is very slow
    i = 0
    data = open(file)
    SNLI_sentences = []
    for pair in data:
        pair = json.loads(pair)  # pair will be string so need json.loads to turn it into a dictionary
        SNLI_sentences.append([pair["sentence1"].lower(), pair["sentence2"].lower(), pair["gold_label"]])
        i = i + 1
        if i > max_num:
            break
    # close open file to free up memory
    data.close()

    return SNLI_sentences


# sentence_list is list of lists where the inner lists are [Premise, Hypothesis, label]
sentence_list = get_data("snli_1.0/snli_1.0_train.jsonl", 100)
test_list = get_data("snli_1.0/snli_1.0_test.jsonl", 5)

print(sentence_list)


def sentence_vectoriser(sentence):
    sentence_array = [glove(item, glove_dimension) for item in sentence.split(" ")]
    return sentence_array


def OHencoder(tensor):
    # manually encoding labels. if there were more classes then could use SKlearn's one hot encoder but for 3 classes
    # this seems like overkill
    # label = tensor[i][2]
    enc_dict = {"entailment": [1, 0, 0], "neutral": [0, 1, 0], "contradiction": [0, 0, 1]}
    # loop backwards here so that when deleting items it doesn't skip forward
    for i in range(len(tensor)):
        label = tensor[-i][2]
        if label in enc_dict:
            tensor[-i][2] = enc_dict[label]

        else:
            del tensor[-i]
    return tensor


def pad_sentence_lengths(tensor):
    max_length = 0
    for sen_arr in tensor:
        max_length = max([max_length, len(sen_arr)])

    padding = [0 for a in range(glove_dimension)]

    for i in range(len(tensor)):
        # tensor[i] = np.lib.pad(tensor[i], (0, max_length - len(tensor[i])), "constant", constant_values = 0)
        for m in range((max_length - len(tensor[i]))):
            tensor[i].append(padding)

    return tensor, max_length

OH_sentence_list = OHencoder(sentence_list)
train_premises, max_plength = pad_sentence_lengths([sentence_vectoriser(i[0]) for i in OH_sentence_list])
train_hypothesis, max_hlength = pad_sentence_lengths([sentence_vectoriser(i[1]) for i in OH_sentence_list])
train_labels = np.array([i[2] for i in OH_sentence_list])
print("done")
print(max_plength)

OH_test_list = OHencoder(test_list)
test_premises, max_tplength = pad_sentence_lengths([sentence_vectoriser(i[0]) for i in OH_test_list])
test_hypothesis, max_thlength = pad_sentence_lengths([sentence_vectoriser(i[1]) for i in OH_test_list])
test_labels = np.array([i[2] for i in OH_test_list])

# I had to define batch size here to be able to run the for loop in the attention function.
# There is a way that this can be avoided using tf.scan but I ran out of time trying to work out how to do this.
# means when calculating the score of the test set I will have to do this in batches.
batch_size = 3
tf.reset_default_graph()


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


# I havent bothered to implement the linear layer here
LSTM_size = glove_dimension

# input data as 3d tensors
input_premises = tf.placeholder(tf.float32, shape=[batch_size, max_plength, glove_dimension])
input_hypothesis = tf.placeholder(tf.float32, shape=[batch_size, max_hlength, glove_dimension])

# must change scope so that the 2 lstms when run don't have the same variable name
with tf.variable_scope("first_lstm"):
    LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_size, forget_bias=1.0, state_is_tuple=True)
    p_out, p_state = tf.nn.dynamic_rnn(LSTM_cell, input_premises, dtype=tf.float32,
                                       sequence_length=length(input_premises))

# now want p_state tensor to iniatlize the hypothesis lstm
with tf.variable_scope("second_lstm"):
    LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_size, forget_bias=1.0, state_is_tuple=True)
    h_out, h_state = tf.nn.dynamic_rnn(LSTM_cell, input_hypothesis, initial_state=p_state, dtype=tf.float32,
                                       sequence_length=length(input_hypothesis))

# define variables for the attention function
Wy = tf.Variable(tf.truncated_normal([LSTM_size, LSTM_size]))
Wh = tf.Variable(tf.truncated_normal([LSTM_size, LSTM_size]))
Wv = tf.Variable(tf.truncated_normal([LSTM_size, 1]))
Wp = tf.Variable(tf.truncated_normal([LSTM_size, LSTM_size]))
Wx = tf.Variable(tf.truncated_normal([LSTM_size, LSTM_size]))

# h_out will be tensor with shape [batch size, max time, cell.output_size]. make max time first axis using perm
h_out = tf.transpose(h_out, perm=[1, 0, 2])
last = tf.gather(h_out, int(h_out.get_shape()[0]) - 1)


def attention(H, Y, WY, WH, w, WP, WX):
    b = Y.get_shape()[0]
    seq_l = Y.get_shape()[1]
    # eL = tf.ones([seq_l, 1])
    gather_list = [0 for i in range(seq_l)]

    for i in range(b):
        Y_i = tf.squeeze(tf.gather(Y, [i]))
        H_i = tf.gather(H, [i])
        # use gather instead of outer product to create the tensor we want
        outer = tf.gather(H_i, gather_list)

        M = tf.tanh(tf.matmul(Y_i, WY) + tf.matmul(outer, WH))
        alpha = tf.nn.softmax(tf.matmul(M, w))

        r = tf.matmul(alpha, Y_i, transpose_a=True)

        h_star = tf.tanh(tf.matmul(r, WP) + tf.matmul(H_i, WX))

        if i == 0:
            H_star = h_star
        else:
            H_star = tf.concat(0, [H_star, h_star])

    print(H_star.get_shape())
    return H_star


Att = attention(last, p_out, Wy, Wh, Wv, Wp, Wx)

# final linear layer to get outputs in shape (batch size, 3).
weight = tf.Variable(tf.truncated_normal([LSTM_size, 3]))
bias = tf.Variable(tf.constant(0.1, shape=[3]))
y_ = tf.placeholder(tf.float32, shape=[batch_size, 3])

y = tf.matmul(Att, weight) + bias
# softmax to get predictions
cross_entropy = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# minimise using the ADAM optimiser
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

# now work out how many the model gets right
correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))


init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

no_of_batches = int(len(train_premises)/batch_size)
ptr = 0
for j in range(no_of_batches):
    inp_p, inp_h, out = train_premises[ptr:ptr+batch_size], train_hypothesis[ptr:ptr+batch_size], train_labels[ptr:ptr+batch_size]
    ptr+=batch_size
    sess.run(minimize,{input_premises: inp_p, input_hypothesis: inp_h, y_: out})

right = sess.run(num_correct, {input_premises: inp_p, input_hypothesis: inp_h, y_: out})
ptr = 0
cumu_right = 0
no_test_batches = int(len(test_premises)/batch_size)
for l in range(no_of_batches):
    test_p, test_h, test_l = train_premises[ptr:ptr+batch_size], train_hypothesis[ptr:ptr+batch_size], train_labels[ptr:ptr+batch_size]
    right = sess.run(num_correct, {input_premises: test_p, input_hypothesis: test_h, y_: test_l})
    cumu_right = cumu_right + right
accuracy = 100 * cumu_right/len(train_premises)
print("accuracy is {0} percent".format(accuracy))

# This is just to help you to be able to get this mechanism working for yourself, not the final result. If you want to implement 
#this for yourself remember to read the paper and add in the Linear layer. Also for speed purposes I suggest you use buckets instead
# of combining your data set into one and then padding. This will mean there a lot less redundant values in your data set.
