#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors:
# Bhargavi Chalasani (bchalasa)
# Parag Juneja (pjuneja)
# Ankit Saxena (ansaxena)
# (based on skeleton code by D. Crandall, Oct 2017)
#
'''
The Initial state, Transition and Letter probabilities are calculated from the input text file in the train_probabilities method.

To calculate Emission probability, each pixel of letters in the test image are compared to the pixels of the letters in the training image
which is a noise-free version of the letters to be predicted. Naive Bayes is used to predict the most likely letter given the observed pixels.

To improve the Emission probability, considered all the pixels unique to a letter by comparing with the rest of the letters given in the training image.
Weights are assigned to each pixel based on the uniqueness of a pixel with respect to the training image.
This approach gave a 30% bump in average letter accuracy over the test data.

For any missing Transition, Initial state or Letter probabilities assumed a small value of 10^-15

'''

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '1' if px[x, y] < 1 else '0' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


# Read in training data file
def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]
    return exemplars

def read_data_others(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += data
    return exemplars

# Initial and transition probabilities
def train_probabilities(data):

    letter_initial = {}
    letter_dict = {}
    letter_transition = {}

    word_list = [w for line in data for w in line[0]]

    space_count = len(word_list) - len(data)
    letter_dict[" "] = space_count

    for word in word_list:
        for i in range(len(word)):
            if word[i] in TRAIN_LETTERS:
                if i == 0:
                    ##Initial probability
                    if word[i] not in letter_initial:
                        letter_initial[word[i]] = 1
                    else:
                        letter_initial[word[i]] += 1

                ##Letter probability
                if word[i] not in letter_dict:
                    letter_dict[word[i]] = 1
                else:
                    letter_dict[word[i]] += 1

                ##Transition probability
                if (i + 1) != len(word):
                    if (word[i], word[i+1]) not in letter_transition:
                        letter_transition[(word[i], word[i+1])] = 1
                    else:
                        letter_transition[(word[i], word[i+1])] += 1
                else:
                    if (word[i], " ") not in letter_transition:
                        letter_transition[(word[i], " ")] = 1
                    else:
                        letter_transition[(word[i], " ")] += 1

    letter_initial = dict((key, float(value) / len(data)) for (key, value) in letter_initial.items())
    letter_dict = dict((key, float(value) / sum(letter_dict.values())) for (key, value) in letter_dict.items())
    letter_transition = dict((key, float(value) / sum(letter_transition.values())) for (key, value) in letter_transition.items())

    return (letter_initial, letter_dict,letter_transition)

def get_pixel_weights(train_ltr):

    ltr_1 = train_ltr
    initial_pixels = train_images[ltr_1]
    set_pixels = np.zeros((25, 14))

    for ltr_2 in TRAIN_LETTERS:
        if ltr_1 != ltr_2:
            comp_pixels = train_images[ltr_2]
            for i in range(0, 25):
                for j in range(0, 14):
                    if initial_pixels[i][j] != comp_pixels[i][j]:
                        set_pixels[i][j] += 1
                    else:
                        set_pixels[i][j] = 0

    cut_1 = np.percentile(set_pixels, 95)
    cut_2 = np.percentile(set_pixels, 90)
    cut_3 = np.percentile(set_pixels, 85)

    weighted_pixels = np.zeros((25, 14))

    for i in range(0, 25):
        for j in range(0, 14):
            if set_pixels[i][j] > 0:

                weighted_pixels[i][j] = 4
                if ltr_1 != " ":
                    if set_pixels[i][j] > cut_3:
                        weighted_pixels[i][j] = 3
                    if set_pixels[i][j] > cut_2:
                        weighted_pixels[i][j] = 2
                    if set_pixels[i][j] > cut_1:
                        weighted_pixels[i][j] = 1

    return weighted_pixels

def emission_probability(test_img, train_ltr):

    cond_prob = 1
    train_img = train_images[train_ltr]
    weighted_img = train_images_weighted[train_ltr]
    weighted_prob = {1.0: 0.99, 2.0: 0.97, 3.0: 0.95, 4.0: 0.90, 0.0: 0.40}

    pstrt, pend = (3, 24)
    for i in range(pstrt,pend):
        for j in range(0,14):
            if test_img[i][j] == train_img[i][j]:
                cond_prob = cond_prob * weighted_prob[weighted_img[i][j]]
            else:
                cond_prob = cond_prob * 0.37

    return cond_prob

# Implement Naive Bayes
def simplified(test_letters):
    pred_test_str = ''

    for sub_img in test_letters:
        max_prob = 0
        max_prob_ltr = ''
        for letter in TRAIN_LETTERS:
            # denominator ignored for faster computation
            if letter in letter_dict:
                prob = float(emission_probability(sub_img, letter) * letter_dict[letter])
                if prob > max_prob:
                    max_prob = prob
                    max_prob_ltr = letter

        pred_test_str = pred_test_str + max_prob_ltr

    return pred_test_str

impute_value = 1e-15

# Implement Variable Elimination
def hmm_ve(test_letters):
    pred_test_str = ''
    var_el = np.zeros((len(TRAIN_LETTERS), len(test_letters)))

    for i in range(len(test_letters)):
        sub_img = test_letters[i]
        for j in range(len(TRAIN_LETTERS)):
            letter = TRAIN_LETTERS[j]

            if i == 0:

                if letter in letter_initial:
                    var_el[j, 0] = letter_initial[letter.lower()] * emission_probability(sub_img, letter)
                else:
                    var_el[j, 0] = impute_value * emission_probability(sub_img, letter)
            else:
                temp_sum = 0.0
                ##Variable elimination
                for k in range(len(TRAIN_LETTERS)):
                    letter_prev = TRAIN_LETTERS[k]
                    if (letter_prev.lower(), letter.lower()) in letter_transition:
                        trans_prob = letter_transition[(letter_prev.lower(), letter.lower())]
                        temp_sum += (var_el[k, i - 1] * trans_prob)
                    else:
                        temp_sum += impute_value

                var_el[j, i] = temp_sum * emission_probability(sub_img, letter)

        max_prob_arg = np.argmax(var_el[:, i])
        pred_test_str += TRAIN_LETTERS[max_prob_arg]

    return pred_test_str

# Implement Viterbi
def hmm_viterbi(test_letters):
    pred_test_str = ''

    vit_pred = np.zeros((len(TRAIN_LETTERS), len(test_letters)))

    for i in range(len(test_letters)):
        sub_img = test_letters[i]
        for j in range(len(TRAIN_LETTERS)):
            letter = TRAIN_LETTERS[j]
            if i == 0:
                if letter.lower() in letter_initial:
                    vit_pred[j, 0] = letter_initial[letter.lower()] * emission_probability(sub_img, letter)
                else:
                    vit_pred[j, 0] = impute_value * emission_probability(sub_img, letter)
            else:
                max_prob = 0.0
                for k in range(len(TRAIN_LETTERS)):
                    letter_prev = TRAIN_LETTERS[k]

                    if (letter_prev.lower(), letter.lower()) in letter_transition:
                        trans_prob = letter_transition[(letter_prev.lower(), letter.lower())]
                        prob = (vit_pred[k, i - 1] * trans_prob)
                    else:
                        prob = impute_value

                    if prob > max_prob:
                        max_prob = prob

                vit_pred[j, i] = max_prob * emission_probability(sub_img, letter)

        max_prob_arg = np.argmax(vit_pred[:, i])
        pred_test_str += TRAIN_LETTERS[max_prob_arg]

    return pred_test_str

####
# Main program

TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_images = load_training_letters(train_img_fname)
test_images = load_letters(test_img_fname)
train_images_weighted = {ltr: get_pixel_weights(ltr) for ltr in TRAIN_LETTERS }

''' Training '''
if train_txt_fname == 'bc.train':
    train_text_data = read_data(train_txt_fname)
else:
    train_text_data = read_data_others(train_txt_fname)

letter_initial, letter_dict, letter_transition = train_probabilities(train_text_data)


''' Implement the three methods for character recognition '''

print " Simple: "+simplified(test_images)
print " HMM VE: "+hmm_ve(test_images)
print "HMM MAP: "+hmm_viterbi(test_images)

