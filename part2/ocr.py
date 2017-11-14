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

# Initial and transition probabilities
def train(data):

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

def emission_probability(test_img, train_ltr):

    cond_prob = 1
    train_img = train_images[train_ltr]

    pstrt, pend = (3, 24)
    for i in range(pstrt,pend):
        for j in range(0,14):
            if test_img[i][j] == train_img[i][j]:
                cond_prob =  cond_prob * 0.95
            else:
                cond_prob = cond_prob * 0.10

    return cond_prob


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

def hmm_ve(test_letters):
    pred_test_str = ''
    var_el = np.zeros((len(TRAIN_LETTERS), len(test_letters)))

    for i in range(len(test_letters)):
        sub_img = test_letters[i]
        for j in range(len(TRAIN_LETTERS)):
            letter = TRAIN_LETTERS[j]

            if i == 0:

                if letter in letter_initial:
                    var_el[j, 0] = letter_initial[letter] * emission_probability(sub_img, letter)
                else:
                    var_el[j, 0] = emission_probability(sub_img, letter)
            else:
                temp_sum = 0.0
                ##Variable elimination
                for k in range(len(TRAIN_LETTERS)):
                    letter_prev = TRAIN_LETTERS[k]
                    if (letter_prev.lower(), letter.lower()) in letter_transition:
                        if (letter_prev, letter.lower()) in letter_transition:
                            trans_prob = min(letter_transition[(letter_prev, letter.lower())], letter_transition[(letter_prev.lower(), letter.lower())])
                        else:
                            trans_prob = letter_transition[(letter_prev.lower(), letter.lower())]
                        temp_sum += (var_el[k, i - 1] * trans_prob)
                    else:
                        temp_sum += 0.000000001

                var_el[j, i] = temp_sum * emission_probability(sub_img, letter)

        max_prob_arg = np.argmax(var_el[:, i])
        pred_test_str += TRAIN_LETTERS[max_prob_arg]

    return pred_test_str


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
                    vit_pred[j, 0] = 0.000000001 * emission_probability(sub_img, letter)
            else:
                max_prob = 0.0
                for k in range(len(TRAIN_LETTERS)):
                    letter_prev = TRAIN_LETTERS[k]

                    if (letter_prev.lower(), letter.lower()) in letter_transition:
                        # if (letter_prev, letter.lower()) in letter_transition:
                        #     trans_prob = min(letter_transition[(letter_prev, letter.lower())], letter_transition[(letter_prev.lower(), letter.lower())])
                        # else:
                        trans_prob = letter_transition[(letter_prev.lower(), letter.lower())]
                        prob = (vit_pred[k, i - 1] * trans_prob)
                    else:
                        prob = 0.000000001

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

''' Training '''
letter_initial, letter_dict, letter_transition = train(read_data(train_txt_fname))


''' Implement the three methods for character recognition '''

print " Simple: "+simplified(test_images)
print " HMM VE: "+hmm_ve(test_images)
print "HMM MAP: "+hmm_viterbi(test_images)

