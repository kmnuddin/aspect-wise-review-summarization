#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import keras.backend as K
import pandas as pd
import numpy as np
import rouge


# In[2]:


reviews = pd.read_csv('reviews2.csv')
summaries = pd.read_csv('summaries2.csv')
reviews = reviews.values
summaries = summaries.values


r_train, r_test, s_train, s_test = train_test_split(reviews, summaries, test_size=0.3, random_state=2)


summary_decoder_input = '<sos> ' + s_train
summary_decoder_target = s_train + ' <eos>'
summary_decoder_input = summary_decoder_input.flatten().tolist()
summary_decoder_target = summary_decoder_target.flatten().tolist()


r_train = r_train.flatten().tolist()
r_test = r_test.flatten().tolist()
s_train = s_train.flatten().tolist()
s_test = s_test.flatten().tolist()

reviews = reviews.flatten().tolist()
summaries = summaries.flatten().tolist()
print('number of examples: ', len(reviews))
print('number of training examples: ', len(r_train))
print('number of testing examples: ', len(r_test))
print('decoder input: ', summary_decoder_input[0])
print('decoder target: ', summary_decoder_target[0])


# In[3]:


MAX_NUM_WORDS = 20000


# In[4]:


tokenizer_r = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_r.fit_on_texts(reviews)
review_sequences = tokenizer_r.texts_to_sequences(reviews)

max_review_len = max(len(s) for s in review_sequences)

tokenizer_s = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_s.fit_on_texts(summaries)
summary_sequences = tokenizer_s.texts_to_sequences(summaries)


max_summary_len = max(len(s) for s in summary_sequences) + 1

print('maximum review length: ', max_review_len)
print('maximum summary length: ', max_summary_len)


# In[5]:


tokenizer_r_train = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_r_train.fit_on_texts(r_train)
r_train_sequences = tokenizer_r_train.texts_to_sequences(r_train)

word2idx_inputs = tokenizer_r_train.word_index

tokenizer_s_train = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_s_train.fit_on_texts(summary_decoder_input + summary_decoder_target)
decoder_input_sequences = tokenizer_s_train.texts_to_sequences(summary_decoder_input)
decoder_target_sequences = tokenizer_s_train.texts_to_sequences(summary_decoder_target)

word2idx_summaries = tokenizer_s_train.word_index

print('Review vocabulary size: ', len(word2idx_inputs))
print('Summary vocabulary size: ', len(word2idx_summaries))


# In[6]:


tokenizer_r_test = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_r_test.fit_on_texts(r_test)
r_test_sequences = tokenizer_r_test.texts_to_sequences(r_test)

test_sequences = pad_sequences(r_test_sequences, maxlen=max_review_len)


# In[7]:


encoder_input = pad_sequences(r_train_sequences, maxlen=max_review_len)

print('encoder input shape: ', encoder_input.shape)

decoder_input = pad_sequences(decoder_input_sequences, maxlen=max_summary_len, padding='post')

print('decoder input shape:', decoder_input.shape)

decoder_target = pad_sequences(decoder_target_sequences, maxlen=max_summary_len, padding='post')

print('decoder target shape:', decoder_target.shape)

num_words_summaries = len(word2idx_summaries) + 1
num_words_reviews = len(word2idx_inputs) + 1

decoder_one_hot_targets = np.zeros((len(r_train), max_summary_len, num_words_summaries), dtype='uint8')

for i, d in enumerate(decoder_target):
    for t, word in enumerate(d):
        decoder_one_hot_targets[i, t, word] = 1
print('decoder one hot targets shape: ', decoder_one_hot_targets.shape)
print('encoder input[0]: ', encoder_input[0])
print('decoder input[0]: ', decoder_input[0])
print('decoder target[0]:', decoder_target[0])


# In[8]:


encoder_model = load_model('enc_model3.h5')
decoder_model = load_model('dec_model3.h5')


# In[9]:


encoder_model.summary()


# In[10]:


decoder_model.summary()


# In[11]:


idx2word_review = {v:k for k, v in word2idx_inputs.items()}
idx2word_summary = {v:k for k, v in word2idx_summaries.items()}
idx2word_summary


# In[12]:


def decode_sequence(input_seq):
  
    states_value = encoder_model.predict(input_seq)


    target_seq = np.zeros((1, 1))


    target_seq[0, 0] = word2idx_summaries['<sos>']


    eos = word2idx_summaries['<eos>']


    output_sentence = []
    for _ in range(max_summary_len):
        output_tokens, h, c = decoder_model.predict(
          [target_seq] + states_value
        )
   
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_summary[idx]
            output_sentence.append(word)


        target_seq[0, 0] = idx

        states_value = [h, c]

    return ' '.join(output_sentence)


# In[13]:


s_pred = []
for i in range(len(test_sequences)):
    s_pred.append(decode_sequence(test_sequences[i:i+1]))


# In[30]:


r_test[0]


# In[28]:


s_test[0]


# In[29]:


s_pred[0]


# In[15]:


s_pred_train = []
for i in range(len(encoder_input)):
    s_pred_train.append(decode_sequence(encoder_input[i:i+1]))


# In[16]:


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


# In[17]:


print('Evaluation with {}'.format('Training Set'))
evaluator = rouge.Rouge(metrics=['rouge-n'],
                       max_n=4,
                       limit_length=False,
                       length_limit=100,
                       length_limit_type='words',
                       apply_avg='Avg',
                       apply_best=False,
                       alpha=0.5, # Default F1_score
                       weight_factor=1.2,
                       stemming=True)

scores = evaluator.get_scores(s_pred_train, s_train)
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(results['p'], results['r'], results['f']))


# In[18]:


print('Evaluation with {}'.format('Testing Set'))
scores = evaluator.get_scores(s_pred, s_test)
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(results['p'], results['r'], results['f']))


# In[32]:


print('---------------Summaries using test set------------------')
print()
while(True):
    i = np.random.randint(0, 3022)
    print('Review:', r_test[i])
    print()
    print('Original Summary: ', s_test[i])
    print()
    print('Generated Summary: ', s_pred[i])
    print()
    ans = input("Continue? [Y/n]")
    print('-------------------------------------------------')
    if ans and ans.lower().startswith('n'):
        break


# In[25]:


print('---------------Summaries using training set------------------')
print()
while(True):
    i = np.random.randint(0, len(r_train))
    print('Review:', r_train[i])
    print()
    print('Original Summary: ', s_train[i])
    print()
    print('Generated Summary: ', s_pred_train[i])
    print()
    ans = input("Continue? [Y/n]")
    print('-------------------------------------------------')
    if ans and ans.lower().startswith('n'):
        break

