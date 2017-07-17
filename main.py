from __future__ import print_function
from math import ceil
import sys

import torch
import torch.optim as optim

import generator
import discriminator
import helpers


CUDA = False
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'

# MAIN
oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN, VOCAB_SIZE, MAX_SEQ_LEN, cuda=CUDA)
oracle.load_state_dict(torch.load(oracle_state_dict_path))
oracle_samples = torch.load(oracle_samples_path)

gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN, VOCAB_SIZE, MAX_SEQ_LEN, cuda=CUDA)

# GENERATOR MLE TRAINING
if CUDA:
    oracle = oracle.cuda()
    gen = gen.cuda()

print('Starting Generator MLE Training...')
optimizer = optim.Adam(gen.parameters())

for epoch in range(MLE_TRAIN_EPOCHS):
    print('epoch %d : ' % (epoch+1), end='')
    sys.stdout.flush()
    total_loss = 0

    for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
        inp, target = helpers.prepare_generator_data(oracle_samples[i:i+BATCH_SIZE], start_letter=START_LETTER,
                                                     cuda=CUDA)
        optimizer.zero_grad()
        loss = gen.batchNLLLoss(inp, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

        if (i/BATCH_SIZE) % ceil(ceil(POS_NEG_SAMPLES/float(BATCH_SIZE))/10.) == 0:  # roughly every 10% of an epoch
            print('.', end='')
            sys.stdout.flush()

    # each loss in a batch is loss per sample
    total_loss = total_loss/ceil(POS_NEG_SAMPLES/float(BATCH_SIZE))/MAX_SEQ_LEN

    # sample from generator and compute oracle NLL
    s = gen.sample(1000)
    inp, target = helpers.prepare_generator_data(s, start_letter=START_LETTER, cuda=CUDA)
    oracle_loss = oracle.batchNLLLoss(inp, target)/MAX_SEQ_LEN

    print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss.data[0]))


