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
ADV_TRAIN_EPOCHS = 100
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'

# MAIN
oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
oracle.load_state_dict(torch.load(oracle_state_dict_path))
oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)

gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, d_steps, epochs):
    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                loss = discriminator.batchBCELoss(inp, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data[0]

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            print(' average_loss = %.4f' % total_loss)


if CUDA:
    oracle = oracle.cuda()
    gen = gen.cuda()
    dis = dis.cuda()
    oracle_samples = oracle_samples.cuda()

# GENERATOR MLE TRAINING
print('Starting Generator MLE Training...')
gen_optimizer = optim.Adam(gen.parameters())

# for epoch in range(MLE_TRAIN_EPOCHS):
#     print('epoch %d : ' % (epoch+1), end='')
#     sys.stdout.flush()
#     total_loss = 0
#
#     for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
#         inp, target = helpers.prepare_generator_batch(oracle_samples[i:i+BATCH_SIZE], start_letter=START_LETTER,
#                                                      gpu=CUDA)
#         gen_optimizer.zero_grad()
#         loss = gen.batchNLLLoss(inp, target)
#         loss.backward()
#         gen_optimizer.step()
#
#         total_loss += loss.data[0]
#
#         if (i/BATCH_SIZE) % ceil(ceil(POS_NEG_SAMPLES/float(BATCH_SIZE))/10.) == 0:  # roughly every 10% of an epoch
#             print('.', end='')
#             sys.stdout.flush()
#
#     # each loss in a batch is loss per sample
#     total_loss = total_loss/ceil(POS_NEG_SAMPLES/float(BATCH_SIZE))/MAX_SEQ_LEN
#
#     # sample from generator and compute oracle NLL
#     s = gen.sample(POS_NEG_SAMPLES/10)
#     inp, target = helpers.prepare_generator_data(s, start_letter=START_LETTER, gpu=CUDA)
#     oracle_loss = oracle.batchNLLLoss(inp, target)/MAX_SEQ_LEN
#
#     print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss.data[0]))

gen.load_state_dict(torch.load(pretrained_gen_path))

# TRAIN DISCRIMINATOR
print('\nStarting Discriminator Training...')
dis_optimizer = optim.Adam(dis.parameters())
train_discriminator(dis, dis_optimizer, oracle_samples, gen, 50, 3)

# ADVERSARIAL TRAINING
print('\nStarting Adersarial Training...')
