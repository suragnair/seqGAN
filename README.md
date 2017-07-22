# seqGAN
A PyTorch implementation of "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient." (Yu, Lantao, et al.). The code is highly simplified, commented and (hopefully) straightforward to understand. The policy gradients implemented are also much simpler than in the original work (https://github.com/LantaoYu/SeqGAN/) and do not involve rollouts- a single reward is used for the entire sentence (inspired by http://karpathy.github.io/2016/05/31/rl/).

The architectures used are different than those in the orignal work. Specifically, a recurrent bidirectional GRU network is used as the discriminator. 

You are encouraged to raise any doubts regarding the working of the code as Issues.

# Hacks and Observations
The following hacks (borrowed from https://github.com/soumith/ganhacks) seem to have worked in this case:
- Training Discriminator a lot more than Generator (Generator is trained only for one batch of examples, and increasing the batch size hurts stability)
- Using Adam for Generator and Adagrad for Discriminator
- Tweaking learning rate for Generator in GAN phase
- Using dropout in both training and testing phase
- Stablity is extremely sensitive to almost every parameter :/