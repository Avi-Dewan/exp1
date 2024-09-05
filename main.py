import argparse
import os
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils.metrics import accuracy, class_accuracy
from utils.util import generated_sample
from models.gan_model import Generator, Discriminator
from gan_trainer.training_step import classifier_train_step, generator_train_step, discriminator_train_step
from gan_trainer.pretraining import gan_pretraining
from data.cifarloader import CIFAR10Loader
from models.resnet import ResNet, BasicBlock

# Argument parser setup
parser = argparse.ArgumentParser(description='Generative Pseudo-label Refinement for Unsupervised Domain Adaptation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data loading parameters
parser.add_argument('--data_path', type=str, default='./datasets')
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--verbose', type=str, default=False, help='Verbose mode')

# GPU
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

# Number of classes
parser.add_argument('--n_classes', type=int, default=5)

# Classifier pretraining parameters
parser.add_argument('--n_epochs_cls_pretraining', type=int, default=1)
parser.add_argument('--lr_cls_pretraining', type=float, default=1e-4)

# GAN pretraining parameters
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--lr_d_pretraining', type=float, default=1e-4)
parser.add_argument('--lr_g_pretraining', type=float, default=1e-4)
parser.add_argument('--n_epochs_gan_pretraining', type=int, default=1)

# Training parameters
parser.add_argument('--lr_cls_training', type=float, default=1e-4)
parser.add_argument('--lr_d_training', type=float, default=1e-4)
parser.add_argument('--lr_g_training', type=float, default=1e-4)
parser.add_argument('--n_epochs_training', type=int, default=1)

# Paths
parser.add_argument('--results_path', type=str, default='./results')
parser.add_argument('--pretraining_path', type=str, default='./results/pretraining')
parser.add_argument('--training_path', type=str, default='./results/training')
parser.add_argument('--cls_pretraining_path', type=str, default='./results/pretraining/classifier_pretrained.pth')

args = parser.parse_args()
args.device = torch.device("cuda" if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

# Updating paths based on args
args.img_training_path = os.path.join(args.training_path, 'images')
args.models_training_path = os.path.join(args.training_path, 'models')

# --------------------
#   Data loading
# --------------------
train_loader = CIFAR10Loader(root=args.data_path, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list=range(5, 10))
eval_loader = CIFAR10Loader(root=args.data_path, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list=range(5, 10))

# --------------------
#   Pretraining
# --------------------
# Classifier pretraining on source data
classifier = ResNet(BasicBlock, [2, 2, 2, 2], args.n_classes).to(args.device)
state_dict = torch.load(args.cls_pretraining_path)
classifier.load_state_dict(state_dict, strict=False)

if args.verbose:
    class_accuracy(classifier, eval_loader, list(range(5, 10)))

# GAN pretraining on target data annotated by classifier
generator = Generator(args.n_classes, args.latent_dim, args.img_size).to(args.device)
discriminator = Discriminator(args.n_classes, args.img_size).to(args.device)

generator, discriminator = gan_pretraining(generator, discriminator, classifier, train_loader,
                                           args.lr_g_pretraining, args.lr_d_pretraining,
                                           args.latent_dim, args.n_classes,
                                           args.n_epochs_gan_pretraining, args.img_size, args.pretraining_path)

if args.verbose:
    generated_sample(generator, args.n_classes, args.latent_dim, args.img_size)
    print('Baseline Algorithm')
    print(f'Test accuracy: {100*accuracy(classifier, eval_loader):.2f}%')
    class_accuracy(classifier, eval_loader, list(range(5, 10)))

# --------------------
#     Training
# --------------------

# Classifier loss and optimizer
cls_criterion = nn.CrossEntropyLoss().to(args.device)
cls_optimizer = optim.Adam(classifier.parameters(), lr=args.lr_cls_training)

# GAN loss and optimizers
gan_criterion = nn.BCELoss().to(args.device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d_training)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr_g_training)

# Create folder for models and generated images
os.makedirs(args.img_training_path, exist_ok=True)
os.makedirs(args.models_training_path, exist_ok=True)

# Training
print('Starting Training GAN')
for epoch in range(args.n_epochs_training):
    print(f'Starting epoch {epoch}/{args.n_epochs_training}...', end=' ')
    g_loss_list = []
    d_loss_list = []
    c_loss_list = []
    for i, ((images, _), _, _) in enumerate(train_loader):
        generator.train()

        # Step 3: Sample latent space and random labels
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).to(args.device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, args.n_classes, args.batch_size))).to(args.device)

        # Step 4: Generate fake images
        fake_images = generator(z, fake_labels).view(args.batch_size, 3, args.img_size, args.img_size)

        # Step 5: Update classifier
        c_loss = classifier_train_step(classifier, fake_images, cls_optimizer, cls_criterion, fake_labels)
        c_loss_list.append(c_loss)

        # Step 6: Sample real images from target data
        real_images = Variable(images).to(args.device)

        # Step 7: Infer labels using classifier
        x = classifier(real_images)
        print(x.shape)
        _, labels_classifier = torch.max(classifier(real_images), dim=1)

        print(labels_classifier.shape)

        break

        # Step 8: Update discriminator
        d_loss = discriminator_train_step(discriminator, generator, d_optimizer, gan_criterion,
                                          real_images, labels_classifier, args.latent_dim, args.n_classes)
        d_loss_list.append(d_loss)

        # Step 9: Update Generator
        g_loss = generator_train_step(discriminator, generator, g_optimizer, gan_criterion,
                                      args.batch_size, args.latent_dim, labels=labels_classifier)
        g_loss_list.append(g_loss)

    generator.eval()

    z = Variable(torch.randn(args.n_classes, args.latent_dim)).to(args.device)
    gen_labels = Variable(torch.LongTensor(np.arange(args.n_classes))).to(args.device)

    gen_imgs = generator(z, gen_labels).view(-1, 3, args.img_size, args.img_size)
    save_image(gen_imgs.data, os.path.join(args.img_training_path, f'epoch_{epoch:02d}.png'), nrow=args.n_classes, normalize=True)
    torch.save(classifier.state_dict(), os.path.join(args.models_training_path, f'{epoch:02d}_cls.pth'))
    torch.save(generator.state_dict(), os.path.join(args.models_training_path, f'{epoch:02d}_gen.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.models_training_path, f'{epoch:02d}_dis.pth'))

    print(f"[D loss: {np.mean(d_loss_list)}] [G loss: {np.mean(g_loss_list)}] [C loss: {np.mean(c_loss_list)}]")
print('Finished Training GAN')
print('\n')

# --------------------
#     Final Model
# --------------------
print(f'MNIST accuracy: {100*accuracy(classifier, eval_loader)::.2f}%')
class_accuracy(classifier, eval_loader, list(range(5, 10)))
if args.verbose:
    generated_sample(generator, args.n_classes, args.latent_dim, args.img_size)
