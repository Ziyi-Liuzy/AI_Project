import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

#--------------------------------Parameter configuration---------------------------------#
class Config():

    # Path setting
    result_save_path = 'results/' 
    d_net_path = 'snapshots/dnet.pth' 
    g_net_path = 'snapshots/gnet.pth' 
    img_path = 'painting/' 

    img_size = 96 
    batch_size = 256 
    max_epoch = 300 
    noise_dim = 100 
    feats_channel = 64 

opt = Config() 

# Generate the result folder and snapshots folder
if not os.path.exists('results'):
    os.mkdir('results')  
if not os.path.exists('snapshots'):
    os.mkdir('snapshots') 

#---------------------------------Generater design----------------------------------#
class Gnet(nn.Module):
    def __init__(self, opt):
        super(Gnet, self).__init__()
        self.feats = opt.feats_channel
        self.generate = nn.Sequential(

           #input = (n, c, h, w = 256, 100, 1, 1)
            nn.ConvTranspose2d(in_channels=opt.noise_dim, out_channels=self.feats * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.ReLU(inplace=True),
            # deconv = (input - 1 ) * stride + k -2* padding = (1- 1)*1  +4-0 = 4
            #output = (256, 800 ,4, 4)

            nn.ConvTranspose2d(in_channels=self.feats * 8, out_channels=self.feats * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (4-1)*2 + 4-2 = 8

            nn.ConvTranspose2d(in_channels=self.feats * 4, out_channels=self.feats * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (8-1)*2 + 4-2 = 16


            nn.ConvTranspose2d(in_channels=self.feats * 2, out_channels=self.feats, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (16-1)*2 + 4-2 = 32

            nn.ConvTranspose2d(in_channels=self.feats, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),

            nn.Tanh(),
            # decon = (input - 1)*stride + k - 2*padding = (32-1)*3 + 5-2 = 96
            #output = (n, c, h, w = 256, 3, 96, 96)

    )

    def forward(self, x):
        return self.generate(x)

#----------------------------------Discriminantor design----------------------------#
class Dnet(nn.Module):
    def __init__(self, opt):
        super(Dnet, self).__init__()
        self.feats = opt.feats_channel
        self.discrim = nn.Sequential(

            #input = （n, c, h, w = 256, 3, 96, 96)

            nn.Conv2d(in_channels=3, out_channels= self.feats, kernel_size= 5, stride= 3, padding= 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),
            #con = (input - k + 2 * padding ) / stride +  1 = (256 - 5 + 2) / 3 + 1 = 128



            nn.Conv2d(in_channels= self.feats, out_channels= self.feats * 2, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.feats* 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels= self.feats * 2, out_channels= self.feats * 4, kernel_size= 4, stride= 2, padding= 1,bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.LeakyReLU(0.2, True),


            nn.Conv2d(in_channels= self.feats * 4, out_channels= self.feats * 8, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.feats *8),
            nn.LeakyReLU(0.2, True),


            nn.Conv2d(in_channels= self.feats * 8, out_channels= 1, kernel_size= 4, stride= 1, padding= 0, bias=True),

            nn.Sigmoid()

            #output = ( n, c, h, w = 256, 1, 1, 1)
        )

    def forward(self, x):
        return self.discrim(x).view(-1) 

g_net, d_net = Gnet(opt), Dnet(opt)

#---------------------------------Data preprocessing---------------------------------#

transforms = torchvision.transforms.Compose([
    # resize Image size
    torchvision.transforms.Resize(opt.img_size), 
    # Center crop image size
    torchvision.transforms.CenterCrop(opt.img_size),
    # array tensor.float(), adapted to the data format of the torch framework
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(root=opt.img_path, transform=transforms)

dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size, 
    num_workers = 0,
    drop_last = True
)

# set device(cpu or cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

g_net.to(device)
d_net.to(device)

optimize_g = torch.optim.Adam(g_net.parameters(), lr= 2e-4, betas=(0.5, 0.999))
optimize_d = torch.optim.Adam(d_net.parameters(), lr= 2e-4, betas=(0.5, 0.999))
# optimize_d = torch.optim.SGD(d_net.parameters(), lr= 2e-4)

#BCEloss, find the binary classification probability
criterions = nn.BCELoss().to(device) 

# Define the tag and start injecting the generator's input noise
true_labels = torch.ones(opt.batch_size).to(device) 
fake_labels = torch.zeros(opt.batch_size).to(device) 

# Generate N(1,1) standard normal distribution, 100 dimensional, 256 numbers of random noise
noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

test_noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

#-----------------------------------Start training----------------------------------#
#Load weight file
try:
    g_net.load_state_dict(torch.load(opt.g_net_path)) 
    d_net.load_state_dict(torch.load(opt.d_net_path))
    print('Load successfully. Continue training')
except:
    print('Load failed, retrain')

for epoch in range(opt.max_epoch):  

    for itertion, (img, _) in tqdm((enumerate(dataloader))):
        real_img = img.to(device)

        # The discriminator is trained 5 times and the generator is trained 1 time
        if itertion % 1 == 0: 

            optimize_d.zero_grad() 

            # Real data input discriminant network
            output = d_net(real_img)
            # The discriminator is expected to identify the real image as a positive sample with a label of 1
            d_real_loss = criterions(output, true_labels) 
            fake_image = g_net(noises.detach()).detach() 

            # Generate data input to the discriminant network
            output = d_net(fake_image) 
            # The discriminator is expected to identify the generated image as a negative sample with a label of 0
            d_fake_loss = criterions(output, fake_labels)

            #loss Fusion calculation
            d_loss = (d_fake_loss + d_real_loss) / 2 
            

            d_loss.backward() 
            optimize_d.step() 

        # Generate network optimizer gradient clear
        if itertion % 1 == 0:
            optimize_g.zero_grad() 
            noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1)) 

            # Calculate the probability that the generated image is real
            fake_image = g_net(noises) 
            output = d_net(fake_image) 
            g_loss = criterions(output, true_labels)

            g_loss.backward() 
            optimize_g.step() 

    # randomly generate 256 noises
    vid_fake_image = g_net(test_noises) 
    # Save the first 16 images
    torchvision.utils.save_image(vid_fake_image.data[:16], "%s/%s.png" % (opt.result_save_path, epoch), normalize=True) 
    torch.save(d_net.state_dict(),  opt.d_net_path)
    torch.save(g_net.state_dict(),  opt.g_net_path)
    #loss visualization
    print('epoch:', epoch, '---D-loss:---', d_loss.item(), '---G-loss:---', g_loss.item()) 

