import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    
    def __init__(self,   
                in_channels, 
                out_channels,              
                kernel_size, 
                padding=0, 
                stride=1, 
                dilation=1,
                groups=1,                                                   
                bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels,
              kernel_size, stride=stride, padding=padding, bias=bias)
        # define the scale v
        size = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        scale = self.weight.data.new(size, size)
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)
        self.scale2 = nn.Parameter(scale, requires_grad=True)

        self.noise = False
        if self.noise:
            self.alpha_w1 = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.02, requires_grad = True)
            self.alpha_w2 = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.02, requires_grad = True)

    def forward(self, input, space1=None, space2=None):

        if space1 is not None or space2 is not None:
            sz =  self.weight.grad.data.size(0)
            if self.noise:
                with torch.no_grad():
                    std = self.weight.std().item()
                    noise = self.weight.clone().normal_(0,std)
            if space2 is None:
                
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                # print(real_scale1.type(), space1.type(), self.weight.type())
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                #[chout, chinxkxk]  [chinxkxk, chinxkxk]
                proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())

                diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight 

            if space1 is None:

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
     
                proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space2, space2.transpose(1,0))).view(self.weight.size())

                #masked_weight = proj_weight + self.weight - diag_weight
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight 
            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight.view(sz,-1),norm_project1).view(self.weight.size())
                diag_weight1 = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight.view(sz,-1),norm_project2).view(self.weight.size())
                diag_weight2 = torch.mm(self.weight.view(sz,-1),torch.mm(space2, space2.transpose(1,0))).view(self.weight.size())

                if self.noise and self.training:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight + ((self.alpha_w2 + self.alpha_w1)/2) * noise * self.noise
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight
       
        else:
            masked_weight = self.weight

        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# Define specific linear layer
class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)


        # define the scale v
        scale = self.weight.data.new(self.weight.size(1), self.weight.size(1))
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)
        self.scale2 = nn.Parameter(scale, requires_grad=True)
        self.alpha_w1 = nn.Parameter(torch.ones(self.weight.size())*0.02, requires_grad = True)
        self.alpha_w2 = nn.Parameter(torch.ones(self.weight.size())*0.02, requires_grad = True)

        self.noise = False
        #self.fixed_scale = scale
    def forward(self, input, space1=None, space2=None):
        if not self.training:
           self.noise = False


        if space1 is not None or space2 is not None:
            sz =  self.weight.grad.data.size(0)
            if self.noise:
                with torch.no_grad():
                    std = self.weight.std().item()
                    noise = self.weight.clone().normal_(0,std)
            if space2 is None:

                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))

                proj_weight = torch.mm(self.weight,norm_project)

                diag_weight = torch.mm(self.weight,torch.mm(space1, space1.transpose(1,0)))
                # masked_weight = proj_weight + self.weight - diag_weight 
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight 

            if space1 is None:

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
     
                proj_weight = torch.mm(self.weight,norm_project)
                diag_weight = torch.mm(self.weight,torch.mm(space2, space2.transpose(1,0)))

                # masked_weight = proj_weight + self.weight - diag_weight
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight 
            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight,norm_project1)
                diag_weight1 = torch.mm(self.weight,torch.mm(space1, space1.transpose(1,0)))

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight,norm_project2)
                diag_weight2 = torch.mm(self.weight,torch.mm(space2, space2.transpose(1,0)))

                # masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight
                if self.noise and self.training:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight + ((self.alpha_w2 + self.alpha_w1)/2) * noise * self.noise
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight
       
        else:
            masked_weight = self.weight
        return F.linear(input, masked_weight, self.bias)
