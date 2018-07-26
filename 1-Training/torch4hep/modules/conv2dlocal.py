"""
 I copied this code from the following link and made a few minor modifications.   
    https://gist.github.com/guillefix/23bff068bdc457649b81027942873ce5           
"""                  
                     
from __future__ import absolute_import             
from __future__ import division
from __future__ import print_function


 
import torch
from torch.nn.parameter import Parameter           
import torch.nn.functional as F                    
from torch.nn.modules.module import Module         
from torch.nn.modules.utils import _pair           

def get_conv_out_length(in_length,
                        kernel_size,
                        padding,
                        stride,
                        dilation=1):
    """
    https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py
    """
    dilated_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    padding = padding.lower()
    if not padding in {"same", "valid", "full"}:
        raise ValueError

    if padding == "same":
        out_length = in_length
    elif padding == "valid":
        out_length = in_length - dilated_kernel_size + 1
    elif padding == "full":
        out_length = in_length + dilated_kernel_size - 1

    # TODO if padding == (0, 0)

    out_length = (out_length + stride - 1) // stride

    return out_length                    
                     
def conv2d_local(input,                            
                 weight,                           
                 bias=None,                       
                 padding=0,                        
                 stride=1,                         
                 dilation=1):                      
                     
    if input.dim() != 4:                           
        raise NotImplementedError(                 
            "Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
                     
    if weight.dim() != 6:                          
        # out_height x out_width x out_channels x in_channels x kernel_height x kernel_width
        raise NotImplementedError(                 
            "Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
                     
    weight_size = weight.size()

    out_height = weight_size[0]                    
    out_width = weight_size[1]                     
    out_channels = weight_size[2]                  
    in_channels = weight_size[3]                   
    kernel_height = weight_size[4]                 
    kernel_width = weight_size[5]                  
                     
    kernel_size = (kernel_height, kernel_width)    
                     
    # torch.nn.functional.unfold                   
    # https://pytorch.org/docs/master/nn.html#torch.nn.Unfold                    
    # Extracts sliding local blocks from an batched input tensor.                
    #                
    # input: (N, C, *)                             
    #   N: the batch dimension                     
    #   C: the channel_dimension                   
    #   *: arbitrary spatial dimensions            
    # output: (N, C x Pi(kernel_size), L)          
    #   Pi(kernel_size): the total number of values with in each block           
    #   L: the total number of such blocks         
    #                
    # 'cols' comes from im2col since F.unfold uses 
    # torch.nn.functional._functions.thnn.fold.Im2Col                            
    cols = F.unfold(input=input,                   
                    kernel_size=kernel_size,       
                    dilation=dilation,             
                    padding=padding,               
                    stride=stride)                 
    # [N, Pi(k), L, 1]                             
    cols = cols.unsqueeze(-1)                      
    # [N, L, 1, Pi(k)]                             
    cols = cols.permute(0, 2, 3, 1)                
                     
    # (H_out * H_out,
    #  C_out,        
    #  C_in * H_kernel * W_kernel)
    weight = weight.view(out_height * out_width,   
                         out_channels,             
                         in_channels * kernel_height * kernel_width)             
    # (H_out * W_out,
    #  C_in * H_kernel * W_kernel,                 
    #  C_out)        
    weight = weight.permute(0, 2, 1)               
                     
    # [N, L, 1, Pi(k)]                             
    out = torch.matmul(cols, weight)               
    out = out.view(cols.size(0), out_height, out_width, out_channels)            
    out = out.permute(0, 3, 1, 2)                  
                     
    if bias is not None:                           
        out = out + bias.expand_as(out)            
                     
    return out       
                     
                     
class Conv2dLocal(Module):                         
    def __init__(self,                             
                 in_height,                        
                 in_width,                         
                 in_channels,                      
                 out_channels,                     
                 kernel_size,                      
                 stride=1,                         
                 padding=0,                        
                 bias=True,                        
                 dilation=1):
        super(Conv2dLocal, self).__init__()        
                     
        self.in_height = in_height                 
        self.in_width = in_width                   
        self.in_channels = in_channels             
        self.out_channels = out_channels           
                     
        self.kernel_size = _pair(kernel_size)      
        self.stride = _pair(stride)                
        self.padding = _pair(padding)              
        self.dilation = _pair(dilation)            
                     
        self.out_height = get_conv_out_length(     
            in_height, kernel_size[0], padding[0], stride[0], dilation[0])       
                     
        self.out_width = get_conv_out_length(      
            in_width, kernel_size[1], padding[1], stride[1], dilation[1])        
                     
                     
        self.weight = Parameter(                   
            torch.Tensor(                          
                self.out_height,                   
                self.out_width,                    
                out_channels,                      
                in_channels,                       
                *self.kernel_size))                
                     
        if bias:     
            self.bias = Parameter(                 
                torch.Tensor(                      
                    out_channels,                  
                    self.out_height,               
                    self.out_width))               
        else:        
            self.register_parameter('bias', None)  
                     
        self.reset_parameters()

    def reset_parameters(self):                    
        num_params_per_channel = np.prod(self.kernel_size)                       
        num_params = self.in_channels * num_params_per_channel                   
        stddev = 1.0 / math.sqrt(num_params)       
        self.weight.data.uniform_(-stddev, stddev) 
        if self.bias is not None:                  
            self.bias.data.uniform_(-stddev, stddev)                             

    # TODO def extra_repr

    def forward(self, input):                      
        return conv2d_local(                       
            input, self.weight, self.bias, stride=self.stride,                   
            padding=self.padding, dilation=self.dilation)
