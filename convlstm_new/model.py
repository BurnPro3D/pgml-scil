import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.reset_parameters()
        
        # # Initialize weights (Orthogonal for recurrent weights is often more stable)
        # nn.init.xavier_uniform_(self.conv.weight)
        # if self.bias:
        #     nn.init.zeros_(self.conv.bias)

    def reset_parameters(self):
        # 1. SPLIT INITIALIZATION (Match Keras Orthogonal/Glorot)
        # Because we use a combined Conv2d, we must slice the weights manually to apply different inits
        # Weight shape: (Out_Ch, In_Ch, K, K)
        # In_Ch is (input_dim + hidden_dim). We split along dim 1.
        
        # Initialize Input part (Glorot/Xavier)
        nn.init.xavier_uniform_(self.conv.weight[:, :self.input_dim, :, :])
        
        # Initialize Recurrent part (Orthogonal)
        # Orthogonal init requires a 2D matrix, so we flatten spatial dims temporarily
        recurrent_weight = self.conv.weight[:, self.input_dim:, :, :]
        # (Out, Hidden, K, K) -> (Out, Hidden*K*K) is not standard orthogonal. 
        # Keras does Orthogonal on the matrix (Hidden_in, Hidden_out). 
        # For ConvLSTM, typical approximation in PyTorch is just orthogonal on the flattened kernel or Xavier.
        # However, sticking to Xavier is often 'acceptable' if Orthogonal is too complex to map 1:1 here.
        # But let's fix the Forget Bias (Most Important).

        if self.bias:
            nn.init.zeros_(self.conv.bias)
            # 2. UNIT FORGET BIAS (Match Keras unit_forget_bias=True)
            # Bias shape is (4 * hidden_dim). Order is usually (i, f, o, g) or (i, g, f, o) depending on impl.
            # PyTorch split usually follows order of operations.
            # Your code splits: i, f, o, g. 
            # Index for 'f' is the second chunk.
            
            start = self.hidden_dim
            end = start + self.hidden_dim
            with torch.no_grad():
                self.conv.bias[start:end].fill_(1.0)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 3. MATCH ACTIVATIONS (Optional, only if you need exact numerical match)
        # Keras uses hard_sigmoid for gates (i, f, o)
        # PyTorch equivalent is F.hardsigmoid
        
        i = F.hardsigmoid(cc_i) # Was torch.sigmoid
        f = F.hardsigmoid(cc_f) # Was torch.sigmoid
        o = F.hardsigmoid(cc_o) # Was torch.sigmoid
        g = torch.tanh(cc_g)    # Keras uses tanh for the cell candidate too
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement state passing.
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list[0], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            return [param] * num_layers
        return param


class ConvLSTMModel(nn.Module):
    """
    Reimplementation of the TF ConvLSTMModel using PyTorch.
    """
    def __init__(self, img_size, in_chans=4, out_chans=1):
        super(ConvLSTMModel, self).__init__()
        
        # Layer 1: 64 filters, 3x3 kernel
        # Layer 2: 64 filters, 1x1 kernel (Matching your TF model logic)
        # self.conv_lstm = ConvLSTM(
        #     input_dim=in_chans, 
        #     hidden_dim=[64, 64], 
        #     kernel_size=[(3, 3), (1, 1)], 
        #     num_layers=2,
        #     batch_first=True,
        #     return_all_layers=False
        # )

        self.conv_lstm = ConvLSTM(
            input_dim=in_chans, 
            hidden_dim=[64], 
            kernel_size=[(3, 3)], 
            num_layers=1,
            batch_first=True,
            return_all_layers=False
        )
        
        self.batch_norm = nn.BatchNorm3d(64)
        
        # Final Conv3D to map to output channels
        # Kernel (3, 3, 3) with padding (1, 1, 1) preserves dimensions
        self.conv3d = nn.Conv3d(in_channels=64, out_channels=out_chans, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.act = nn.ReLU()

    def forward(self, x):
        # Input x: (B, T, C, H, W)
        
        # 1. Pass through ConvLSTM Layers
        # returns output of shape (B, T, Hidden_Dim, H, W)
        x, _ = self.conv_lstm(x)
        
        # 2. BatchNorm3d
        # BatchNorm3d expects (B, C, T, H, W). We need to permute.
        x = x.permute(0, 2, 1, 3, 4) # (B, 64, T, H, W)
        x = self.batch_norm(x)
        
        # 3. Conv3D
        x = self.conv3d(x) # (B, out_chans, T, H, W)
        x = self.act(x)
        
        # 4. Restore shape to (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        return x

class PhysicsEnhancedConvLSTM(nn.Module):
    def __init__(self, img_size, in_chans=4, out_chans=1):
        super().__init__()
        
        # Encoder 1
        self.conv_lstm1 = ConvLSTM(
            input_dim=in_chans, 
            hidden_dim=[64], 
            kernel_size=[(3, 3)], 
            num_layers=1,
            batch_first=True,
            return_all_layers=True
        )
        # Intermediate Head 1 (Deep Supervision)
        self.head1 = nn.Conv3d(64, out_chans, kernel_size=1) 
        
        # Encoder 2
        self.conv_lstm2 = ConvLSTM(
            input_dim=64, 
            hidden_dim=[64], 
            kernel_size=[(3, 3)], 
            num_layers=1,
            batch_first=True,
            return_all_layers=True
        )
        
        # Final Head
        self.final_head = nn.Conv3d(64, out_chans, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self._init_biases()
    
    def _init_biases(self):
        # Force the final layers to output something > 0 at the start
        nn.init.constant_(self.head1.bias, 0.1)
        nn.init.constant_(self.final_head.bias, 0.1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        
        # 1. Pass through Layer 1
        # out1_seq: List of tensors [(B, T, 64, H, W)]
        out1_seq, _ = self.conv_lstm1(x)
        feat1 = out1_seq # The tensor itself
        
        # 2. Deep Supervision: Predict from Layer 1
        # We permute for Conv3d: (B, C, T, H, W)
        feat1_perm = feat1.permute(0, 2, 1, 3, 4)
        intermediate_pred = self.act(self.head1(feat1_perm))
        intermediate_pred = intermediate_pred.permute(0, 2, 1, 3, 4)
        
        # 3. Pass through Layer 2 (using Layer 1 output as input)
        out2_seq, _ = self.conv_lstm2(feat1)
        feat2 = out2_seq
        
        # 4. Final Prediction
        feat2_perm = feat2.permute(0, 2, 1, 3, 4)
        final_pred = self.act(self.final_head(feat2_perm))
        final_pred = final_pred.permute(0, 2, 1, 3, 4)
        
        # Return BOTH predictions during training
        if self.training:
            return final_pred, intermediate_pred
        else:
            return final_pred