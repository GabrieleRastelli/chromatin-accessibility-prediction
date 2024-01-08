import torch.nn as nn
import torch
import torch.nn.functional as F


class DeopenRegressionModel(nn.Module):
    def __init__(self):
        super(DeopenRegressionModel, self).__init__()
        self.l = 2114
        pool_size = 5
        test_size1 = 13
        test_size2 = 7
        test_size3 = 5
        kernel1 = 128
        kernel2 = 128
        kernel3 = 128
        num_cells = 19

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=3138,
            nhead=6,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.layer1 = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.layer2_f = nn.Flatten()
        self.layer3 = nn.Conv2d(1, kernel1, kernel_size=(4, test_size1))
        self.layer4 = nn.Conv2d(kernel1, kernel1, kernel_size=(1, test_size1))
        self.layer5 = nn.Conv2d(kernel1, kernel1, kernel_size=(1, test_size1))
        self.layer6 = nn.MaxPool2d(kernel_size=(1, pool_size))
        self.layer7 = nn.Conv2d(kernel1, kernel2, kernel_size=(1, test_size2))
        self.layer8 = nn.Conv2d(kernel2, kernel2, kernel_size=(1, test_size2))
        self.layer9 = nn.Conv2d(kernel2, kernel2, kernel_size=(1, test_size2))
        self.layer10 = nn.MaxPool2d(kernel_size=(1, pool_size))
        self.layer11 = nn.Conv2d(kernel2, kernel3, kernel_size=(1, test_size3))
        self.layer12 = nn.Conv2d(kernel3, kernel3, kernel_size=(1, test_size3))
        self.layer13 = nn.Conv2d(kernel3, kernel3, kernel_size=(1, test_size3))
        self.layer14 = nn.MaxPool2d(kernel_size=(1, pool_size))
        self.layer14_d = nn.Linear(256, 256)
        self.layer16_d = nn.Linear(12552, 256)
        self.layer17 = nn.Linear(640, 256)
        self.network = nn.Linear(256, num_cells)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_transformer = x.squeeze(1)
        transformer_output = self.transformer(x_transformer)
        transformer_output = transformer_output.unsqueeze(1)

        # Forward pass through the layers
        x = self.layer1(x)
        layer2_1 = x.narrow(-1, 0, self.l)
        layer2_2 = x.narrow(-1, self.l, x.size(-1) - self.l)
        layer2_3 = layer2_2.narrow(-2, 0, 4)
        layer2_f = self.layer2_f(layer2_3)

        layer3 = F.relu(self.layer3(layer2_1))
        layer4 = F.relu(self.layer4(layer3))
        layer5 = F.relu(self.layer5(layer4))
        layer6 = self.layer6(layer5)
        layer7 = F.relu(self.layer7(layer6))
        layer8 = F.relu(self.layer8(layer7))
        layer9 = F.relu(self.layer9(layer8))
        layer10 = self.layer10(layer9)
        layer11 = F.relu(self.layer11(layer10))
        layer12 = F.relu(self.layer12(layer11))
        layer13 = F.relu(self.layer13(layer12))
        layer14 = self.layer14(layer13)

        # Flatten layer
        layer2_f_size = layer2_f.size(1)

        # Dense layers
        layer14_d = F.relu(self.layer14_d(layer14.view(-1, 256)))
        layer3_2 = nn.Linear(layer2_f_size, 128).to(device)
        layer3_2 = F.relu(layer3_2(layer2_f))
        collapsed_layer14_d = layer14_d.mean(dim=0)
        replicated_layer14_d = collapsed_layer14_d.repeat(layer3_2.shape[0], 1)
        layer15 = torch.cat([replicated_layer14_d, layer3_2], dim=1)

        layer16_d = F.relu(self.layer16_d(transformer_output.view(transformer_output.size(0), -1)))
        layer16_d = torch.cat([layer15, layer16_d], dim=1)

        # Layer 17
        layer17 = F.relu(self.layer17(layer16_d))

        # Output layer
        network_output = self.network(layer17)

        return network_output

if __name__ == "__main__":
    model = DeopenRegressionModel()
    out = model(torch.randn((4, 1, 4, 3138)))
    assert out.shape == (4, 19)
