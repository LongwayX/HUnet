import torch
import torch.nn as nn


class encode_layers(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(encode_layers, self).__init__()
        # 第一次卷积，层内卷积
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding = 'same')
        # bug:ValueError padiing='same' is not supported for strided convolutions
        self.bn1 = nn.BatchNorm3d(out_channels)
        # 第二次卷积，层间卷积
        self.conv2 = nn.Conv3d(out_channels, out_channels, (3, 1, 1), padding = 'same')
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)
        conv1 = self.bn1(conv1)
        conv1 = self.drop(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv2 = self.bn2(conv2)
        conv2 = self.drop(conv2)
        res = conv1 + conv2
        return res


class down_operation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(down_operation, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=(1, 2, 2),padding=(0,1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.down(x)


class decode_layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, code_layer):
        super(decode_layer, self).__init__()
        self.code_layer = code_layer
        self.deconv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size, stride=(1, 2, 2),padding=(0, 1, 1))
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,padding = 'same')
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        deconv = self.deconv(x)
        deconv = self.relu(deconv)
        merge = torch.cat((deconv, self.code_layer), dim=4)
        conv = self.conv(merge)
        conv = self.relu(conv)
        conv = self.drop(conv)
        res = deconv + conv
        return res


class HUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, drop=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop = drop
        # 卷积层1
        self.conv1_1 = encode_layers(1, 32, (1, 3, 3))
        # 下采样1
        self.down1 = down_operation(32, 64, (1, 3, 3))
        # 卷积层2
        self.conv2_1 = encode_layers(64, 64, (1, 3, 3))
        # 下采样2
        self.down2 = down_operation(64, 128, (1, 3, 3))
        # 卷积层3
        self.conv3_1 = encode_layers(128, 128, (1, 3, 3))
        # 下采样3
        self.down3 = down_operation(128, 256, (1, 3, 3))
        # 卷积层4
        self.conv4_1 = encode_layers(256, 256, (1, 3, 3))
        # 下采样4
        self.down4 = down_operation(256, 512, (1, 3, 3))
        # 卷积层5
        self.conv5_1 = encode_layers(512, 512, (1, 3, 3))
        self.conv5_2 = nn.Conv3d(512, 512, (3, 1, 1),padding = 'valid')
        self.relu = nn.ReLU()
        # 反卷积6
        self.deconv6 = nn.ConvTranspose3d(512, 256, (1, 3, 3), (1, 2, 2), (0, 1, 1), (0, 1, 1))
        self.conv4_2 = nn.Conv3d(256, 256, (3, 1, 1),padding='valid')
        self.conv6 = nn.Conv3d(512, 256, (1, 3, 3),padding='same')
        self.add6 = nn.ConvTranspose3d(256, 128, (1, 3, 3), (1, 2, 2),(0, 1, 1), (0, 1, 1))
        # 反卷积7
        self.deconv7 = nn.ConvTranspose3d(256, 128, (1, 3, 3), (1, 2, 2),(0, 1, 1), (0, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 128, (3, 1, 1), padding='valid')
        self.conv7 = nn.Conv3d(256, 128, (1, 3, 3),padding='same')
        self.drop = nn.Dropout(p=0.5)
        self.add7 = nn.ConvTranspose3d(128, 64, (1, 3, 3), (1, 2, 2),(0, 1, 1), (0, 1, 1))
        # 反卷积8
        self.deconv8 = nn.ConvTranspose3d(128, 64, (1, 3, 3), (1, 2, 2), (0, 1, 1), (0, 1, 1))
        self.conv2_2 = nn.Conv3d(64, 64, (3, 1, 1), padding='valid')
        self.conv8 = nn.Conv3d(128, 64, (1, 3, 3),padding='same')
        self.add8 = nn.ConvTranspose3d(64, 32, (1, 3, 3), (1, 2, 2),(0, 1, 1), (0, 1, 1))
        # 反卷积9
        self.deconv9 = nn.ConvTranspose3d(64, 32, (1, 3, 3), (1, 2, 2), (0, 1, 1), (0, 1, 1))
        self.conv1_2 = nn.Conv3d(32, 32, (3, 1, 1), padding='valid')
        self.conv9 = nn.Conv3d(64, 32, (1, 3, 3),padding='same')

        self.conv10 = nn.Conv3d(32, 3, (1, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        #print('conv1_1:',conv1_1.shape)
        down1 = self.down1(conv1_1)
        #print('down1:',down1.shape)
        conv2_1 = self.conv2_1(down1)
        #print('conv2_1:',conv2_1.shape)
        down2 = self.down2(conv2_1)
        #print('down2:',down2.shape)
        conv3_1 = self.conv3_1(down2)
        #print('conv3_1:',conv3_1.shape)
        down3 = self.down3(conv3_1)
        #print('down3:',down3.shape)
        conv4_1 = self.conv4_1(down3)
        #print('conv4_1:',conv4_1.shape)
        down4 = self.down4(conv4_1)
        #print('down4:',down4.shape)
        conv5_1 = self.conv5_1(down4)
        #print('conv5_1:',conv5_1.shape)
        conv5_2 = self.conv5_2(conv5_1)
        #print('conv5_2:',conv5_2.shape)
        conv5_2 = self.relu(conv5_2)
        deconv6 = self.deconv6(conv5_2)
        #print('deconv6:',deconv6.shape)
        deconv6 = self.relu(deconv6)
        conv4_2 = self.conv4_2(conv4_1)
        #print('conv4_2:',conv4_2.shape)
        conv4_2 = self.relu(conv4_2)
        merge6 = torch.cat((deconv6,conv4_2),dim=1)
        #print('merge6:',merge6.shape)
        conv6 = self.conv6(merge6)
        conv6 = self.relu(conv6)
        conv6 = self.drop(conv6)
        res6 = deconv6+conv6
        add6 = deconv6+res6
        add6 = self.add6(add6)
        add6 = self.relu(add6)
        #print('add6.shape:',add6.shape)
        deconv7 = self.deconv7(res6)
        deconv7 = self.relu(deconv7)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.relu(conv3_2)
        merge7 = torch.cat((deconv7,conv3_2),dim=1)
        conv7 = self.conv7(merge7)
        conv7 = self.relu(conv7)
        conv7 = self.drop(conv7)
        res7 = deconv7+conv7
        #print('res7.shape:',res7.shape)
        add7 = res7+add6 
        add7 = self.add7(add7)
        add7 = self.relu(add7)
        deconv8 = self.deconv8(res7)
        deconv8 = self.relu(deconv8)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = self.relu(conv2_2)
        merge8 = torch.cat((deconv8,conv2_2),dim=1)
        conv8 = self.conv8(merge8)
        conv8 = self.relu(conv8)
        conv8 = self.drop(conv8)
        res8 = deconv8+conv8
        add8 = add7+res8
        add8 = self.add8(add8)
        add8 = self.relu(add8)
        deconv9 = self.deconv9(res8)
        deconv9 = self.relu(deconv9)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.relu(conv1_2)
        merge9 = torch.cat((deconv9,conv1_2),dim=1)
        conv9 = self.conv9(merge9)
        conv9 = self.relu(conv9)
        conv9 = self.drop(conv9)
        res9 = deconv9+conv9
        add9 = add8+res9
        conv10 = self.conv10(add9)
        return conv10

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm3d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()





