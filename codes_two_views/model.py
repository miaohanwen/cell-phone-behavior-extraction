import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d, Unit3D
from weight_init import weight_init


class TAL_Net(nn.Module):
    def __init__(self, num_classes):
        super(TAL_Net, self).__init__()
        self.num_classes = num_classes
        self.I3D_1 = InceptionI3d(3, in_channels=3)
        self.I3D_2 = InceptionI3d(3, in_channels=3)
        
#         for param in self.I3D.parameters():
#             param.requires_grad = False
            
        self.dropout = nn.Dropout(p=0.5)   
        self.predictor = nn.Sequential(
            Unit3D(in_channels=2*(384 + 384 + 128 + 128), 
                   output_channels=256,
                   kernel_shape=[1, 1, 1],
                   name='layer1'),
            nn.Dropout(p=0.5),
            Unit3D(in_channels=256, 
                   output_channels=self.num_classes + 2,
                   kernel_shape=[1, 1, 1],
                   activation_fn=None,
                   use_batch_norm=False,
                   use_bias=True,
                   name='layer2')
        )
#         self.predictor = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self.num_classes+2,
#                                 kernel_shape=[1, 1, 1],
#                                 activation_fn=None,
#                                 use_batch_norm=False,
#                                 use_bias=True)
               
        self.predictor.apply(weight_init)

    def forward(self, cabin_clips, face_clips, class_labels=None, start_labels=None, end_labels=None, weight=0.25):
        B, C, T, H, W = cabin_clips.shape
        face_clips = F.interpolate(face_clips, size=(T, H, W), mode='trilinear')
        feat1 = self.I3D_1.extract_features(cabin_clips)
        feat2 = self.I3D_2.extract_features(face_clips)
        feat = torch.cat((feat1, feat2), 1)
        feat = self.dropout(feat)
        preds = self.predictor(feat)
        preds = preds.squeeze(3).squeeze(3)
        preds = torch.max(preds, dim=2)[0]
#         preds = torch.mean(preds, dim=2)
        # shape (B,num_classes+2)
        class_scores = preds[:, :self.num_classes].sigmoid()
        start_scores = preds[:, self.num_classes].sigmoid()
        end_scores = preds[:, self.num_classes+1].sigmoid()
        
        if class_labels is not None and start_labels is not None:
            class_loss = F.cross_entropy(class_scores, class_labels)
            all_chunk_inclusion_loss = 1/2*(F.binary_cross_entropy(start_scores, start_labels, reduction='none')
                                        + F.binary_cross_entropy(end_scores, end_labels, reduction='none'))
            indicator = (class_labels > 0).to(class_loss.dtype)
            chunk_inclusion_loss = torch.sum(indicator*all_chunk_inclusion_loss)/(torch.sum(indicator)+1e-10)
            total_loss = class_loss + weight*chunk_inclusion_loss
            return total_loss, class_loss, chunk_inclusion_loss, class_scores, start_scores, end_scores
        else:
            return class_scores, start_scores, end_scores