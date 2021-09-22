import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


pathology_list = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Nodule', 'Pneumothorax', 'Atelectasis',
                  'Pleural_Thickening', 'Mass','Edema', 'Consolidation', 'Infiltration', 'Fibrosis', 'Pneumonia']


def view_classify (img, model, device):

    img = img.to(device)
#     lbl = lbl.to(device)
    output = torch.sigmoid (model (img))

    class_name = pathology_list
    classes = np.array(class_name)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img [0].cpu ().numpy ().transpose (1, 2, 0) * std + mean
    output = output.cpu().data.numpy().squeeze()
    df = pd.DataFrame (output, index = pathology_list, columns = ['Prob'])
#     lbl = lbl.cpu ()
#     class_labels = list (np.where(lbl==1)[0])

#     if not class_labels :
#         title = 'No Findings'
#     else : 
#         title = itemgetter(*class_labels)(class_name)


    fig, (ax1, ax2) = plt.subplots(figsize=(8,12), ncols=2)
    ax1.imshow(img)
    #ax1.set_title('Ground Truth : {}'.format(title))
    ax1.axis('off')
    ax2.barh(classes, output)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return fig, df