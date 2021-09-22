import torch
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from train import load_checkpoint


def compute_class_freqs(labels):

    labels = np.array(labels)
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis = 0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


def weighted_loss(pos_weights, neg_weights, outputs, labels, epsilon = 1e-7):

    loss = 0.0
    for i in range(len (pos_weights)):
        loss_pos = -1 * torch.mean(pos_weights [i] * labels [:,i] * torch.log (outputs [:,i] + epsilon))
        loss_neg = -1 * torch.mean(neg_weights [i] * (1 - labels [:,i]) * torch.log ((1 - outputs [:,i]) + epsilon))
        loss += loss_pos + loss_neg
    return loss


def imshow (img, title = None):
    img = img.numpy ().transpose (1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img * std + mean
    img = np.clip (img, 0, 1)
    
    plt.figure (figsize = (10, 8))
    plt.axis ('off')
    plt.imshow (img)
    if title:
        plt.title (title)
        

def plot_acc_loss (loss, val_loss, acc, val_acc):
    plt.figure (figsize = (12, 4))
    plt.subplot (1, 2, 1)
    plt.plot (range (len (loss)), loss, 'b-', label = 'Training')
    plt.plot (range (len (loss)), val_loss, 'bo-', label = 'Validation')
    plt.title ('Loss')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (range (len (acc)), acc, 'b-', label = 'Training')
    plt.plot (range (len (acc)), val_acc, 'bo-', label = 'Validation')
    plt.title ('Accuracy')
    plt.legend ()

    plt.show ()


def view_classify (test_dl, pathology_list, checkpoint_path, load_model = False):

    if load_model:
        load_checkpoint (torch.load (checkpoint_path), model)
    images, labels = next (iter (test_dl))
    images, labels = images.to(device), labels.to(device)
    outputs = torch.sigmoid (model (images))

    idx = np.random.choice (len (outputs))
    class_name = pathology_list
    classes = np.array(class_name)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = images [idx].cpu ().numpy ().transpose (1, 2, 0) * std + mean
    output = outputs [idx].cpu().data.numpy().squeeze()
    label = labels [idx].cpu ()
    class_labels = list (np.where(label==1)[0])

    if not class_labels :
        title = 'No Findings'
    else : 
        title = itemgetter(*class_labels)(class_name)


    fig, (ax1, ax2) = plt.subplots(figsize=(8,12), ncols=2)
    ax1.imshow(image)
    ax1.set_title('Ground Truth : {}'.format(title))
    ax1.axis('off')
    ax2.barh(classes, output)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None