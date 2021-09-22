import sys
import time
import torch
import numpy as np


def save_checkpoint (state, filename):
    """ saving model's weights """
    print ('=> saving checkpoint')
    torch.save (state, filename)

    
def load_checkpoint (checkpoint, model):
    """ loading model's weights """
    print ('=> loading checkpoint')
    model.load_state_dict (checkpoint ['state_dict'])


def accuracy (outputs, labels, thresh = 0.5, per_class = False):
    """ calculate accuracy, precision, recall, F1score """
    correct_classes = [0 for i in range (outputs.shape [1])]
    per_class_accuracy = [0 for i in range (outputs.shape [1])]
    preds = (outputs >= thresh).float ()

    for cls_idx in range (outputs.shape [1]):
        pred = preds [:, cls_idx: cls_idx +1]
        label = labels [:, cls_idx: cls_idx + 1]
        correct_classes [cls_idx] += int ((pred == label).sum ())

        per_class_accuracy = [(i/len(outputs))*100.0 for i in correct_classes]
        accuracy = sum (correct_classes) / (len (outputs) * outputs.shape [1])

    return per_class_accuracy if per_class else accuracy


def train (model, loader, loss_fn, optimizer, metric_fn, pos_weights, device):
    """ training one epoch and calculate loss and metrics """
    
    # Training model
    model.train ()
    losses = 0.0
    metrics = 0.0
    steps = len (loader)

    for i, (inputs, labels) in enumerate (loader):
        # Place to gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model (inputs)

        # Calculate loss
        loss = loss_fn (pos_weights, 1 - pos_weights, outputs, labels).to(device)
        # Adding L2 regularization
        #loss += L2_lambda * sum ([torch.norm (param, p = 2) for param in model.parameters ()])
        losses += loss

        # Backpropagation and update weights
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()

        # Calculate metrics
        metric = metric_fn (outputs, labels)
        metrics += metric

        # report
        sys.stdout.flush ()
        sys.stdout.write ('\r Step: [%2d/%2d], loss: %.4f - acc: %.4f' % (i, steps, loss.item (), metric))
    sys.stdout.write ('\r')
    return losses.item () / len (loader), metrics / len (loader)


def evaluate (model, loader, loss_fn, metric_fn, pos_weights, device, per_class = False):
    """ Evaluate trained weights using calculate loss and metrics """
    # Evaluate model
    model.eval ()
    losses = 0.0
    metrics = 0.0
    class_metrics = [0 for i in range (14)]

    with torch.no_grad ():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model (inputs)
              
            loss = loss_fn (pos_weights, 1 - pos_weights, outputs, labels).to(device)
            #loss += L2_lambda * sum ([torch.norm (param, p = 2) for param in model.parameters ()])
            losses += loss
            metrics += metric_fn (outputs, labels)

            if per_class:
                class_metrics = [class_metrics [i] + metric_fn (outputs, labels, per_class = True) [i] for i in range (14)]
            

    return [x / len (loader) for x in class_metrics] if per_class else (losses.item () / len (loader), metrics / len (loader))


def fit (model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, metric_fn, pos_weights, device, checkpoint_path, scheduler = None, load_model = False):
    """ fiting model to dataloaders, saving best weights and showing results """
    losses, val_losses, accs, val_accs = [], [], [], []
    best_loss = np.Inf

    # to continue training from saved weights
    if load_model:
        load_checkpoint (torch.load (checkpoint_path), model)

    since = time.time ()

    for epoch in range (num_epochs):

        loss, acc = train (model, train_dl, loss_fn, optimizer, metric_fn, pos_weights, device)
        val_loss, val_acc = evaluate (model, valid_dl, loss_fn, metric_fn, pos_weights, device)

        # learning rate scheduler
        if scheduler is not None:
            scheduler.step (val_loss)

        losses.append (loss)
        accs.append (acc)
        val_losses.append (val_loss)
        val_accs.append (val_acc)

        # save weights if improved
        if val_loss < best_loss:
            checkpoint = {'state_dict': model.state_dict (), 'optimizer': optimizer.state_dict ()}
            save_checkpoint (checkpoint, checkpoint_path)
            best_loss = val_loss

        print ('Epoch [{}/{}], loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}'.format (epoch + 1, num_epochs, loss, acc, val_loss, val_acc))

    period = time.time () - since
    print ('Training complete in {:.0f}m {:.0f}s'.format (period // 60, period % 60))

    return dict (loss = losses, val_loss = val_losses, acc = accs, val_acc = val_accs)