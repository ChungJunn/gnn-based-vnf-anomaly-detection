import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def eval_main(model, validiter, device, dim_out=7, neptune=None):
    model.eval()

    preds = []
    targets = []

    # forward the whole dataset and obtain result
    for xs, ys, end_of_data in validiter:
        xs = xs.to(device)

        outs = model(xs)
        outs = outs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy().reshape(-1,1)

        preds.append(outs)
        targets.append(ys)

        if end_of_data == 1: break

    # obtain results using metrics
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    preds = np.argmax(preds, axis=1)

    if dim_out == 2:
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds)
        rec = recall_score(targets, preds)
        f1 = f1_score(targets, preds)

        if neptune is not None:
            neptune.set_property('test acc', acc)
            neptune.set_property('test prec', prec)
            neptune.set_property('test rec', rec)
            neptune.set_property('test f1', f1)

        print('acc:{:.4f} | prec:{:.4f} | rec:{:.4f} | f1:{:.4f}'.format(acc, prec, rec, f1))
        
    else:
        acc = accuracy_score(targets, preds)
        if neptune is not None:
            neptune.set_property('test acc', acc)
        print('acc:{:.4f}'.format(acc))

        if dim_out == 7:
            target_names = ['normal', 'fw', 'ids', 'fm', 'dpi', 'lb', 'traffic']
        elif dim_out == 6:
            target_names = ['normal', 'fw', 'fm', 'dpi', 'ids', 'traffic']
            
        print(classification_report(targets, preds, target_names=target_names, digits=4))

    return
