import argparse
import neptun

from ad_eval import eval_main

'''
eval loads model trained from different datset and measure detection performance in another dataset
'''
if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--direction', type=str)

    args = parser.parse_args()

    params = vars(args)

    # set neptune
    neptune.init('cjlee/AnomalyDetection-GNN')
    experiment = neptune.create_experiment(name=args.exp_name, params=params)

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    # load model trained from a given dataset
    model = torch.load(args.model_path).to(device)

    # load different dataset
    from ad_data import ad_gnn_iterator
    testiter = ad_gnn_iterator(tvt='sup_test', dataset=args.dataset, direction=args.direction)

    # evaluate the model and measure performance 
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=none)

    # print results and logging
    print('acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}'.format(acc, prec, rec, f1))
    neptune.set_property('acc', acc)
    neptune.set_property('prec', prec)
    neptune.set_property('rec', rec)
    neptune.set_property('f1', f1)



