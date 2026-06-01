import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, accuracy_score, fowlkes_mallows_score, normalized_mutual_info_score, adjusted_mutual_info_score

from config import Config as Config


def target_distribution(q):
    # clustering target distribution for self-training
    # q (batch,n_clusters): similarity between embedded point and cluster center
    # p (batch,n_clusters): target distribution
    weight = q**2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p


def update_cluster(encoder_model, task_model, dataloader, device, momentum_model):
    q = []
    encoder_model.eval()
    task_model.eval()

    labels = []
    vecs = []
    # torch.save(model.bert.state_dict(), 'before_update_cluster.pt')

    # torch.save({"encoder_q": encoder_model.clmodel.encoder_q.state_dict(),
    #             "trajcluster": task_model.state_dict()},
    #            'before_update_cluster.pt')

    # for i, (
    # input_ids, masked_tokens, masked_pos, input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,
    # label) in enumerate(dataloader):
    #     # context, q_i, head_in, head_cl = model(input_ids_o.to(device),lengths.to(device),id_mask.to(device),timestamp.to(device),momentum_model)
    #     context, q_i, head_in, head_cl = model(input_ids_o.to(device), lengths.to(device), id_mask.to(device),
    #                                            timestamp_o.to(device), momentum_model)

    for i_batch, batch in enumerate(dataloader):

        # trajs_emb, trajs_emb_p, trajs_len, label, _, _, _, _, _, _  = batch
        trajs_emb, trajs_emb_p, trajs_len, label = batch
        context = encoder_model.interpret(trajs_emb, trajs_emb_p, trajs_len)
        q_i, head_in, head_cl = task_model(context)

        q.append(q_i.cpu().data)
        vecs.append(context.cpu().data)
        labels.append(torch.tensor(label, device=Config.device))  # Ensure label is a tensor

    # (datasize,n_clusters)
    q = torch.cat(q)
    labels = torch.cat(labels)
    vecs = torch.cat(vecs)

    # torch.save({"encoder_q": encoder_model.clmodel.encoder_q.state_dict(),
    #             "trajcluster": task_model.state_dict()}, 'after_update_cluster.pt')

    encoder_model.train()
    task_model.train()

    return q, target_distribution(q), labels, vecs


def cluster_acc(y_true, y_pred):
    """
    Calculate unsupervised clustering accuracy. Requires scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def nmi_score(y, y_pred):
    return normalized_mutual_info_score(y, y_pred)


def ari_score(y, y_pred):
    return adjusted_rand_score(y, y_pred)