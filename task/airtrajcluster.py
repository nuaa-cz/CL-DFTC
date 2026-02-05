import logging
import time
import pickle
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import pair_confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import Parameter

from utils.cluster_tool import update_cluster, cluster_acc, nmi_score, ari_score
from utils.data_loader import read_trajcluster_traj_dataset
from utils.traj import merc2cell2, generate_spatial_features
from utils.losses import Loss_Function
from utils.traj import *


class ClusertLayer(nn.Module):
    def __init__(self, n_clusters, alpha=1):
        super(ClusertLayer, self).__init__()

        # self.clusters = Parameter(torch.Tensor(n_clusters, cluster_d), requires_grad=True).cuda()

        self.clusters = Parameter(torch.Tensor(n_clusters, Config.seq_embedding_dim), requires_grad=True)  # .cuda()

        self.alpha = alpha

    def forward(self, emb):
        # clustering: caculate Student’s t-distribution
        # clusters (n_clusters, hidden_size * num_directions)
        # emb (batch, hidden_size * num_directions)
        # q (batch,n_clusters): similarity between embedded point and cluster center
        # distance = torch.sum(torch.pow(emb.unsqueeze(1) - self.clusters, 2), 2)

        distance = emb.unsqueeze(1).cpu() - self.clusters.cpu()
        distance = torch.sum(torch.pow(distance, 2), 2)
        q = 1.0 / (1.0 + distance / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class AirTrajClusterModule(nn.Module):
    def __init__(self, n_clusters):
        super(AirTrajClusterModule, self).__init__()
        # self.args = args
        # self.bert = model_mlm
        self.clusterlayer = ClusertLayer(n_clusters)
        # self.dropout = dropout
        self.mlp_in = nn.Sequential(
            nn.Linear(Config.seq_embedding_dim, Config.seq_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(Config.seq_embedding_dim, 128)
        )
        self.mlp_cl = nn.Sequential(
            nn.Linear(Config.seq_embedding_dim, Config.seq_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(Config.seq_embedding_dim, n_clusters)
        )
        self.norm = nn.LayerNorm(n_clusters)

    def forward(self, trajs):
        q = self.clusterlayer(trajs)
        head_in = self.mlp_in(trajs)
        head_cl = self.mlp_cl(trajs).t()

        return q, head_in, head_cl


class AirTrajCluster:
    def __init__(self, encoder, str_aug1, str_aug2):
        super(AirTrajCluster, self).__init__()

        self.aug1 = get_aug_fn(str_aug1)
        self.aug2 = get_aug_fn(str_aug2)

        self.trajclustermodule = None
        self.encoder = encoder
        self.dic_datasets = AirTrajCluster.load_trajcluster_dataset()

        self.embedding_filepath = '{}/{}_trajcluster_{}_{}_{}_cell{}_emb{}_pre{}_clu{}_init{}' \
            .format(Config.emb_dir,
                    Config.dataset_prefix,
                    Config.trajcluster_encoder_name,
                    Config.trajcl_aug1,
                    Config.trajcl_aug2,
                    Config.cell_size,
                    Config.seq_embedding_dim,
                    Config.trajcl_training_epochs,
                    Config.trajcluster_epochs,
                    Config.init_clusters)

        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.cellembs = pickle.load(open(Config.dataset_embs_file, 'rb')).to(Config.device)  # tensor

    def train(self):
        training_starttime = time.time()
        # training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_trajcluster start.@={:.3f}".format(training_starttime))

        self.trajclustermodule = AirTrajClusterModule(Config.init_clusters)
        self.trajclustermodule.to(Config.device)
        # 选择损失函数
        criterion = self._select_criterion()
        criterion.to(Config.device)

        optimizer = torch.optim.Adam([{'params': self.trajclustermodule.parameters(),
                                       'lr': Config.trajcluster_learning_rate,
                                       'weight_decay': Config.trajcluster_learning_weight_decay},
                                      {'params': self.encoder.clmodel.encoder_q.parameters(),
                                       'lr': Config.trajcluster_learning_rate}])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        vecsForKmeans, kmeansCenters, init_acc = self.init_clusterlayer()

        if Config.trajcluster_epochs == 0:
            return init_acc

        best_acc = 0
        best_nmi = 0
        best_ari = 0
        best_epoch = 0
        best_emb = None

        for i_ep in range(Config.trajcluster_epochs):
            _time_ep = time.time()
            infoNCE_losses = []
            main_clustering_losses = []
            inter_cluster_losses = []
            cluster_level_contrastive_losses = []
            train_losses = []
            train_gpus = []
            train_rams = []

            self.encoder.train()
            self.trajclustermodule.train()

            if i_ep % 1 == 0:
                with torch.no_grad():
                    tmp_q, p, labels, vecs = update_cluster(self.encoder, self.trajclustermodule,
                                                            self.trajcluster_dataset_generator_batchi(augment=False),
                                                            Config.device, None)
                # cc = nn.KLDivLoss(reduction='sum').cuda()
                y = labels.cpu()
                y_pred = tmp_q.numpy().argmax(1)
                # print(len(np.unique(y_pred)))
                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)

                if i_ep == 0:
                    np.save(self.embedding_filepath + '_init_vec', vecs)
                    np.save(self.embedding_filepath + '_init_true_label', y.numpy())
                    np.save(self.embedding_filepath + '_init_pre_label', y_pred)

                if (best_nmi < nmi):
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_epoch = i_ep + 1
                    best_emb = vecs
                    best_true_label = y.numpy()
                    best_pre_label = y_pred

                    np.save(self.embedding_filepath + '_best_vec', best_emb)
                    np.save(self.embedding_filepath + '_best_pre_label', best_pre_label)
                    np.save(self.embedding_filepath + '_best_true_label', best_true_label)
                    matrix = pair_confusion_matrix(y.numpy(), y_pred)


                logging.info("[Training] ep={}: acc={:.3f}, nmi={:.3f}, ari={:.3f}" \
                             .format(i_ep + 1, acc, nmi, ari))

            for i_batch, batch in enumerate(self.trajcluster_dataset_generator_batchi(Config.trajcluster_aug)):
                _time_batch = time.time()
                optimizer.zero_grad()

                # Initialize losses to 0
                infoNCE_loss = torch.tensor(0.0, device=Config.device)
                assignment_loss = torch.tensor(0.0, device=Config.device)
                inter_cluster_loss = torch.tensor(0.0, device=Config.device)
                cluster_level_contrastive_loss = torch.tensor(0.0, device=Config.device)

                if Config.trajcluster_aug:
                    # Unpack with augmentation
                    trajs_emb, trajs_emb_p, trajs_len, sub_label, trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len = batch

                    if Config.infoNCE_loss:
                        model_rtn = self.encoder(trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p,
                                                 trajs2_len)
                        infoNCE_loss = self.encoder.loss(*model_rtn)

                else:
                    # Unpack without augmentation
                    trajs_emb, trajs_emb_p, trajs_len, sub_label = batch

                if Config.assignment_loss:
                    emb = self.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
                    q, head_in, head_cl = self.trajclustermodule(emb)
                    p_select = p[i_batch * Config.trajcluster_batch_size:(i_batch + 1) * Config.trajcluster_batch_size]

                    assignment_loss = criterion.clusteringLoss(
                        self.trajclustermodule.clusterlayer,
                        emb,
                        p_select,
                        Config.device,
                        Config.device,
                        q
                    )

                if Config.inter_cluster_distance_loss:
                    inter_cluster_loss = criterion.inter_cluster_distance_loss(
                        self.trajclustermodule.clusterlayer.clusters)

                if Config.cluster_level_contrastive_loss:
                    # Get trajectory embeddings and hard cluster assignments
                    emb = self.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
                    q, _, _ = self.trajclustermodule(emb)

                    # Get hard cluster assignments (argmax of soft assignments)
                    hard_assignments = torch.argmax(q, dim=1)

                    # Compute cluster-level contrastive loss
                    cluster_level_contrastive_loss = criterion.cluster_level_contrastive_loss(
                        emb, hard_assignments, temperature=0.05, mu=1.0)

                total_loss = Config.alpha * infoNCE_loss + Config.beta * assignment_loss + Config.gamma * cluster_level_contrastive_loss + Config.omega * inter_cluster_loss

                # Backpropagation
                loss_train = total_loss
                loss_train.backward()
                optimizer.step()

                # Append losses for logging
                infoNCE_losses.append(infoNCE_loss.item())
                main_clustering_losses.append(assignment_loss.item())
                inter_cluster_losses.append(inter_cluster_loss.item())
                cluster_level_contrastive_losses.append(cluster_level_contrastive_loss.item())
                train_losses.append(loss_train.item())
                train_gpus.append(tool_funcs.GPUInfo.mem()[0])
                train_rams.append(tool_funcs.RAMInfo.mem())

                if i_batch % 200 == 0 and i_batch:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}, gpu={}, ram={}" \
                                  .format(i_ep + 1, i_batch, loss_train.item(),
                                          time.time() - _time_batch, tool_funcs.GPUInfo.mem(),
                                          tool_funcs.RAMInfo.mem()))

            scheduler.step()  # decay before optimizer when pytorch < 1.1

            logging.info(
                "[Training] ep={}, train_losses={:.4f}, infoNCE_loss={:.4f}, main_clustering_losses={:.4f}, inter_cluster_loss={:.4f}, cluster_level_contrastive_loss={:.4f}, @={:.3f}" \
                    .format(i_ep + 1,
                            tool_funcs.mean(train_losses),
                            tool_funcs.mean(infoNCE_losses),
                            tool_funcs.mean(main_clustering_losses),
                            tool_funcs.mean(inter_cluster_losses),
                            tool_funcs.mean(cluster_level_contrastive_losses),
                            time.time() - _time_ep))

        return {'Best Epoch': best_epoch,
                'Best Acc': best_acc,
                'Best Nmi': best_nmi,
                'Best Ari': best_ari}

    # single batchy data generator - for training
    def trajcluster_dataset_generator_batchi(self, augment=False):
        """
        Generator for trajectory clustering dataset with optional data augmentation.

        Args:
            augment (bool): Whether to generate augmented data (next 6 elements).

        Yields:
            Tuple of trajectory data and optionally augmented data.
        """
        datasets_label = self.dic_datasets['trains_traj_label']
        datasets = self.dic_datasets['trains_traj']

        cur_index = 0
        len_datasets = len(datasets)

        while cur_index < len_datasets:
            # Define the end index for the current batch
            end_index = cur_index + Config.trajcluster_batch_size \
                if cur_index + Config.trajcluster_batch_size < len_datasets \
                else len_datasets

            # Get the current batch of trajectories
            trajs = [datasets[d_idx] for d_idx in range(cur_index, end_index)]

            # Extract corresponding labels for the batch of trajectories
            trajs_label = datasets_label[cur_index:end_index]  # slice labels for the batch

            # Process trajectory data into cell and spatial representations
            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            # trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = [torch.tensor(generate_semantics_features(t)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first=False).to(Config.device)

            trajs_emb_cell = [self.cellembs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first=False).to(
                Config.device)  # [seq_len, batch_size, emb_dim]

            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype=torch.long, device=Config.device)

            if augment:
                # Data augmentation
                trajs1 = [self.aug1(t) for t in trajs]
                trajs2 = [self.aug2(t) for t in trajs]

                trajs1_cell, trajs1_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs1])
                trajs2_cell, trajs2_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs2])

                trajs1_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace), dtype=torch.float32) for t in
                                trajs1_p]
                trajs2_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace), dtype=torch.float32) for t in
                                trajs2_p]

                trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first=False).to(Config.device)
                trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first=False).to(Config.device)

                trajs1_emb_cell = [self.cellembs[list(t)] for t in trajs1_cell]
                trajs2_emb_cell = [self.cellembs[list(t)] for t in trajs2_cell]

                trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first=False).to(
                    Config.device)  # [seq_len, batch_size, emb_dim]
                trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first=False).to(
                    Config.device)  # [seq_len, batch_size, emb_dim]

                trajs1_len = torch.tensor(list(map(len, trajs1_cell)), dtype=torch.long, device=Config.device)
                trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype=torch.long, device=Config.device)

                # Yield the batch including the augmented data
                yield trajs_emb_cell, trajs_emb_p, trajs_len, trajs_label, trajs1_emb_cell, trajs1_emb_p, trajs1_len, trajs2_emb_cell, trajs2_emb_p, trajs2_len
            else:
                # Yield the batch without augmented data
                yield trajs_emb_cell, trajs_emb_p, trajs_len, trajs_label

            cur_index = end_index

    @staticmethod
    def load_trajcluster_dataset():
        # read (1) traj dataset for trajcluster, (2) simi matrix dataset for trajcluster
        trajcluster_traj_dataset_file = Config.dataset_file

        trains, evals, tests = read_trajcluster_traj_dataset(trajcluster_traj_dataset_file)
        trains_traj, evals_traj, tests_traj = trains.merc_seq.values, evals.merc_seq.values, tests.merc_seq.values
        trains_traj_label, evals_traj_label, test_traj_label = trains.label.values, evals.label.values, tests.label.values

        # trains_traj : [[[lon, lat_in_merc], [], ..], [], ...]
        # trains_simi : list of list
        return {'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj,
                'trains_traj_label': trains_traj_label, 'evals_traj_label': evals_traj_label,
                'test_traj_label': test_traj_label}

    def init_clusterlayer(self):

        self.encoder.eval()
        self.trajclustermodule.eval()

        vecs = []
        y = []
        # data_dict = {'traj': [], 'time': [], 'label': []}
        with torch.no_grad():
            # for i, (
            # input_ids, masked_tokens, masked_pos, input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask,
            # lengths, label) in enumerate(self.cluster_loader):
            #
            #     emb, q_i, head_in, head_cl = self.cluster_model(input_ids_o.to(self.device),
            #                                                         lengths.to(self.device), id_mask.to(self.device),
            #                                                         timestamp_o.to(self.device), None)
            #     # emb, q_i, head_in, head_cl = self.cluster_model(input_ids.to(self.device), lengths.to(self.device), id_mask.to(self.device),timestamp.to(self.device))
            #     data_dict['traj'].append(input_ids_o.cpu().data)
            #     data_dict['time'].append(timestamp_o.cpu().data)
            #     data_dict['label'].append(label.cpu().data)

            for i_batch, batch in enumerate(self.trajcluster_dataset_generator_batchi()):
                # trajs_emb, trajs_emb_p, trajs_len, label, _, _, _, _, _, _ = batch
                trajs_emb, trajs_emb_p, trajs_len, label = batch
                emb = self.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)

                vecs.append(emb.cpu().data)
                y.append(torch.tensor(label, device=Config.device))  # Ensure label is a tensor

            vecs = torch.cat(vecs).cpu().numpy()
            y = torch.cat(y).cpu().numpy()

            # data_dict['traj'] = torch.cat(data_dict['traj']).cpu().numpy().tolist()
            # data_dict['time'] = torch.cat(data_dict['time']).cpu().numpy().tolist()
            # data_dict['label'] = torch.cat(data_dict['label']).cpu().numpy().tolist()
            # df = pd.DataFrame(data_dict)
            # df.to_csv('my_data2.csv', index=False)

            # torch.save(self.cluster_model.bert.state_dict(), 'before_initcluster.pt')
            if True:
                print('-' * 20 + 'init cluster layer' + '-' * 20)

                kmeans = KMeans(n_clusters=Config.init_clusters, n_init=10, random_state=58).fit(vecs)

                y_pred = kmeans.fit_predict(vecs)
                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                # print('KMeans\tAcc: {0:.4f}\tnmi: {1:.4f}\tari: {2:.4f}'.format(acc, nmi, ari))
                self.trajclustermodule.clusterlayer.clusters.data = torch.Tensor(
                    kmeans.cluster_centers_).to(Config.device)

                # with open(self.args.kname+".pkl", "wb") as file:
                #     pickle.dump(kmeans, file)
                # np.save(self.args.kname+'_vec',vecs )
                # np.save(self.args.kname+'_y',y )

                # torch.save(self.cluster_model.clusterlayer.clusters.data, './pth_model/init_clusterLayer_epoch_%s.pth' % str(self.args.pretrain_epoch))
                # torch.save({'model': self.cluster_model}, './pth_model/init_clusterLayer_epoch_%s.pth' % str(self.args.pretrain_epoch))
            # torch.save(self.cluster_model.bert.state_dict(), 'after_initcluster.pt')

        self.encoder.train()
        self.trajclustermodule.train()  # 将 cluster_model 恢复为训练模式，以便后续的模型训练和调整

        return vecs, kmeans.cluster_centers_, acc

    def _select_criterion(self):
        """选择损失函数"""
        criterion = Loss_Function()
        return criterion