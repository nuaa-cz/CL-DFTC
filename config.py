import os
import random
import torch
import numpy


def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    debug = True
    dumpfile_uniqueid = ''
    seed = 2000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10]  # dont use os.getcwd()
    checkpoint_dir = root_dir + '/exp/snapshots'
    emb_dir = root_dir + '/exp/embeddings'

    dataset = ''
    dataset_prefix = ''
    dataset_file = ''
    dataset_cell_file = ''
    dataset_embs_file = ''

    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 600
    min_traj_len = 100
    cell_size = 4000.0
    cellspace_buffer = 10000.0

    # ===========CL-DFTC=============
    trajcl_batch_size = 128
    cell_embedding_dim = 256
    seq_embedding_dim = 512
    moco_proj_dim = seq_embedding_dim // 2
    moco_nqueue = 2048
    moco_temperature = 0.05

    trajcl_training_epochs = 10
    trajcl_training_bad_patience = 5
    trajcl_training_lr = 0.001
    trajcl_training_lr_degrade_gamma = 0.5
    trajcl_training_lr_degrade_step = 5
    trajcl_aug1 = 'simplify'
    trajcl_aug2 = 'shift'
    trajcl_local_mask_sidelen = cell_size * 11

    # Transformer
    trans_attention_dropout = 0.1
    trans_pos_encoder_dropout = 0.1

    # MVFTT
    stru_embedding_dim = 256
    trans_stru_hidden_dim = 2048
    trans_stru_attention_head = 4
    trans_stru_attention_layer = 2
    sem_embedding_dim = 2
    trans_sem_hidden_dim = 16
    trans_sem_attention_head = 1
    trans_sem_attention_layer = 4

    traj_simp_dist = 5000
    traj_shift_decay_factor = 5
    traj_shift_max_offset = 20000
    traj_mask_ratio = 0.3
    
    # truncated_rand
    truncated_rand_bound_lo = -10000
    truncated_rand_bound_hi = 10000

    # ===========Joint train=============
    trajcluster_encoder_name = 'AirTrajCL'
    trajcluster_batch_size = 128
    trajcluster_epochs = 10
    trajcluster_training_bad_patience = 10
    trajcluster_learning_rate = 0.0001
    trajcluster_learning_weight_decay = 0.0001
    trajcluster_finetune_lr_rescale = 0.5
    trajcluster_aug = 0

    init_clusters = 40

    infoNCE_loss = 0
    assignment_loss = 1
    cluster_level_contrastive_loss = 0
    inter_cluster_distance_loss = 0

    alpha = 0.1
    beta = 1
    gamma = 0.01
    omega = 0.01

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()

    @classmethod
    def post_value_updates(cls):
        if 'pvg' == cls.dataset:
            cls.dataset_prefix = 'pvg'
            cls.min_lon = 119.20118497159169
            cls.min_lat = 28.98681600416685
            cls.max_lon = 124.40961545795538
            cls.max_lat = 33.40690796666688

        else:
            pass

        cls.dataset_file = cls.root_dir + f'/data_{cls.dataset}/' + cls.dataset_prefix
        cls.dataset_cell_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_cellspace.pkl'
        cls.dataset_embs_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_embdim' + str(
            cls.cell_embedding_dim) + '_embs.pkl'
        set_seed(cls.seed)

        cls.moco_proj_dim = cls.seq_embedding_dim // 2

    @classmethod
    def to_str(cls):  # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
            dic.items() \
            ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
