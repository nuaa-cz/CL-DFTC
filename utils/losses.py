import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()

    def clusterloss(self, q, p, loss_cuda):
        """
        caculate the KL loss for clustering
        """
        q, p = q.to(loss_cuda), p.to(loss_cuda)
        criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
        return criterion(q.log(), p)

    def clusteringLoss(self, clusterlayer, context, p, cuda2, loss_cuda, q):
        """
        One batch cluster KL loss

        Input:
        context: (batch, hidden_size * num_directions) last hidden layer from encoder
        clusterlayer: caculate Student's t-distribution with clustering center

        p: (batch_size,n_clusters)target distribution

        Output:loss
        """
        batch = context.size(0)
        assert batch == p.size(0)
        kl_loss = self.clusterloss(q, p, loss_cuda)

        return kl_loss.div(batch)

    def Cross_Entropy_Loss(self, logit_lm, ground_truth):
        _, num_classes = logit_lm.size()
        p_i = torch.softmax(logit_lm, dim=1)
        y = F.one_hot(ground_truth, num_classes=num_classes)
        loss = y * torch.log(p_i + 0.0000001)
        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss

    def infonce(self, features_1, features_2, temperature=0.1):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        features = F.normalize(features, dim=1)
        features_1, features_2 = features.chunk(2, 0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)

        loss_pos = (- torch.log(pos / (Ng + pos))).mean()

        return loss_pos

    def nce(self, z1, z2, device):
        sim_mat = torch.mm(z1, z2.t())
        batch_size = z1.shape[0]
        pos_mask = torch.eye(batch_size).to(device)  # .cuda()
        return nn.BCEWithLogitsLoss(reduction='none')(sim_mat, pos_mask).sum(1).mean()

    def contrastive_loss_simclr(self, z1, z2, temperature=0.1, similarity='inner'):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)  # .to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / temperature

        # loss_res = Cross_Entropy_Loss(logits, labels)
        loss_res = self.Cross_Entropy_Loss(logits, labels)  # 源代码好像有错误，上面的是源代码
        return loss_res

    def inter_cluster_distance_loss(self, clusters):
        """
        Inter-cluster distance loss based on the paper formula:
        L_d(θ) = Σ_i Σ_{j≠i} exp(-(\mu_i - \mu_j)^2)

        Args:
            clusters: tensor of shape (n_clusters, feature_dim) - cluster centers

        Returns:
            inter-cluster distance loss
        """
        n_clusters = clusters.size(0)

        # Calculate pairwise squared differences between cluster centers
        # clusters[:, None] - clusters creates a tensor of shape (n_clusters, n_clusters, feature_dim)
        diff = clusters[:, None] - clusters  # (n_clusters, n_clusters, feature_dim)

        # Calculate squared L2 norm for each pair: (\mu_i - \mu_j)^2
        squared_distances = torch.sum(diff ** 2, dim=2)  # (n_clusters, n_clusters)

        # Apply exponential: exp(-(\mu_i - \mu_j)^2)
        exp_neg_squared_distances = torch.exp(-squared_distances)

        # Mask out diagonal entries (i == j cases) since we only want j ≠ i
        mask = 1 - torch.eye(n_clusters, device=clusters.device)

        # Sum over all i and j where j ≠ i
        loss = (exp_neg_squared_distances * mask).sum()

        return loss

    def cluster_level_contrastive_loss(self, trajectory_embeddings, cluster_assignments, temperature=0.5, mu=1.0):
        """
        Cluster-level contrastive loss based on the paper formula:
        L_e = -1/n * sum_{i=1}^n log(exp(sim(T_i, T_i^+)/tau) / (exp(sim(T_i, T_i^+)/tau) + mu * exp(sim(T_i, T_i^-)/tau)))

        Args:
            trajectory_embeddings: tensor of shape (batch_size, embedding_dim) - trajectory embeddings
            cluster_assignments: tensor of shape (batch_size,) - hard cluster assignments for each trajectory
            temperature: temperature parameter tau
            mu: weight parameter for negative samples

        Returns:
            cluster-level contrastive loss
        """
        batch_size = trajectory_embeddings.size(0)
        device = trajectory_embeddings.device

        # Ensure cluster_assignments is on the same device as trajectory_embeddings
        cluster_assignments = cluster_assignments.to(device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(trajectory_embeddings, dim=1)

        # Compute similarity matrix (cosine similarity)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        total_loss = 0.0
        valid_samples = 0

        for i in range(batch_size):
            anchor_cluster = cluster_assignments[i]

            # Find positive samples (same cluster, excluding anchor)
            positive_mask = (cluster_assignments == anchor_cluster) & (torch.arange(batch_size, device=device) != i)

            # Find negative samples (different clusters)
            negative_mask = (cluster_assignments != anchor_cluster)

            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                # Get similarities for positive and negative samples
                pos_similarities = similarity_matrix[i][positive_mask]
                neg_similarities = similarity_matrix[i][negative_mask]

                # Use the maximum positive similarity (closest positive sample)
                pos_sim = pos_similarities.max()

                # Compute the contrastive loss for this anchor
                pos_exp = torch.exp(pos_sim / temperature)
                neg_exp = torch.exp(neg_similarities / temperature).sum() * mu

                sample_loss = -torch.log(pos_exp / (pos_exp + neg_exp))
                total_loss += sample_loss
                valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
