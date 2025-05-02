import torch, torch.nn.functional as F, pickle
from sklearn.preprocessing import normalize

class SubtractingRankingSet():
    def __init__(self, file_path, idf_weights, device = "cuda"):
        '''
            Creates a ranking set. Assumes the file specified at file_path is a pickle file of 
            a numpy array of fingerprints.

            self.data should be (n, 6144)
        '''
        self.device = device
        with torch.no_grad():
          with open(file_path, "rb") as f:
            self.data = F.normalize(torch.load(f).type(torch.FloatTensor), dim=1, p=2.0).to(device)
          self.idf = idf_weights.to(device)
          self.idf_data = F.normalize(self.data - self.idf, dim=1, p=2.0).to(device)

            
    @staticmethod
    def round(fp):
        hi = torch.max(fp)
        where = torch.where(fp == hi)
        fp = torch.zeros(6144)
        fp[where] = 1
        return fp

    @staticmethod
    def normalized_to_nonzero(fp):
        hi = torch.max(fp)
        nonzero = torch.nonzero(torch.isclose(fp, hi))
        return tuple(nonzero[:,0].tolist())

    def dot_prod_rank(self, data, queries, truths, thresh):
        with torch.no_grad():
            q = queries.size()[0]
            n = data.size()[0]
            query_products = data @ queries.T # (n x 6144) * (6144 x q) = (n, q)
            truth_products = data @ truths.T # (n, q)

            ct_ones = torch.sum(torch.isclose(truth_products, torch.ones(n, q).to(self.device)), dim = 0) # (q)
            ct_greater = torch.sum(query_products > thresh, dim = 0) # ((n, q) >= (q)) -> (q)
            return ct_greater - ct_ones

    def batched_rank_tfidf(self, queries, truths):
        '''
            Perform a batched ranking with tfidf

            query(predictions) should be (q, 6144)
            truths should be (q, 6144)

            returns the number of items that have a higher dot product similarity (query_i (dot) truth_i)
        '''
        if self.idf_data is None:
            raise Exception("No idf weights")
        with torch.no_grad():
            q = queries.size()[0]
            queries = F.normalize(queries.to(self.device) - self.idf, dim=1, p=2.0)
            truths = F.normalize(truths.to(self.device) - self.idf, dim=1, p=2.0)
            thresh = torch.stack([torch.dot(queries[i,:], truths[i,:]) for i in range(q)]).to(self.device) # (q)
            
            return self.dot_prod_rank(self.idf_data, queries, truths, thresh)

