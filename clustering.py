from classifier import ClassifierFindU
import sklearn.cluster as sk_cl
from sklearn.mixture import GaussianMixture

class ClusterFindU:
    def __init__(self, user_id, df):
        self.user_id = user_id
        self.dataset = df

    def run_cluster(self): #self, func, df
        #BIRCH
        mdl = self.clust_BIRCH(0.05)
        yhat = mdl.fit_predict(self.dataset)
        print(f'BIRCH, {(yhat,len(yhat))}')
        yhat = [max(yhat)+1 if each == -1 else each for each in yhat]
        ClassifierFindU(self.dataset, yhat,'BIRCH', self.user_id)

        #DBSCAN
        mdl = self.clust_DBSCAN(0.05, 5)
        yhat = mdl.fit_predict(self.dataset)
        print(f'DBSCAN, {(yhat,len(yhat))}')
        yhat = [max(yhat)+1 if each == -1 else each for each in yhat]
        ClassifierFindU(self.dataset, yhat,'DBSCAN', self.user_id)

        #OPTICS
        mdl = self.clust_OPTICS(0.05, 50)
        yhat = mdl.fit_predict(self.dataset)
        print(f'OPTICS, {(yhat,len(yhat))}')
        yhat = [max(yhat)+1 if each == -1 else each for each in yhat]
        ClassifierFindU(self.dataset, yhat, 'OPTICS', self.user_id)


    def clust_BIRCH(self, thresh): # 0.05
        # define the model
        return sk_cl.Birch(threshold=thresh)


    def clust_DBSCAN(self, eps, min_samp): # 0.05, 5
        # define the model
        return sk_cl.DBSCAN(eps=eps, min_samples=min_samp)


    def clust_OPTICS(self, eps, min_samp): # 0.05, 50
        return sk_cl.OPTICS(eps=eps, min_samples=min_samp)  
