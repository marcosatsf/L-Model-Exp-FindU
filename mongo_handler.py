from bson.objectid import ObjectId
from datetime import datetime
from clustering import ClusterFindU
from pymongo import MongoClient
import os

class MongoConnection:
    def __init__(self):
        mongo_url = os.environ['mongo_client_url']
        self.client = MongoClient(mongo_url)
        self.database = self.client['findu']
        self.query_result = []

    def count_users_locations(self):
        self.__load_collection('locations')
        self.__query(self.collection.find())
        counting = {}
        for each_item in self.query_result:
            if counting.get(each_item['owner']):
                counting[each_item['owner']] += 1
            else:
                counting[each_item['owner']] = 1
        return counting

    def parse_info(self):
        self.__load_collection('locations')
        self.__query(self.collection.find({'owner':ObjectId('5f70e3dbd120710a34229b4b')}))
        coord_timestamp = []
        for item in self.query_result:
            #print(item)
            try:
                if item.get('location').get('coordinates'):
                    internal_struct = [
                        item['location']['coordinates'][0],
                        item['location']['coordinates'][1]
                    ]
                    if item.get('timestamp'):
                        internal_struct.append(
                            int(item['timestamp'])/1e8
                        )
                    if len(internal_struct) == 3:
                        coord_timestamp.append(internal_struct)
            except AttributeError:
                continue

        return coord_timestamp, str(ObjectId('5f70e3dbd120710a34229b4b'))


    def get_collection(self, coll_name):
        self.__load_collection(coll_name)
        self.__query(self.collection.find())
        return self.ret_query_json()

    def __load_collection(self, coll_name):
        self.collection = self.database[coll_name]

    def __query(self, cursor):
        self.query_result = []
        for each_row in cursor:
            self.query_result.append(each_row)

    def ret_query_json(self):
        list_row = []
        for each_row in self.query_result:
            list_row.append(each_row)
        return {'response':list_row}
        
    def info(self):
        return 'client: {}, db: {}, query_result: {}'.format(self.client,
                                                            self.database,
                                                            self.query_result)

    
# if __name__ == "__main__":
#     conn = MongoConnection()
#     dataset, id_str = conn.parse_info()

#     cluster = ClusterFindU(id_str, dataset)
#     cluster.run_cluster()
    