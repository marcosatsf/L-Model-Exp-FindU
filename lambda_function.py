import json
from mongo_handler import MongoConnection
from clustering import ClusterFindU

def lambda_handler(event, context):
    
    request = eval(event['body'])
    #print(request)
    
    mongo_h = MongoConnection()
    #print(mongo_h.info())
    dataset, id_str = mongo_h.parse_info()

    cluster = ClusterFindU(id_str, dataset)
    cluster.run_cluster()

    #result = mongo_h.get_collection(request["table"])
    #print(result)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'user_id':id_str})
    }
