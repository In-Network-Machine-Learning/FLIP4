########################################################################
# Copyright (c) Mingyuan Zang, Tomasz Koziak, Networks Technology and Service Platforms Group, Technical University of Denmark
# This work was partly done with Changgang Zheng at Computing Infrastructure Group, University of Oxford
# All rights reserved.
# E-mail: mingyuanzang@outlook.com
# Licensed under the Apache License, Version 2.0 (the 'License')
########################################################################
import socket, pickle, threading, hashlib, json, jwt, datetime, random
from FL.supported_models import Supported_models
from sklearn.utils.class_weight import compute_sample_weight
import torch
from torch import nn
from sklearn.metrics import f1_score
from copy import deepcopy



from federated_gbdt.core.loss_functions import SigmoidBinaryCrossEntropyLoss, BinaryRFLoss, SoftmaxCrossEntropyLoss

class Fedavg:
    def __init__(self, name, learning_rate, model_name):
        self.name = name
        self.model = None
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.accuracy = 0
        self.ip = '127.0.0.1'
        self.port = 5001
        self.clients = []
        self.hashtable = None
        self.secret = '5791628bb0b13ce0c676dfde280ba245'
        self.socket = None
        self.loss = BinaryRFLoss()

        try:
            with open('ghost.json', 'r') as f:
                self.hashtable = json.loads(f.read())
        except FileNotFoundError:
            self.hashtable = None

        self.socket = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
        try:
            self.socket.bind((self.ip, self.port))
        except socket.error as e:
            print(str(e))

        print('[INFO] Waiting for a Connection...')
        self.socket.listen(5)


    def client_login(self, connection):
            connection.send(str.encode('ENTER USERNAME : ')) # Request Username
            name = connection.recv(2048)
            connection.send(str.encode('ENTER PASSWORD : ')) # Request Password
            password = connection.recv(2048)
            password = password.decode()
            name = name.decode()
            password=hashlib.sha256(str.encode(password)).hexdigest() # Password hash using SHA256
        # REGISTERATION PHASE
        # If new user,  regiter in Hashtable Dictionary
            if name not in self.hashtable:
                self.hashtable[name]=password
                connection.send(str.encode('Registeration Successful'))
                print('[INFO] Registered : ',name)
                print("{:<8} {:<20}".format('USER','PASSWORD'))
                for k, v in self.hashtable.items():
                    label, num = k,v
                    print("{:<8} {:<20}".format(label, num))
                print("-------------------------------------------")
                connection.close()
            else:
        # If already existing user, check if the entered password is correct
                if(self.hashtable[name] == password):
                    token = jwt.encode(
                    {
                        "name": name,
                        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
                    },
                        self.secret
                    )
                    print('[INFO] Connected : ',token)
                    connection.send(token.encode())
                    connection.close()
                else:
                    connection.send(str.encode('False')) # Response code for login failed
                    print('[INFO] Connection denied : ',name)
                    connection.close()

    def check_token(self, token):
            try:
                data = jwt.decode(token, self.secret, algorithms=['HS256'])
                current_user = data['name']
            except:
                return False
            return True


    def init_global_model(self, model):
        self.model = model



    def register_client(self, clients):
        self.clients = clients

    def update_global_model(self, applicable_models, round_weights, model_name):
        # Average models parameters
        coefs = []
        intercept = []
        if self.model_name == Supported_models.gradient_boosting_classifier:
            for model in applicable_models:
                for tree_i in range(0, len(model.trees)):
                    window = model.es_window
                    if len(model.trees) >= 2*model.es_window and model.early_stopping:
                        if model.es_metric == "leaf_hess":
                            es_metric = model.train_monitor.leaf_gradient_tracker[1]
                        elif model.es_metric == "leaf_grad":
                            es_metric = model.train_monitor.leaf_gradient_tracker[0]
                        elif model.es_metric == "root_grad":
                            es_metric = model.train_monitor.root_gradient_tracker[0]
                        else:
                            es_metric = model.train_monitor.root_gradient_tracker[1]

                        current_window_hess = abs(np.mean(es_metric[-window:])) if "grad" in model.es_metric else np.mean(es_metric[-window:])
                        previous_window_hess = abs(np.mean(es_metric[-2*window:-window])) if "grad" in model.es_metric else np.mean(es_metric[-2*window:-window])
                        per_change = (previous_window_hess/current_window_hess-1)*100

                        if ("standard" in model.early_stopping or model.early_stopping == "rollback" or "average" in model.early_stopping) and per_change < threshold_change:
                            print("[INFO] Early Stopping at round", i+1)
                            if model.early_stopping == "rollback":
                                model.trees = model.trees[:-1]
                                model.train_monitor.y_weights -= model.train_monitor.current_tree_weights
                            break
                        elif model.early_stopping == "retry" and per_change < threshold_change: #es_metric[-2] - es_metric[-1] < 0
                            prune_step = -2 if "root" in model.es_metric else -1
                            model.trees = model.trees[:prune_step]
                            model.train_monitor.gradient_info = model.train_monitor.gradient_info[:prune_step]
                            model.train_monitor.root_gradient_tracker[0] = model.train_monitor.root_gradient_tracker[0][:prune_step]
                            model.train_monitor.root_gradient_tracker[1] = model.train_monitor.root_gradient_tracker[1][:prune_step]
                            model.train_monitor.leaf_gradient_tracker[0] = model.train_monitor.leaf_gradient_tracker[0][:prune_step]
                            model.train_monitor.leaf_gradient_tracker[1] = model.train_monitor.leaf_gradient_tracker[1][:prune_step]
                            model.train_monitor.y_weights -= model.train_monitor.current_tree_weights + (prune_step+1)*-1*model.train_monitor.previous_tree_weights # If root then remove 2 trees, if leaf remove 1

                    model.train_monitor.end_timing_event("server", "post-tree ops")

                    # Reset tracking vars
                    model.train_monitor._update_comm_stats(model.split_method, model.training_method)
                    model.train_monitor.reset()


        if self.model_name == Supported_models.NN_classifier:
            fc1_mean_weight = torch.zeros(size=applicable_models[0].fc1.weight.shape)
            fc1_mean_bias = torch.zeros(size=applicable_models[0].fc1.bias.shape)

            fc2_mean_weight = torch.zeros(size=applicable_models[0].fc2.weight.shape)
            fc2_mean_bias = torch.zeros(size=applicable_models[0].fc2.bias.shape)

            fc3_mean_weight = torch.zeros(size=applicable_models[0].fc3.weight.shape)
            fc3_mean_bias = torch.zeros(size=applicable_models[0].fc3.bias.shape)

            i = 0

            for model in applicable_models:
                fc1_mean_weight += model.fc1.weight.data.clone() * round_weights[i]
                fc1_mean_bias += model.fc1.bias.data.clone() * round_weights[i]
                fc2_mean_weight += model.fc2.weight.data.clone() * round_weights[i]
                fc2_mean_bias += model.fc2.bias.data.clone() * round_weights[i]
                fc3_mean_weight += model.fc3.weight.data.clone() * round_weights[i]
                fc3_mean_bias += model.fc3.bias.data.clone() * round_weights[i]
                i += 1

            model.fc1.weight.data = fc1_mean_weight.data.clone()
            model.fc2.weight.data = fc2_mean_weight.data.clone()
            model.fc3.weight.data = fc3_mean_weight.data.clone()
            model.fc1.bias.data = fc1_mean_bias.data.clone()
            model.fc2.bias.data = fc2_mean_bias.data.clone()
            model.fc3.bias.data = fc3_mean_bias.data.clone()

        if self.model_name == Supported_models.BNN_classifier:
            fc_mean_weight = torch.zeros(size=applicable_models[0].fc.weight.shape)
            fc_mean_bias = torch.zeros(size=applicable_models[0].fc.bias.shape)
            i = 0

            for model in applicable_models:
                fc_mean_weight += model.fc.weight.data.clone() * round_weights[i]
                fc_mean_bias += model.fc.bias.data.clone() * round_weights[i]
                i += 1

            model.fc.weight.data = fc_mean_weight.data.clone()
            model.fc.bias.data = fc_mean_bias.data.clone()

    def test_model_f1(self, y_test=None, X_test=None):
        if self.model_name == Supported_models.NN_classifier:
            test_x = np.float32(X_test)
            test_x = torch.FloatTensor(X_test)
            output = self.model(test_x)
            prediction = output.argmax(dim=1, keepdim=True)
            return f1_score(prediction,y_test, average="binary")
        if self.model_name == Supported_models.BNN_classifier:
            test_x = np.float32(X_test)
            test_x = torch.FloatTensor(X_test)
            output = self.model(test_x)
            prediction = output.argmax(dim=1, keepdim=True)
            return f1_score(prediction,y_test, average="binary")
        if self.model is None:
            print("[WARN] Model not trained yet.")
            return 0
        if y_test is None:
            y_hat = self.model.predict(self.x_test)
            return f1_score(self.y_test, y_hat, average="binary")
        else:
            y_hat = self.model.predict(X_test)
            return f1_score(y_test, y_hat, average="binary")


    def wait_for_data(self,connection):
        print('[INFO] Waiting for a Connection...')
        data = b""
        while True:
            packet = connection.recv(4096)
            if not packet:
                break
            data += packet
        d = pickle.loads(data)

        token = d[0]
        self.check_token(token)
        struct = d[1]

        self.clients.append(struct)
        connection.close()

    def send_request(self, connection, msg):
        print('[INFO] Waiting for a Connection...')
        data_string = pickle.dumps(msg)
        connection.send(data_string)
        connection.close()
        print("[INFO] Data Sent to Server")

    def login(self):
        HOST = self.server_address
        PORT = self.server_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            response = s.recv(2048)
            # Input UserName
            name = input(response.decode())
            s.send(str.encode(name))
            response = s.recv(2048)
            # Input Password
            password = input(response.decode())
            s.send(str.encode(password))
            ''' Response : Status of Connection :
                1 : Registeration successful
                2 : Connection Successful
                3 : Login Failed
            '''
            # Receive response
            response = s.recv(2048)
            response = response.decode()

            self.token = response



# Start server for rounds of federated learning
if __name__ == "__main__":
    NUMBER_OF_CLIENTS = 2
    NUMBER_OF_ROUNDS = 5
    selected_model = Supported_models.gradient_boosting_classifier
    fedavg = Fedavg("global", 0.1, selected_model)
    ThreadCount = 0
    threads = []
    epochs = 10
    max_score = 0
    optimal_model = None

    for round in range(NUMBER_OF_ROUNDS):
        print(f'[INFO] Starting new round!')
        print(round, end=' ')

        applicable_models = []
        applicable_name = []
        round_weights = []
        threads = []
        dataset_size = 0

        while True:
            Client, address = fedavg.socket.accept()
            client_handler = threading.Thread(
                target=fedavg.wait_for_data,
                args=(Client,)
            )
            client_handler.start()
            threads.append(client_handler)
            if len(threads) == NUMBER_OF_CLIENTS:
                break

        # Wait for all of them to finish
        for x in threads:
            x.join()

        applicable_clients = random.sample((fedavg.clients),2)

        if round == 0:
            fedavg.model = applicable_clients[0].model

        for client in applicable_clients:
            print(f'.', end='')
            round_weights.append(client.dataset_size)
            dataset_size += client.dataset_size
            applicable_models.append(client.model)

        round_weights = np.array(round_weights) / dataset_size
        print('[INFO] aggregate and update global model...')
        fedavg.update_global_model(applicable_models, round_weights, selected_model)

        threads = []
        while True:
            Client, address = fedavg.socket.accept()
            client_handler = threading.Thread(
                target=fedavg.send_request,
                args=(Client,fedavg.model)
            )
            client_handler.start()
            threads.append(client_handler)
            if len(threads) == NUMBER_OF_CLIENTS:
                break
        for x in threads:
                x.join()

    fedavg.socket.close()
