from pipeline import *
import copy
from model import *
from ewc import *
from arf import *
import torch
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy
import logging
import argparse
import time
import random
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from armnet import *
from skmultiflow.data import SEAGenerator, HyperplaneGenerator, STAGGERGenerator, RandomRBFGeneratorDrift, LEDGeneratorDrift, WaveformGenerator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='model used in training')
    parser.add_argument('--gbdt', type=int, default=0, help='whether to use gbdt for tree model')
    parser.add_argument('--dataset', type=str, default='selected', help='dataset used for training')
    parser.add_argument('--alg', type=str, default='naive', help='training algorithm')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs for each window')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--layers', type=int, default=3, help='number of layers to test model heaviness')
    parser.add_argument('--reg', type=float, default=1, help='regularization factor')
    parser.add_argument('--buffer', type=int, default=100, help='the number of examplars allowed to store')
    parser.add_argument('--ensemble', type=int, default=1, help='ensemble size')
    parser.add_argument('--window_factor', type=float, default=1, help='factor to multiply window size')
    parser.add_argument('--missing_fill', type=str, default='knn2', help='method to fill missing value')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='repo to store logging file')
    parser.add_argument('--device', type=str, default='cpu', help='device to train')
    parser.add_argument('--init_seed', type=int, default=0, help='initial random seed')
    
    args = parser.parse_args()
    return args

def compute_result(net, window_x, window_y, task, tree=False):
    if task == "classification":
        if tree:
            pred_label = torch.LongTensor(net.predict(window_x))
        else:
            out = net(window_x)
            _, pred_label = torch.max(out.data, 1)
        acc = (pred_label == window_y).sum().item()/window_y.shape[0]
        return 1-acc
    else:
        if tree:
            out = torch.Tensor(net.predict(window_x))
        else:
            out = net(window_x).reshape(-1).detach()
        loss = torch.mean(torch.square(out - window_y))
        return loss.item()
    
def compute_result_constant(constant_output, window_x, window_y, task):
    if task == "classification":
        acc = (window_y == constant_output).sum().item()/window_y.shape[0]
        return 1-acc
    else:
        loss = torch.mean(torch.square(window_y - constant_output))
        return loss.item()
    
def compute_result_ensemble(net_ensemble, cnt, window_x, window_y, task, tree=False):
    if task == "classification":
        if tree:
            for i in range(cnt):
                net = net_ensemble[i]
                pred_label = torch.LongTensor(net.predict(window_x))
                if i == 0:
                    final = pred_label.unsqueeze(1)
                else:
                    final = torch.cat((final, pred_label.unsqueeze(1)), dim=1)
        else:
            for i in range(cnt):
                net = net_ensemble[i]
                out = net(window_x)
                _, pred_label = torch.max(out.data, 1)
                if i == 0:
                    final = pred_label.unsqueeze(1)
                else:
                    final = torch.cat((final, pred_label.unsqueeze(1)), dim=1)
        
        pred_label, _ = torch.mode(final, dim=1)
        acc = (pred_label == window_y).sum().item()/window_y.shape[0]
        return 1-acc
    else:
        if tree:
            for i in range(cnt):
                net = net_ensemble[i]
                out = torch.Tensor(net.predict(window_x))
                if i == 0:
                    final = out
                else:
                    final += out
        else:
            for i in range(cnt):
                net = net_ensemble[i]
                out = net(window_x).reshape(-1).detach()
                if i == 0:
                    final = out
                else:
                    final += out
        out = final/cnt

        loss = torch.mean(torch.square(out - window_y))
        return loss.item()
    
def fill_missing_value(window_x, missing_fill):
    if missing_fill.startswith("knn"):
        num = eval(missing_fill[3:])
        imp = KNNImputer(n_neighbors=num, weights="uniform", keep_empty_features=True)
        filled = imp.fit_transform(window_x.numpy())
        return torch.tensor(filled)
    elif missing_fill == "regression":
        imp = IterativeImputer(keep_empty_features=True)
        filled = imp.fit_transform(window_x.numpy())
        return torch.tensor(filled)
    elif missing_fill == "avg":
        column_means = torch.mean(window_x, dim=0)
        column_means = torch.nan_to_num(column_means, nan=0.0)
        nan_mask = torch.isnan(window_x)
        filled = torch.where(nan_mask, column_means, window_x)
        return filled
    elif missing_fill == "zero": 
        filled = torch.nan_to_num(window_x, nan=0.0)
        return filled
    

def train_naive(input, target, window_size, task, net, args):
    device = args.device
    result_record = []
    if args.model not in ("tree", "tabnet"):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        if task == "classification":
            criterion = nn.CrossEntropyLoss().to(device)
            target = torch.LongTensor(target)
        else:
            criterion = nn.MSELoss().to(device)
            target = target.float()
    constant_output = 0
    for ind in range(0,target.shape[0], window_size):
        window_x = input[ind:ind+window_size].float()
        window_y = target[ind:ind+window_size]
        if torch.isnan(window_x).any():
            window_x = fill_missing_value(window_x, args.missing_fill)
        if args.model == "armnet":
            id = torch.LongTensor(range(window_x.shape[1]))
        if ind > 0:
            try:
                if args.model == "armnet":
                    x_tmp = dict({})
                    x_tmp['value'] = window_x
                    x_tmp['id'] = id.repeat((window_x.shape[0],1))
                    result = compute_result(net, x_tmp, window_y, task, tree=(args.model in ("tree", "tabnet")))
                else:
                    result = compute_result(net, window_x, window_y, task, tree=(args.model in ("tree", "tabnet")))
                result_record.append(result)
            except:
                result = compute_result_constant(constant_output, window_x, window_y, task)
                logger.info("abnormal")
                result_record.append(result)
        length = window_y.shape[0]
        for epoch in range(args.epochs):
            if args.model in ("mlp", "armnet"):
                for batch_ind in range(0,length,args.batch_size):
                    x = window_x[batch_ind:batch_ind+args.batch_size]
                    y = window_y[batch_ind:batch_ind+args.batch_size]
                    if args.model == "armnet":
                        x_tmp = dict({})
                        x_tmp['value'] = x
                        x_tmp['id'] = id.repeat((x.shape[0],1))
                        x = x_tmp
                    optimizer.zero_grad()
                    out = net(x)
                    if task=="regression":
                        out = out.reshape(-1)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
            elif args.model == "tabnet":
                try:
                    window_x = window_x.numpy()
                    window_y = window_y.numpy()
                except:
                    pass
                if task == "regression":
                    window_y = window_y.reshape(-1,1)
                net.fit(window_x, window_y, batch_size=args.batch_size, virtual_batch_size=args.batch_size, max_epochs=args.epochs)
                break
            else:
                try:
                    net.fit(window_x, window_y)
                except:
                    constant_output = window_y[0]
                break # no epoch needed in tree model

    logger.info(result_record)
    mean_result = torch.mean(torch.Tensor(result_record))
    if task == "classification":
        logger.info("avg error: %f"%mean_result)
    else:
        logger.info("avg loss: %f"%mean_result)

def train_LwF(input, target, window_size, task, net, net_copy, args):
    device = args.device
    result_record = []
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    if task == "classification":
        criterion = nn.CrossEntropyLoss().to(device)
        target = torch.LongTensor(target)
    else:
        criterion = nn.MSELoss().to(device)
        target = target.float()

    T=2

    for ind in range(0,target.shape[0], window_size):
        window_x = input[ind:ind+window_size].float()
        window_y = target[ind:ind+window_size]
        if torch.isnan(window_x).any():
            window_x = fill_missing_value(window_x, args.missing_fill)
        if ind > 0:
            result = compute_result(net, window_x, window_y, task)
            result_record.append(result)
        length = window_y.shape[0]
        for epoch in range(args.epochs):
            for batch_ind in range(0,length,args.batch_size):
                x = window_x[batch_ind:batch_ind+args.batch_size]
                y = window_y[batch_ind:batch_ind+args.batch_size]
                optimizer.zero_grad()
                out = net(x)
                if task=="regression":
                    out = out.reshape(-1)
                loss = criterion(out, y)
                if ind>0:
                    if task == "classification":
                        # refer to https://github.com/MasLiang/Learning-without-Forgetting-using-Pytorch
                        soft_target = net_copy(x)
                        outputs_S = F.softmax(out/T,dim=1)
                        outputs_T = F.softmax(soft_target/T,dim=1)
                        loss2 = outputs_T.mul(-1*torch.log(outputs_S))
                        loss2 = loss2.sum(1)
                        loss2 = loss2.mean()
                        loss += args.reg*loss2
                    else:
                        soft_target = net_copy(x)
                        loss += args.reg*criterion(out, soft_target)

                loss.backward()
                optimizer.step()
        net_copy.load_state_dict(net.state_dict())

    logger.info(result_record)
    mean_result = torch.mean(torch.Tensor(result_record))
    if task == "classification":
        logger.info("avg error: %f"%mean_result)
    else:
        logger.info("avg loss: %f"%mean_result)

def train_EWC(input, target, window_size, task, net, args):
    device = args.device
    input = input.float()
    result_record = []
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    if task == "classification":
        criterion = nn.CrossEntropyLoss().to(device)
        target = torch.LongTensor(target)
    else:
        criterion = nn.MSELoss().to(device)
        target = target.float()

    for ind in range(0,target.shape[0], window_size):
        window_x = input[ind:ind+window_size]
        window_y = target[ind:ind+window_size]
        if torch.isnan(window_x).any():
            window_x = fill_missing_value(window_x, args.missing_fill)
            input[ind:ind+window_size] = window_x
        if ind > 0:
            result = compute_result(net, window_x, window_y, task)
            result_record.append(result)
            ids = torch.arange(ind-window_size,ind) #torch.randperm(ind)[:args.buffer]
            ewc = EWC(net, input, target, ids, task)
        length = window_y.shape[0]
        for epoch in range(args.epochs):
            for batch_ind in range(0,length,args.batch_size):
                x = window_x[batch_ind:batch_ind+args.batch_size]
                y = window_y[batch_ind:batch_ind+args.batch_size]
                optimizer.zero_grad()
                out = net(x)
                if task=="regression":
                    out = out.reshape(-1)
                loss = criterion(out, y)
                if ind>0:
                    loss += args.reg * ewc.penalty(net)
                loss.backward()
                optimizer.step()

    logger.info(result_record)
    mean_result = torch.mean(torch.Tensor(result_record))
    if task == "classification":
        logger.info("avg error: %f"%mean_result)
    else:
        logger.info("avg loss: %f"%mean_result)


def train_icarl(input, target, window_size, task, net, args):
    device = args.device
    result_record = []
    examples = [[] for i in range(args.output_dim)]
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    if task == "classification":
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
        target = target.float()
    net.to(device)

    buffer_class = int(args.buffer/args.output_dim)
    for ind in range(0,target.shape[0], window_size):
        window_x = input[ind:ind+window_size].float()
        window_y = target[ind:ind+window_size]
        if torch.isnan(window_x).any():
            window_x = fill_missing_value(window_x, args.missing_fill)
        if ind > 0:
            result = compute_result(net, window_x, window_y, task)
            result_record.append(result)
        length = window_y.shape[0]
        for epoch in range(args.epochs):
            for batch_ind in range(0,length,args.batch_size):
                x = window_x[batch_ind:batch_ind+args.batch_size].to(device)
                y = window_y[batch_ind:batch_ind+args.batch_size].to(device)
                optimizer.zero_grad()
                out = net(x)
                if task=="regression":
                    out = out.reshape(-1)
                loss = criterion(out, y)
                if ind>0:
                    out_e = net(x_example)
                    y_example = y_example.to(device)
                    loss += criterion(out_e, y_example)
                loss.backward()
                optimizer.step()
        if task == "regression":
            if ind == 0:
                features = net.feature_extractor(window_x).detach()
                avg = torch.mean(features, dim=0).unsqueeze(0)
                distance = torch.norm(features-avg,dim=1)
                _, indices = torch.topk(distance, k=args.buffer, largest=False)
                x_example = window_x[indices]
                y_example = window_y[indices]
        else:
            update = False
            for i in range(args.output_dim):
                if len(examples[i]) < buffer_class and i in window_y:
                    data_class = window_x[window_y==i]
                    features = net.feature_extractor(data_class).detach()
                    avg = torch.mean(features, dim=0).unsqueeze(0)
                    distance = torch.norm(features-avg,dim=1)
                    _, indices = torch.topk(distance, k=min(buffer_class-len(examples[i]),len(distance)), largest=False)
                    if len(examples[i])==0:
                        examples[i] = data_class[indices]
                    else:
                        examples[i] = torch.cat((examples[i],data_class[indices]),dim=0)
                    update = True
            if update:
                x_example = []
                for i in range(args.output_dim):
                    if len(examples[i])==0:
                        continue
                    if len(x_example)==0:
                        x_example = examples[i]
                        y_example = torch.zeros(examples[i].shape[0])
                    else:
                        x_example = torch.cat((x_example, examples[i]),dim=0)
                        y_example = torch.cat((y_example,torch.ones(examples[i].shape[0])*i),dim=0)
                if task == "classification":
                    y_example = y_example.long()

    logger.info(result_record)
    mean_result = torch.mean(torch.Tensor(result_record))
    if task == "classification":
        logger.info("avg error: %f"%mean_result)
    else:
        logger.info("avg loss: %f"%mean_result)

def train_arf(input, target, window_size, task, net, args):
    result_record = []
    target = torch.LongTensor(target)

    for ind in range(0,target.shape[0], window_size):
        window_x = input[ind:ind+window_size].float()
        window_y = target[ind:ind+window_size]
        if torch.isnan(window_x).any():
            window_x = fill_missing_value(window_x, args.missing_fill)
        if ind > 0:
            result = compute_result(net, window_x, window_y, task, tree=True)
            result_record.append(result)
        length = window_y.shape[0]
        net.partial_fit(window_x, window_y) # no epoch needed for decision tree
                           

    logger.info(result_record)
    mean_result = torch.mean(torch.Tensor(result_record))
    logger.info("avg error: %f"%mean_result)

def copy_model(net_target, net, tree=False):
    if tree:
        net_target.__dict__.update(net.__dict__)
    else:
        net_target.load_state_dict(net.state_dict())

def train_sea(input, target, window_size, task, net, net_ensemble, args):
    device = args.device
    result_record = []
    if args.model != "tree":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        if task == "classification":
            criterion = nn.CrossEntropyLoss().to(device)
            target = torch.LongTensor(target)
        else:
            criterion = nn.MSELoss().to(device)
            target = target.float()

    cnt = 0
    for ind in range(0,target.shape[0], window_size):
        window_x = input[ind:ind+window_size].float()
        window_y = target[ind:ind+window_size]
        if torch.isnan(window_x).any():
            window_x = fill_missing_value(window_x, args.missing_fill)
        if ind > 0:
            result = compute_result(net, window_x, window_y, task, tree=(args.model=="tree"))
            #result_record.append(result)
            if cnt<=args.ensemble:
                copy_model(net_ensemble[cnt-1], net, tree=(args.model=="tree"))
            else:
                for i in range(args.ensemble):
                    if result < compute_result(net_ensemble[i], window_x, window_y, task, tree=(args.model=="tree")):
                        copy_model(net_ensemble[i], net, tree=(args.model=="tree"))
                        break

            if cnt == 1:
                result_record.append(result)
            else:
                result_ensemble = compute_result_ensemble(net_ensemble, min(cnt,args.ensemble), window_x, window_y, task, tree=(args.model=="tree"))
                result_record.append(result_ensemble)

        length = window_y.shape[0]
        for epoch in range(args.epochs):
            if args.model == "mlp":
                for batch_ind in range(0,length,args.batch_size):
                    x = window_x[batch_ind:batch_ind+args.batch_size]
                    y = window_y[batch_ind:batch_ind+args.batch_size]
                    optimizer.zero_grad()
                    out = net(x)
                    if task=="regression":
                        out = out.reshape(-1)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
            else:
                net.fit(window_x, window_y)
                break # no epoch needed in tree model
        cnt+=1

    logger.info(result_record)
    mean_result = torch.mean(torch.Tensor(result_record))
    if task == "classification":
        logger.info("avg error: %f"%mean_result)
    else:
        logger.info("avg loss: %f"%mean_result)


if __name__ == "__main__":
    args = get_args()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    mkdirs(args.log_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file_name = '%sexperiment_log-%s.log' % (args.log_dir,datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))

    logging.basicConfig(
        filename=log_file_name,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.dataset == "selected":
        selected_dataset = ['dataset_experiment_info/room_occupancy', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/tetouan']
    elif args.dataset == "generated":
        selected_dataset = ["SEAGenerator", "HyperplaneGenerator", "STAGGERGenerator", "RandomRBFGeneratorDrift", "LEDGeneratorDrift", "WaveformGenerator"]
        generated_dataset = dict({})
        for i in range(4):
            stream = SEAGenerator(classification_function = i, balance_classes = False)
            if i==0:
                x_total, y_total = stream.next_sample(12500)
            else:
                x, y = stream.next_sample(12500)
                x_total = np.concatenate((x_total,x),axis=0)
                y_total = np.concatenate((y_total,y),axis=0)
        generated_dataset["SEAGenerator"] = dict({})
        generated_dataset["SEAGenerator"]["input"] = x_total
        generated_dataset["SEAGenerator"]["target"] = y_total

        stream = HyperplaneGenerator(mag_change=0.1)
        x, y = stream.next_sample(50000)
        generated_dataset["HyperplaneGenerator"] = dict({})
        generated_dataset["HyperplaneGenerator"]["input"] = x
        generated_dataset["HyperplaneGenerator"]["target"] = y

        for i in range(3):
            stream = STAGGERGenerator(classification_function = i, balance_classes = False)
            if i==0:
                x_total, y_total = stream.next_sample(16700)
            else:
                x, y = stream.next_sample(16700)
                x_total = np.concatenate((x_total,x),axis=0)
                y_total = np.concatenate((y_total,y),axis=0)
        generated_dataset["STAGGERGenerator"] = dict({})
        generated_dataset["STAGGERGenerator"]["input"] = x_total
        generated_dataset["STAGGERGenerator"]["target"] = y_total
        
        stream = RandomRBFGeneratorDrift(n_classes=4, change_speed=0.87)
        x, y = stream.next_sample(50000)
        generated_dataset["RandomRBFGeneratorDrift"] = dict({})
        generated_dataset["RandomRBFGeneratorDrift"]["input"] = x
        generated_dataset["RandomRBFGeneratorDrift"]["target"] = y

        stream = LEDGeneratorDrift(noise_percentage = 0.28,has_noise= True, n_drift_features=4)
        x, y = stream.next_sample(50000)
        generated_dataset["LEDGeneratorDrift"] = dict({})
        generated_dataset["LEDGeneratorDrift"]["input"] = x
        generated_dataset["LEDGeneratorDrift"]["target"] = y

        stream = WaveformGenerator(has_noise= True)
        x, y = stream.next_sample(50000)
        generated_dataset["WaveformGenerator"] = dict({})
        generated_dataset["WaveformGenerator"]["input"] = x
        generated_dataset["WaveformGenerator"]["target"] = y
        
    else:
        selected_dataset = [args.dataset]

    device = args.device

    for dataset_path_prefix in selected_dataset:
        logger.info(dataset_path_prefix)
        if args.dataset == "generated":
            window_size = 500
            input = generated_dataset[dataset_path_prefix]["input"]
            target = generated_dataset[dataset_path_prefix]["target"]
            task = "classification"
            column_count = input.shape[1]
            output_dim = np.max(target) + 1
            input = torch.tensor(input).to(device)
            target = torch.tensor(target).to(device)
        else:
            data_path, schema_path, task = schema_parser(dataset_path_prefix)

            input, target, window_size, task, column_count, output_dim = data_preprocessing(dataset_path_prefix, data_path, schema_path, task, logger, delete_null_target=True)
            
            input = input.astype(float)
            target = target.astype(float)

            input = torch.tensor(input.values).to(device)
            target = torch.tensor(target.values).to(device)

        if task == "classification":
            target = target.long()
        input_avg = torch.nanmean(input[:window_size],dim=0).unsqueeze(0).to(device)
        input_std = torch.tensor(np.nanstd(input[:window_size].cpu().numpy(),axis=0)).unsqueeze(0).to(device) + 0.1
        input = (input-input_avg)/input_std
        target = target.reshape(-1)
        if task == "regression":
            target_avg = torch.mean(target)
            target_std = torch.std(target) + 0.1
            target = (target-target_avg)/target_std

        logger.info(window_size)
        logger.info(column_count)
        logger.info(output_dim)
        args.output_dim = output_dim
        window_size = int(window_size*args.window_factor)

        if args.model == "mlp":
            if args.layers == 3:
                hidden_layers = [32, 16, 8]
            elif args.layers == 5:
                hidden_layers = [32, 32, 16, 16, 8]
            elif args.layers == 7:
                hidden_layers = [32, 32, 32, 16, 16, 16, 8]
            net = FcNet(column_count, hidden_layers, output_dim)
            net_copy = FcNet(column_count, hidden_layers, output_dim)
            net_ensemble = [FcNet(column_count, hidden_layers, output_dim) for i in range(args.ensemble)]
        elif args.alg == "arf" and args.model == "tree":
            net = AdaptiveRandomForest(nb_features=output_dim, nb_trees=args.ensemble, pretrain_size=window_size)
        elif args.model == "tree":
            if task == "classification":
                if args.gbdt:
                    net = GradientBoostingClassifier()
                    net_ensemble = [GradientBoostingClassifier() for i in range(args.ensemble)]
                else:
                    net = DecisionTreeClassifier()
                    net_ensemble = [DecisionTreeClassifier() for i in range(args.ensemble)]
            else:
                if args.gbdt:
                    net = GradientBoostingRegressor()
                    net_ensemble = [GradientBoostingRegressor() for i in range(args.ensemble)]
                else:
                    net = DecisionTreeRegressor()
                    net_ensemble = [DecisionTreeRegressor() for i in range(args.ensemble)]
        elif args.model == "tabnet":
            if task == "classification":
                net = TabNetClassifier(seed=args.init_seed)
            else:
                net = TabNetRegressor(seed=args.init_seed)
        elif args.model == "armnet":
            net = ARMNetModel(column_count, column_count, column_count, 2, 1.7, 32,
                        args.layers, 16, 0, False, args.layers, 16, noutput=output_dim)
        start_time = time.time()

        if args.alg == "naive":
            train_naive(input, target, window_size, task, net, args)
        elif args.alg == "ewc":
            train_EWC(input, target, window_size, task, net, args)
        elif args.alg == "lwf":
            train_LwF(input, target, window_size, task, net, net_copy, args)
        elif args.alg == "icarl":
            train_icarl(input, target, window_size, task, net, args)
        elif args.alg == "arf" and task == "classification": # Adaptive Random Forest can only be trained on classification tasks
            train_arf(input, target, window_size, task, net, args)
        elif args.alg == "sea":
            train_sea(input, target, window_size, task, net, net_ensemble, args)


        time_elapse = time.time() - start_time
        logger.info("time: %f"%time_elapse)
            

