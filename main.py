import torch
import torch.nn as nn
import argparse
import logging
import pickle
import time
import os
import random
from TransformerLSR import TransformerLSR
from functions import get_tensors
from loss import (long_loss_LSR, surv_loss,inten_loss)


# Other Python libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None




def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)              
    parser.add_argument("--epoch", default=20, type=int)          
    parser.add_argument("--batch_size", default=16, type=int)      # Mini batch size for networks
    parser.add_argument("--num_enc_layer", default=4, type=int)  
    parser.add_argument("--num_dec_layer", default=4, type=int)  
    parser.add_argument("--d_long", default=3, type=int)    
    parser.add_argument("--num_head", default=4, type=int)      
    parser.add_argument("--model_size", default=32, type=int)     
    parser.add_argument('--suffix', type=str, default='train')
    parser.add_argument('--model', type=str, default='LSR')
    parser.add_argument('--data', type=str, default='DIVAT_sim_1000_visit_1000_long_3')
    parser.add_argument("--local", action="store_true")   # local test mode
    parser.add_argument("--Y1_missing", default=0, type=float)
    parser.add_argument("--Y2_missing", default=0, type=float)
    parser.add_argument("--Y3_missing", default=0, type=float)
    parser.add_argument("--inten_weight", default=0.01, type=float)
    parser.add_argument("--surv_weight", default=0.1, type=float)
    parser.add_argument("--lr", default=0.0003, type=float) # learning rate
    args = parser.parse_args()


    # make logger here
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    work_dir = os.path.join('./work_dir',
                                time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    time_prefix = time.strftime("%H:%M:%S", time.localtime())
    full_path = work_dir + '/'+time_prefix +'_'+args.data+'_'+args.model+'_'+"head_"+str(args.num_head)+'_'+ \
                                        "enc_layer_"+str(args.num_enc_layer)+'_'+"dec_layer_"+str(args.num_dec_layer)+'_'+"size_"+str(args.model_size)+'_'+ \
                                        '_'+"visit_weight_"+str(args.inten_weight)+'_'+"surv_weight_"+str(args.surv_weight)+ \
                                        '_'+"lr_"+str(args.lr)+"Y1miss_"+str(args.Y1_missing)+"Y2miss_"+str(args.Y2_missing)+"Y3miss_"+str(args.Y3_missing)+args.suffix +'-log.txt'

    if not args.local:
        fHandler = logging.FileHandler(full_path, mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)


    # log meta-data
    logger.info(args)
    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./models/info"):
        os.makedirs("./models/info")
    
    
    
    Y_str_list = []
    for i in range(args.d_long):
        Y_str = "Y"+str(i+1)
        Y_str_list.append(Y_str)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load dag
    dag_info_path = f'data/{args.data}_info.pkl' 
    with open(dag_info_path, 'rb') as f:
        dag_info = pickle.load(f)
    n_epoch = args.epoch
    train_values = np.zeros([args.seed,n_epoch])
    vali_values = np.zeros([args.seed,n_epoch])

    seed = args.seed
    seednum = seed
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)

    # load dataset
    dataset_path = f'data/{args.data}_seed_{seed}.pkl' 
    data_all = pd.read_pickle(dataset_path) 
    I = data_all["id"].values[-1]+1

    logger.info('=' * 50)
    logger.info(f'Starting training for dataset: {args.data}')
    logger.info(f'{args.num_head} heads, {args.num_enc_layer} enc layers,{args.num_dec_layer} dec layers, {args.model_size} model dimension')
    logger.info(f'Data containing total {I} trajectories' )
    logger.info('=' * 50)

    data = data_all[data_all.obstime <= data_all.time]

    random_id = range(I) #np.random.permutation(range(I))

    train_id = random_id[0:int(0.6*I)]
    
    ## Scale data using Min-Max Scaler
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))

    vali_id = random_id[int(0.6*I):int(0.8*I)]
    test_id = random_id[int(0.8*I):I]

    train_data = data[data["id"].isin(train_id)]
    
    vali_data = data[data["id"].isin(vali_id)]
    test_data = data[data["id"].isin(test_id)]

    train_data.loc[:,Y_str_list] = minmax_scaler.fit_transform(train_data.loc[:,Y_str_list])
    vali_data.loc[:,Y_str_list] = minmax_scaler.transform(vali_data.loc[:,Y_str_list])
    test_data.loc[:,Y_str_list] = minmax_scaler.transform(test_data.loc[:,Y_str_list])


    model_save_path ='./models/'+args.data+'_'+'seed'+str(seednum)+'_'+args.model+'_'+\
                    "head_"+str(args.num_head)+'_'+"enc_layer_"+str(args.num_enc_layer)+'_'+"dec_layer_"+str(args.num_dec_layer)+'_'+"size_"+str(args.model_size)+\
                        '_'+"visit_weight_"+str(args.inten_weight)+'_'+"surv_weight_"+str(args.surv_weight)+\
                            '_'+"lr_"+str(args.lr)+"Y1miss_"+str(args.Y1_missing)+"Y2miss_"+str(args.Y2_missing)+"Y3miss_"+str(args.Y3_missing)+'.pt'

    logger.info('=' * 50)
    logger.info(f'Training for seed {seednum}')
    

    model = TransformerLSR(d_long=args.d_long,d_base=3,dag_info=dag_info, d_model=args.model_size, nhead=args.num_head,
                num_encoder_layers=args.num_enc_layer,num_decoder_layers=args.num_dec_layer,device=device)
    long_loss = long_loss_LSR
        
    
    
    # modify training and vali data for missingness
    if args.model == "LSR_missing":
        Y1_nan_inds = np.random.choice(train_data.index,size = int(args.Y1_missing*len(train_data)),replace=False)
        train_data["Y1"][Y1_nan_inds] = float('nan') 
        Y2_nan_inds = np.random.choice(train_data.index,size = int(args.Y2_missing*len(train_data)),replace=False)
        train_data["Y2"][Y2_nan_inds] = float('nan') 
        Y3_nan_inds = np.random.choice(train_data.index,size = int(args.Y3_missing*len(train_data)),replace=False)
        train_data["Y3"][Y3_nan_inds] = float('nan') 

        Y1_nan_inds_vali = np.random.choice(vali_data.index,size = int(args.Y1_missing*len(vali_data)),replace=False)
        vali_data["Y1"][Y1_nan_inds_vali] = float('nan') 
        Y2_nan_inds_vali = np.random.choice(vali_data.index,size = int(args.Y2_missing*len(vali_data)),replace=False)
        vali_data["Y2"][Y2_nan_inds_vali] = float('nan') 
        Y3_nan_inds_vali = np.random.choice(vali_data.index,size = int(args.Y3_missing*len(vali_data)),replace=False)
        vali_data["Y3"][Y3_nan_inds_vali] = float('nan') 
    
    
    
    
    
    
    
    
    model.to(device=device)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    
    batch_size = args.batch_size
    
    curr_best = 100000000

    for epoch in range(n_epoch):
        logger.info(f'epoch: {epoch}')
        #train loop
        model = model.train()
        running_loss = 0
        tokens = 0
        train_id = np.random.permutation(train_id)
        vali_id = np.random.permutation(vali_id)

        for batch in range(0, len(train_id), batch_size):
            optimizer.zero_grad()
            indices = train_id[batch:batch+batch_size]
            batch_data = train_data[train_data["id"].isin(indices)]

            batch  = get_tensors(batch_data.copy(),long=Y_str_list,device=device)

            long_preds,visit_inten,surv_inten,Lambda,Zeta = model(batch)
            
            
            loss1,full_loss1,num_tokens = long_loss(long_preds,batch)
            loss2,full_loss2 = surv_loss(surv_inten,Zeta,batch)
            loss3,full_loss3 = inten_loss(visit_inten,Lambda,batch)

            loss = loss1 + args.surv_weight*loss2 + args.inten_weight*loss3
            loss.backward()
            optimizer.step()
            running_loss += (full_loss1+args.surv_weight*full_loss2+args.inten_weight*full_loss3).item()
            tokens += num_tokens



        train_show = (running_loss/tokens)
        logger.info(f"longloss:{loss1.item():.2f}")
        logger.info(f"survloss:{loss2.item():.2f}")
        logger.info(f"intenloss:{loss3.item():.2f}")

        #train_values[seed,epoch] = train_show


        





        # validation loop
        model = model.eval()
        vali_loss = 0
        tokens = 0
        for batch in range(0, len(vali_id), batch_size):
            indices = vali_id[batch:batch+batch_size]
            batch_data = vali_data[vali_data["id"].isin(indices)]
            batch  = get_tensors(batch_data.copy(),long=Y_str_list,device=device)
            with torch.no_grad():
                long_preds,visit_inten,surv_inten,Lambda,Zeta = model(batch)
                loss1,full_loss1,num_tokens = long_loss(long_preds,batch)
                loss2,full_loss2 = surv_loss(surv_inten,Zeta,batch)
                loss3,full_loss3 = inten_loss(visit_inten,Lambda,batch)

            vali_loss += (full_loss1+args.surv_weight*full_loss2+args.inten_weight*full_loss3).item()
            tokens += num_tokens

            



        event_ll = (torch.log(visit_inten[0])*batch["longmask"][0,1:]).sum()
        non_event_ll = (Lambda[0]*batch["mask"][0]).sum()
        visit_ll = event_ll - non_event_ll
        #logger.info(f"sample trajectory visit log likelihood:{visit_ll.item():.2f}")
        logger.info(f"sample trajectory visit event intensity:{event_ll.item():.2f}")
        logger.info(f"sample trajectory visit non event intensity:{non_event_ll.item():.2f}")
        ground_truth_ll = batch_data["event_ll"].to_numpy()[0]
        logger.info(f"ground truth visit event intensity:{ground_truth_ll:.2f}")
        ground_truth_non_ll = batch_data["event_non_ll"].to_numpy()[0]
        logger.info(f"ground truth visit NON-event intensity:{ground_truth_non_ll:.2f}")

        # more detailed comparison
        first_traj_len = torch.sum(batch["mask"][0],dim=-1).cpu().numpy()
        logger.info(f"sample trajectory visit intensities:{np.log(visit_inten[0,:first_traj_len-1].detach().cpu().numpy())}")
        ground_intensities = batch_data["true_inten"].to_numpy()[:first_traj_len-1]
        logger.info(f"ground truth visit event intensities:{np.log(ground_intensities)}")
        total_time = batch["obstime"][0][1:first_traj_len]
        logger.info(f"times:{total_time}")

        #sample survival examination for last batch first trajectory
        surv_event_ll = (torch.log(surv_inten[0])*batch["intenmask"][0,1:]).sum(dim=-1)
        non_surv_event_ll = (Zeta[0]*batch["mask"][0]).sum(dim=-1)
        surv_ll = surv_event_ll - non_surv_event_ll
        logger.info(f"sample trajectory survival event intensity:{surv_event_ll.item():.2f}")
        logger.info(f"sample trajectory survival non event intensity:{non_surv_event_ll.item():.2f}")
        ground_truth_surv_ll = batch_data["surv_ll"].to_numpy()[0]
        logger.info(f"ground truth survival event intensity:{ground_truth_surv_ll:.2f}")
        ground_truth_surv_non_ll = batch_data["surv_non_ll"].to_numpy()[0]
        logger.info(f"ground truth survival NON-event intensity:{ground_truth_surv_non_ll:.2f}")

        # more detailed comparison
        first_traj_len = torch.sum(batch["mask"][0],dim=-1).cpu().numpy()
        logger.info(f"sample trajectory surv intensities:{np.log(surv_inten[0,:first_traj_len].detach().cpu().numpy())}")
        ground_surv_intensities = batch_data["true_surv"].to_numpy()[:first_traj_len]
        logger.info(f"ground truth surv intensities:{np.log(ground_surv_intensities)}")
        total_time = batch["totaltime"][0][1:first_traj_len+1]
        logger.info(f"times:{total_time}")




        vali_show = (vali_loss/tokens)
        #vali_values[seed,epoch] = vali_show
        if vali_show < curr_best:
            curr_best = vali_show
            logger.info(f"updated at epoch: {epoch}")
            logger.info(f"current best validation loss: {curr_best:.2f}")
            torch.save(model.state_dict(), model_save_path)
                
    # info_path = './models/info/'+args.data+'_'+'seed'+str(args.seed)+'_'+args.model+'_'+\
    #                     "head_"+str(args.num_head)+'_'+ "enc_layer_"+str(args.num_enc_layer)+'_'+"dec_layer_"+str(args.num_dec_layer)+'_'+"size_"+str(args.model_size)+'_train_info.pkl'
    # train_info={}
    # train_info["train"] = train_values
    # train_info["vali"] = vali_values

    # with open(info_path, 'wb') as f:
    #     pickle.dump(train_info,f)

    logger.info("training info saved, training completed")

if __name__ == '__main__':
    main()