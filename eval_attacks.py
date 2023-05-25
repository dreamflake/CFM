import torch
from PIL import Image
import csv
import numpy as np
import os
import traceback
import random
from attacks import *
from utils import load_model, WrapperModel, load_model_cifar10
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from PIL import Image
import os
import easypyxl
from datetime import datetime
import time
from config import *
import neptune.new as neptune



now = datetime.now()
today_string = now.strftime("%m-%d|%H-%M")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) - 1 )
            label_tar_list.append( int(row['TargetClass']) - 1 )


    return image_id_list,label_ori_list,label_tar_list

def load_ground_truth_cifar10(img_dir):
    filenames = os.listdir(img_dir)
    image_id_list = [os.path.splitext(x)[0] for x in filenames]
    label_ori_list = [int(x.split('_')[1]) for x in filenames]
    label_tar_list = [int(os.path.splitext(x.split('_')[2])[0]) for x in filenames]
    return image_id_list, label_ori_list, label_tar_list


def plot_img(img_tensor, file_name):
    img = np.array(img_tensor.cpu().detach().numpy()).transpose(1, 2, 0) * 255.
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save("imgs/" + file_name + ".png")
    
def main(args):


    if args.neptune:
        run = neptune.init(
        project="", 
        api_token="", # your credentials
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load experiment configuration
    exp_settings=exp_configuration[args.config_idx]


    print(args)
    print(exp_settings,flush=True)


    target_model_names=exp_settings['target_model_names']
    source_model_names=exp_settings['source_model_names']
    dset=exp_settings['dataset']
    
    if 'seed' not in exp_settings:
        exp_settings['seed']=42
    seed=exp_settings['seed']
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    small_sized_models=['vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit']
    # pre-process input image
    if dset=='ImageNet':
        mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        trn = transforms.Compose([transforms.ToTensor(),])
        image_id_list,label_ori_list,label_tar_list = load_ground_truth('./dataset/images.csv')
        if seed!=42:
            random.seed(seed)
            c = list(zip(image_id_list, label_ori_list,label_tar_list))
            random.shuffle(c)
            image_id_list, label_ori_list,label_tar_list= zip(*c)

        img_size = 299
        transfer_models = [WrapperModel(load_model(x), mean, stddev, True if x in small_sized_models else False ).to(device) for x in target_model_names] #,resize=False if x in 'inception_v3' else True



    elif dset=='Cifar10':


        # CIFAR-10
        mean = [0.4914, 0.4822, 0.4465]
        stddev = [0.2471, 0.2435, 0.2616]
        mean_dv = [0.4914, 0.4822, 0.4465]
        stddev_dv = [0.2023, 0.1994, 0.2010]
        trn = transforms.Compose([transforms.ToTensor(),])
        image_id_list,label_ori_list,label_tar_list = load_ground_truth_cifar10('./dataset/CIFAR10_targeted')
        img_size = 32
        transfer_models = [WrapperModel(load_model_cifar10(x), mean, stddev).to(device)
                    if 'ens3' not in x
                    else WrapperModel(load_model_cifar10(x), mean_dv, stddev_dv).to(device)
                    for x in target_model_names]

    print('Models are loaded',flush=True)


    total_img_num=exp_settings['num_images']

    ###################### IMPORTANT ####################
    # Comment the below line (-> # total_img_num=100) when you use full test set (1000 images). 
    total_img_num=100
    ###################### IMPORTANT ####################


    image_id_list=image_id_list[:total_img_num]
    label_ori_list=label_ori_list[:total_img_num]
    label_tar_list=label_tar_list[:total_img_num]


    # easypyxl settings
    excel_path='./results/NEW_EXP_'+str(args.config_idx)+'.xlsx'
    wb = easypyxl.Workbook(excel_path)
    exp_info_cursor = wb.new_cursor("Experiment Info", "A2", 2, overwrite=True)
    exp_info_cursor.write_cell(['Date',today_string])
    exp_info_cursor.write_cell(['Args',str(args)])
    exp_info_cursor.write_cell(['exp_settings',str(exp_settings)])
    
    if args.neptune:
        run["arguments"] = args
        run["exp_settings"] = exp_settings
        run["comment"] = exp_settings['comment']
        run["dataset"] = exp_settings['dataset']
        run["targeted"] = exp_settings['targeted']
        run["epsilon"] = exp_settings['epsilon']
        
        run["max_iterations"] = exp_settings['max_iterations']    
        run["date"] = today_string
        run["config_idx"] = args.config_idx



    
    succs_cursors=[wb.new_cursor('Succ_'+str((n+1)*20), "A2", 2+len(target_model_names), overwrite=True) for n in range(exp_settings['max_iterations']//20)]
    accs_cursors=[wb.new_cursor('Accs_'+str((n+1)*20), "A2", 2+len(target_model_names), overwrite=True) for n in range(exp_settings['max_iterations']//20)]

    for c in succs_cursors:
        c.write_cell(["Source", "Attack"])
        c.write_cell(target_model_names)
    for c in accs_cursors:
        c.write_cell(["Source", "Attack"])
        c.write_cell(target_model_names)
    
    attack_methods=exp_settings['attack_methods']


    for model_i, source_model_name in enumerate(source_model_names):
        print(source_model_name)
        
        if args.neptune:
            run["current_source"] = source_model_name
        torch.cuda.empty_cache()
        batch_size=args.batch_size
        # load models
        if dset=='ImageNet':
            source_model = WrapperModel(load_model(source_model_name), mean, stddev).to(device)
        elif dset=='Cifar10':
            source_model = WrapperModel(load_model_cifar10(source_model_name), mean, stddev).to(device)

        source_model = source_model.eval()


        def iter_source():
            num_images = 0
            target_accs = {m: {k: ([0.] *(exp_settings['max_iterations']//20)) for k in attack_methods.keys()} for m in target_model_names}
            target_succs = {m: {k: ([0.] * (exp_settings['max_iterations']//20)) for k in attack_methods.keys()} for m in target_model_names}
            
            num_batches = np.int(np.ceil(len(image_id_list) / batch_size))
            total_time=0.
            for k in range(0,num_batches):
                
                batch_size_cur = min(batch_size,len(image_id_list) - k * batch_size)        
                img = torch.zeros(batch_size_cur,3,img_size,img_size).to(device)
                for i in range(batch_size_cur):
                    if dset=='ImageNet':
                        img[i] = trn(Image.open("./dataset/images/" + image_id_list[k * batch_size + i] + '.png'))
                    elif dset=="Cifar10":
                        img[i] = trn(Image.open("./dataset/CIFAR10_targeted/" + image_id_list[k * batch_size + i] + '.png'))

                labels = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
                target_labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
                num_images += batch_size_cur

                source_model.eval()
                start=time.time()
                # Generate adversarial examples
                output_dict = {key: advanced_fgsm(atk,source_model, img, labels, target_labels,num_iter=exp_settings['max_iterations'],max_epsilon=exp_settings['epsilon'],count=k,config_idx=args.config_idx) for key, atk in
                            attack_methods.items()}
                end=time.time()
                total_time+=end-start


                # Save Image
                createFolder('./imgs/'+ str(args.config_idx))
                createFolder('./imgs/'+ str(args.config_idx)+'/'+source_model_name)
                n= exp_settings['max_iterations']//20-1
                for key, value in output_dict.items():
                    imgs=value[n]
                    for nn in range(imgs.size()[0]):
                        plot_img(imgs[nn],str(args.config_idx)+'/'+source_model_name+'/'+str(k*batch_size+nn))
                
                if args.neptune:
                    run['Iteration'].log(k)
                    run['Processed_imgs'].log(k * batch_size+batch_size_cur)
                
                avg_succ=0
                avg_acc=0
                for j, mod in enumerate(transfer_models):
                    mod.eval()
                    for n in range(exp_settings['max_iterations']//20):
                        with torch.no_grad():
                            transfer_results_dict = {key: F.softmax(mod(value[n]), dim=1).max(dim=1) for key, value in
                                                    output_dict.items()}
                        for a in attack_methods.keys():
                            target_succs[target_model_names[j]][a][n] += (
                                torch.sum((transfer_results_dict[a][1] == target_labels).float())).item()
                            target_accs[target_model_names[j]][a][n] += (
                                torch.sum((transfer_results_dict[a][1] == labels).float())).item()
                            if n == exp_settings['max_iterations']//20-1:
                                succ = (target_succs[target_model_names[j]][a][exp_settings['max_iterations']//20-1]) / num_images
                                acc = (target_accs[target_model_names[j]][a][exp_settings['max_iterations']//20-1]) / num_images
                                
                                if args.neptune:
                                    run['Succ/'+source_model_name+"/"+target_model_names[j]].log(succ*100)
                                    run['Acc/'+source_model_name+"/"+target_model_names[j]].log(acc*100)
                                avg_succ+=succ*100
                                avg_acc+=acc*100
                                print(f'[{k * batch_size+batch_size_cur}/{len(image_id_list) }]Success Rate (%) on {target_model_names[j]} with {a} : {succ*100:.2f} | Acc (%) : {acc*100:.2f}',flush=True)
                
                if args.neptune:
                    run['Succ/'+source_model_name+'/Avg'].log(avg_succ/len(transfer_models))
                    run['Acc/'+source_model_name+'/Avg'].log(avg_acc/len(transfer_models))
                print('AVG_Succ:',avg_succ/len(transfer_models))
            return target_accs, target_succs,total_time

        tot_time=0.
        while True:
            try:
                print(f"batch={batch_size}",flush=True)
                target_accs,target_succs,tot_time = iter_source()
            except Exception: 
                print("Error",flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
                time.sleep(5)
                batch_size = int(batch_size / 1.1) # Auto adjust the batch size within the GPU memory
                if batch_size<1:
                    break
                continue
            print(datetime.now().strftime("%m-%d|%H-%M"),flush=True)
            break

        for a in attack_methods.keys(): # Export experimental results
            for n in range(exp_settings['max_iterations']//20):
                succs_cursors[n].write_cell([source_model_name, a])
                accs_cursors[n].write_cell([source_model_name, a])

                for j, mod in enumerate(transfer_models):
                    final_succ = (target_succs[target_model_names[j]][a][n]) / total_img_num
                    final_acc = (target_accs[target_model_names[j]][a][n]) / total_img_num
                    succs_cursors[n].write_cell(final_succ*100)
                    accs_cursors[n].write_cell(final_acc*100)
                    
                    if args.neptune:
                        run['Final_Succ_'+str((n+1)*20)+'/'+source_model_name+"/"+target_model_names[j]]=final_succ*100
                        run['Final_Acc_'+str((n+1)*20)+'/'+source_model_name+"/"+target_model_names[j]]=final_acc*100
        
        if args.neptune:
            run['avg_time']=tot_time/total_img_num
            run['end_time']=datetime.now().strftime("%m-%d|%H-%M")
        exp_info_cursor.write_cell([source_model_name,str(tot_time/total_img_num)])
        print('AVG TIME: ',tot_time/total_img_num)
        print(datetime.now().strftime("%m-%d|%H-%M"),flush=True)

    if args.neptune:
        run.stop()

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="batch_size as an integer")
    parser.add_argument("--config_idx", default=923, type=int, help="experiment config index")
    parser.add_argument("--neptune", default=False, type=bool, help="experiment config index")
    return parser

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)

