import easypyxl
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse


def argument_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fig_name", default="result.png")
    parser.add_argument("--common_legend", default="True")
    parser.add_argument("--fig_size_x", default="4")


    return parser



def main(args):



    cfg_idx=[401, 402,605,604,403,579,578]# 

    methods=['DI','RDI','SI-RDI','VT-RDI', 'Admix-RDI','ODI','CFM-RDI']
    result_dir="results/"
    target_models=['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit']
    source_model_names=['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3']


    target_source_idx=[1,3 ]
    display_idx_set=[['mobilenet_v2'], ['xception']]
    src_name=['RN-50','Inc-v3']
    tgt_name=['MB-v2','Xcep'] 
    num_iterations=300
    values=np.zeros((len(cfg_idx),len(source_model_names),len(target_models),num_iterations//20+1), dtype=np.float32)





    workbook=[]
    cursor=[[] for i in range(num_iterations//20)]

    for i in range(len(cfg_idx)):
        workbook.append(easypyxl.Workbook(result_dir+"NEW_EXP_"+str(cfg_idx[i])+".xlsx", backup=False))
        for j in range(num_iterations//20):
            cursor[i].append(workbook[i].new_cursor("Succ_"+str((j+1)*20), "C2", len(target_models), reader=True))
            title=cursor[i][j].read_line()


    for i in range(len(source_model_names)):
        for c in range(len(cfg_idx)):
            for j in range(num_iterations//20):
                for k in range(len(target_models)):
                    item=cursor[c][j].read_cell()
                    values[c][i][k][j+1]=item
                    if item is None:
                        print(c)


    fig_size_x = 3.5
    fig_size_y = fig_size_x * len(display_idx_set) * 1.5
    fig, axes = plt.subplots(1,len(display_idx_set),figsize=(fig_size_y,fig_size_x), dpi=300)

    plt.subplots_adjust(bottom = 0.3, wspace = 0.25)
    for idx, ax in enumerate(fig.axes):
        for s in range(len(source_model_names)):
            source_idx=s
            if source_idx==target_source_idx[idx]:
                cur_idx_set=display_idx_set[idx]
                colors = ['tab:purple',
                    'tab:blue',
                    'tab:green',
                    'tab:orange',
                    'tab:pink',
                    'tab:brown',
                    'tab:red',]
                markers = ['o',
                    'o',
                            'o',
                            'o',
                            'o',
                            'o',
                            'o',]
                x=np.arange(0,num_iterations+20,20)
                avg_value=np.zeros((len(cfg_idx),num_iterations//20+1))
                max=0
                for i in range(len(cfg_idx)):
                    ct=0
                    for t in cur_idx_set:
                        ct+=1
                        target_idx=target_models.index(t)
                        if ct==1:
                            avg_value[i]=values[i,source_idx,target_idx]
                        else:
                            avg_value[i]+=values[i,source_idx,target_idx]
                    avg_value[i]/=len(cur_idx_set)
                    if max<avg_value[i].max():
                        max=avg_value[i].max()




                x=np.arange(0,num_iterations+20,20)
                print(max)
                ax.set_xlim(0, 300)
                ax.set_ylim(0, (max//10+1)*10)
                ax.grid(color='gainsboro', linestyle='-', linewidth=1)

                ax.set_xlabel('Iteration', fontsize = 13)
                ax.set_ylabel('Average attack success rate (%)', fontsize = 13) 
                ax.title.set_fontsize(14)
                ax.title.set_text(src_name[idx] +' (Source) → '+ tgt_name[idx] +' (Target)')

                for i in range(len(cfg_idx)):
                    y=avg_value[i]
                    ax.plot(x,y,color=colors[i],marker=markers[i],linewidth=2,aa=True,markersize=4)

    if args.common_legend == 'True':
        fig.legend(labels = methods,loc = (0.042, 0.02), ncol = 7,fontsize=12)

    plt.savefig('result.pdf',bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)

