import easypyxl



result_dir="results/"
target_models=['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit']
source_model_names=['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3']

# Table 1 
cfg_idx=[(401,'DI'),(402,'RDI'),(605,'SI-RDI'),(604,'VT-RDI'),(403,'Admix-RDI'),(579,'ODI'),(578,'CFM-RDI')]
display_idx=['vgg16','ResNet18', 'ResNet50', 'DenseNet121','xception','mobilenet_v2','efficientnet_b0',
        'inception_resnet_v2', 'inception_v3','inception_v4_timm'] 

# Table 2
# cfg_idx=[(401,'DI'),(402,'RDI'),(605,'SI-RDI'),(604,'VT-RDI'),(403,'Admix-RDI'),(579,'ODI'),(578,'CFM-RDI')]
# display_idx=['resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'] 

# Table 3
# -> gen_table_cifar10.py

# Table 4
# cfg_idx=[(578,'V/V/V'),(592,'V/V/N'),(591,'V/N/V'),(603,'V/N/N'),(593,'N/V/V'),(594,'N/N/V')]
# display_idx=['xception','mobilenet_v2','efficientnet_b0','inception_resnet_v2','inception_v4_timm' ,'vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'] 

# Appendix Table 1
# cfg_idx=[(401,'DI'),(402,'RDI'),(403,'Admix-RDI'),(1412,'SI-Admix-RDI'),(605,'SI-RDI'),(604,'VT-RDI'),(579,'ODI'),(578,'CFM-RDI'),(1408,'SI-CFM-RDI')]
# display_idx=['vgg16','ResNet18', 'ResNet50', 'DenseNet121','xception','mobilenet_v2','efficientnet_b0',
#         'inception_resnet_v2', 'inception_v3','inception_v4_timm'] 

# Appendix Table 2
# cfg_idx=[(401,'DI'),(402,'RDI'),(403,'Admix-RDI'),(1412,'SI-Admix-RDI'),(605,'SI-RDI'),(604,'VT-RDI'),(579,'ODI'),(578,'CFM-RDI'),(1408,'SI-CFM-RDI')]
# display_idx=['resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'] 

# Appendix Table 3
# -> gen_table_cifar10.py

# Appendix Table 4
# cfg_idx=[(709,'MI-TI'),(402,'RDI'),(605,'SI-RDI'),(604,'VT-RDI'),(579,'ODI'),(703,'ODI-RDI-MI-TI'),(403,'Admix-RDI'),(1412,'SI-Admix-RDI'),(1410,'VT-Admix-RDI'),(702,'CFM-MI-TI'),(578,'CFM-RDI'),(580,'CFM-ODI'),(1408,'SI-CFM-RDI'),(1409,'VT-CFM-RDI'),(598,'Admix-CFM-RDI'),(1406,'SI-Admix-CFM-RDI')] 
# display_idx=['xception','mobilenet_v2','efficientnet_b0','inception_resnet_v2','inception_v4_timm' ,'vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'] 

# Appendix Table 5
#cfg_idx=[(583,'0.05/0.5'),(586,'0.05/0.75'),(588,'0.05/1.'),(584,'0.1/0.5'),(578,'0.1/0.75'),(589,'0.1/1.'),(585,'0.15/0.5'),(587,'0.15/0.75'),(590,'0.15/1.')]
#display_idx=['xception','mobilenet_v2','efficientnet_b0','inception_resnet_v2','inception_v4_timm' ,'vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'] 


workbook=[]
cursor=[]
time_cursor=[]

for i in range(len(cfg_idx)):
    workbook.append(easypyxl.Workbook(result_dir+"NEW_EXP_"+str(cfg_idx[i][0])+".xlsx", backup=False))
    time_cursor.append(workbook[i].new_cursor("Experiment Info", "B5",1, reader=True))
    cursor.append(workbook[i].new_cursor("Succ_300", "C2", len(target_models), reader=True))
    title=cursor[i].read_line()
    if i==len(cfg_idx)-1:
        for j in display_idx:
            idx_j=target_models.index(j)
            print(title[idx_j],end='& ')
        print('')

end_string=' & '
for s in range(len(source_model_names)):
    print(source_model_names[s])

    for i in range(len(cfg_idx)):
        items=cursor[i].read_line()
        compuation_time=time_cursor[i].read_cell()
        print(cfg_idx[i][1],end=end_string)
        sum_acc=0
        for j in display_idx:
            #print(j)
            idx_j=target_models.index(j)
            sum_acc+=float(items[idx_j])
            print(f"{items[idx_j]:2.1f}",end=end_string)

        print(f"{float(sum_acc/len(display_idx)):2.1f} ",end='& ')
        print(f"{float(compuation_time):2.2f} ",end='\\\\')
        print('')