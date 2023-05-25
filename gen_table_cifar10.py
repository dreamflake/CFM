import easypyxl


result_dir="results/"
target_models=['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_advt','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge']
source_model_names=['resnet50','vgg16_bn','densenet121','inception_v3']


# Table 3
cfg_idx=[(801,'DI'),(920,'RDI'), (935,'SI-RDI'),(934,'VT-RDI'),(931,'Admix-RDI'),(923,'CFM-RDI')]
display_idx=['vgg16_bn', 'resnet18', 'mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'] 

# Appendix Table 3
# cfg_idx=[(801,'DI'),(920,'RDI'), (935,'SI-RDI'),(934,'VT-RDI'),(931,'Admix-RDI'),(1501,'SI-Admix-RDI'),(923,'CFM-RDI'),(1505,'SI-CFM-RDI')] 
# display_idx=['vgg16_bn', 'resnet18', 'mobilenet_v2','inception_v3','densenet121',
#         'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'] 


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
            print(title[idx_j],end=', ')
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