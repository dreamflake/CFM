
exp_configuration={
    401 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'DI-TI-MI': 'DTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        
        #####################################
        'comment':'DI-TI-MI'
    },
        402 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'RDI-TI-MI': 'RTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'RDI-TI-MI'
    },
    403 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'Admix-RDI-TI-MI': 'IRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'Admix-RDI-TI-MI'
    },
                      578:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI Main Result'
    },
                 579:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'ODI-TI-MI': 'OTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        # ODI params
        'shininess':0.5,
        'source_3d_models':['pack','pillow','book'],
        'rand_elev':(-35,35),
        'rand_azim':(-35,35),
        'rand_angle':(-35,35),
        'min_dist':0.8, 'rand_dist':0.4,
        'light_location':[0.0, 0.0,4.0],
        'rand_light_location':4,
        'rand_ambient_color':0.3,
        'ambient_color':0.6,
        'rand_diffuse_color':0.5,
        'diffuse_color':0.0,
        'specular_color':0.0,
        'background_type':'random_pixel',
        'texture_type':'random_solid',
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'ODI Main Result'
    },
                     580:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-ODI-TI-MI': 'COTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        # ODI params
        'shininess':0.5,
        'source_3d_models':['pack','pillow','book'],
        'rand_elev':(-35,35),
        'rand_azim':(-35,35),
        'rand_angle':(-35,35),
        'min_dist':0.8, 'rand_dist':0.4,
        'light_location':[0.0, 0.0,4.0],
        'rand_light_location':4,
        'rand_ambient_color':0.3,
        'ambient_color':0.6,
        'rand_diffuse_color':0.5,
        'diffuse_color':0.0,
        'specular_color':0.0,
        'background_type':'random_pixel',
        'texture_type':'random_solid',
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-ODI Main Result'
    },
      583:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.5,
        'mix_prob':0.05,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.05/0.5'
    },
          584:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.5,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.1/0.5'
    },
          585:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.5,
        'mix_prob':0.15,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.15/0.5'
    },
          586:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.05,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.05/0.75'
    },
          587:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.15,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.15/0.75'
    },
          588:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':1.,
        'mix_prob':0.05,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.05/1.'
    },
              589:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':1.,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.1/1.'
    },
              590:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':1.,
        'mix_prob':0.15,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'0.15/1.'
    },
              591:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':False,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'V/N/V'
    },
       592:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'None', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'V/V/N'
    },
       593:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'A', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'N/V/V'
    },
      594:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'A', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':False,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'N/N/V'
    },
              598:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"

        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'], #'resnet18_l2_eps0_1','resnet18_l2_eps0_1',
        'attack_methods': {'CFM-Admix-RDI-TI-MI': 'CIRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################

        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': Shuffle unconditionally, 'NonSelfShuffle': Shuffle to always set different index
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0.0, # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'start_layer_ratio':0.0,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        #####################################
        'comment':''
    },
          603:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"

        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        # Self-mix-up in feature domain
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'None', # 'None': Without shuffle, 'SelfShuffle': Shuffle unconditionally, 'NonSelfShuffle': Shuffle to always set different index
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'start_layer_ratio':0.0,
        'channelwise':False,
        'mixup_layer':'conv_linear_include_last', # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        #####################################
        'comment':'V/N/N'
    },
      
      604 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'VT-RDI-TI-MI': 'VRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'None', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':False,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'VT-RDI-TI-MI'
    },
          605 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'SI-RDI-TI-MI': 'SRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'None', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':False,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'SI-RDI-TI-MI'
    },
      




        # CIFAR-10
         801:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'DI-TI-MI': 'DTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        
        #####################################
        'comment':'DI-TI-MI'
    },
     920:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'RDI-TI-MI': 'RTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'RDI-TI-MI'
    },
                             923:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.25,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI-TI-MI'
    },
         931:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'Admix-RDI-TI-MI': 'IRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':1.,
        'mix_prob':0.4,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'Admix-RDI-TI-MI'
    },
      934:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        

        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'VT-RDI-TI-MI': 'VRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':1.,
        'mix_prob':0.4,
        'divisor':4,
        
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'VT-RDI-TI-MI'
    },
      935:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'SI-RDI-TI-MI': 'SRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
          'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':1.,
        'mix_prob':0.4,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'SI-RDI-TI-MI'
    },
    
       1406 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        

        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'Admix-CFM-RDI-TI-MI-SI': 'CIRTMS'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'Admix-CFM-RDI-TI-MI-SI'
    },

            1408 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
         

        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI-SI': 'CRTMS'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI-TI-MI-SI'
    },
       1409 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
         
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI-VT': 'CRTMV'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI-TI-MI-VT'
    },
           1410 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
         

        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'Admix-RDI-TI-MI-VT': 'IRTMV'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'Admix-RDI-TI-MI-VT'
    },
    1412 :{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'], #'resnet18_l2_eps0_1','resnet18_l2_eps0_1',
        'attack_methods': {'Admix-RDI-TI-MI-SI': 'IRTMS'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, #'protion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        # Self-mix-up in feature domain
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': Shuffle unconditionally, 'NonSelfShuffle': Shuffle to always set different index
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'start_layer_ratio':0.0,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        #####################################
        'comment':'Admix-RDI-TI-MI-SI'
    },
    

     1501:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
         
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'SI-Admix-RDI-TI-MI': 'IRTMS'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.25,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'SI-Admix-RDI-TI-MI'
    },
     1505:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
         
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'SI-CFM-RDI-TI-MI': 'CRTMS'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.25,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'SI-CFM-RDI-TI-MI'
    },
         1506:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
         
        
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'VT-CFM-RDI-TI-MI': 'CRTMV'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.25,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last',
        #####################################
        'comment':'VT-CFM-RDI-TI-MI'
    },
    701:{
        'dataset':'Cifar10',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50','vgg16_bn','densenet121','inception_v3'],
        # 
        'target_model_names':['vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121',
        'ens3_res20_baseline','ens3_res20_adp','ens3_res20_gal','ens3_res20_dverge'],
        #
        'attack_methods': {'CFM-TI-MI': 'CTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.25,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last',
        #####################################
        'comment':'CFM-TI-MI Cifar10'
    },
    7022:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-TI-MI': 'CTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-TI-MI ImageNet'
    },
    702:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-TI-MI': 'CTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-TI-MI ImageNet'
    },
        703:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'ODI-RDI-TI-MI': 'ORTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        # ODI params
        'shininess':0.5,
        'source_3d_models':['pack','pillow','book'],
        'rand_elev':(-35,35),
        'rand_azim':(-35,35),
        'rand_angle':(-35,35),
        'min_dist':0.8, 'rand_dist':0.4,
        'light_location':[0.0, 0.0,4.0],
        'rand_light_location':4,
        'rand_ambient_color':0.3,
        'ambient_color':0.6,
        'rand_diffuse_color':0.5,
        'diffuse_color':0.0,
        'specular_color':0.0,
        'background_type':'random_pixel',
        'texture_type':'random_solid',
        #####################################
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'ODI-RDI-TI-MI ImageNet'
    },
    704:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI Main Result - BS40'
    },
    705:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI Main Result - BS30'
    },
     706:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI Main Result - BS10'
    },
    707:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-MI': 'CRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI Main Result - BS5'
    },
        708:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'CFM-RDI-TI-NI': 'CRTN'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'CFM-RDI-NI-TI - BS20'
    },
      709:{
        'dataset':'ImageNet',
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'TI-MI': 'TM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'MI-TI'
    },
     714 :{
        'dataset':'ImageNet',
        'targeted':False,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"
        
        
        
        'source_model_names':['resnet50_l2_eps0_1','ResNet50','DenseNet121','inception_v3'],
        'target_model_names':['ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0',
        'DenseNet121', 'mobilenet_v2','inception_resnet_v2',
        'inception_v4_timm','xception','resnet50_l2_eps0_1','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'],
        'attack_methods': {'VT-RDI-TI-MI': 'VRTM'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        #####################################
        
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'None', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':False,
        'mixup_layer':'conv_linear_include_last', 
        #####################################
        'comment':'VT-RDI-TI-MI'
    },
}