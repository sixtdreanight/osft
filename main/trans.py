#!/usr/bin/env python3
"""
TopologyGAN TF -> PyTorch 权重转换脚本
将TensorFlow checkpoint转换为PyTorch state_dict
"""

import tensorflow as tf
import torch
import numpy as np
from collections import OrderedDict

def convert_topologygan_to_pytorch(tf_checkpoint_path, output_path):
    """
    转换TopologyGAN权重到PyTorch格式
    
    Args:
        tf_checkpoint_path: TF checkpoint路径 (如 './checkpoint/model_gan_se_res_unet/model_gan_se_res_unet-498')
        output_path: 输出PyTorch权重路径
    """
    
    # 启动TF会话加载权重
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        # 加载meta graph和权重
        saver = tf.train.import_meta_graph(tf_checkpoint_path + '.meta')
        saver.restore(sess, tf_checkpoint_path)
        
        # 获取所有变量
        all_vars = tf.global_variables()
        
        # 构建PyTorch state_dict
        pytorch_state_dict = OrderedDict()
        
        # 变量名映射表（TF -> PyTorch）
        name_mapping = {
            # Generator encoder
            'g_e1_conv/w:0': 'encoder.0.weight',
            'g_e1_conv/biases:0': 'encoder.0.bias',
            'g_e2_conv/w:0': 'encoder.2.weight',
            'g_e2_bn/beta:0': 'encoder.3.bias',  # BN beta
            'g_e2_bn/gamma:0': 'encoder.3.weight',  # BN gamma
            'g_e3_conv/w:0': 'encoder.5.weight',
            'g_e3_bn/beta:0': 'encoder.6.bias',
            'g_e3_bn/gamma:0': 'encoder.6.weight',
            
            # SE-ResNet blocks（需要手动映射32个residual block）
            # 格式: rsenet{i}/x1_conv/w:0 -> res_blocks.{i-1}.conv1.weight
            
            # Decoder
            'g_d1/w:0': 'decoder.0.weight',
            'g_d1/biases:0': 'decoder.0.bias',
            'g_d1_bn/beta:0': 'decoder.1.bias',
            'g_d1_bn/gamma:0': 'decoder.1.weight',
            # ... 继续映射其他层
        }
        
        for var in all_vars:
            name = var.name
            if name in name_mapping:
                pytorch_name = name_mapping[name]
                value = sess.run(var)
                
                # 转换维度（TF: [H,W,C_in,C_out], PyTorch: [C_out,C_in,H,W]）
                if len(value.shape) == 4:  # Conv权重
                    value = np.transpose(value, (3, 2, 0, 1))
                elif len(value.shape) == 2:  # Linear权重
                    value = np.transpose(value, (1, 0))
                
                pytorch_state_dict[pytorch_name] = torch.from_numpy(value)
                print(f"Converted: {name} -> {pytorch_name}, shape: {value.shape}")
        
        # 保存
        torch.save(pytorch_state_dict, output_path)
        print(f"\nSaved PyTorch weights to: {output_path}")
        
        return pytorch_state_dict


if __name__ == '__main__':
    # 使用示例
    tf_path = './checkpoint/model_gan_se_res_unet/model_gan_se_res_unet-498'
    pt_path = 'topologygan_se_res_unet_498.pth'
    convert_topologygan_to_pytorch(tf_path, pt_path)