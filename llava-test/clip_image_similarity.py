#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Hugging Face CLIP 计算「主图 vs 多个文件夹内图片」的图像嵌入余弦相似度。
运行前可设置 CUDA_VISIBLE_DEVICES，勿在脚本内写死 GPU。
"""
import argparse
import glob
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

def get_all_images_from_folders(folder_paths: List[str]) -> List[str]:
    """
    从多个文件夹中获取所有图片文件路径
    
    Args:
        folder_paths: 文件夹路径列表
        
    Returns:
        所有图片文件路径列表
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    all_images = []
    
    for folder_path in folder_paths:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for ext in image_extensions:
                pattern = os.path.join(folder_path, ext)
                images = glob.glob(pattern)
                all_images.extend(images)
                # 也搜索子文件夹
                sub_pattern = os.path.join(folder_path, '**', ext)
                sub_images = glob.glob(sub_pattern, recursive=True)
                all_images.extend(sub_images)
        else:
            print(f"文件夹不存在或无效: {folder_path}")
    
    # 去重并排序
    all_images = list(set(all_images))
    all_images.sort()
    
    return all_images

def get_images_from_single_folder(folder_path: str) -> List[str]:
    """
    从单个文件夹中获取所有图片文件路径
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        该文件夹中所有图片文件路径列表
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    images = []
    
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for ext in image_extensions:
            pattern = os.path.join(folder_path, ext)
            found_images = glob.glob(pattern)
            images.extend(found_images)
            # 也搜索子文件夹
            sub_pattern = os.path.join(folder_path, '**', ext)
            sub_images = glob.glob(sub_pattern, recursive=True)
            images.extend(sub_images)
    else:
        print(f"文件夹不存在或无效: {folder_path}")
    
    # 去重并排序
    images = list(set(images))
    images.sort()
    
    return images


def load_clip_model(clip_path: str):
    """
    加载CLIP模型
    
    Args:
        clip_path: CLIP模型路径
        
    Returns:
        model, preprocess: CLIP模型和预处理函数
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
        
        print(f"正在从 {clip_path} 加载CLIP模型...")
        model = CLIPModel.from_pretrained(clip_path)
        processor = CLIPProcessor.from_pretrained(clip_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"模型已加载到设备: {device}")
        
        return model, processor
        
    except ImportError:
        print("错误: 需要安装transformers库")
        print("请运行: pip install transformers")
        return None, None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None

def load_and_process_image(image_path: str, processor, device: str) -> torch.Tensor:
    """
    加载并预处理图片
    
    Args:
        image_path: 图片路径
        processor: CLIP处理器
        device: 设备类型 (cuda/cpu)
        
    Returns:
        处理后的图片张量
    """
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(device)
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

def compute_similarity(image1_features: torch.Tensor, image2_features: torch.Tensor) -> float:
    """
    计算两张图片的余弦相似度
    
    Args:
        image1_features: 第一张图片的特征
        image2_features: 第二张图片的特征
        
    Returns:
        相似度分数 (0-1之间，1表示完全相同)
    """
    # 确保两个张量具有相同的数据类型
    if image1_features.dtype != image2_features.dtype:
        # 将两个张量都转换为float32进行计算
        image1_features = image1_features.to(torch.float32)
        image2_features = image2_features.to(torch.float32)
    
    image1_norm = F.normalize(image1_features, p=2, dim=1)
    image2_norm = F.normalize(image2_features, p=2, dim=1)
    similarity = torch.mm(image1_norm, image2_norm.t())
    return similarity.item()

def compute_main_vs_comparisons(main_image_path: str, comparison_image_paths: List[str], model, processor, device: str) -> List[Tuple[str, float]]:
    """
    计算主图像与所有对比图像的相似度
    
    Args:
        main_image_path: 主图像路径
        comparison_image_paths: 对比图像路径列表
        model: CLIP模型
        processor: CLIP处理器
        device: 设备类型
        
    Returns:
        相似度结果列表
    """
    results = []
    
    # 加载主图像特征
    main_feature = load_and_process_image(main_image_path, processor, device)
    if main_feature is None:
        print(f"无法加载主图像: {main_image_path}")
        return results
    
    with torch.no_grad():
        main_features = model.get_image_features(pixel_values=main_feature)
        # 确保主图像特征使用bfloat16类型，与文件夹特征保持一致
        if main_features.dtype != torch.bfloat16:
            main_features = main_features.to(torch.bfloat16)
    
    # 计算主图像与每个对比图像的相似度
    for comp_path in comparison_image_paths:
        if os.path.exists(comp_path):
            comp_feature = load_and_process_image(comp_path, processor, device)
            if comp_feature is not None:
                with torch.no_grad():
                    comp_features = model.get_image_features(pixel_values=comp_feature)
                    # 确保对比图像特征使用bfloat16类型，与主图像特征保持一致
                    if comp_features.dtype != torch.bfloat16:
                        comp_features = comp_features.to(torch.bfloat16)
                similarity = compute_similarity(main_features, comp_features)
                results.append((comp_path, similarity))
            else:
                print(f"跳过无效对比图像: {comp_path}")
        else:
            print(f"对比图像不存在: {comp_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用 CLIP 计算主图与多文件夹图片的相似度")
    parser.add_argument(
        "--clip_path",
        type=str,
        required=True,
        help="Hugging Face 格式 CLIP 模型目录或模型名",
    )
    parser.add_argument("--main_image", type=str, required=True, help="主图像路径")
    parser.add_argument(
        "--comparison_folders",
        nargs="+",
        required=True,
        help="一个或多个对比图片所在文件夹路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similarity_results.txt",
        help="输出结果文件路径",
    )

    args = parser.parse_args()
    
    # 加载CLIP模型
    model, processor = load_clip_model(args.clip_path)
    if model is None or processor is None:
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== 按文件夹分别比较模式 ===")
    print(f"主图像: {os.path.basename(args.main_image)}")
    print(f"对比文件夹数量: {len(args.comparison_folders)}\n")
    
    # 存储所有文件夹的结果
    all_folder_results = []
    
    # 分别处理每个文件夹
    for folder_path in args.comparison_folders:
        folder_name = os.path.basename(folder_path.rstrip('/'))
        print(f"\n--- 处理文件夹: {folder_name} ---")
        print(f"路径: {folder_path}")
        
        # 获取该文件夹中的所有图片
        comparison_images = get_images_from_single_folder(folder_path)
        print(f"找到 {len(comparison_images)} 张图片")
        
        if not comparison_images:
            print(f"文件夹 {folder_name} 中没有找到任何图片，跳过")
            continue
        
        # 计算该文件夹的相似度
        print(f"正在计算主图像与该文件夹中 {len(comparison_images)} 张图片的相似度...")
        similarities = compute_main_vs_comparisons(args.main_image, comparison_images, model, processor, device)
        
        if not similarities:
            print(f"文件夹 {folder_name} 没有计算出相似度结果，跳过")
            continue
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 计算该文件夹的平均相似度
        avg_similarity = sum(sim for _, sim in similarities) / len(similarities)
        
        # 存储结果
        folder_result = {
            'folder_name': folder_name,
            'folder_path': folder_path,
            'image_count': len(similarities),
            'avg_similarity': avg_similarity,
            'similarities': similarities
        }
        all_folder_results.append(folder_result)
        
        # 输出该文件夹的结果
        print(f"文件夹 {folder_name} 结果:")
        print(f"  - 图片数量: {len(similarities)}")
        print(f"  - 平均相似度: {avg_similarity:.4f}")
        print(f"  - 最高相似度: {similarities[0][1]:.4f} ({os.path.basename(similarities[0][0])})")
        print(f"  - 最低相似度: {similarities[-1][1]:.4f} ({os.path.basename(similarities[-1][0])})")
    
    # 输出汇总结果
    print("\n" + "=" * 60)
    print("=== 汇总结果 ===")
    print(f"主图像: {os.path.basename(args.main_image)}")
    print(f"成功处理的文件夹数: {len(all_folder_results)}")
    print("\n各文件夹平均相似度排名:")
    
    # 按平均相似度排序
    all_folder_results.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    for i, result in enumerate(all_folder_results, 1):
        print(f"{i}. {result['folder_name']}: {result['avg_similarity']:.4f} (图片数: {result['image_count']})")
    
    # 保存结果到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("主图像与各文件夹对比图像相似度结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"主图像: {os.path.basename(args.main_image)}\n")
        f.write(f"主图像路径: {args.main_image}\n")
        f.write(f"成功处理的文件夹数: {len(all_folder_results)}\n\n")
        
        f.write("各文件夹详细结果:\n")
        f.write("-" * 60 + "\n")
        
        for result in all_folder_results:
            f.write(f"\n文件夹: {result['folder_name']}\n")
            f.write(f"路径: {result['folder_path']}\n")
            f.write(f"图片数量: {result['image_count']}\n")
            f.write(f"平均相似度: {result['avg_similarity']:.4f}\n")
            f.write(f"最高相似度: {result['similarities'][0][1]:.4f} ({os.path.basename(result['similarities'][0][0])})\n")
            f.write(f"最低相似度: {result['similarities'][-1][1]:.4f} ({os.path.basename(result['similarities'][-1][0])})\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("按平均相似度排名:\n")
        for i, result in enumerate(all_folder_results, 1):
            f.write(f"{i}. {result['folder_name']}: {result['avg_similarity']:.4f} (图片数: {result['image_count']})\n")
    
    print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main()